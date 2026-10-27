"""PatchCore Training Pipeline.

Extracts dense patch embeddings from nominal (good) manufacturing images,
constructs a coreset memory bank using greedy minimax facility location,
and serializes the trained model weights and configuration.
"""

import os
import sys
import time
import json
import yaml
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from src.dataset import get_dataloaders, MVTecDataset
from src.model import PatchCore
from src.download_mvtec import check_dataset_exists


def train_patchcore(
    category: str = "bottle",
    data_dir: str = "data/mvtec",
    config_path: str = None,
    backbone: str = "resnet18",
    coreset_ratio: float = 0.02,
    project_dim: int = 128,
    img_size: int = 224,
    crop_size: int = 224,
    batch_size: int = 16,
    device: str = "auto",
    checkpoint_dir: str = None,
) -> Path:
    """Trains a PatchCore model on nominal samples and saves the checkpoint."""
    # Optional YAML config loading
    if config_path and Path(config_path).is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        category = cfg.get("dataset", {}).get("category", category)
        data_dir = cfg.get("dataset", {}).get("data_dir", data_dir)
        backbone = cfg.get("model", {}).get("backbone", backbone)
        coreset_ratio = cfg.get("model", {}).get("coreset_sampling_ratio", coreset_ratio)
        project_dim = cfg.get("model", {}).get("project_dim", project_dim)
        img_size = cfg.get("dataset", {}).get("img_size", img_size)
        crop_size = cfg.get("dataset", {}).get("crop_size", crop_size)
        batch_size = cfg.get("dataset", {}).get("batch_size", batch_size)
        device = cfg.get("model", {}).get("device", device)

    cat_path = Path(data_dir) / category
    status = check_dataset_exists(cat_path)
    if not status["exists"]:
        raise FileNotFoundError(
            f"Dataset for '{category}' not found at {cat_path}.\n"
            f"Please run: python src/download_mvtec.py --category {category}\n"
            f"Reason: {status['reason']}"
        )

    print("=" * 70)
    print(f"PATCHCORE TRAINING PIPELINE - CATEGORY: {category.upper()}")
    print("=" * 70)
    print(f"Data directory:      {cat_path.resolve()}")
    print(f"Normal train images: {status['train_count']}")
    print(f"Backbone:            {backbone}")
    print(f"Coreset ratio:       {coreset_ratio * 100:.1f}%")
    print(f"Hardware device:     {'cuda' if (device == 'auto' and torch.cuda.is_available()) or device == 'cuda' else 'cpu'}")

    # Build DataLoader
    train_loader, _ = get_dataloaders(
        root_dir=data_dir,
        category=category,
        img_size=img_size,
        crop_size=crop_size,
        batch_size=batch_size,
        num_workers=0,
    )

    # Initialize PatchCore model
    model = PatchCore(
        backbone_name=backbone,
        layers=("layer2", "layer3"),
        patch_size=3,
        patch_stride=1,
        coreset_sampling_ratio=coreset_ratio,
        project_dim=project_dim,
        device=device,
    )

    # Extract patch embeddings from all normal training images
    print(f"\n[1/3] Extracting patch embeddings from {status['train_count']} normal images...")
    start_time = time.time()
    collected_patches = []

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Feature extraction"):
            images = batch["image"].to(model.device)
            embeddings, feat_shape = model.feature_extractor(images)
            # embeddings: (B, N_patches, D)
            b, n_p, d = embeddings.shape
            collected_patches.append(embeddings.reshape(b * n_p, d).cpu())

    all_patches = torch.cat(collected_patches, dim=0)
    extract_time = time.time() - start_time
    print(f"Feature extraction completed in {extract_time:.2f}s.")
    print(f"Total patch vectors: {all_patches.shape[0]:,} with feature dim: {all_patches.shape[1]}")

    # Fit memory bank via greedy k-center coreset selection
    print(f"\n[2/3] Constructing coreset memory bank...")
    bank_start = time.time()
    model.fit_memory_bank(all_patches)
    bank_time = time.time() - bank_start
    print(f"Memory bank selection completed in {bank_time:.2f}s.")

    # Save model checkpoint
    if checkpoint_dir is None:
        checkpoint_dir = Path("checkpoints") / category
    else:
        checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_path = checkpoint_dir / f"patchcore_{category}.pt"
    config_save_path = checkpoint_dir / "config.json"

    meta = {
        "category": category,
        "backbone": backbone,
        "train_samples": status["train_count"],
        "coreset_ratio": coreset_ratio,
        "coreset_bank_size": model.memory_bank.shape[0] if model.memory_bank is not None else 0,
        "feature_dim": model.feature_dim,
        "feat_shape": feat_shape,
        "train_duration_sec": round(extract_time + bank_time, 2),
    }

    model.save(save_path, extra_metadata=meta)
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[3/3] Training finished successfully!")
    print(f"Model saved to:       {save_path.resolve()}")
    print(f"Config saved to:      {config_save_path.resolve()}")
    print("=" * 70)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore on MVTec category")
    parser.add_argument("--category", type=str, default="bottle", help="Product category")
    parser.add_argument("--data_dir", type=str, default="data/mvtec", help="MVTec dataset directory")
    parser.add_argument("--config", type=str, default="configs/bottle_config.yaml", help="Path to config YAML")
    parser.add_argument("--backbone", type=str, default="resnet18", help="ResNet backbone (resnet18, resnet50, wide_resnet50_2)")
    parser.add_argument("--coreset_ratio", type=float, default=0.02, help="Coreset subsampling fraction (e.g. 0.02 = 2 percent)")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")
    args = parser.parse_args()

    train_patchcore(
        category=args.category,
        data_dir=args.data_dir,
        config_path=args.config if Path(args.config).is_file() else None,
        backbone=args.backbone,
        coreset_ratio=args.coreset_ratio,
        device=args.device,
    )


if __name__ == "__main__":
    main()

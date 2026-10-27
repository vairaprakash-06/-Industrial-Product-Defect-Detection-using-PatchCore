"""Configuration management and project paths for Industrial Defect Detection."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Base project directories
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "mvtec"
DATASET_DIR = DATA_DIR  # Alias for standard naming
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODELS_DIR = CHECKPOINTS_DIR  # Alias for standard naming
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLES_DIR = PROJECT_ROOT / "samples"

# Ensure output directory exists
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Official MVTec AD Categories
SUPPORTED_CATEGORIES = [
    "bottle",
    "metal_nut",
    "capsule",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
    "cable",
    "carpet",
    "hazelnut",
    "leather",
    "tile",
    "wood",
    "grid",
]

DEFAULT_CATEGORY = "bottle"


def load_config(category: str = DEFAULT_CATEGORY, config_path: Optional[str or Path] = None) -> Dict[str, Any]:
    """Loads YAML configuration for the specified product category."""
    if config_path is not None and Path(config_path).is_file():
        cfg_file = Path(config_path)
    else:
        # Check category specific config
        cat_file = CONFIGS_DIR / f"{category}_config.yaml"
        if cat_file.is_file():
            cfg_file = cat_file
        else:
            cfg_file = CONFIGS_DIR / "default_config.yaml"

    if cfg_file.is_file():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    # Defaults fallback
    dataset_cfg = cfg.setdefault("dataset", {})
    dataset_cfg.setdefault("category", category)
    dataset_cfg.setdefault("data_dir", str(DATA_DIR))
    dataset_cfg.setdefault("img_size", 224)
    dataset_cfg.setdefault("crop_size", 224)
    dataset_cfg.setdefault("batch_size", 16)
    dataset_cfg.setdefault("num_workers", 0)

    model_cfg = cfg.setdefault("model", {})
    model_cfg.setdefault("backbone", "resnet18")
    model_cfg.setdefault("layers", ["layer2", "layer3"])
    model_cfg.setdefault("patch_size", 3)
    model_cfg.setdefault("patch_stride", 1)
    model_cfg.setdefault("coreset_sampling_ratio", 0.02)
    model_cfg.setdefault("project_dim", 128)
    model_cfg.setdefault("device", "auto")

    eval_cfg = cfg.setdefault("evaluation", {})
    eval_cfg.setdefault("sigma", 4.0)
    eval_cfg.setdefault("n_neighbors", 9)
    eval_cfg.setdefault("calc_pixel_auroc", True)

    ckpt_cfg = cfg.setdefault("checkpoints", {})
    ckpt_cfg.setdefault("checkpoint_dir", str(CHECKPOINTS_DIR / category))
    ckpt_cfg.setdefault("model_filename", f"patchcore_{category}.pt")
    ckpt_cfg.setdefault("metrics_filename", "metrics.json")

    return cfg

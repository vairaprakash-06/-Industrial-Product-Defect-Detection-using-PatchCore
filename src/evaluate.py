"""Evaluation Pipeline for PatchCore Model.

Calculates:
- Image-level AUROC
- Pixel-level AUROC (using ground-truth defect masks)
- Optimal decision threshold (maximizing F1 score)
- Defect-type breakdown (e.g. broken_large, contamination, good)
- Updates checkpoint with calibrated threshold and saves metrics JSON.
"""

import sys
import time
import json
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import PatchCore
from src.utils import find_optimal_threshold, compute_auroc_metrics


def evaluate_patchcore(
    category: str = "bottle",
    data_dir: str = "data/mvtec",
    checkpoint_path: str = None,
    batch_size: int = 16,
    device: str = "auto",
    calc_pixel_auroc: bool = True,
) -> dict:
    """Evaluates the PatchCore model on MVTec test split and updates threshold."""
    if checkpoint_path is None:
        checkpoint_path = Path("checkpoints") / category / f"patchcore_{category}.pt"
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}.\n"
            f"Please run: python src/train.py --category {category}"
        )

    print("=" * 70)
    print(f"PATCHCORE EVALUATION PIPELINE - CATEGORY: {category.upper()}")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path.resolve()}")

    model = PatchCore.load_from_checkpoint(checkpoint_path, device=device)
    print(f"Model backbone: {model.backbone_name}, Memory bank shape: {tuple(model.memory_bank.shape)}")

    # Load test dataset
    _, test_loader = get_dataloaders(
        root_dir=data_dir,
        category=category,
        batch_size=1,  # batch size 1 for precise per-image heatmap and mask evaluation
        num_workers=0,
    )

    image_labels = []
    image_scores = []
    defect_type_scores = defaultdict(list)
    defect_type_labels = defaultdict(list)

    pixel_labels_list = []
    pixel_scores_list = []

    print(f"\nRunning inference on {len(test_loader)} test images...")
    start_time = time.time()

    for idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        img = batch["image"]  # (1, 3, H, W)
        gt_mask = batch["mask"][0, 0].numpy()  # (H, W)
        label = int(batch["label"].item())  # 0 or 1
        defect_type = batch["defect_type"][0]

        res = model.predict_image(img, return_heatmap=calc_pixel_auroc)
        score = res["anomaly_score"]
        heatmap = res["anomaly_map"]

        image_labels.append(label)
        image_scores.append(score)
        defect_type_scores[defect_type].append(score)
        defect_type_labels[defect_type].append(label)

        if calc_pixel_auroc and heatmap is not None:
            # Flatten 2D mask and heatmap for pixel-level AUROC
            pixel_labels_list.append(gt_mask.flatten().astype(np.uint8))
            pixel_scores_list.append(heatmap.flatten().astype(np.float32))

    total_time = time.time() - start_time
    avg_latency_ms = (total_time / len(test_loader)) * 1000

    y_true = np.array(image_labels)
    y_scores = np.array(image_scores)

    # Pixel-level AUROC
    if calc_pixel_auroc and len(pixel_labels_list) > 0:
        flat_pixel_gt = np.concatenate(pixel_labels_list)
        flat_pixel_pred = np.concatenate(pixel_scores_list)
    else:
        flat_pixel_gt, flat_pixel_pred = None, None

    auroc_results = compute_auroc_metrics(
        image_labels=y_true,
        image_scores=y_scores,
        pixel_labels=flat_pixel_gt,
        pixel_scores=flat_pixel_pred,
    )

    # Optimal threshold based on F1
    best_threshold, threshold_metrics = find_optimal_threshold(y_true, y_scores)

    # Update model threshold and score bounds
    model.threshold = best_threshold
    model.min_score = float(np.min(y_scores))
    model.max_score = float(np.max(y_scores))
    model.save(checkpoint_path)

    # Defect breakdown
    defect_breakdown = {}
    for d_type in sorted(defect_type_scores.keys()):
        scores_arr = np.array(defect_type_scores[d_type])
        preds_arr = (scores_arr >= best_threshold).astype(int)
        labels_arr = np.array(defect_type_labels[d_type])
        correct = int(np.sum(preds_arr == labels_arr))
        total_d = len(labels_arr)
        defect_breakdown[d_type] = {
            "sample_count": total_d,
            "accuracy": round(correct / total_d, 4),
            "mean_anomaly_score": round(float(np.mean(scores_arr)), 4),
            "min_score": round(float(np.min(scores_arr)), 4),
            "max_score": round(float(np.max(scores_arr)), 4),
        }

    summary = {
        "category": category,
        "test_samples": len(y_true),
        "normal_samples": int(np.sum(y_true == 0)),
        "anomalous_samples": int(np.sum(y_true == 1)),
        "image_auroc": round(auroc_results.get("image_auroc", 0.0), 4),
        "pixel_auroc": round(auroc_results.get("pixel_auroc", 0.0), 4) if calc_pixel_auroc else "N/A",
        "optimal_threshold": round(best_threshold, 4),
        "f1_score": round(threshold_metrics["best_f1"], 4),
        "precision": round(threshold_metrics["precision"], 4),
        "recall": round(threshold_metrics["recall"], 4),
        "specificity": round(threshold_metrics["specificity"], 4),
        "confusion_matrix": {
            "tp": threshold_metrics["tp"],
            "fp": threshold_metrics["fp"],
            "tn": threshold_metrics["tn"],
            "fn": threshold_metrics["fn"],
        },
        "average_inference_ms": round(avg_latency_ms, 2),
        "defect_breakdown": defect_breakdown,
    }

    metrics_file = checkpoint_path.parent / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Output formatted report
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Image-Level AUROC:      {summary['image_auroc'] * 100:.2f}%")
    if calc_pixel_auroc:
        print(f"Pixel-Level AUROC:      {summary['pixel_auroc'] * 100:.2f}%")
    print(f"Optimal Threshold:      {summary['optimal_threshold']:.4f}")
    print(f"F1 Score:               {summary['f1_score']:.4f}")
    print(f"Precision:              {summary['precision']:.4f}")
    print(f"Recall (Sensitivity):   {summary['recall']:.4f}")
    print(f"Specificity:            {summary['specificity']:.4f}")
    print(f"Confusion Matrix:       TP={threshold_metrics['tp']}, FP={threshold_metrics['fp']}, TN={threshold_metrics['tn']}, FN={threshold_metrics['fn']}")
    print(f"Mean Latency / Image:   {summary['average_inference_ms']:.2f} ms")
    print("-" * 70)
    print("Performance by Defect Subtype:")
    for d_name, d_stat in defect_breakdown.items():
        print(f"  - {d_name:<16}: Acc={d_stat['accuracy']*100:.1f}%, Mean Score={d_stat['mean_anomaly_score']:.4f} (n={d_stat['sample_count']})")
    print("=" * 70)
    print(f"Metrics saved to: {metrics_file.resolve()}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate PatchCore on MVTec test set")
    parser.add_argument("--category", type=str, default="bottle", help="Product category")
    parser.add_argument("--data_dir", type=str, default="data/mvtec", help="Dataset directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--no_pixel", action="store_true", help="Skip pixel-level AUROC")
    args = parser.parse_args()

    evaluate_patchcore(
        category=args.category,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
        calc_pixel_auroc=not args.no_pixel,
    )


if __name__ == "__main__":
    main()

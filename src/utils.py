"""Utility functions for visualization, metrics, report generation, and post-processing."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix


def normalize_heatmap(
    heatmap: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """Normalizes an anomaly heatmap to [0, 1]."""
    if min_val is None:
        min_val = float(np.min(heatmap))
    if max_val is None:
        max_val = float(np.max(heatmap))

    if max_val - min_val > 1e-8:
        norm = (heatmap - min_val) / (max_val - min_val)
    else:
        norm = np.zeros_like(heatmap)
    return np.clip(norm, 0.0, 1.0)


def generate_defect_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    threshold: Optional[float] = None,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    draw_contours: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates an anomaly heatmap, overlaid composite image, and defect mask.
    Args:
        image_rgb: RGB image numpy array uint8 (H, W, 3)
        heatmap: 2D float array (H, W) of anomaly scores
        threshold: Score threshold to segment defect regions (if None, use 50% max)
        alpha: Blending weight for heatmap overlay
        colormap: OpenCV colormap constant (COLORMAP_JET or COLORMAP_TURBO)
        draw_contours: Whether to draw contour outlines around defect areas
    Returns:
        overlay: (H, W, 3) uint8 RGB blended image
        heatmap_colored: (H, W, 3) uint8 RGB colorized heatmap
        defect_mask: (H, W) uint8 binary mask {0, 255}
    """
    h, w = image_rgb.shape[:2]
    # Resize heatmap if shape does not match
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    norm_map = normalize_heatmap(heatmap)
    heatmap_uint8 = (norm_map * 255).astype(np.uint8)

    # Colorize heatmap (OpenCV returns BGR, convert to RGB)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    # Blend original image and heatmap
    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_colored, alpha, 0)

    # Binary defect mask
    if threshold is not None:
        defect_mask = (heatmap >= threshold).astype(np.uint8) * 255
    else:
        # Default mask threshold at 50% relative anomaly
        defect_mask = (norm_map >= 0.5).astype(np.uint8) * 255

    if draw_contours and np.any(defect_mask > 0):
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw high-visibility contours (bright yellow/lime green) and bounding boxes
        contour_img = overlay.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) > 25:  # filter negligible noise specks
                cv2.drawContours(contour_img, [cnt], -1, (255, 230, 0), 2)
                x, y, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(contour_img, (x, y), (x + bw, y + bh), (255, 50, 50), 2)
        overlay = contour_img

    return overlay, heatmap_colored, defect_mask


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Finds the optimal anomaly decision threshold maximizing F1-score on evaluation data.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Avoid zero division
    denominator = precisions + recalls
    f1_scores = np.where(denominator > 0, 2 * (precisions * recalls) / denominator, 0)
    best_idx = int(np.argmax(f1_scores))

    if best_idx < len(thresholds):
        best_threshold = float(thresholds[best_idx])
    else:
        best_threshold = float(np.median(y_scores))

    preds = (y_scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    metrics = {
        "best_threshold": best_threshold,
        "best_f1": float(f1_scores[best_idx]),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    return best_threshold, metrics


def compute_auroc_metrics(
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    pixel_labels: Optional[np.ndarray] = None,
    pixel_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculates Image-level and optional Pixel-level AUROC."""
    results = {}

    # Image-level AUROC
    if len(np.unique(image_labels)) > 1:
        img_auroc = float(roc_auc_score(image_labels, image_scores))
    else:
        img_auroc = 1.0
    results["image_auroc"] = img_auroc

    # Pixel-level AUROC
    if pixel_labels is not None and pixel_scores is not None and len(pixel_labels) > 0:
        if len(np.unique(pixel_labels)) > 1:
            pix_auroc = float(roc_auc_score(pixel_labels, pixel_scores))
        else:
            pix_auroc = 1.0
        results["pixel_auroc"] = pix_auroc

    return results


def generate_inspection_report(
    product_id: str,
    category: str,
    filename: str,
    image_shape: Tuple[int, int],
    anomaly_score: float,
    threshold: float,
    is_anomaly: bool,
    inference_time_ms: float,
    defect_coverage_pct: float,
    notes: Optional[str] = None,
) -> Dict:
    """Generates a structured visual inspection record."""
    report = {
        "inspection_metadata": {
            "report_id": f"INSP-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{product_id}",
            "timestamp": datetime.now().isoformat(),
            "operator": "AI Visual Inspection System (PatchCore)",
            "product_category": category,
            "image_filename": filename,
            "image_resolution": f"{image_shape[1]}x{image_shape[0]} px",
        },
        "model_assessment": {
            "algorithm": "PatchCore (Towards Total Recall in Industrial Anomaly Detection)",
            "uncalibrated_anomaly_score": round(anomaly_score, 4),
            "decision_threshold": round(threshold, 4),
            "score_interpretation": "Raw nearest-neighbor Euclidean distance in patch feature space (not a probability)",
            "status": "ANOMALY_DETECTED" if is_anomaly else "NORMAL_PASSED",
            "defect_detected": is_anomaly,
            "defect_surface_coverage_pct": round(defect_coverage_pct, 2),
            "inference_latency_ms": round(inference_time_ms, 2),
        },
        "disclaimer": (
            "The anomaly score represents an uncalibrated geometric distance between query patch features "
            "and nominal memory bank representations. Products with scores above the threshold require manual review."
        ),
        "notes": notes or "Automated optical inspection completed."
    }
    return report

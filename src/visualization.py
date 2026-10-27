"""Visualization and Defect Plotting Utilities.

Provides multi-panel Matplotlib figures, OpenCV blending, colormaps, and contour drawing.
"""

from pathlib import Path
from typing import Tuple, Optional, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils import generate_defect_overlay, normalize_heatmap


def plot_inspection_results(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    defect_mask: np.ndarray,
    title: str = "Industrial Visual Inspection",
    anomaly_score: Optional[float] = None,
    threshold: Optional[float] = None,
    is_anomaly: Optional[bool] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 4),
) -> plt.Figure:
    """
    Renders a clean 4-panel inspection figure using Matplotlib.

    Panels:
        1. Original Camera Image
        2. Continuous Anomaly Heatmap
        3. Heatmap Overlay with Contours & Bounding Boxes
        4. Binary Segmented Defect Mask
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # 1. Original
    axes[0].imshow(original_rgb)
    axes[0].set_title("1. Original Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Heatmap
    norm_heat = normalize_heatmap(heatmap)
    im2 = axes[1].imshow(norm_heat, cmap="jet")
    axes[1].set_title("2. Anomaly Heatmap", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("3. Defect Overlay", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # 4. Binary Mask
    axes[3].imshow(defect_mask, cmap="gray")
    axes[3].set_title("4. Defect Mask", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    # Header title with verdict
    if is_anomaly is not None and anomaly_score is not None and threshold is not None:
        status_text = "DEFECTIVE (ANOMALY DETECTED)" if is_anomaly else "NORMAL (PASSED)"
        status_color = "red" if is_anomaly else "green"
        fig.suptitle(
            f"{title} | Result: {status_text} | Score: {anomaly_score:.4f} (Thresh: {threshold:.4f})",
            fontsize=13,
            fontweight="bold",
            color=status_color,
            y=1.02,
        )
    else:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
        print(f"Saved inspection plot to {save_path}")

    return fig


__all__ = [
    "generate_defect_overlay",
    "normalize_heatmap",
    "plot_inspection_results",
]

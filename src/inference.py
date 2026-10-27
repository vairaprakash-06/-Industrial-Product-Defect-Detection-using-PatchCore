"""Inference Engine for Industrial Defect Detection.

Provides the high-level DefectInspector class used by both CLI and Streamlit UI.
Features:
- Image format validation and standardized preprocessing.
- PatchCore inference with exact feature extraction.
- Anomaly score computation and threshold comparison.
- Generation of color-coded anomaly heatmaps and defect overlays.
- Defect surface area coverage calculation.
- Structured inspection report generation.
"""

import time
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import torch

from src.model import PatchCore
from src.dataset import get_transforms
from src.utils import generate_defect_overlay, generate_inspection_report, normalize_heatmap


class DefectInspector:
    """Industrial visual inspection inference service."""

    def __init__(
        self,
        category: str = "bottle",
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        img_size: int = 224,
        crop_size: int = 224,
    ):
        self.category = category
        self.device = device
        self.img_size = img_size
        self.crop_size = crop_size

        if checkpoint_path is None:
            self.checkpoint_path = Path("checkpoints") / category / f"patchcore_{category}.pt"
        else:
            self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found for category '{category}' at '{self.checkpoint_path}'.\n"
                f"Please train the model first by running:\n"
                f"  python src/train.py --category {category}\n"
                f"  python src/evaluate.py --category {category}"
            )

        self.model = PatchCore.load_from_checkpoint(self.checkpoint_path, device=device)
        self.img_transform, _ = get_transforms(img_size=img_size, crop_size=crop_size)

    @property
    def threshold(self) -> float:
        return self.model.threshold

    @threshold.setter
    def threshold(self, value: float):
        self.model.threshold = float(value)

    def inspect_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        custom_threshold: Optional[float] = None,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Dict:
        """
        Performs full visual inspection on an image.
        Returns a dictionary containing:
            - original_image: (H, W, 3) uint8 RGB
            - anomaly_score: float (raw uncalibrated distance)
            - threshold: float
            - is_anomaly: bool
            - heatmap: 2D float array
            - heatmap_colored: (H, W, 3) uint8 RGB
            - overlay: (H, W, 3) uint8 RGB blended visualization
            - defect_mask: (H, W) uint8 binary mask {0, 255}
            - defect_coverage_pct: float
            - inference_time_ms: float
            - report: Dict
        """
        # Validate and convert input to PIL Image
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.is_file():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            filename = img_path.name
            pil_img = Image.open(img_path).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            filename = "uploaded_inspection_image.png"
            if image_input.dtype != np.uint8:
                image_input = (image_input * 255).astype(np.uint8)
            pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            filename = getattr(image_input, "filename", "uploaded_inspection_image.png")
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        orig_w, orig_h = pil_img.size
        rgb_np = np.array(pil_img)

        # Preprocess for model
        tensor_img = self.img_transform(pil_img)

        # Run PatchCore inference
        start_time = time.perf_counter()
        pred = self.model.predict_image(tensor_img, return_heatmap=True)
        latency_ms = (time.perf_counter() - start_time) * 1000

        score = pred["anomaly_score"]
        heatmap_small = pred["anomaly_map"]  # (crop_size, crop_size)

        # Resize heatmap back to match original image dimensions
        heatmap_full = cv2.resize(heatmap_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        active_threshold = custom_threshold if custom_threshold is not None else self.threshold
        is_anomaly = bool(score >= active_threshold)

        # Generate overlay and defect segmentation
        overlay, heatmap_colored, defect_mask = generate_defect_overlay(
            image_rgb=rgb_np,
            heatmap=heatmap_full,
            threshold=active_threshold,
            alpha=alpha,
            colormap=colormap,
            draw_contours=True,
        )

        # Defect area coverage percentage
        defect_pixels = int(np.sum(defect_mask > 0))
        total_pixels = orig_w * orig_h
        defect_coverage_pct = (defect_pixels / total_pixels) * 100.0

        report = generate_inspection_report(
            product_id=filename.split(".")[0],
            category=self.category,
            filename=filename,
            image_shape=(orig_h, orig_w),
            anomaly_score=score,
            threshold=active_threshold,
            is_anomaly=is_anomaly,
            inference_time_ms=latency_ms,
            defect_coverage_pct=defect_coverage_pct,
        )

        return {
            "original_image": rgb_np,
            "anomaly_score": score,
            "reweighted_score": pred.get("reweighted_score", score),
            "threshold": active_threshold,
            "is_anomaly": is_anomaly,
            "heatmap": heatmap_full,
            "heatmap_colored": heatmap_colored,
            "overlay": overlay,
            "defect_mask": defect_mask,
            "defect_coverage_pct": defect_coverage_pct,
            "inference_time_ms": latency_ms,
            "report": report,
        }

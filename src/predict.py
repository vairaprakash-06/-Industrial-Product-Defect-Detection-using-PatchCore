"""Prediction and Single-Image Inspection CLI.

Allows running inference from the command line using trained PatchCore checkpoints:
    python -m src.predict --image samples/defect_broken_large_01.png --category bottle
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Union
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import DefectInspector
from src.preprocessing import load_and_validate_image


def predict_image(
    image_path: Union[str, Path],
    category: str = "bottle",
    checkpoint_path: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
    save_output: bool = True,
    output_dir: Union[str, Path] = "outputs",
    device: str = "auto",
) -> Dict:
    """
    Performs visual defect detection on a single image.

    Args:
        image_path: Path to the image file to inspect.
        category: Industrial product category (e.g., 'bottle').
        checkpoint_path: Optional path to .pt checkpoint.
        threshold: Optional override for decision threshold.
        save_output: Whether to save visualization and report to outputs/.
        output_dir: Destination directory for artifacts.
        device: 'auto', 'cuda', or 'cpu'.

    Returns:
        Result dictionary with scores, prediction, and visualizations.
    """
    image_path = Path(image_path)
    inspector = DefectInspector(
        category=category,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    result = inspector.inspect_image(
        image_input=image_path,
        custom_threshold=threshold,
    )

    score = result["anomaly_score"]
    active_thresh = result["threshold"]
    is_anomaly = result["is_anomaly"]
    pred_label = "DEFECTIVE" if is_anomaly else "NORMAL"
    latency = result["inference_time_ms"]

    print("=" * 65)
    print(f"PATCHCORE INSPECTION RESULT: {image_path.name}")
    print("=" * 65)
    print(f"Product Category:   {category.upper()}")
    status_tag = "[DEFECTIVE - ANOMALY DETECTED]" if is_anomaly else "[NORMAL - CONFORMANT]"
    print(f"Prediction:         {status_tag}")
    print(f"Anomaly Score:      {score:.4f} (Euclidean patch distance)")
    print(f"Decision Threshold: {active_thresh:.4f}")
    print(f"Defect Coverage:    {result['defect_coverage_pct']:.2f}% of surface area")
    print(f"Inference Latency:  {latency:.2f} ms")
    print("=" * 65)

    if save_output:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        overlay_file = out_path / f"{stem}_overlay.png"
        heatmap_file = out_path / f"{stem}_heatmap.png"

        # Save OpenCV BGR images
        cv2.imwrite(str(overlay_file), cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(heatmap_file), cv2.cvtColor(result["heatmap_colored"], cv2.COLOR_RGB2BGR))
        print(f"Saved overlay image to: {overlay_file}")
        print(f"Saved heatmap image to: {heatmap_file}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run PatchCore Defect Prediction on an image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (.png, .jpg)")
    parser.add_argument("--category", type=str, default="bottle", help="Product category")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold override")
    parser.add_argument("--no_save", action="store_true", help="Do not save output images to disk")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")
    args = parser.parse_args()

    predict_image(
        image_path=args.image,
        category=args.category,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        save_output=not args.no_save,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()

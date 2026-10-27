"""Basic Unit and Regression Tests for PatchCore Defect Detection."""

import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config, SUPPORTED_CATEGORIES
from src.preprocessing import get_transforms, load_and_validate_image, denormalize_tensor
from src.visualization import generate_defect_overlay, plot_inspection_results
from src.inference import DefectInspector


def test_config_loader():
    """Verifies default and category-specific YAML configs load properly."""
    cfg = load_config("bottle")
    assert cfg["dataset"]["category"] == "bottle"
    assert "model" in cfg
    assert "evaluation" in cfg
    assert "checkpoints" in cfg
    assert "bottle" in SUPPORTED_CATEGORIES


def test_preprocessing_and_transforms():
    """Verifies image transform pipelines and input validation."""
    img_transform, mask_transform = get_transforms(img_size=224, crop_size=224)
    
    # Create synthetic test image
    pil_dummy = Image.new("RGB", (256, 256), color=(128, 128, 128))
    tensor = img_transform(pil_dummy)
    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)

    # Test denormalization
    restored_np = denormalize_tensor(tensor)
    assert restored_np.shape == (224, 224, 3)
    assert restored_np.dtype == np.uint8

    # Test load_and_validate_image with numpy array
    dummy_np = np.ones((100, 100, 3), dtype=np.uint8) * 200
    pil_out, arr_out, filename = load_and_validate_image(dummy_np)
    assert arr_out.shape == (100, 100, 3)
    assert isinstance(pil_out, Image.Image)


def test_visualization_utilities():
    """Verifies overlay generation and Matplotlib figure rendering."""
    dummy_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_heatmap = np.zeros((100, 100), dtype=np.float32)
    dummy_heatmap[40:60, 40:60] = 4.5  # Artificial hotspot

    overlay, colored_heat, mask = generate_defect_overlay(
        image_rgb=dummy_rgb,
        heatmap=dummy_heatmap,
        threshold=2.0,
    )

    assert overlay.shape == (100, 100, 3)
    assert colored_heat.shape == (100, 100, 3)
    assert mask.shape == (100, 100)
    assert np.any(mask > 0)

    # Verify Matplotlib figure rendering
    fig = plot_inspection_results(
        original_rgb=dummy_rgb,
        heatmap=dummy_heatmap,
        overlay=overlay,
        defect_mask=mask,
        anomaly_score=4.5,
        threshold=2.0,
        is_anomaly=True,
    )
    assert fig is not None


def test_live_defect_inspector():
    """Verifies end-to-end inference on pre-packaged sample images."""
    sample_normal = PROJECT_ROOT / "samples" / "normal_bottle_01.png"
    sample_defect = PROJECT_ROOT / "samples" / "defect_broken_large_01.png"

    if sample_normal.is_file() and (PROJECT_ROOT / "checkpoints" / "bottle" / "patchcore_bottle.pt").is_file():
        inspector = DefectInspector("bottle")
        
        # Test normal sample
        res_norm = inspector.inspect_image(sample_normal)
        assert res_norm["is_anomaly"] is False
        assert res_norm["anomaly_score"] < res_norm["threshold"]

        # Test defect sample
        res_defect = inspector.inspect_image(sample_defect)
        assert res_defect["is_anomaly"] is True
        assert res_defect["anomaly_score"] >= res_defect["threshold"]


if __name__ == "__main__":
    test_config_loader()
    test_preprocessing_and_transforms()
    test_visualization_utilities()
    test_live_defect_inspector()
    print("All basic unit and regression tests passed successfully!")

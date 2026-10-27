"""Unit and Integration Tests for PatchCore Defect Detection Pipeline."""

import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import PatchCore, PatchCoreFeatureExtractor
from src.utils import generate_defect_overlay, find_optimal_threshold, compute_auroc_metrics, generate_inspection_report


def test_feature_extractor_shape():
    """Verifies multi-scale intermediate feature concatenation and pooling."""
    extractor = PatchCoreFeatureExtractor(
        backbone_name="resnet18",
        layers=("layer2", "layer3"),
        patch_size=3,
        patch_stride=1,
    )
    dummy_input = torch.randn(2, 3, 224, 224)
    embeddings, feat_shape = extractor(dummy_input)

    # ResNet18: layer2 has 128 channels, layer3 has 256 channels -> concatenated = 384
    # 224x224 input yields 28x28 spatial grid for layer2
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 28 * 28
    assert embeddings.shape[2] == 384
    assert feat_shape == (28, 28)


def test_patchcore_fit_and_predict():
    """Verifies memory bank construction and prediction on synthetic data."""
    model = PatchCore(
        backbone_name="resnet18",
        layers=("layer2", "layer3"),
        coreset_sampling_ratio=0.1,
        project_dim=64,
        device="cpu",
    )

    # Synthetic normal patch features (500 patches of dim 384)
    dummy_patches = torch.randn(500, 384)
    model.fit_memory_bank(dummy_patches)

    assert model.memory_bank is not None
    # 10% of 500 = 50 patches
    assert model.memory_bank.shape == (50, 384)

    # Predict on a test image
    test_img = torch.randn(3, 224, 224)
    result = model.predict_image(test_img, return_heatmap=True)

    assert "anomaly_score" in result
    assert isinstance(result["anomaly_score"], float)
    assert result["anomaly_map"].shape == (224, 224)
    assert result["patch_scores"].shape == (28, 28)


def test_overlay_and_report_generation():
    """Verifies visualization generation and structured report output."""
    dummy_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_heatmap = np.zeros((100, 100), dtype=np.float32)
    # Add a synthetic defect hotspot
    dummy_heatmap[30:50, 30:50] = 5.0

    overlay, color_map, mask = generate_defect_overlay(
        image_rgb=dummy_rgb,
        heatmap=dummy_heatmap,
        threshold=2.0,
    )

    assert overlay.shape == (100, 100, 3)
    assert color_map.shape == (100, 100, 3)
    assert mask.shape == (100, 100)
    assert np.any(mask > 0)

    report = generate_inspection_report(
        product_id="TEST-001",
        category="bottle",
        filename="test_sample.png",
        image_shape=(100, 100),
        anomaly_score=4.5,
        threshold=2.0,
        is_anomaly=True,
        inference_time_ms=35.0,
        defect_coverage_pct=4.0,
    )

    assert report["model_assessment"]["status"] == "ANOMALY_DETECTED"
    assert report["model_assessment"]["defect_detected"] is True
    assert "inspection_metadata" in report


def test_metrics_calculation():
    """Verifies AUROC and optimal threshold calculation."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.15, 0.3, 0.8, 0.9, 0.85, 0.95])

    thresh, metrics = find_optimal_threshold(y_true, y_scores)
    assert metrics["best_f1"] == 1.0
    assert 0.3 <= thresh <= 0.8

    auroc = compute_auroc_metrics(y_true, y_scores)
    assert auroc["image_auroc"] == 1.0


if __name__ == "__main__":
    test_feature_extractor_shape()
    test_patchcore_fit_and_predict()
    test_overlay_and_report_generation()
    test_metrics_calculation()
    print("All unit tests passed successfully!")

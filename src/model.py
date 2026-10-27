"""PatchCore Architecture for Industrial Defect Detection.

Reference:
Towards Total Recall in Industrial Anomaly Detection
Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler
CVPR 2022 (Oral)

Features:
- Multi-layer CNN feature extraction (e.g. ResNet18, ResNet50, WideResNet50_2).
- Locally aggregated neighborhood feature pooling (patch pooling).
- Fast Johnson-Lindenstrauss random projection accelerated greedy minimax k-center coreset subsampling.
- Memory bank nearest-neighbor anomaly scoring and spatially smoothed defect heatmap localization.
- Complete serialization (save/load checkpoint with threshold and metadata).
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PatchCoreFeatureExtractor(nn.Module):
    """Extracts locally aggregated multi-layer feature maps from a CNN backbone."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        layers: Tuple[str, ...] = ("layer2", "layer3"),
        patch_size: int = 3,
        patch_stride: int = 1,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.layers = list(layers)
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Load standard ImageNet pretrained backbone
        if backbone_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif backbone_name == "wide_resnet50_2":
            backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Neighborhood patch pooling
        self.patch_pool = nn.AvgPool2d(
            kernel_size=patch_size,
            stride=patch_stride,
            padding=patch_size // 2,
        )

        self._hook_features: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        def _get_hook(name: str):
            def hook(module, input, output):
                self._hook_features[name] = output
            return hook

        for name, module in self.backbone.named_children():
            if name in self.layers:
                module.register_forward_hook(_get_hook(name))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extracts multi-scale features and returns aggregated patch embeddings.
        Args:
            x: Input images of shape (B, 3, H, W)
        Returns:
            patch_embeddings: Tensor of shape (B, H_feat * W_feat, D)
            feat_shape: Spatial tuple (H_feat, W_feat)
        """
        self._hook_features.clear()
        _ = self.backbone(x)

        extracted = [self._hook_features[layer] for layer in self.layers]
        # Target spatial resolution is the highest resolution among selected layers (layer2)
        target_h, target_w = extracted[0].shape[-2:]

        processed_features = []
        for feat in extracted:
            pooled = self.patch_pool(feat)
            if pooled.shape[-2:] != (target_h, target_w):
                pooled = F.interpolate(
                    pooled,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            processed_features.append(pooled)

        # Concatenate along channel dimension: (B, D, H_feat, W_feat)
        concatenated = torch.cat(processed_features, dim=1)
        b, d, h, w = concatenated.shape

        # Reshape to (B, H_feat * W_feat, D)
        embeddings = concatenated.permute(0, 2, 3, 1).reshape(b, h * w, d)
        return embeddings, (h, w)


class PatchCore:
    """End-to-end PatchCore model for training memory bank and conducting anomaly inference."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        layers: Tuple[str, ...] = ("layer2", "layer3"),
        patch_size: int = 3,
        patch_stride: int = 1,
        coreset_sampling_ratio: float = 0.02,
        project_dim: int = 128,
        device: str = "auto",
        sigma: float = 4.0,
        n_neighbors: int = 9,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone_name = backbone_name
        self.layers = tuple(layers)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.project_dim = project_dim
        self.sigma = sigma
        self.n_neighbors = n_neighbors

        self.feature_extractor = PatchCoreFeatureExtractor(
            backbone_name=backbone_name,
            layers=self.layers,
            patch_size=patch_size,
            patch_stride=patch_stride,
        ).to(self.device)

        # Memory bank holding normal patch representations: shape (M, D)
        self.memory_bank: Optional[torch.Tensor] = None
        self.threshold: float = 0.5
        self.min_score: float = 0.0
        self.max_score: float = 1.0
        self.feature_dim: Optional[int] = None
        self.feat_shape: Optional[Tuple[int, int]] = None

    def fit_memory_bank(self, all_patches: torch.Tensor):
        """
        Subsamples and stores the memory bank from all normal training patch embeddings.
        Args:
            all_patches: Tensor of shape (N_total_patches, D)
        """
        all_patches = all_patches.cpu()
        n_total, d = all_patches.shape
        self.feature_dim = d
        target_size = max(10, int(n_total * self.coreset_sampling_ratio))

        print(f"[PatchCore] Total normal training patches: {n_total:,} (dim={d})")
        print(f"[PatchCore] Coreset target size: {target_size:,} ({self.coreset_sampling_ratio*100:.1f}%)")

        if target_size >= n_total:
            self.memory_bank = all_patches.to(self.device)
            return

        # Fast Greedy Minimax Facility Location (k-center) with JL random projection
        selected_indices = self._greedy_kcenter_selection(all_patches, target_size, self.project_dim)
        self.memory_bank = all_patches[selected_indices].to(self.device)
        print(f"[PatchCore] Memory bank successfully fitted. Bank shape: {tuple(self.memory_bank.shape)}")

    @staticmethod
    def _greedy_kcenter_selection(
        patches: torch.Tensor,
        target_size: int,
        project_dim: int = 128,
        seed: int = 42,
    ) -> List[int]:
        """Greedy k-center selection using Johnson-Lindenstrauss random projection for efficiency."""
        torch.manual_seed(seed)

        n, d = patches.shape
        proj_dim = min(project_dim, d)

        # Random Gaussian projection matrix: D x proj_dim
        proj_matrix = torch.randn(d, proj_dim, dtype=patches.dtype) / np.sqrt(proj_dim)
        projected = torch.matmul(patches, proj_matrix)  # (N, proj_dim)

        # Precompute squared norms: ||x||^2 shape (N,)
        proj_sq_norm = torch.sum(projected ** 2, dim=-1)

        selected_indices: List[int] = []
        # Random initial point
        initial_idx = int(torch.randint(0, n, (1,)).item())
        selected_indices.append(initial_idx)

        # Distance to initial point: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * (x . y)
        curr_point = projected[initial_idx]
        min_distances = proj_sq_norm + torch.dot(curr_point, curr_point) - 2.0 * torch.mv(projected, curr_point)
        min_distances = torch.clamp(min_distances, min=0.0)

        step_interval = max(1, target_size // 5)
        for step in range(1, target_size):
            next_idx = int(torch.argmax(min_distances).item())
            selected_indices.append(next_idx)

            new_point = projected[next_idx]
            new_dists = proj_sq_norm + torch.dot(new_point, new_point) - 2.0 * torch.mv(projected, new_point)
            min_distances = torch.minimum(min_distances, torch.clamp(new_dists, min=0.0))

            if step % step_interval == 0 or step == target_size - 1:
                progress = (step + 1) / target_size * 100
                print(f"[PatchCore Coreset] Selected {step+1}/{target_size} points ({progress:.1f}%)")

        return selected_indices

    def predict_image(
        self,
        image_tensor: torch.Tensor,
        return_heatmap: bool = True,
    ) -> Dict[str, float or np.ndarray or bool]:
        """
        Runs inference on a single image tensor of shape (3, H, W) or (1, 3, H, W).
        Returns:
            anomaly_score: Raw Euclidean patch distance (uncalibrated)
            is_anomaly: True if score >= threshold
            anomaly_map: 2D numpy array of size (H, W) smoothed anomaly scores
            patch_scores: 2D numpy array of shape (H_feat, W_feat)
        """
        if self.memory_bank is None:
            raise RuntimeError("PatchCore memory bank is not initialized! Please train or load a checkpoint.")

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        b, c, orig_h, orig_w = image_tensor.shape

        with torch.no_grad():
            embeddings, feat_shape = self.feature_extractor(image_tensor)
            self.feat_shape = feat_shape
            # embeddings: (1, N_patches, D)
            patches = embeddings[0]  # (N_patches, D)

            # Pairwise distance matrix between query patches and memory bank
            # patches: (N, D), memory_bank: (M, D)
            # Efficient chunked or direct distance computation:
            # ||p - m||^2 = ||p||^2 + ||m||^2 - 2 * p * m^T
            patches_norm_sq = (patches ** 2).sum(dim=-1, keepdim=True)  # (N, 1)
            bank_norm_sq = (self.memory_bank ** 2).sum(dim=-1, keepdim=True).t()  # (1, M)
            dot = torch.matmul(patches, self.memory_bank.t())  # (N, M)
            dist_sq = patches_norm_sq + bank_norm_sq - 2.0 * dot
            dist_sq = torch.clamp(dist_sq, min=0.0)
            distances = torch.sqrt(dist_sq)  # (N, M)

            # Minimum distance to nearest neighbor in memory bank for each patch
            min_dists, min_indices = torch.min(distances, dim=-1)  # (N,)
            patch_scores_np = min_dists.cpu().numpy().reshape(feat_shape)

            # Roth et al. image anomaly score: maximum patch score
            max_patch_score = float(torch.max(min_dists).item())

            # Optional nearest-neighbor reweighting (Roth et al., Eq. 7)
            if self.n_neighbors > 1 and distances.shape[-1] >= self.n_neighbors:
                topk_dists, _ = torch.topk(distances, k=self.n_neighbors, dim=-1, largest=False)
                # Max patch index
                max_patch_idx = torch.argmax(min_dists)
                n_b = topk_dists[max_patch_idx]
                weights = 1.0 - (torch.exp(n_b[0]) / torch.sum(torch.exp(n_b))).item()
                reweighted_score = max_patch_score * weights
            else:
                reweighted_score = max_patch_score

        # Heatmap generation and Gaussian smoothing
        if return_heatmap:
            # Gaussian blur on the feature-map grid
            blurred_map = ndimage.gaussian_filter(patch_scores_np, sigma=self.sigma)
            # Bilinear upsample to input image resolution
            blurred_tensor = torch.from_numpy(blurred_map).unsqueeze(0).unsqueeze(0).float()
            upsampled = F.interpolate(
                blurred_tensor,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            anomaly_map = upsampled.squeeze().numpy()
        else:
            anomaly_map = None

        is_anomaly = bool(max_patch_score >= self.threshold)

        return {
            "anomaly_score": max_patch_score,
            "reweighted_score": reweighted_score,
            "threshold": self.threshold,
            "is_anomaly": is_anomaly,
            "anomaly_map": anomaly_map,
            "patch_scores": patch_scores_np,
        }

    # pyrefly: ignore[invalid-annotation]
    def save(self, save_path: str or Path, extra_metadata: Optional[Dict] = None):
        """Saves memory bank weights and configuration."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "backbone_name": self.backbone_name,
            "layers": self.layers,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "project_dim": self.project_dim,
            "sigma": self.sigma,
            "n_neighbors": self.n_neighbors,
            "threshold": self.threshold,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "feature_dim": self.feature_dim,
            "feat_shape": self.feat_shape,
            "memory_bank": self.memory_bank.cpu() if self.memory_bank is not None else None,
            "extra_metadata": extra_metadata or {},
        }
        torch.save(payload, str(save_path))
        print(f"[PatchCore] Checkpoint saved successfully to {save_path}")

    @classmethod
    # pyrefly: ignore[invalid-annotation]
    def load_from_checkpoint(cls, checkpoint_path: str or Path, device: str = "auto") -> "PatchCore":
        """Loads a trained PatchCore model from a serialized checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        payload = torch.load(str(checkpoint_path), map_location="cpu")
        model = cls(
            backbone_name=payload.get("backbone_name", "resnet18"),
            layers=tuple(payload.get("layers", ("layer2", "layer3"))),
            patch_size=payload.get("patch_size", 3),
            patch_stride=payload.get("patch_stride", 1),
            coreset_sampling_ratio=payload.get("coreset_sampling_ratio", 0.02),
            project_dim=payload.get("project_dim", 128),
            device=device,
            sigma=payload.get("sigma", 4.0),
            n_neighbors=payload.get("n_neighbors", 9),
        )

        model.threshold = float(payload.get("threshold", 0.5))
        model.min_score = float(payload.get("min_score", 0.0))
        model.max_score = float(payload.get("max_score", 1.0))
        model.feature_dim = payload.get("feature_dim")
        model.feat_shape = payload.get("feat_shape")

        bank = payload.get("memory_bank")
        if bank is not None:
            model.memory_bank = bank.to(model.device)

        return model

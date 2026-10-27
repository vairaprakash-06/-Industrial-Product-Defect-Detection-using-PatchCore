"""MVTec AD Dataset Loader for PatchCore.

Preserves the official MVTec AD folder structure:
  <category>/
    train/good/
    test/good/
    test/<defect_type>/
    ground_truth/<defect_type>/<image_name>_mask.png

Handles image preprocessing, standard ImageNet normalization,
and corresponding pixel-level binary ground-truth masks.
"""

from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224, crop_size: int = 224) -> Tuple[T.Compose, T.Compose]:
    """Returns transform pipelines for image and binary mask."""
    # Standard industrial inspection transform: resize and center crop
    img_transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    mask_transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
        T.CenterCrop(crop_size),
        T.ToTensor(),
    ])

    return img_transform, mask_transform


class MVTecDataset(Dataset):
    """PyTorch Dataset for MVTec Anomaly Detection benchmark."""

    def __init__(
        self,
        # pyrefly: ignore[invalid-annotation]
        root_dir: str or Path,
        category: str = "bottle",
        split: str = "train",
        img_size: int = 224,
        crop_size: int = 224,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.img_size = img_size
        self.crop_size = crop_size

        self.category_dir = self.root_dir / self.category
        if not self.category_dir.is_dir():
            raise FileNotFoundError(
                f"Category folder not found at '{self.category_dir}'. "
                f"Please run 'python src/download_mvtec.py --category {self.category}' "
                f"or follow the manual download steps in the README."
            )

        self.img_transform, self.mask_transform = get_transforms(img_size, crop_size)
        self.samples: List[Dict] = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        samples = []
        split_dir = self.category_dir / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        gt_base = self.category_dir / "ground_truth"

        if self.split == "train":
            # For PatchCore training, strictly normal ("good") images are used
            good_dir = split_dir / "good"
            if not good_dir.is_dir():
                raise FileNotFoundError(f"Missing train/good directory at {good_dir}")
            for img_path in sorted(list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpg"))):
                samples.append({
                    "img_path": str(img_path),
                    "label": 0,  # 0 = normal
                    "defect_type": "good",
                    "mask_path": None,
                })
        else:
            # Test split contains both "good" and various anomaly defect types
            for defect_dir in sorted(split_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect_name = defect_dir.name
                is_anomaly = defect_name != "good"
                label = 1 if is_anomaly else 0

                for img_path in sorted(list(defect_dir.glob("*.png")) + list(defect_dir.glob("*.jpg"))):
                    mask_path = None
                    if is_anomaly:
                        # Ground truth mask naming convention: <stem>_mask.png
                        expected_mask = gt_base / defect_name / f"{img_path.stem}_mask.png"
                        if expected_mask.is_file():
                            mask_path = str(expected_mask)
                        else:
                            # Fallback check without _mask suffix
                            alt_mask = gt_base / defect_name / img_path.name
                            if alt_mask.is_file():
                                mask_path = str(alt_mask)

                    samples.append({
                        "img_path": str(img_path),
                        "label": label,
                        "defect_type": defect_name,
                        "mask_path": mask_path,
                    })

        if len(samples) == 0:
            raise RuntimeError(f"No valid images found in {split_dir}.")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor or str or int]:
        item = self.samples[idx]
        img = Image.open(item["img_path"]).convert("RGB")
        orig_w, orig_h = img.size
        img_tensor = self.img_transform(img)

        # Ground truth mask: (1, H, W) binary tensor {0, 1}
        if item["mask_path"] is not None and Path(item["mask_path"]).is_file():
            mask_img = Image.open(item["mask_path"]).convert("L")
            mask_tensor = self.mask_transform(mask_img)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = torch.zeros((1, self.crop_size, self.crop_size), dtype=torch.float32)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "label": item["label"],
            "defect_type": item["defect_type"],
            "img_path": item["img_path"],
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
        }


def get_dataloaders(
    # pyrefly: ignore[invalid-annotation]
    root_dir: str or Path,
    category: str = "bottle",
    img_size: int = 224,
    crop_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Creates train (normal) and test (normal + anomalous) DataLoaders."""
    train_dataset = MVTecDataset(root_dir, category, split="train", img_size=img_size, crop_size=crop_size)
    test_dataset = MVTecDataset(root_dir, category, split="test", img_size=img_size, crop_size=crop_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, test_loader

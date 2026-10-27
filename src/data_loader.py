"""MVTec AD Data Loader and Dataset Management.

Provides high-level dataset construction, split management, and PyTorch DataLoaders.
"""

from pathlib import Path
from typing import Tuple, Optional, Union
from torch.utils.data import DataLoader

from src.dataset import MVTecDataset, get_dataloaders, get_transforms
from src.download_mvtec import check_dataset_exists, download_category


def create_dataloaders(
    data_dir: Union[str, Path] = "data/mvtec",
    category: str = "bottle",
    img_size: int = 224,
    crop_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs train and test DataLoader instances for the given category.

    Args:
        data_dir: Base directory containing MVTec category folders.
        category: Industrial product category (e.g., 'bottle').
        img_size: Image resize dimension.
        crop_size: Center crop dimension.
        batch_size: Mini-batch size.
        num_workers: PyTorch DataLoader workers.

    Returns:
        (train_loader, test_loader)
    """
    return get_dataloaders(
        root_dir=data_dir,
        category=category,
        img_size=img_size,
        crop_size=crop_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )


__all__ = [
    "MVTecDataset",
    "get_dataloaders",
    "create_dataloaders",
    "get_transforms",
    "check_dataset_exists",
    "download_category",
]

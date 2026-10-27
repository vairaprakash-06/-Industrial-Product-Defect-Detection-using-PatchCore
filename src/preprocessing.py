"""Image Preprocessing and Validation Pipeline for PatchCore.

Provides standard ImageNet normalization, format verification, and transform pipelines.
"""

from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Supported image file extensions
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_transforms(
    img_size: int = 224,
    crop_size: int = 224,
) -> Tuple[T.Compose, T.Compose]:
    """
    Returns torchvision transform pipelines for product images and binary defect masks.
    
    Args:
        img_size: Size to resize input image (bilinear/bicubic interpolation).
        crop_size: Center crop dimension.
    
    Returns:
        (image_transform, mask_transform)
    """
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


def load_and_validate_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    expected_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Image.Image, np.ndarray, str]:
    """
    Validates, opens, and converts an industrial product image into RGB.

    Args:
        image_input: File path, PIL Image, or NumPy array.
        expected_size: Optional (width, height) check.

    Returns:
        (pil_image, rgb_numpy_array, filename)
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{img_path.suffix}'. "
                f"Supported formats: {', '.join(sorted(VALID_EXTENSIONS))}"
            )
        filename = img_path.name
        pil_img = Image.open(img_path).convert("RGB")

    elif isinstance(image_input, Image.Image):
        filename = getattr(image_input, "filename", "inspection_image.png")
        if filename is not None and Path(filename).name:
            filename = Path(filename).name
        else:
            filename = "inspection_image.png"
        pil_img = image_input.convert("RGB")

    elif isinstance(image_input, np.ndarray):
        filename = "numpy_inspection_image.png"
        if image_input.ndim == 2:
            # Grayscale to RGB
            image_input = np.stack([image_input] * 3, axis=-1)
        elif image_input.ndim == 3 and image_input.shape[2] == 4:
            # RGBA to RGB
            image_input = image_input[:, :, :3]

        if image_input.dtype != np.uint8:
            if np.max(image_input) <= 1.0:
                image_input = (image_input * 255.0).astype(np.uint8)
            else:
                image_input = image_input.astype(np.uint8)
        pil_img = Image.fromarray(image_input)

    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Check minimum image dimensions
    w, h = pil_img.size
    if w < 16 or h < 16:
        raise ValueError(f"Image dimensions ({w}x{h}) are too small for inspection.")

    rgb_np = np.array(pil_img)
    return pil_img, rgb_np, filename


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized PyTorch tensor (3, H, W) back to uint8 RGB numpy array.
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    t = tensor.detach().cpu().clone()
    for c in range(3):
        t[c] = t[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    np_img = t.permute(1, 2, 0).numpy()
    np_img = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)
    return np_img

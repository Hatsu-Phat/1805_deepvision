"""Frame preprocessing that follows the DeMamba data loader.

Reference: chenhaoxing/DeMamba, dataloader.py (Ours_Dataset_train / Ours_Dataset_val):
  * crop_center_by_percentage(image, p) cuts p of the longer side on both ends;
    train: only `real/youku` videos with p = 0.15; val/test: every video with p = 0.10
  * albumentations.Resize(224, 224)   (aspect ratio is not preserved, no letterbox padding)
  * albumentations.Normalize(ImageNet mean/std, max_pixel_value=255)
  * clips shorter than the requested length are padded with all-zero frames *after* normalization

Training-time augmentations (flip, JPEG, noise, blur, gray) are not part of this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_center_by_percentage(image: np.ndarray, percentage: float) -> np.ndarray:
    height, width = image.shape[:2]
    if width > height:
        left = int(width * percentage)
        right = int(width * percentage)
        return image[:, left:width - right]
    up = int(height * percentage)
    down = int(height * percentage)
    return image[up:height - down, :]


def demamba_crop_percentage(split: str, generator: str, label: int) -> float:
    """Crop ratio DeMamba applies to a video (0.0 means no crop)."""
    if split in ("train", "train2"):
        return 0.15 if (label == 0 and generator.lower() == "youku") else 0.0
    return 0.10


@dataclass
class FrameTransform:
    size: int = 224
    interpolation: int = cv2.INTER_LINEAR
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD
    normalize: bool = True

    def __post_init__(self):
        import albumentations as A  # imported here: the clip pipeline (crop + cv2 resize) does not need it

        ops = [A.Resize(self.size, self.size, interpolation=self.interpolation)]
        if self.normalize:
            ops.append(A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0, p=1.0))
        self._pipeline = A.Compose(ops)

    def __call__(self, image_rgb: np.ndarray, crop: float = 0.0) -> np.ndarray:
        """HWC uint8 RGB -> CHW float32 (or uint8 when normalize=False)."""
        if crop:
            image_rgb = crop_center_by_percentage(image_rgb, crop)
        out = self._pipeline(image=image_rgb)["image"]
        return np.ascontiguousarray(out.transpose(2, 0, 1))


# Per-model settings used by the baselines in this project.
PROFILES = {
    # DeMamba / XCLIP-style models
    "demamba": dict(interpolation=cv2.INTER_LINEAR, normalize=True),
    # ReStraV resizes with bicubic and feeds [0, 1] pixels to DINOv2 (ReStraV/dinov2_features.py)
    "restrav": dict(interpolation=cv2.INTER_CUBIC, normalize=False),
    # SSTGNN uses cv2.resize defaults (bilinear) (SSTGNN/dataset/utils.py)
    "sstgnn": dict(interpolation=cv2.INTER_LINEAR, normalize=False),
}


def build_transform(profile: str = "demamba", size: int = 224) -> FrameTransform:
    return FrameTransform(size=size, **PROFILES[profile])


def pad_clip(frames: np.ndarray, length: int, mode: str = "zero") -> tuple[np.ndarray, np.ndarray]:
    """Pad a (T, C, H, W) clip to `length` frames.

    mode="zero":   append zero frames (DeMamba behaviour)
    mode="repeat": repeat the last frame
    Returns the padded clip and a validity mask of shape (length,).
    """
    t = frames.shape[0]
    mask = np.zeros(length, dtype=bool)
    mask[:min(t, length)] = True
    if t >= length:
        return frames[:length], mask
    if t == 0:
        raise ValueError("cannot pad an empty clip")
    if mode == "zero":
        pad = np.zeros((length - t, *frames.shape[1:]), dtype=frames.dtype)
    elif mode == "repeat":
        pad = np.repeat(frames[-1:], length - t, axis=0)
    else:
        raise ValueError(f"unknown pad mode {mode!r}")
    return np.concatenate([frames, pad], axis=0), mask

import albumentations
import cv2
import numpy as np
import pytest

from deepvision.data.transforms import (FrameTransform, build_transform, crop_center_by_percentage,
                                        demamba_crop_percentage, pad_clip)


# Copied verbatim from chenhaoxing/DeMamba dataloader.py
def crop_center_by_percentage_reference(image, percentage):
    height, width = image.shape[:2]

    if width > height:
        left_pixels = int(width * percentage)
        right_pixels = int(width * percentage)
        start_x = left_pixels
        end_x = width - right_pixels
        cropped_image = image[:, start_x:end_x]
    else:
        up_pixels = int(height * percentage)
        down_pixels = int(height * percentage)
        start_y = up_pixels
        end_y = height - down_pixels
        cropped_image = image[start_y:end_y, :]

    return cropped_image


def demamba_val_reference(image):
    # Ours_Dataset_val: crop 0.1, then Resize + Normalize, then transpose(2, 0, 1)
    trans = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ])
    image = crop_center_by_percentage_reference(image, 0.1)
    return trans(image=image)["image"].transpose(2, 0, 1)


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            for h, w in [(720, 1280), (1280, 720), (240, 320), (576, 1024), (300, 300), (1088, 1920)]]


@pytest.mark.parametrize("pct", [0.0, 0.1, 0.15])
def test_crop_matches_reference(images, pct):
    for img in images:
        assert np.array_equal(crop_center_by_percentage(img, pct), crop_center_by_percentage_reference(img, pct))


def test_val_transform_matches_reference(images):
    t = build_transform("demamba")
    for img in images:
        ours = t(img, crop=0.1)
        ref = demamba_val_reference(img)
        assert ours.shape == (3, 224, 224) and ours.dtype == np.float32
        assert np.array_equal(ours, ref)


def test_crop_policy():
    assert demamba_crop_percentage("train", "youku", 0) == 0.15
    assert demamba_crop_percentage("train", "Pika", 1) == 0.0
    assert demamba_crop_percentage("test", "MSR-VTT", 0) == 0.10
    assert demamba_crop_percentage("test", "Sora", 1) == 0.10


def test_resize_is_bilinear_by_default():
    img = np.random.default_rng(1).integers(0, 256, (100, 150, 3), dtype=np.uint8)
    out = FrameTransform(normalize=False)(img)
    expected = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    assert np.array_equal(out, expected)


def test_profiles():
    img = np.random.default_rng(2).integers(0, 256, (100, 150, 3), dtype=np.uint8)
    out = build_transform("restrav")(img)
    expected = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
    assert np.array_equal(out, expected)


def test_pad_zero_matches_demamba():
    clip = np.ones((5, 3, 4, 4), dtype=np.float32)
    padded, mask = pad_clip(clip, 8, "zero")
    assert padded.shape == (8, 3, 4, 4)
    assert np.all(padded[5:] == 0) and np.all(padded[:5] == 1)
    assert mask.tolist() == [True] * 5 + [False] * 3


def test_pad_repeat():
    clip = np.arange(3, dtype=np.float32).reshape(3, 1, 1, 1)
    padded, mask = pad_clip(clip, 6, "repeat")
    assert padded.ravel().tolist() == [0, 1, 2, 2, 2, 2]
    assert mask.sum() == 3


def test_pad_no_op_when_long_enough():
    clip = np.zeros((10, 3, 2, 2), dtype=np.float32)
    padded, mask = pad_clip(clip, 8)
    assert padded.shape[0] == 8 and mask.all()

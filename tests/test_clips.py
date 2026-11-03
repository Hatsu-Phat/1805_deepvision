from pathlib import Path

import albumentations
import cv2
import numpy as np
import pandas as pd
import pytest

from deepvision.data.clips import build_clip, convert_one, read_clip, shrink_jpeg, write_clip
from deepvision.data.dataset import FrameClipDataset
from deepvision.data.extract import extract_one

from .test_extract import job_for
from .test_transforms import crop_center_by_percentage_reference

CLIP_OPTS = {"size": 224, "crop": "demamba", "jpeg_quality": 95, "keep_native": ["test"]}


def encode(img, q=95):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


@pytest.fixture
def jpeg():
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 256, (360, 640, 3), dtype=np.uint8), (7, 7), 0)
    return encode(img)


@pytest.mark.parametrize("crop", [0.0, 0.1, 0.15])
def test_shrink_matches_demamba_geometry(jpeg, crop):
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    ref = albumentations.Resize(224, 224)(image=crop_center_by_percentage_reference(img, crop))["image"]
    assert shrink_jpeg(jpeg, crop, 224, 95) == encode(ref)


def test_resize_to_same_size_is_identity():
    img = np.random.default_rng(1).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    assert np.array_equal(albumentations.Resize(224, 224)(image=img)["image"], img)


def test_clip_roundtrip(tmp_path, jpeg):
    frames = [jpeg, jpeg[:-1] + b"\xd9", b"x" * 1000]
    size = write_clip(tmp_path / "a.tar", frames, {"k": 1})
    got, meta = read_clip(tmp_path / "a.tar")
    assert got == frames and meta == {"k": 1}
    assert size == (tmp_path / "a.tar").stat().st_size
    assert not (tmp_path / "a.tar.tmp").exists()


def clip_job(src, tmp_path, name, duration, split):
    job = job_for(src, tmp_path / "frames" / name, duration) | {"split": split}
    job["clip"] = {"dst": str(tmp_path / "clips" / f"{name}.tar"), "root": str(tmp_path / "clips"),
                   "opts": CLIP_OPTS, "delete_native": split != "test"}
    return job


@pytest.mark.parametrize("split", ["train", "test"])
def test_direct_clip_equals_native_then_convert(video_dir, tmp_path, split):
    """Both paths must give byte-identical clips (same compression history)."""
    src = video_dir / "long_8s_24fps.mp4"
    direct = extract_one(clip_job(src, tmp_path / "a", "v", 8.0, split))
    assert "error" not in direct, direct
    assert not (tmp_path / "a" / "frames" / "v").exists()

    native = extract_one(job_for(src, tmp_path / "b" / "frames" / "v", 8.0) | {"split": split})
    assert "error" not in native
    conv = extract_one(clip_job(src, tmp_path / "b", "v", 8.0, split))
    assert "error" not in conv, conv
    assert read_clip(tmp_path / "a" / "clips" / "v.tar")[0] == read_clip(tmp_path / "b" / "clips" / "v.tar")[0]
    assert direct["crop"] == conv["crop"] == (0.1 if split == "test" else 0.0)
    assert (direct["height"], direct["width"]) == (224, 224)
    assert (direct["native_height"], direct["native_width"]) == (240, 320)
    # train native frames are removed after conversion, test native frames are kept
    assert (tmp_path / "b" / "frames" / "v").exists() == (split == "test")


def test_clip_skip_existing(video_dir, tmp_path):
    src = video_dir / "long_8s_24fps.mp4"
    job = clip_job(src, tmp_path, "v", 8.0, "train")
    first = extract_one(job)
    again = extract_one(job)
    assert again.get("skipped") and again["clip_bytes"] == first["clip_bytes"]


def test_convert_one_reports_missing_frames(tmp_path):
    res = convert_one({"video_id": "x", "path": "x.mp4", "split": "train", "label": 1, "generator": "g",
                       "src": str(tmp_path / "none"), "dst": str(tmp_path / "x.tar"), "clips_root": str(tmp_path),
                       "opts": CLIP_OPTS, "delete_native": True})
    assert "error" in res


def test_dataset_reads_clips_like_native(video_dir, tmp_path):
    src = video_dir / "long_12s_25fps.mp4"
    native = extract_one(job_for(src, tmp_path / "frames" / "v", 12.0))
    pd.DataFrame([native]).to_csv(tmp_path / "frames.csv", index=False)
    job = clip_job(src, tmp_path, "v", 12.0, "test")
    job["clip"]["delete_native"] = False
    clip = extract_one(job)
    pd.DataFrame([clip]).to_csv(tmp_path / "clips.csv", index=False)

    a, la, ma, _ = FrameClipDataset(tmp_path / "frames.csv", tmp_path / "frames")[0]
    b, lb, mb, info = FrameClipDataset(tmp_path / "clips.csv", tmp_path / "clips")[0]
    assert a.shape == b.shape == (8, 3, 224, 224) and la == lb and ma.all() and mb.all()
    assert info["indices"] == [0, 2, 4, 6, 9, 11, 13, 15]
    # only difference: one more JPEG encode of the 224x224 frame (testsrc2 has very sharp edges)
    assert np.abs(a - b).mean() < 0.1

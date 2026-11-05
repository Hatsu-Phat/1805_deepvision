"""Map-style dataset over the extracted frames (works with torch DataLoader; torch is optional).

Reads either native frame folders (<split>_frames.csv + frames/) or resized clips
(<split>_clips.csv + clips/). Clips are already cropped and resized, so no crop is applied again.
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .sampling import uniform_indices
from .transforms import build_transform, demamba_crop_percentage, pad_clip


def decode_rgb(data: np.ndarray, name: str = "") -> np.ndarray:
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {name}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_rgb(path: Path) -> np.ndarray:
    return decode_rgb(np.fromfile(str(path), dtype=np.uint8), str(path))


def read_clip_frames(path: Path, indices) -> list[np.ndarray]:
    with tarfile.open(path, "r") as tar:
        return [decode_rgb(np.frombuffer(tar.extractfile(f"{j:03d}.jpg").read(), np.uint8), f"{path}:{j}")
                for j in indices]


class FrameClipDataset:
    """Yields (clip, label, mask, info).

    clip  float32 (num_frames, 3, size, size)
    mask  bool (num_frames,) True for real frames, False for padding
    info  dict with video_id / generator / selected frame indices

    Frames are chosen evenly over the stored window (`uniform_indices`), e.g. 8 of 16 -> 0,2,4,6,9,11,13,15.
    Pass `indices_fn` to plug in a different selector (e.g. a learned one) later.
    """

    def __init__(self, index_csv: str | Path, frames_root: str | Path, num_frames: int = 8,
                 profile: str = "demamba", size: int = 224, pad_mode: str = "zero",
                 crop: str = "demamba", indices_fn=None):
        self.df = pd.read_csv(index_csv)
        self.frames_root = Path(frames_root)
        self.num_frames = num_frames
        self.transform = build_transform(profile, size)
        self.pad_mode = pad_mode
        self.crop = crop
        self.indices_fn = indices_fn or uniform_indices
        self.clips = "clip_path" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def crop_for(self, row) -> float:
        if self.clips:
            return 0.0  # applied when the clip was written
        if self.crop == "demamba":
            return demamba_crop_percentage(row.split, row.generator, int(row.label))
        return float(self.crop or 0.0)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        n_valid = int(row.n_frames)
        idx = self.indices_fn(n_valid, self.num_frames)
        crop = self.crop_for(row)
        if self.clips:
            images = read_clip_frames(self.frames_root / row.clip_path, idx)
        else:
            folder = self.frames_root / row.frames_dir
            images = [read_rgb(folder / f"{j:03d}.jpg") for j in idx]
        frames = np.stack([self.transform(img, crop) for img in images])
        clip, mask = pad_clip(frames, self.num_frames, self.pad_mode)
        info = {"video_id": row.video_id, "generator": row.generator, "indices": idx.tolist()}
        return clip.astype(np.float32, copy=False), int(row.label), mask, info

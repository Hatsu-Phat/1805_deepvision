"""Map-style dataset over the extracted frames (works with torch DataLoader; torch is optional)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .sampling import uniform_indices
from .transforms import build_transform, demamba_crop_percentage, pad_clip


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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

    def __len__(self) -> int:
        return len(self.df)

    def crop_for(self, row) -> float:
        if self.crop == "demamba":
            return demamba_crop_percentage(row.split, row.generator, int(row.label))
        return float(self.crop or 0.0)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        folder = self.frames_root / row.frames_dir
        n_valid = int(row.n_frames)
        idx = self.indices_fn(n_valid, self.num_frames)
        crop = self.crop_for(row)
        frames = np.stack([self.transform(read_rgb(folder / f"{j:03d}.jpg"), crop) for j in idx])
        clip, mask = pad_clip(frames, self.num_frames, self.pad_mode)
        info = {"video_id": row.video_id, "generator": row.generator, "indices": idx.tolist()}
        return clip.astype(np.float32, copy=False), int(row.label), mask, info

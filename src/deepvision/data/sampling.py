"""Frame sampling rules.

DeMamba (Preprocess/video2frame.py) extracts frames with ffmpeg `-r <rate>` where
    rate = 2                  if duration >= 4 s
    rate = ceil(8 / duration) otherwise
which guarantees at least 8 frames per video. We keep the same rule but target 16 frames,
so the duration threshold and the numerator scale with the frame count:
    rate = base_fps                       if duration >= num_frames / base_fps   (8 s for 16 frames)
    rate = ceil(num_frames / duration)    otherwise
"""
from __future__ import annotations

import math

import numpy as np


def sampling_rate(duration: float, num_frames: int = 16, base_fps: int = 2,
                  min_duration_for_base_fps: float | None = None) -> int:
    if duration <= 0:
        raise ValueError("duration must be positive")
    threshold = min_duration_for_base_fps if min_duration_for_base_fps is not None else num_frames / base_fps
    if duration >= threshold:
        return base_fps
    return math.ceil(num_frames / duration)


def window_start(n_total: int, num_frames: int = 16, mode: str = "center") -> int:
    """First index of a window of `num_frames` consecutive frames out of `n_total`."""
    if n_total <= num_frames:
        return 0
    if mode == "center":
        return (n_total - num_frames) // 2
    if mode == "first":
        return 0
    raise ValueError(f"unknown window mode {mode!r}")


def uniform_indices(n_valid: int, k: int = 8) -> np.ndarray:
    """Pick `k` evenly spaced indices from `n_valid` stored frames (e.g. 8 of 16 -> 0,2,4,6,9,11,13,15).

    If fewer than `k` frames exist, every frame is returned once (padding is the loader's job).
    """
    if n_valid <= 0:
        return np.zeros(0, dtype=np.int64)
    if n_valid <= k:
        return np.arange(n_valid, dtype=np.int64)
    return np.round(np.linspace(0, n_valid - 1, k)).astype(np.int64)

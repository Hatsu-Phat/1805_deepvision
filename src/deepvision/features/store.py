"""Read the feature store written by features.extract.

    X, rows = load_features(out, "resnet50", "train", pooling="mean", device="cuda")
X is a float16 tensor (N, D) (or (N, 8, D) with pooling="frames") built shard by shard directly
on `device`, so a 2 GB train matrix never has to fit in the 4 GiB of host RAM.
"<encoder>:video" selects the X-CLIP video embedding instead of the per-frame features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

POOLINGS = ("mean", "frames", "first")


def parse_name(name: str) -> tuple[str, bool]:
    encoder, _, part = name.partition(":")
    if part not in ("", "video"):
        raise ValueError(f"unknown feature part {part!r}")
    return encoder, part == "video"


def shard_files(out: Path, encoder: str, split: str) -> list[Path]:
    """Finished shards (their .csv is written last)."""
    d = Path(out) / encoder / split
    return sorted(p for p in d.glob(f"{split}-*.csv"))


def load_features(out: Path, name: str, split: str, pooling: str = "mean", device="cpu",
                  keep: set[str] | None = None) -> tuple[torch.Tensor, pd.DataFrame]:
    """pooling: mean over the 8 frames | frames (keep all) | first (frame 0 only, no temporal information)."""
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}")
    encoder, video = parse_name(name)
    parts = shard_files(out, encoder, split)
    if not parts:
        raise FileNotFoundError(f"no finished shards in {Path(out) / encoder / split}")
    rows = [pd.read_csv(p).assign(shard=p.stem + ".tar", row=lambda d: np.arange(len(d))) for p in parts]
    masks = [np.ones(len(r), bool) if keep is None else r["video_id"].isin(keep).to_numpy() for r in rows]
    n = int(sum(m.sum() for m in masks))
    first = np.load(parts[0].with_name(parts[0].stem + (".video.npy" if video else ".npy")), mmap_mode="r")
    feat_shape = first.shape[1:] if video or pooling == "frames" else first.shape[2:]
    X = torch.empty((n, *feat_shape), dtype=torch.float16, device=device)
    i = 0
    for p, mask in zip(parts, masks):
        arr = np.load(p.with_name(p.stem + (".video.npy" if video else ".npy")), mmap_mode="r")
        chunk = torch.from_numpy(np.ascontiguousarray(arr[mask]))
        if not video and pooling == "mean":
            chunk = chunk.float().mean(1).half()
        elif not video and pooling == "first":
            chunk = chunk[:, 0]
        X[i:i + len(chunk)] = chunk.to(device)
        i += len(chunk)
    index = pd.concat([r[m] for r, m in zip(rows, masks)], ignore_index=True)
    return X, index

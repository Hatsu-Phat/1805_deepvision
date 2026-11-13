"""T0' (2026-11-13): structure function S(k) of low-level signals, straight from the local clips (no torch).

Per video (16 clip frames, grey, [0, 1]):
    pix: S(k) = mean_{t, pixels} (x(t+k) - x(t))^2
    res: same on the high-pass residual r = x - GaussianBlur(x, sigma)  (noise / fine texture)
Then the same statistics as features.structure (bootstrap CI, beta, AUROC), grouped by rate.

    python -m deepvision.features.lowlevel --clips H:/.../vif_native/clips \\
        --manifest H:/.../vif_native/manifests/vif_test_clips.csv --dest reports/runs/t0n
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LOG = logging.getLogger("deepvision")
SIGNALS = ("pix", "res")


def grey_frames(frames: list[bytes]) -> np.ndarray:
    """JPEG bytes -> float32 (T, H, W) grey in [0, 1]."""
    return np.stack([cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE) for b in frames]
                    ).astype(np.float32) / 255.0


def residual(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return np.stack([f - cv2.GaussianBlur(f, (0, 0), sigma) for f in x])


def curves(x: np.ndarray, sigma: float = 1.0) -> dict[str, np.ndarray]:
    """(T, H, W) -> {signal: (T-1,)} with S(k), k = 1..T-1."""
    out = {}
    for name, a in (("pix", x), ("res", residual(x, sigma))):
        out[name] = np.array([((a[k:] - a[:-k]) ** 2).mean() for k in range(1, len(a))], dtype=np.float64)
    return out


def _one(path: str) -> dict[str, np.ndarray]:
    from ..data.clips import read_clip

    frames, _ = read_clip(Path(path))
    return curves(grey_frames(frames))


def main(argv=None) -> None:
    from ..data.clips import clip_path_for
    from .structure import analyse_curves, video_kind

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="clips root of the data config")
    ap.add_argument("--manifest", required=True, help="<split>_clips.csv")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = pd.read_csv(args.manifest)
    rows = rows[rows["n_frames"] == 16].reset_index(drop=True)
    paths = [str(clip_path_for(Path(args.clips), r)) for r in rows.to_dict("records")]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(_one, paths, chunksize=16))
    s = {name: np.stack([r[name] for r in res]) for name in SIGNALS}
    rows["kind"] = [video_kind(g, l) for g, l in zip(rows["generator"], rows["label"])]
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    split = rows["split"].iloc[0]
    np.savez(dest / f"t0n_lowlevel_{split}_S.npz", **s, key=(rows["generator"] + "/" + rows["video_id"]).to_numpy())
    out = analyse_curves(s, rows, args.boot, args.seed)
    for name, df in out.items():
        df.to_csv(dest / f"t0n_lowlevel_{split}_{name}.csv", index=False)
    (dest / f"t0n_lowlevel_{split}.json").write_text(json.dumps(
        {"split": split, "videos": len(rows), "videos_16_unique": int((rows["n_unique"] == 16).sum()),
         "boot": args.boot, "seed": args.seed, "sigma": 1.0}, indent=2))
    LOG.info("%s", out["auroc"].query("group == 'fake_all'").round(3).to_dict("records"))


if __name__ == "__main__":
    main()

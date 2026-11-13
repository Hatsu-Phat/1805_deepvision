"""T0 (2026-11-13): the temporal structure function of frozen frame features, real vs fake.

For one video with 16 clip frames z(0..15) (even store + odd store, features.extract --frames odd)
    S(k) = mean_t || z(t+k) - z(t) ||^2,   k = 1..15,   dt = k / rate seconds (rate = clip sampling fps)
on L2-normalised features ("cos", 2 - 2 cos) and on raw features ("raw"). Per video we also fit the
log-log slope beta of S(k) over k = 1..8 (shape of the curve, independent of the amount of motion).
Groups are compared only on the same time grid (same rate), with a bootstrap 95 % CI over videos.

    python -m deepvision.features.structure --encoders vit_b16 dinov2_s14 xclip_b32 --split vif_test
Outputs in --dest: t0_<encoder>_<split>_{curves,diff,auroc}.csv and t0_<encoder>_<split>.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .extract import FRAME_SETS

LOG = logging.getLogger("deepvision")
K = np.arange(1, 16)
SLOPE_K = 8  # beta is fitted on k = 1..SLOPE_K
METRICS = ("cos", "raw")


def video_kind(generator: str, label: int) -> str:
    """I2V / T2V from the generator name; closed-source models whose mode is not in the name -> 'closed'."""
    if label == 0:
        return "real"
    if generator.endswith("-I"):  # Wan2.2-TI2V-5B-I / -T: the suffix is the mode, "TI2V" is the model
        return "I2V"
    if generator.endswith("-T"):
        return "T2V"
    if re.search(r"I2V", generator):
        return "I2V"
    if re.search(r"T2V", generator) or generator in ("HunyuanVideo", "SkyReels-V2"):
        return "T2V"
    return "closed"


def merge_frames(even: np.ndarray, odd: np.ndarray) -> np.ndarray:
    """(N, 8, D) even + (N, 8, D) odd -> (N, 16, D) in clip order."""
    z = np.empty((even.shape[0], 16, even.shape[2]), dtype=np.float32)
    z[:, FRAME_SETS["even"]] = even
    z[:, FRAME_SETS["odd"]] = odd
    return z


def structure_function(z: np.ndarray, normalise: bool) -> np.ndarray:
    """(N, T, D) -> (N, T-1): S(k) for k = 1..T-1."""
    if normalise:
        z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    return np.stack([((z[:, k:] - z[:, :-k]) ** 2).sum(-1).mean(1) for k in range(1, z.shape[1])], axis=1)


def loglog_slope(s: np.ndarray, kmax: int = SLOPE_K) -> np.ndarray:
    """Least-squares slope of log S(k) on log k, k = 1..kmax, per row."""
    x = np.log(np.arange(1, kmax + 1))
    y = np.log(np.maximum(s[:, :kmax], 1e-12))
    x = x - x.mean()
    return ((y - y.mean(1, keepdims=True)) * x).sum(1) / (x ** 2).sum()


def bootstrap_mean(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """(n, ...) -> (n_boot, ...) means of resampled rows."""
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return x[idx].mean(1)


def compare(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    """Mean of b - mean of a with a 95 % CI (independent bootstrap of both groups)."""
    d = bootstrap_mean(b, rng, n_boot) - bootstrap_mean(a, rng, n_boot)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    return {"diff": b.mean(0) - a.mean(0), "lo": lo, "hi": hi}


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def load_split(even_out: Path, odd_out: Path, encoder: str, split: str, index: pd.DataFrame
               ) -> tuple[np.ndarray, pd.DataFrame]:
    """16-frame features of the videos found in both stores, joined to the shard index on `key`."""
    from .store import load_features

    e, ei = load_features(even_out, encoder, split, pooling="frames")
    o, oi = load_features(odd_out, encoder, split, pooling="frames")
    pos = pd.Series(np.arange(len(oi)), index=oi["key"])
    common = ei["key"].isin(pos.index).to_numpy()
    z = merge_frames(e.numpy()[common], o.numpy()[pos[ei.loc[common, "key"]].to_numpy()])
    rows = ei[common].reset_index(drop=True)
    meta = index.drop_duplicates("key").set_index("key")
    for col in ("rate", "n_unique"):
        rows[col] = meta.loc[rows["key"], col].to_numpy()
    rows["kind"] = [video_kind(g, l) for g, l in zip(rows["generator"], rows["label"])]
    LOG.info("%s/%s: %d even, %d odd, %d joined", encoder, split, len(ei), len(oi), len(rows))
    return z, rows


def groups(rows: pd.DataFrame) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    """(name, rate, real mask, fake mask): all fakes, per kind and per generator, each against the reals
    with the same rate."""
    out = []
    for rate in sorted(rows["rate"].unique()):
        r = rows["rate"].to_numpy() == rate
        real = r & (rows["label"].to_numpy() == 0)
        fake = r & (rows["label"].to_numpy() == 1)
        if real.sum() < 20 or fake.sum() < 20:
            continue
        out.append(("fake_all", int(rate), real, fake))
        for col in ("kind", "generator"):
            for v in sorted(rows.loc[fake, col].unique()):
                m = fake & (rows[col].to_numpy() == v)
                if m.sum() >= 20:
                    out.append((f"{col}={v}", int(rate), real, m))
    return out


def analyse(z: np.ndarray, rows: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    keep = rows["n_unique"].to_numpy() == 16  # clips with repeated frames would give S(k) = 0
    z, rows = z[keep], rows[keep].reset_index(drop=True)
    curves, diffs, aurocs = [], [], []
    for metric in METRICS:
        s = structure_function(z, normalise=metric == "cos")
        beta = loglog_slope(s)
        for name, rate, real, fake in groups(rows):
            dt = K / rate
            for who, m in (("real", real), ("fake", fake)):
                if who == "real" and name != "fake_all":
                    continue  # the real curve of a rate is the same for every fake group
                bm = bootstrap_mean(s[m], rng, n_boot)
                lo, hi = np.percentile(bm, [2.5, 97.5], axis=0)
                curves.append(pd.DataFrame({"metric": metric, "rate": rate, "group": "real" if who == "real" else name,
                                            "n": int(m.sum()), "k": K, "dt": dt, "mean": s[m].mean(0),
                                            "lo": lo, "hi": hi}))
            c = compare(s[real], s[fake], rng, n_boot)
            rel = c["diff"] / s[real].mean(0)
            diffs.append(pd.DataFrame({"metric": metric, "rate": rate, "group": name, "n_real": int(real.sum()),
                                       "n_fake": int(fake.sum()), "k": K, "dt": dt, "fake_minus_real": c["diff"],
                                       "lo": c["lo"], "hi": c["hi"], "rel": rel,
                                       "ci_excludes_0": (c["lo"] > 0) | (c["hi"] < 0)}))
            cb = compare(beta[real], beta[fake], rng, n_boot)
            row = {"metric": metric, "rate": rate, "group": name, "n_real": int(real.sum()), "n_fake": int(fake.sum()),
                   "beta_real": float(beta[real].mean()), "beta_fake": float(beta[fake].mean()),
                   "beta_diff": float(cb["diff"]), "beta_lo": float(cb["lo"]), "beta_hi": float(cb["hi"]),
                   "auroc_beta": auroc(beta[real], beta[fake])}
            for k in (1, 4, 8, 15):
                row[f"auroc_S{k}"] = auroc(s[real, k - 1], s[fake, k - 1])
            aurocs.append(row)
    return {"curves": pd.concat(curves, ignore_index=True), "diff": pd.concat(diffs, ignore_index=True),
            "auroc": pd.DataFrame(aurocs)}


def main(argv=None) -> None:
    from ..data.stream import read_index

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", default=["vit_b16", "dinov2_s14", "xclip_b32"])
    ap.add_argument("--split", nargs="+", default=["vif_test", "vif_train"])
    ap.add_argument("--even", default="~/deepvision_features")
    ap.add_argument("--odd", default="~/deepvision_features_odd")
    ap.add_argument("--root", default="gdrive:DeepVision/vif_v1", help="where shards/<split>_index.csv is")
    ap.add_argument("--dest", default="~/deepvision_runs/t0")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rclone", default="rclone")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dest = Path(os.path.expanduser(args.dest))
    dest.mkdir(parents=True, exist_ok=True)
    even, odd = Path(os.path.expanduser(args.even)), Path(os.path.expanduser(args.odd))
    for split in args.split:
        index = read_index(args.root, split, args.rclone)
        for enc in args.encoders:
            z, rows = load_split(even, odd, enc, split, index)
            res = analyse(z, rows, args.boot, args.seed)
            stem = f"t0_{enc}_{split}"
            for name, df in res.items():
                df.to_csv(dest / f"{stem}_{name}.csv", index=False)
            d = res["diff"]
            summary = {"encoder": enc, "split": split, "videos": len(rows),
                       "videos_16_unique": int((rows["n_unique"] == 16).sum()), "boot": args.boot, "seed": args.seed,
                       "k_with_ci_excluding_0": {f"{m}/{r}/{g}": int(x["ci_excludes_0"].sum())
                                                 for (m, r, g), x in d.groupby(["metric", "rate", "group"])}}
            (dest / f"{stem}.json").write_text(json.dumps(summary, indent=2))
            LOG.info("%s: %s", stem, res["auroc"].query("group == 'fake_all'").round(3).to_dict("records"))
            del z


if __name__ == "__main__":
    main()

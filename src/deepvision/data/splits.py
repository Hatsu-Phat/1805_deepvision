"""Phase 2 splits built from the shard indices -> manifests/splits_<version>.csv (+ splits_<version>_counts.json).

Every indexed video gets one `split`:
    train     train videos used for fitting (everything not held out below)
    ood_real  `ood_real_frac` of every real train generator, the reals of val_ood, never trained on
    holdout   all videos of the held-out fake generator(s) (default SEINE), never trained on
    val_id    optional (`val_frac`, default 0): stratified share of every trained generator
    test      the test split (GenVideo-Val); also used to pick checkpoints (user decision 2026-11-05)
Evaluation sets (see eval_set): val_ood = holdout + ood_real; val_id; test.

    python -m deepvision.data.splits -c configs/data.yaml                      # v1: SEINE + 2% reals held out
    python -m deepvision.data.splits -c configs/data.yaml --version v2 --ood-real-frac 0 --holdout
                                       # v2: train on every train video, test only (user decision 2026-11-06)
    ... --version v3 --ood-real-frac 0 --holdout   # v3: as v2, plus the extra Youku reals of train2
The training pool is every <split>_index.csv of TRAIN_SPLITS that exists (train, and train2 once built).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .config import TRAIN_SPLITS, add_config_args, config_from_args

COLUMNS = ["video_id", "key", "shard", "label", "generator", "split"]
EVAL_SETS = ("val_id", "val_ood", "test")


def _sample(train: pd.DataFrame, mask: np.ndarray, frac: float, name: str, rng) -> None:
    """Move round(frac * n) videos of every generator selected by `mask` to split `name`."""
    for gen in sorted(train.loc[mask, "generator"].unique()):
        rows = np.flatnonzero(mask & (train["generator"] == gen).to_numpy())
        n = int(round(frac * len(rows)))
        train.loc[rng.choice(rows, n, replace=False), "split"] = name


def build_splits(train_index: pd.DataFrame, test_index: pd.DataFrame, ood_real_frac: float = 0.02,
                 holdout: tuple[str, ...] = ("SEINE",), val_frac: float = 0.0, seed: int = 2024) -> pd.DataFrame:
    train = train_index[COLUMNS[:-1]].sort_values("video_id").reset_index(drop=True)
    unknown = set(holdout) - set(train["generator"])
    if unknown:
        raise ValueError(f"holdout generators not in train: {sorted(unknown)}")
    if train.loc[train["generator"].isin(holdout), "label"].ne(1).any():
        raise ValueError("holdout generators must be fake")
    train["split"] = np.where(train["generator"].isin(holdout), "holdout", "train")
    rng = np.random.default_rng(seed)
    _sample(train, ((train["split"] == "train") & (train["label"] == 0)).to_numpy(), ood_real_frac, "ood_real", rng)
    if val_frac > 0:
        _sample(train, (train["split"] == "train").to_numpy(), val_frac, "val_id", rng)
    test = test_index[COLUMNS[:-1]].assign(split="test")
    out = pd.concat([train, test], ignore_index=True)
    check_splits(out)
    return out


def eval_set(splits: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "val_ood":
        mask = splits["split"].isin(["holdout", "ood_real"])
    elif name in ("train", "val_id", "test"):
        mask = splits["split"] == name
    else:
        raise ValueError(f"unknown set {name!r}")
    return splits[mask]


def check_splits(splits: pd.DataFrame) -> None:
    if splits["video_id"].duplicated().any():
        raise ValueError("a video_id appears in more than one split")
    trained_gens = set(splits.loc[splits["split"] == "train", "generator"])
    ood_fake_gens = set(eval_set(splits, "val_ood").query("label == 1")["generator"])
    if trained_gens & ood_fake_gens:
        raise ValueError(f"val_ood fakes come from trained generators: {sorted(trained_gens & ood_fake_gens)}")


def split_counts(splits: pd.DataFrame) -> dict:
    out = {}
    for name in ("train", *EVAL_SETS):
        df = eval_set(splits, name)
        out[name] = {"videos": len(df), "real": int((df["label"] == 0).sum()), "fake": int((df["label"] == 1).sum()),
                     "by_generator": {g: int(n) for g, n in df.groupby("generator").size().items()}}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--ood-real-frac", type=float, default=0.02)
    ap.add_argument("--holdout", nargs="*", default=["SEINE"], help="no value: hold out no generator")
    ap.add_argument("--val-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--version", default="v1")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    pool = [pd.read_csv(p) for s in TRAIN_SPLITS if (p := cfg.shards_dir / f"{s}_index.csv").exists()]
    splits = build_splits(pd.concat(pool, ignore_index=True), pd.read_csv(cfg.shards_dir / "test_index.csv"),
                          args.ood_real_frac, tuple(args.holdout), args.val_frac, args.seed)
    out = cfg.manifest_dir / f"splits_{args.version}.csv"
    splits.to_csv(out, index=False)
    counts = {"version": args.version, "ood_real_frac": args.ood_real_frac, "holdout": args.holdout,
              "val_frac": args.val_frac, "seed": args.seed} | split_counts(splits)
    (cfg.manifest_dir / f"splits_{args.version}_counts.json").write_text(json.dumps(counts, indent=2))
    print(out)
    print(json.dumps({k: {kk: v[kk] for kk in ("videos", "real", "fake")} for k, v in counts.items()
                      if isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()

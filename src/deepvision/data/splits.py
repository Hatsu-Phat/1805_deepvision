"""Phase 2 splits built from the shard indices -> manifests/splits_v1.csv (+ splits_v1_counts.json).

Every indexed video gets one `split`:
    train    train videos used for fitting
    val_id   `val_frac` of every train generator (real included), stratified, fixed seed
    holdout  all videos of the held-out fake generator(s) (default SEINE), never trained on
    test     the test split (GenVideo-Val), evaluated once at the end
Evaluation sets (see eval_set): val_id; val_ood = holdout fakes + val_id reals; test.

    python -m deepvision.data.splits -c configs/data.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .config import add_config_args, config_from_args

VERSION = "v1"
COLUMNS = ["video_id", "key", "shard", "label", "generator", "split"]
EVAL_SETS = ("val_id", "val_ood", "test")


def build_splits(train_index: pd.DataFrame, test_index: pd.DataFrame, val_frac: float = 0.02,
                 holdout: tuple[str, ...] = ("SEINE",), seed: int = 2024) -> pd.DataFrame:
    train = train_index[COLUMNS[:-1]].sort_values("video_id").reset_index(drop=True)
    unknown = set(holdout) - set(train["generator"])
    if unknown:
        raise ValueError(f"holdout generators not in train: {sorted(unknown)}")
    if train.loc[train["generator"].isin(holdout), "label"].ne(1).any():
        raise ValueError("holdout generators must be fake")
    train["split"] = np.where(train["generator"].isin(holdout), "holdout", "train")
    rng = np.random.default_rng(seed)
    for gen in sorted(train.loc[train["split"] == "train", "generator"].unique()):
        rows = np.flatnonzero((train["generator"] == gen).to_numpy())
        n_val = int(round(val_frac * len(rows)))
        train.loc[rng.choice(rows, n_val, replace=False), "split"] = "val_id"
    test = test_index[COLUMNS[:-1]].assign(split="test")
    out = pd.concat([train, test], ignore_index=True)
    check_splits(out)
    return out


def eval_set(splits: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "val_ood":
        mask = (splits["split"] == "holdout") | ((splits["split"] == "val_id") & (splits["label"] == 0))
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
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--holdout", nargs="+", default=["SEINE"])
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    splits = build_splits(pd.read_csv(cfg.shards_dir / "train_index.csv"), pd.read_csv(cfg.shards_dir / "test_index.csv"),
                          args.val_frac, tuple(args.holdout), args.seed)
    out = cfg.manifest_dir / f"splits_{VERSION}.csv"
    splits.to_csv(out, index=False)
    counts = {"version": VERSION, "val_frac": args.val_frac, "holdout": args.holdout, "seed": args.seed} \
        | split_counts(splits)
    (cfg.manifest_dir / f"splits_{VERSION}_counts.json").write_text(json.dumps(counts, indent=2))
    print(out)
    print(json.dumps({k: {kk: v[kk] for kk in ("videos", "real", "fake")} for k, v in counts.items()
                      if isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()

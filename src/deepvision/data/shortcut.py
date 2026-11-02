"""Metadata shortcut check: can real vs fake be told apart from file metadata alone?

A gradient boosting classifier is trained on metadata only (duration, fps, resolution, bitrate, ...).
High AUROC means the dataset carries a non-visual shortcut that a video model could also exploit.

Reports
    train -> test AUROC (cross-dataset)
    5-fold AUROC inside each split
    single-feature AUROC on test
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import add_config_args, config_from_args
from .io import setup_logging

FEATURES = ["duration", "fps", "width", "height", "aspect", "pixels", "bit_rate", "bytes_per_s", "nb_frames"]


def features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["duration"] = df["duration"]
    x["fps"] = df["fps"]
    x["width"] = df["width"]
    x["height"] = df["height"]
    x["aspect"] = df["width"] / df["height"].replace(0, np.nan)
    x["pixels"] = df["width"] * df["height"]
    x["bit_rate"] = pd.to_numeric(df["bit_rate"], errors="coerce")
    x["bytes_per_s"] = df["size_bytes"] / df["duration"].replace(0, np.nan)
    x["nb_frames"] = pd.to_numeric(df["nb_frames"], errors="coerce")
    return x[FEATURES].astype(float)


def _model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, random_state=seed)


def cv_auroc(x: pd.DataFrame, y: pd.Series, seed: int) -> float | None:
    if y.nunique() < 2:
        return None
    scores = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        scores[te] = _model(seed).fit(x.iloc[tr], y.iloc[tr]).predict_proba(x.iloc[te])[:, 1]
    return round(float(roc_auc_score(y, scores)), 4)


def single_feature_auroc(x: pd.DataFrame, y: pd.Series) -> dict:
    out = {}
    for c in x.columns:
        v = x[c].fillna(x[c].median())
        if v.nunique() < 2:
            continue
        auc = roc_auc_score(y, v)
        out[c] = round(float(max(auc, 1 - auc)), 4)  # direction-free
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--max-train", type=int, default=200_000, help="subsample train rows for speed")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    log = setup_logging(cfg.logs_dir / "shortcut.log")

    inv = pd.read_csv(cfg.inventory_dir / "inventory.csv")
    status = pd.read_csv(cfg.inventory_dir / "video_status.csv")
    df = inv[inv["path"].isin(status.loc[status["status"] == "ok", "path"])]
    report = {}
    splits = {s: g for s, g in df.groupby("split")}
    if "train" in splits and len(splits["train"]) > args.max_train:
        splits["train"] = splits["train"].sample(args.max_train, random_state=args.seed)

    for name, g in splits.items():
        report[f"{name}_cv_auroc"] = cv_auroc(features(g), g["label"], args.seed)
    if "train" in splits and "test" in splits:
        tr, te = splits["train"], splits["test"]
        clf = _model(args.seed).fit(features(tr), tr["label"])
        report["train_to_test_auroc"] = round(float(roc_auc_score(te["label"], clf.predict_proba(features(te))[:, 1])), 4)
    if "test" in splits:
        report["test_single_feature_auroc"] = single_feature_auroc(features(splits["test"]), splits["test"]["label"])
    if "train" in splits:
        report["train_single_feature_auroc"] = single_feature_auroc(features(splits["train"]), splits["train"]["label"])

    (cfg.inventory_dir / "shortcut_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("shortcut report:\n%s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Phase 2C: train heads on frozen features and evaluate them.

Heads (on the mean of the 8 frame features, standardized with train statistics):
    linear  logistic regression
    mlp     Linear(D, hidden) - GELU - Dropout - Linear(hidden, 1)
Splits come from manifests/splits_v1.csv: train on `train`; checkpoints are selected on test AUROC
(user decision 2026-11-05); val_ood (held-out SEINE fakes + held-out reals) is never used to train or select.

Each run <runs_root>/p2_<feature>_<head>_s<seed>/ gets config.yaml, env.txt, train.log, metrics.jsonl,
ckpt/{last,best}.pt, predictions_{test,val_ood}.csv and final.json with
    protocol A (threshold 0.5): AUROC, AP, acc, bAcc, F1, recall, TNR, precision
    TNR-95: same metrics at the threshold that keeps 95% of the val_ood reals below it
    bootstrap 95% CI of AUROC, per-generator AUROC / AP / recall
Finished runs are skipped, so the command can be rerun after an interruption.

    python -m deepvision.features.heads --features resnet50 xclip_b32 xclip_b32:video --heads linear mlp --seeds 0 1 2
    python -m deepvision.features.heads --summary      # one row per run -> <runs_root>/p2_summary.csv
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset

from ..data.splits import eval_set
from ..data.stream import read_text
from ..eval.metrics import binary_metrics, evaluate, threshold_at_tnr
from ..train.loop import Trainer, TrainConfig, predict
from ..train.run import expand, prepare_run, set_seed, setup_device
from .encoders import ENCODERS
from .store import load_features

LOG = logging.getLogger("deepvision")
FEATURES = (*ENCODERS, "xclip_b32:video")
DEFAULT_TRAIN = {"epochs": 20, "batch_size": 4096, "lr": 1e-3, "weight_decay": 1e-4, "warmup_steps": 100,
                 "sampler": "balanced", "log_every": 50, "select": "test/auroc", "amp": "off"}


class Standardize(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(1e-6))

    def forward(self, x):
        return (x.float() - self.mean) / self.std


def build_head(kind: str, dim: int, mean: torch.Tensor, std: torch.Tensor, hidden: int = 512,
               dropout: float = 0.5) -> nn.Module:
    if kind == "linear":
        body = nn.Linear(dim, 1)
    elif kind == "mlp":
        body = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    else:
        raise ValueError(f"unknown head {kind!r}")
    return nn.Sequential(Standardize(mean, std), body, nn.Flatten(0))


def feature_stats(X: torch.Tensor, chunk: int = 65536) -> tuple[torch.Tensor, torch.Tensor]:
    s = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
    s2 = torch.zeros_like(s)
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk].double()
        s += x.sum(0)
        s2 += (x * x).sum(0)
    mean = s / len(X)
    return mean, (s2 / len(X) - mean ** 2).clamp_min(0).sqrt()


def read_splits(path: str, rclone: str = "rclone") -> pd.DataFrame:
    if ":" in path and not Path(path).exists():  # remote file
        root, _, rel = path.rpartition("/")
        return pd.read_csv(io.StringIO(read_text(root, rel, rclone)))
    return pd.read_csv(path)


def load_sets(out: Path, feature: str, splits: pd.DataFrame, device) -> dict:
    """train / test / val_ood tensors and their rows (labels and generators from the feature index)."""
    sets = {}
    for name, split in (("train", "train"), ("val_ood", "train"), ("test", "test")):
        ids = set(eval_set(splits, name)["video_id"])
        X, rows = load_features(out, feature, split, "mean", device, keep=ids)
        missing = len(ids) - len(rows)
        if missing:
            LOG.warning("%s/%s: %d videos of the split have no features yet", feature, name, missing)
        sets[name] = (X, rows)
    return sets


def run_one(cfg: dict, feature: str, head: str, seed: int, sets: dict, device) -> dict:
    run_id = f"p2_{feature.replace(':', '-')}_{head}_s{seed}"
    run_cfg = cfg | {"feature": feature, "head": head, "train": cfg["train"] | {"seed": seed}}
    run_dir = expand(cfg["runs_root"]) / run_id
    if (run_dir / "final.json").exists():
        return json.loads((run_dir / "final.json").read_text())
    run_dir, log = prepare_run(run_cfg, run_id, resume=(run_dir / "config.yaml").exists())
    tcfg = TrainConfig.from_dict(run_cfg["train"])
    Xtr, rtr = sets["train"]
    ytr = torch.as_tensor(rtr["label"].to_numpy(copy=True), device=device)
    mean, std = feature_stats(Xtr)
    set_seed(seed)
    model = build_head(head, Xtr.shape[1], mean, std, **cfg.get("head", {})).to(device)
    evalsets = {k: (TensorDataset(sets[k][0], torch.zeros(len(sets[k][0]), device=device)), sets[k][1])
                for k in ("test", "val_ood")}

    def evaluate_fn(m):
        return {k: binary_metrics(r["label"], predict(m, ds, 8192, device, tcfg.amp)) for k, (ds, r) in evalsets.items()}

    t0 = time.time()
    Trainer(model, tcfg, run_dir, device, evaluate_fn).fit(TensorDataset(Xtr, ytr), rtr["label"].to_numpy())
    best = torch.load(run_dir / "ckpt" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    scores = {k: predict(model, ds, 8192, device, tcfg.amp) for k, (ds, _) in evalsets.items()}
    ood_rows = evalsets["val_ood"][1]
    t95 = threshold_at_tnr(scores["val_ood"][ood_rows["label"].to_numpy() == 0], 0.95)
    final = {"run_id": run_id, "feature": feature, "head": head, "seed": seed, "best_epoch_step": best["progress"]["best_step"],
             "train_videos": len(rtr), "train_seconds": round(time.time() - t0, 1), "tnr95_threshold": t95,
             "params": sum(p.numel() for p in model.parameters())}
    for k, (_, rows) in evalsets.items():
        df = rows[["video_id", "generator", "label"]].assign(score=scores[k])
        df.to_csv(run_dir / f"predictions_{k}.csv", index=False)
        final[k] = evaluate(df, 0.5, t95, n_boot=cfg.get("n_boot", 1000), seed=seed)
    (run_dir / "final.json").write_text(json.dumps(final, indent=2, default=float))
    log.info("%s: test AUROC %.4f AP %.4f F1 %.4f | val_ood AUROC %.4f", run_id, final["test"]["auroc"],
             final["test"]["ap"], final["test"]["f1"], final["val_ood"]["auroc"])
    return final


SUMMARY_KEYS = ("auroc", "ap", "acc", "bacc", "f1", "recall", "tnr", "precision")


def summary(runs_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs_root.glob("p2_*/final.json")):
        f = json.loads(path.read_text())
        row = {k: f[k] for k in ("run_id", "feature", "head", "seed", "train_videos")}
        for s in ("test", "val_ood"):
            row |= {f"{s}_{k}": f[s][k] for k in SUMMARY_KEYS}
            row |= {f"{s}_auroc_ci_low": f[s]["auroc_ci"][0], f"{s}_auroc_ci_high": f[s]["auroc_ci"][1]}
            row |= {f"{s}_tnr95_{k}": f[s]["at_tnr"][k] for k in ("acc", "bacc", "f1", "recall", "tnr")}
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    ap.add_argument("--heads", nargs="+", choices=["linear", "mlp"], default=["linear", "mlp"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="~/deepvision_features")
    ap.add_argument("--runs-root", default="~/deepvision_runs")
    ap.add_argument("--splits", default="gdrive:DeepVision/dataset_v1/manifests/splits_v1.csv")
    ap.add_argument("--set", action="append", default=[], help="train option override, e.g. epochs=30")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runs_root = expand(args.runs_root)
    if args.summary:
        df = summary(runs_root)
        df.to_csv(runs_root / "p2_summary.csv", index=False)
        print(df.groupby(["feature", "head"])[["test_auroc", "test_ap", "test_f1", "val_ood_auroc"]].agg(["mean", "std"]))
        return
    import yaml

    train = DEFAULT_TRAIN | {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = {"runs_root": str(runs_root), "features_dir": args.out, "splits": args.splits, "n_boot": args.n_boot,
           "gpu_memory_fraction": 0.8, "train": train, "head": {"hidden": 512, "dropout": 0.5}}
    device = setup_device(cfg["gpu_memory_fraction"])
    splits = read_splits(args.splits)
    for feature in args.features:
        sets = load_sets(expand(args.out), feature, splits, device)
        for head in args.heads:
            for seed in args.seeds:
                run_one(cfg, feature, head, seed, sets, device)
        del sets
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

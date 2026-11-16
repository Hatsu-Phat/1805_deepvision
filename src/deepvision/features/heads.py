"""Phase 2C: train heads on frozen features and evaluate them.

Heads (on the mean of the 8 frame features, standardized with train statistics):
    linear  logistic regression
    mlp     Linear(D, hidden) - GELU - Dropout - Linear(hidden, 1)
Splits come from manifests/splits_<version>.csv: train on `train`; checkpoints are selected on test AUROC
(user decision 2026-11-05). splits_v1 holds out val_ood (SEINE fakes + 2% of the reals), never used to train
or select; splits_v2 trains on every train video and evaluates on test only (user decision 2026-11-06).

Each run <runs_root>/<prefix>_<feature>_<head>_s<seed>/ gets config.yaml, env.txt, train.log, metrics.jsonl,
ckpt/{last,best}.pt, predictions_<set>.csv and final.json with
    protocol A (threshold 0.5): AUROC, AP, acc, bAcc, F1, recall, TNR, precision
    TNR-95: same metrics at the threshold that keeps 95% of the reals below it, the val_ood reals when the
            splits have a val_ood, otherwise the test reals (then it is TPR at 5% FPR on test)
    bootstrap 95% CI of AUROC, per-generator AUROC / AP / recall
    demamba: GenVideo protocol of DeMamba (per-generator R / P / F1 / AP, FP shared over all reals, mean)
Finished runs are skipped, so the command can be rerun after an interruption.

    python -m deepvision.features.heads --features resnet50 xclip_b32 xclip_b32:video --heads linear mlp --seeds 0 1 2
    python -m deepvision.features.heads --summary      # one row per run -> <runs_root>/<prefix>_summary.csv
    python -m deepvision.features.heads --splits gdrive:DeepVision/dataset_v1/manifests/splits_v2.csv --prefix p2f
    # DeMamba protocol (2026-11-09): MLP only, checkpoint picked on overall test accuracy at 0.5 like upstream
    python -m deepvision.features.heads --splits .../splits_v3.csv --heads mlp --set select=test/acc --prefix p2d
    # ViF (2026-11-09): train on ViF-CoT-4K, test on ViF-Bench with the paired protocol of Skyra
    python -m deepvision.features.heads --train-on vif_train --test-on vif_test --paired --heads mlp \
        --set select=test/vif_avg_bacc --set batch_size=256 --prefix p2v
    # zero-shot: score the best checkpoints of finished runs (e.g. p2d, trained on GenVideo) on ViF-Bench
    python -m deepvision.features.heads --crosseval p2d --test-on vif_test --paired --heads mlp
    python -m deepvision.features.heads --summary --prefix p2d --crosseval-tag vif_test
    # the other way (2026-11-11): ViF-CoT-4K heads on the GenVideo test split, under another name
    python -m deepvision.features.heads --crosseval p2v --test-on test --crosseval-tag genvideo_test --heads mlp
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

from ..data.config import TRAIN_SPLITS
from ..data.splits import eval_set
from ..data.stream import read_text
from ..eval.metrics import binary_metrics, demamba_protocol, evaluate, threshold_at_tnr, vif_paired_protocol
from ..train.loop import Trainer, TrainConfig, predict
from ..train.run import expand, prepare_run, set_seed, setup_device
from .encoders import ENCODERS
from .store import load_features, parse_name

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


def load_sets(out: Path, feature: str, splits: pd.DataFrame | None, device, train_on: list[str] | None = None,
              test_on: str = "test") -> dict:
    """train / test / val_ood (when the splits hold one out) tensors and their rows (labels and generators
    from the feature index). Train videos are read from every feature split of TRAIN_SPLITS that exists.
    With splits=None: every video of the feature splits `train_on` (if any) is trained on, every video of `test_on`
    is tested."""
    if splits is None:
        sets = {}
        for name, fsplits in (("train", train_on or []), ("test", [test_on])):
            if not fsplits:
                continue
            parts = [load_features(out, feature, s, "mean", device) for s in fsplits]
            X = parts[0][0] if len(parts) == 1 else torch.cat([p[0] for p in parts])
            sets[name] = (X, pd.concat([p[1] for p in parts], ignore_index=True))
        return sets
    sets = {}
    pool = [s for s in TRAIN_SPLITS if (Path(out) / parse_name(feature)[0] / s).exists()]
    for name, fsplits in (("train", pool), ("val_ood", pool), ("test", ["test"])):
        ids = set(eval_set(splits, name)["video_id"])
        if not ids and name == "val_ood":
            continue
        parts = [load_features(out, feature, s, "mean", device, keep=ids) for s in fsplits]
        X = parts[0][0] if len(parts) == 1 else torch.cat([p[0] for p in parts])
        rows = pd.concat([p[1] for p in parts], ignore_index=True)
        del parts
        missing = len(ids) - len(rows)
        if missing:
            LOG.warning("%s/%s: %d videos of the split have no features yet", feature, name, missing)
        sets[name] = (X, rows)
    return sets


def run_one(cfg: dict, feature: str, head: str, seed: int, sets: dict, device) -> dict:
    run_id = f"{cfg.get('prefix', 'p2')}_{feature.replace(':', '-')}_{head}_s{seed}"
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
                for k in ("test", "val_ood") if k in sets}

    def evaluate_fn(m):
        out = {}
        for k, (ds, r) in evalsets.items():
            s = predict(m, ds, 8192, device, tcfg.amp)
            out[k] = binary_metrics(r["label"], s)
            if "generator" in r and (r["label"] == 0).any() and (r["label"] == 1).any():
                d = demamba_protocol(r[["label", "generator"]].assign(score=s))
                out[k] |= {f"demamba_avg_{x}": d[f"avg_{x}"] for x in ("recall", "precision", "f1", "ap", "auroc")}
                if cfg.get("paired"):
                    v = vif_paired_protocol(r[["video_id", "label", "generator"]].assign(score=s))
                    out[k] |= {f"vif_avg_{x}": v[f"avg_{x}"] for x in VIF_KEYS}
        return out

    t0 = time.time()
    Trainer(model, tcfg, run_dir, device, evaluate_fn).fit(TensorDataset(Xtr, ytr), rtr["label"].to_numpy())
    best = torch.load(run_dir / "ckpt" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    scores = {k: predict(model, ds, 8192, device, tcfg.amp) for k, (ds, _) in evalsets.items()}
    ref = "val_ood" if "val_ood" in evalsets else "test"
    t95 = threshold_at_tnr(scores[ref][evalsets[ref][1]["label"].to_numpy() == 0], 0.95)
    final = {"run_id": run_id, "feature": feature, "head": head, "seed": seed, "best_epoch_step": best["progress"]["best_step"],
             "train_videos": len(rtr), "train_seconds": round(time.time() - t0, 1), "tnr95_threshold": t95,
             "tnr95_from": ref,
             "params": sum(p.numel() for p in model.parameters())}
    for k, (_, rows) in evalsets.items():
        df = rows[["video_id", "generator", "label"]].assign(score=scores[k])
        df.to_csv(run_dir / f"predictions_{k}.csv", index=False)
        final[k] = evaluate(df, 0.5, t95, n_boot=cfg.get("n_boot", 1000), seed=seed, paired=cfg.get("paired", False))
    (run_dir / "final.json").write_text(json.dumps(final, indent=2, default=float))
    dm = final["test"].get("demamba", {})
    log.info("%s: test AUROC %.4f AP %.4f F1 %.4f | DeMamba avg R %.4f F1 %.4f AP %.4f", run_id, final["test"]["auroc"],
             final["test"]["ap"], final["test"]["f1"], dm.get("avg_recall", float("nan")), dm.get("avg_f1", float("nan")),
             dm.get("avg_ap", float("nan")))
    return final


def load_head(run_dir: Path, device) -> nn.Module:
    """Best checkpoint of a finished run; dim, hidden size, mean and std come from the saved weights."""
    import yaml

    kind = yaml.safe_load((run_dir / "config.yaml").read_text())["head"]
    state = torch.load(run_dir / "ckpt" / "best.pt", map_location=device, weights_only=False)["model"]
    mean, std = state["0.mean"], state["0.std"]
    extra = {"hidden": state["1.0.weight"].shape[0]} if kind == "mlp" else {}
    model = build_head(kind, len(mean), mean, std, **extra)
    model.load_state_dict(state)
    return model.to(device).eval()


def crosseval(runs_root: Path, src_prefix: str, feature: str, heads: list[str], test: tuple, tag: str, device,
              paired: bool = False, n_boot: int = 1000) -> list[dict]:
    """Score the best checkpoint of every finished run <src_prefix>_<feature>_<head>_s* on another feature split
    (no training). Writes predictions_<tag>.csv and crosseval_<tag>.json into the run folder; done ones are skipped.
    The TNR-95 threshold is the one the run fitted on its own test reals."""
    X, rows = test
    ds = TensorDataset(X, torch.zeros(len(X), device=device))
    results = []
    for head in heads:
        for run_dir in sorted(runs_root.glob(f"{src_prefix}_{feature.replace(':', '-')}_{head}_s*")):
            out_json = run_dir / f"crosseval_{tag}.json"
            if out_json.exists():
                results.append(json.loads(out_json.read_text()))
                continue
            if not (run_dir / "final.json").exists():
                continue
            final = json.loads((run_dir / "final.json").read_text())
            model = load_head(run_dir, device)
            s = predict(model, ds, 8192, device, "off")
            df = rows[["video_id", "generator", "label"]].assign(score=s)
            df.to_csv(run_dir / f"predictions_{tag}.csv", index=False)
            res = {k: final[k] for k in ("run_id", "feature", "head", "seed", "train_videos")} | {
                "tag": tag, "tnr95_threshold": final["tnr95_threshold"],
                tag: evaluate(df, 0.5, final["tnr95_threshold"], n_boot=n_boot, seed=final["seed"], paired=paired)}
            out_json.write_text(json.dumps(res, indent=2, default=float))
            v = res[tag].get("vif_paired", {})
            LOG.info("%s on %s: AUROC %.4f AP %.4f | paired bAcc %.4f F1 %.4f", final["run_id"], tag, res[tag]["auroc"],
                     res[tag]["ap"], v.get("avg_bacc", float("nan")), v.get("avg_f1", float("nan")))
            results.append(res)
    return results


SUMMARY_KEYS = ("auroc", "ap", "acc", "bacc", "f1", "recall", "tnr", "precision")
VIF_KEYS = ("bacc", "recall", "f1", "real_acc", "ap", "auroc")


def summary(runs_root: Path, prefix: str = "p2", tag: str | None = None) -> pd.DataFrame:
    """One row per run from final.json, or from crosseval_<tag>.json when tag is given (its set is named tag)."""
    rows = []
    name = f"crosseval_{tag}.json" if tag else "final.json"
    for path in sorted(runs_root.glob(f"{prefix}_*/{name}")):
        f = json.loads(path.read_text())
        row = {k: f[k] for k in ("run_id", "feature", "head", "seed", "train_videos")}
        for s in (tag,) if tag else ("test", "val_ood") if "val_ood" in f else ("test",):
            row |= {f"{s}_{k}": f[s][k] for k in SUMMARY_KEYS}
            row |= {f"{s}_auroc_ci_low": f[s]["auroc_ci"][0], f"{s}_auroc_ci_high": f[s]["auroc_ci"][1]}
            row |= {f"{s}_tnr95_{k}": f[s]["at_tnr"][k] for k in ("acc", "bacc", "f1", "recall", "tnr")}
            if "demamba" in f[s]:
                d = f[s]["demamba"]
                row |= {f"{s}_demamba_real_acc": d["real_acc"]}
                row |= {f"{s}_demamba_avg_{k}": d[f"avg_{k}"] for k in ("recall", "precision", "f1", "ap", "auroc")}
            if "vif_paired" in f[s]:
                row |= {f"{s}_vif_avg_{k}": f[s]["vif_paired"][f"avg_{k}"] for k in VIF_KEYS}
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
    ap.add_argument("--prefix", default="p2", help="run id prefix; p2 = splits_v1 runs, p2f = splits_v2 runs")
    ap.add_argument("--train-on", nargs="+", default=None,
                    help="feature splits to train on (every video, no splits file), e.g. vif_train")
    ap.add_argument("--test-on", default=None, help="feature split to test on (every video), e.g. vif_test")
    ap.add_argument("--paired", action="store_true", help="also score the ViF-Bench paired protocol")
    ap.add_argument("--crosseval", default=None, metavar="PREFIX",
                    help="no training: score the finished runs PREFIX_* on --test-on (zero-shot)")
    ap.add_argument("--crosseval-tag", default=None,
                    help="name of the crosseval results (default: --test-on); with --summary: summarize crosseval_<tag>.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runs_root = expand(args.runs_root)
    if args.summary:
        tag = args.crosseval_tag
        df = summary(runs_root, args.prefix, tag)
        df.to_csv(runs_root / (f"{args.prefix}_{tag}_summary.csv" if tag else f"{args.prefix}_summary.csv"), index=False)
        s = tag or "test"
        cols = [c for c in (f"{s}_auroc", f"{s}_ap", f"{s}_f1", f"{s}_demamba_avg_f1", f"{s}_demamba_avg_ap",
                            f"{s}_vif_avg_bacc", f"{s}_vif_avg_f1", f"{s}_vif_avg_ap", "val_ood_auroc") if c in df]
        print(df.groupby(["feature", "head"])[cols].agg(["mean", "std"]))
        return
    import yaml

    train = DEFAULT_TRAIN | {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    vif = args.train_on is not None or args.crosseval is not None
    if vif and not args.test_on:
        ap.error("--train-on / --crosseval need --test-on")
    cfg = {"prefix": args.prefix, "runs_root": str(runs_root), "features_dir": args.out,
           "splits": None if vif else args.splits, "train_on": args.train_on, "test_on": args.test_on,
           "paired": args.paired, "n_boot": args.n_boot,
           "gpu_memory_fraction": 0.8, "train": train, "head": {"hidden": 512, "dropout": 0.5}}
    device = setup_device(cfg["gpu_memory_fraction"])
    splits = None if vif else read_splits(args.splits)
    for feature in args.features:
        if args.crosseval:
            test = load_sets(expand(args.out), feature, None, device, None, args.test_on)["test"]
            crosseval(runs_root, args.crosseval, feature, args.heads, test, args.crosseval_tag or args.test_on, device,
                      args.paired, args.n_boot)
            del test
            continue
        sets = load_sets(expand(args.out), feature, splits, device, args.train_on, args.test_on or "test")
        for head in args.heads:
            for seed in args.seeds:
                run_one(cfg, feature, head, seed, sets, device)
        del sets
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

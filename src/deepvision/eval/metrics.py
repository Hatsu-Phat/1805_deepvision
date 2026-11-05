"""Binary detection metrics (label 1 = fake, score = probability of fake).

    auroc                 main metric (rank based, equal to sklearn.metrics.roc_auc_score)
    average_precision     AP of the fake class (sklearn.metrics.average_precision_score)
    binary_metrics        acc / bAcc / F1 / TNR / recall / precision at a threshold (Protocol A: 0.5)
    threshold_at_tnr      threshold that keeps `tnr` of the reals below it (fit on val reals)
    bootstrap_auroc_ci    stratified bootstrap CI
    per_generator         AUROC of each fake generator against all reals of the set, recall, CI
    evaluate              everything above for one prediction table
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


def _arrays(y, s) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(np.int64).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError("labels and scores differ in length")
    return y, s


def auroc(y, s) -> float:
    """Mann-Whitney U / (n_pos * n_neg) with tied scores counted as 1/2; NaN if one class is missing."""
    y, s = _arrays(y, s)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y, s) -> float:
    y, s = _arrays(y, s)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, s))


def binary_metrics(y, s, threshold: float = 0.5) -> dict:
    y, s = _arrays(y, s)
    pred = s >= threshold
    pos, neg = y == 1, y == 0
    tp, fn = int((pred & pos).sum()), int((~pred & pos).sum())
    tn, fp = int((~pred & neg).sum()), int((pred & neg).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return {"n": len(y), "n_fake": int(pos.sum()), "n_real": int(neg.sum()), "threshold": float(threshold),
            "auroc": auroc(y, s), "ap": average_precision(y, s), "acc": (tp + tn) / len(y) if len(y) else float("nan"),
            "bacc": (recall + tnr) / 2,
            "f1": f1, "tnr": tnr, "recall": recall, "precision": precision}


def threshold_at_tnr(real_scores, tnr: float = 0.95) -> float:
    """Smallest threshold t with mean(real_scores < t) >= tnr (a score >= t is called fake)."""
    s = np.sort(np.asarray(real_scores, dtype=np.float64).ravel())
    if len(s) == 0:
        raise ValueError("no real scores")
    k = math.ceil(tnr * len(s) - 1e-9)  # 0.95 * 100 is 95.00000000000001 in floating point
    if k == 0:
        return float(s[0])
    return float(np.nextafter(s[k - 1], np.inf))


def bootstrap_auroc_ci(y, s, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile CI; reals and fakes are resampled separately so every replicate has both classes."""
    y, s = _arrays(y, s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = auroc(labels, np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def per_generator(df: pd.DataFrame, threshold: float = 0.5, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """df: label, score, generator. One row per fake generator (vs all reals) plus one per real source."""
    reals = df[df["label"] == 0]
    rows = []
    for gen, part in df.groupby("generator", sort=True):
        if (part["label"] == 0).all():
            rows.append({"generator": gen, "label": 0, "n": len(part), "auroc": float("nan"), "ap": float("nan"),
                         "ci_low": float("nan"), "ci_high": float("nan"), "recall": float("nan"),
                         "tnr": float((part["score"] < threshold).mean())})
            continue
        fakes = part[part["label"] == 1]
        y = np.r_[np.ones(len(fakes)), np.zeros(len(reals))]
        s = np.r_[fakes["score"].to_numpy(), reals["score"].to_numpy()]
        lo, hi = bootstrap_auroc_ci(y, s, n_boot, seed=seed)
        rows.append({"generator": gen, "label": 1, "n": len(fakes), "auroc": auroc(y, s),
                     "ap": average_precision(y, s), "ci_low": lo,
                     "ci_high": hi, "recall": float((fakes["score"] >= threshold).mean()), "tnr": float("nan")})
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, threshold: float = 0.5, tnr_threshold: float | None = None,
             n_boot: int = 1000, seed: int = 0) -> dict:
    """df: label, score[, generator]. tnr_threshold: e.g. threshold_at_tnr(val reals) for the TNR-95 protocol."""
    y, s = df["label"].to_numpy(), df["score"].to_numpy()
    out = binary_metrics(y, s, threshold)
    out["auroc_ci"] = list(bootstrap_auroc_ci(y, s, n_boot, seed=seed))
    if tnr_threshold is not None:
        out["at_tnr"] = binary_metrics(y, s, tnr_threshold)
    if "generator" in df:
        out["per_generator"] = per_generator(df, threshold, n_boot, seed).to_dict("records")
    return out

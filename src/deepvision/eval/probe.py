"""G3 (2026-11-14): frozen-feature probes on ViF with a held-out validation split and nuisance-balanced metrics.

CPU only (numpy + scikit-learn), so it runs where torch does not. Features: a store written by
features.extract (<store>/<encoder>/<split>/<split>-NNNNN.{csv,npy}, (N, T, D) float16).

Protocol
  train  = ViF-CoT-4K minus a 10 % validation split drawn by video_id (a real and its paired fake share
           the id, so a pair never straddles train and val); seed-controlled.
  model  = StandardScaler + LogisticRegression (C in a grid) or MLP(512) with early stopping; the model and
           its hyper-parameter are picked on validation AUROC only. Input: mean over frames, optionally
           concatenated with the frame std ("meanstd").
  test   = ViF-Bench, scored once. Thresholds: 0.5 on the class-balanced model; TPR at 5 % FPR uses the
           threshold of the validation reals (tpr_fpr5_val) and, for reference, of the test reals.
Nuisance-balanced metrics (C2): given per-video nuisance covariates z (processing features that must not carry
the label), a logistic propensity e(z) = P(fake | z) is fit on the test set itself; real videos get weight
1 / (1 - e), fake 1 / e (each class normalised to its count). Weighted AUROC / bAcc then compare reals and
fakes as if z had the same distribution in both. `nuisance_check` reports the weighted AUROC of every
covariate (should be ~0.5 after weighting).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_store(store: Path, encoder: str, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = Path(store) / encoder / split
    parts = sorted(d.glob(f"{split}-*.csv"))
    if not parts:
        raise FileNotFoundError(f"no feature parts in {d}")
    rows = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    feats = np.concatenate([np.load(p.with_suffix(".npy")).astype(np.float32) for p in parts])
    if len(rows) != len(feats):
        raise ValueError(f"{d}: {len(rows)} rows but {len(feats)} feature rows")
    return rows, feats


def pool(feats: np.ndarray, mode: str = "mean") -> np.ndarray:
    m = feats.mean(1)
    return np.concatenate([m, feats.std(1)], 1) if mode == "meanstd" else m


def split_val(rows: pd.DataFrame, frac: float = 0.1, seed: int = 0) -> np.ndarray:
    """Boolean mask of validation rows, drawn by video_id."""
    ids = np.array(sorted(rows["video_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    val_ids = set(rng.choice(ids, size=max(1, int(round(frac * len(ids)))), replace=False))
    return rows["video_id"].astype(str).isin(val_ids).to_numpy()


def candidates(seed: int):
    for c in (0.001, 0.01, 0.1, 1.0):
        yield f"logreg_C{c}", make_pipeline(StandardScaler(), LogisticRegression(
            C=c, max_iter=3000, class_weight="balanced", random_state=seed))
    yield "mlp512", make_pipeline(StandardScaler(), MLPClassifier(
        (512,), alpha=1e-3, early_stopping=True, validation_fraction=0.1, max_iter=200, random_state=seed))


def fit_select(Xtr, ytr, Xva, yva, seed: int = 0):
    best = None
    for name, m in candidates(seed):
        m.fit(Xtr, ytr)
        a = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        if best is None or a > best[1]:
            best = (name, a, m)
    return best


def propensity_weights(z: np.ndarray, y: np.ndarray, seed: int = 0, model: str = "hgb", folds: int = 5,
                       clip: float = 0.02) -> np.ndarray:
    """Inverse-propensity weights. model="hgb": gradient boosting, cross-fitted over `folds` stratified folds
    (each video's propensity comes from a model that did not see it), so non-linear nuisances such as a
    categorical width are balanced; model="logreg": a linear logistic fit on everything."""
    z = np.nan_to_num(np.asarray(z, float))
    y = np.asarray(y)
    if model == "logreg":
        e = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, random_state=seed)).fit(z, y)
        p = e.predict_proba(z)[:, 1]
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold

        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(z, y):
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                               min_samples_leaf=20, random_state=seed).fit(z[tr], y[tr])
            p[te] = m.predict_proba(z[te])[:, 1]
    p = np.clip(p, clip, 1 - clip)
    w = np.where(y == 1, 1 / p, 1 / (1 - p))
    for c in (0, 1):
        w[y == c] *= (y == c).sum() / w[y == c].sum()
    return w


def weighted_bacc(y, pred, w):
    return 0.5 * sum(np.average(pred[y == c] == c, weights=w[y == c]) for c in (0, 1))


def metrics(y, s, thr=0.5, w=None, thr_val_fpr5=None) -> dict:
    y, s = np.asarray(y), np.asarray(s)
    out = {"auroc": roc_auc_score(y, s), "bacc": balanced_accuracy_score(y, s > thr)}
    t_test = np.quantile(s[y == 0], 0.95)
    out["tpr_fpr5_test"] = float(np.mean(s[y == 1] > t_test))
    if thr_val_fpr5 is not None:
        out["tpr_fpr5_val"] = float(np.mean(s[y == 1] > thr_val_fpr5))
        out["fpr_at_val_thr"] = float(np.mean(s[y == 0] > thr_val_fpr5))
    if w is not None:
        out["auroc_w"] = roc_auc_score(y, s, sample_weight=w)
        out["bacc_w"] = weighted_bacc(y, (s > thr).astype(int), w)
    return out


def nuisance_check(z: pd.DataFrame, y, w) -> pd.DataFrame:
    rows = []
    for c in z.columns:
        v = np.nan_to_num(z[c].to_numpy(float))
        rows.append({"covariate": c, "auroc": roc_auc_score(y, v), "auroc_w": roc_auc_score(y, v, sample_weight=w)})
    return pd.DataFrame(rows)

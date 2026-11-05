import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from deepvision.eval.metrics import (auroc, binary_metrics, bootstrap_auroc_ci, evaluate, per_generator,
                                     threshold_at_tnr)


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.random(500) * 0.6 + 0.4 * y, 2)  # rounded -> many ties
    return y, s


def test_auroc_matches_sklearn(scores):
    y, s = scores
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5
    assert math.isnan(auroc([1, 1], [0.1, 0.2]))


def test_binary_metrics_match_sklearn(scores):
    y, s = scores
    m = binary_metrics(y, s, 0.5)
    pred = (s >= 0.5).astype(int)
    assert m["acc"] == pytest.approx((pred == y).mean())
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["bacc"] == pytest.approx(balanced_accuracy_score(y, pred))
    assert m["tnr"] == pytest.approx(((pred == 0) & (y == 0)).sum() / (y == 0).sum())
    assert m["n_fake"] + m["n_real"] == m["n"] == 500


def test_threshold_at_tnr():
    real = np.arange(100) / 100
    t = threshold_at_tnr(real, 0.95)
    assert (real < t).mean() == 0.95
    assert binary_metrics(np.zeros(100), real, t)["tnr"] == pytest.approx(0.95)
    tied = np.full(10, 0.3)
    assert (tied < threshold_at_tnr(tied, 0.95)).mean() == 1.0


def test_bootstrap_ci(scores):
    y, s = scores
    lo, hi = bootstrap_auroc_ci(y, s, n_boot=200, seed=1)
    assert lo < auroc(y, s) < hi and hi - lo < 0.15
    assert (lo, hi) == bootstrap_auroc_ci(y, s, n_boot=200, seed=1)


def test_per_generator_and_evaluate():
    df = pd.DataFrame({"label": [0, 0, 0, 0, 1, 1, 1, 1], "generator": ["R"] * 4 + ["A", "A", "B", "B"],
                       "score": [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.25, 0.6]})
    pg = per_generator(df, n_boot=50).set_index("generator")
    assert pg.loc["A", "auroc"] == 1.0 and pg.loc["A", "recall"] == 1.0
    assert pg.loc["B", "auroc"] == pytest.approx(roc_auc_score([1, 1, 0, 0, 0, 0], [0.25, 0.6, 0.1, 0.2, 0.3, 0.4]))
    assert pg.loc["R", "tnr"] == 1.0 and pg.loc["B", "n"] == 2
    out = evaluate(df, tnr_threshold=threshold_at_tnr(df.query("label == 0")["score"], 0.75), n_boot=50)
    assert out["auroc"] == pytest.approx(roc_auc_score(df["label"], df["score"]))
    assert out["at_tnr"]["tnr"] == 0.75 and len(out["per_generator"]) == 3

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from deepvision.eval import probe


def test_split_val_keeps_pairs_together():
    rows = pd.DataFrame({"video_id": np.repeat([f"v{i}" for i in range(50)], 2), "label": np.tile([0, 1], 50)})
    m = probe.split_val(rows, 0.2, seed=0)
    assert 15 <= m.sum() <= 25
    per_id = rows.assign(m=m).groupby("video_id")["m"].nunique()
    assert (per_id == 1).all()


def test_propensity_weights_remove_a_confounder():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    z = rng.normal(size=n) + 1.5 * y  # nuisance strongly tied to the label
    assert roc_auc_score(y, z) > 0.8
    w = probe.propensity_weights(z[:, None], y)
    assert abs(roc_auc_score(y, z, sample_weight=w) - 0.5) < 0.05
    chk = probe.nuisance_check(pd.DataFrame({"z": z}), y, w)
    assert abs(chk["auroc_w"].iloc[0] - 0.5) < 0.05


def test_fit_select_and_metrics():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] + 0.3 * rng.normal(size=400) > 0).astype(int)
    name, a, m = probe.fit_select(X[:300], y[:300], X[300:], y[300:])
    assert a > 0.9
    s = m.predict_proba(X[300:])[:, 1]
    out = probe.metrics(y[300:], s, w=np.ones(100), thr_val_fpr5=0.9)
    assert out["auroc"] > 0.9 and abs(out["auroc_w"] - out["auroc"]) < 1e-9

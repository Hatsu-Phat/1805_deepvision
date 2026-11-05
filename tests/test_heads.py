import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from deepvision.features.heads import build_head, feature_stats, load_sets, run_one, summary  # noqa: E402
from deepvision.features.store import load_features  # noqa: E402


def write_store(out, encoder, split, shards, per_shard, dim, video_dim=0, seed=0):
    """Fake extract output: label 1 shifts the features, so the task is learnable."""
    rng = np.random.default_rng(seed)
    d = out / encoder / split
    d.mkdir(parents=True)
    rows = []
    for s in range(shards):
        stem = f"{split}-{s:05d}"
        labels = rng.integers(0, 2, per_shard)
        gens = np.where(labels == 1, np.where(np.arange(per_shard) % 3 == 0, "SEINE", "G"), "real")
        if split == "test":
            gens = np.where(labels == 1, "T", "R")
        feats = rng.normal(size=(per_shard, 8, dim)) + labels[:, None, None] * 1.0
        feats[:, :, 0] = np.arange(8)  # frame index in dim 0
        np.save(d / f"{stem}.npy", feats.astype(np.float16))
        if video_dim:
            np.save(d / f"{stem}.video.npy", (rng.normal(size=(per_shard, video_dim)) + labels[:, None]).astype(np.float16))
        ids = [f"{split}{s}_{i}" for i in range(per_shard)]
        pd.DataFrame({"video_id": ids, "label": labels, "generator": gens, "key": ids}).to_csv(d / f"{stem}.csv", index=False)
        rows.append(pd.DataFrame({"video_id": ids, "label": labels, "generator": gens}))
    (d / f"{split}-{shards:05d}.tmp.npy").write_bytes(b"unfinished")  # ignored
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def store(tmp_path):
    out = tmp_path / "feat"
    tr = write_store(out, "enc", "train", 3, 200, 6, video_dim=4)
    te = write_store(out, "enc", "test", 1, 150, 6, video_dim=4, seed=1)
    split = np.where(tr["generator"] == "SEINE", "holdout", "train").astype(object)
    real = np.flatnonzero(tr["label"].to_numpy() == 0)[:20]
    split[real] = "ood_real"
    splits = pd.concat([tr.assign(split=split), te.assign(split="test")], ignore_index=True)
    return out, splits


def test_load_features_poolings(store):
    out, _ = store
    X, rows = load_features(out, "enc", "train", "mean")
    assert X.shape == (600, 6) and X.dtype == torch.float16 and len(rows) == 600
    assert torch.allclose(X[:, 0].float(), torch.full((600,), 3.5), atol=1e-2)  # mean of frame indices 0..7
    F, _ = load_features(out, "enc", "train", "frames")
    assert F.shape == (600, 8, 6)
    first, _ = load_features(out, "enc", "train", "first")
    assert (first[:, 0] == 0).all()
    V, rows = load_features(out, "enc:video", "test", keep={"test0_1", "test0_5"})
    assert V.shape == (2, 4) and rows["video_id"].tolist() == ["test0_1", "test0_5"]
    assert rows["row"].tolist() == [1, 5] and rows["shard"].iloc[0] == "test-00000.tar"


def test_head_shapes_and_stats():
    X = torch.randn(1000, 5) * 3 + 2
    mean, std = feature_stats(X.half(), chunk=128)
    assert torch.allclose(mean.float(), X.mean(0), atol=0.05) and torch.allclose(std.float(), X.std(0), atol=0.05)
    for kind in ("linear", "mlp"):
        assert build_head(kind, 5, mean, std, hidden=8)(X.half()).shape == (1000,)
    with pytest.raises(ValueError):
        build_head("svm", 5, mean, std)


@pytest.mark.parametrize("feature", ["enc", "enc:video"])
def test_run_one_trains_evaluates_and_resumes(store, tmp_path, feature):
    out, splits = store
    sets = load_sets(out, feature, splits, "cpu")
    assert len(sets["train"][1]) == len(splits.query("split == 'train'"))
    assert set(sets["val_ood"][1]["generator"]) == {"SEINE", "real"}
    cfg = {"runs_root": str(tmp_path / "runs"), "n_boot": 20, "head": {"hidden": 16, "dropout": 0.1},
           "train": {"epochs": 3, "batch_size": 64, "lr": 0.01, "sampler": "balanced", "log_every": 5,
                     "select": "test/auroc", "amp": "off"}}
    final = run_one(cfg, feature, "mlp", 0, sets, "cpu")
    assert final["test"]["auroc"] > 0.9 and 0 < final["test"]["ap"] <= 1
    assert {"f1", "recall", "precision", "bacc", "tnr", "auroc_ci", "at_tnr", "per_generator"} <= set(final["test"])
    assert final["val_ood"]["at_tnr"]["tnr"] >= 0.95
    run_dir = tmp_path / "runs" / f"p2_{feature.replace(':', '-')}_mlp_s0"
    preds = pd.read_csv(run_dir / "predictions_test.csv")
    assert len(preds) == 150 and preds["score"].between(0, 1).all()
    assert run_one(cfg, feature, "mlp", 0, sets, "cpu") == json.loads((run_dir / "final.json").read_text())
    row = summary(tmp_path / "runs").iloc[0]
    assert row["test_auroc"] == final["test"]["auroc"] and row["val_ood_tnr95_tnr"] >= 0.95

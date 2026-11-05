import json
import subprocess
import sys

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import TensorDataset  # noqa: E402

from deepvision.eval.metrics import binary_metrics  # noqa: E402
from deepvision.train.checkpoint import CheckpointManager  # noqa: E402
from deepvision.train.loop import EpochSampler, Trainer, TrainConfig, cosine_schedule, predict  # noqa: E402
from deepvision.train.run import load_run_config, prepare_run, set_seed  # noqa: E402


def toy_data(n=96, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = (torch.rand(n, generator=g) < 0.25).long()  # imbalanced like the dataset
    x = torch.randn(n, d, generator=g) + y[:, None] * 1.5
    return TensorDataset(x, y), y.numpy()


def make_model(seed=0):
    set_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1))


def eval_fn(data, labels):
    def evaluate(m):
        return {"val_id": binary_metrics(labels, predict(m, data, batch_size=32))}
    return evaluate


def records(run_dir):
    return [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]


def test_epoch_sampler():
    labels = np.array([0] * 90 + [1] * 10)
    s = EpochSampler(labels, "balanced", seed=3)
    order = s.order()
    assert len(order) == 100 and 0.35 < labels[order].mean() < 0.65
    assert np.array_equal(order, EpochSampler(labels, "balanced", seed=3).order())
    s.set_epoch(1, start=40)
    assert len(s) == 60 and list(s) == s.order()[40:].tolist() and not np.array_equal(s.order(), order)
    sh = EpochSampler(labels, "shuffle", seed=0)
    assert sorted(sh.order()) == list(range(100))


def test_cosine_schedule():
    f = cosine_schedule(100, warmup_steps=10, min_ratio=0.1)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(100) == pytest.approx(0.1) and f(55) == pytest.approx(0.55)


def test_checkpoint_atomic(tmp_path):
    ck = CheckpointManager(tmp_path)
    assert ck.load() is None
    ck.save({"a": 1}, is_best=True)
    ck.save({"a": 2})
    assert ck.load()["a"] == 2 and ck.load("best")["a"] == 1
    assert not list(ck.dir.glob("*.tmp"))


def test_trainer_learns_and_keeps_best(tmp_path):
    data, labels = toy_data()
    cfg = TrainConfig(epochs=8, batch_size=16, lr=0.02, log_every=2, select="val_id/auroc")
    progress = Trainer(make_model(), cfg, tmp_path, evaluate_fn=eval_fn(data, labels)).fit(data, labels)
    evals = [r for r in records(tmp_path) if r["set"] == "val_id"]
    assert len(evals) == 8 and evals[-1]["auroc"] > 0.9
    assert progress["best"] == max(r["auroc"] for r in evals)
    best = CheckpointManager(tmp_path).load("best")
    assert best["progress"]["best_step"] == progress["best_step"]


@pytest.mark.parametrize("kill_at", [7, 13, 30])  # mid-epoch, right after a checkpoint, an epoch end
def test_resume_matches_uninterrupted(tmp_path, kill_at):
    data, labels = toy_data()
    cfg = TrainConfig(epochs=5, batch_size=16, lr=0.01, ckpt_every=4, log_every=1, sampler="balanced",
                      select="val_id/auroc")
    full = Trainer(make_model(), cfg, tmp_path / "a", evaluate_fn=eval_fn(data, labels)).fit(data, labels)

    Trainer(make_model(), cfg, tmp_path / "b", evaluate_fn=eval_fn(data, labels)).fit(data, labels, stop_after=kill_at)
    resumed = Trainer(make_model(seed=123), cfg, tmp_path / "b", evaluate_fn=eval_fn(data, labels)).fit(data, labels)

    assert resumed == full

    def strip(rs):
        return [{k: v for k, v in r.items() if k != "samples_per_s"} for r in rs]
    assert strip(records(tmp_path / "b")) == strip(records(tmp_path / "a"))
    sa = CheckpointManager(tmp_path / "a").load()["model"]
    sb = CheckpointManager(tmp_path / "b").load()["model"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_fresh_run_ignores_old_metrics(tmp_path):
    data, labels = toy_data()
    cfg = TrainConfig(epochs=1, batch_size=32, log_every=1)
    Trainer(make_model(), cfg, tmp_path).fit(data, labels)
    Trainer(make_model(), cfg, tmp_path).fit(data, labels, resume=False)
    assert [r["step"] for r in records(tmp_path)] == [1, 2, 3]


def test_config_and_run_dir(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"runs_root": str(tmp_path / "runs"), "train": {"lr": 0.1}}))
    cfg = load_run_config(path, ["train.lr=0.5", "train.sampler=shuffle", "data.n_videos=8"])
    assert cfg["train"] == {"lr": 0.5, "sampler": "shuffle"} and cfg["data"] == {"n_videos": 8}
    with pytest.raises(ValueError, match="unknown train options"):
        TrainConfig.from_dict({"lr": 1, "lrr": 2})
    run_dir, _ = prepare_run(cfg, "r1")
    assert (run_dir / "config.yaml").exists() and "torch" in (run_dir / "env.txt").read_text()
    with pytest.raises(FileExistsError):
        prepare_run(cfg, "r1")
    prepare_run(cfg, "r1", resume=True)
    with pytest.raises(ValueError, match="config differs"):
        prepare_run(cfg | {"train": {"lr": 1.0}}, "r1", resume=True)


def test_smoke_overfit_and_kill_resume(tmp_path):
    """The server smoke script end to end on a local shard (real subprocess kill)."""
    from tests.test_stream import make_shards

    paths, _ = make_shards(tmp_path / "data", n_shards=1, per_shard=8)
    cfg = {"runs_root": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache"),
           "data": {"root": str(tmp_path / "data"), "shard": paths[0], "split": "train", "n_videos": 8},
           "model": {"width": 4},
           "train": {"epochs": 6, "batch_size": 4, "lr": 0.01, "sampler": "shuffle", "ckpt_every": 3,
                     "log_every": 1, "select": "train_eval/acc"},
           "resume": {"kill_at_step": 5}}
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = subprocess.run([sys.executable, "-m", "deepvision.train.smoke", "resume", "-c", str(path),
                          "--run-id", "t"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    result = json.loads(out.stdout[out.stdout.rindex("\n{") + 1:])  # the report printed after the log lines
    assert result["child_exit_code"] != 0 and result["max_metric_diff"] == 0
    # across processes CPU conv kernels may differ in the last float32 bit
    assert result["max_train_loss_diff"] < 1e-6 and result["steps_compared"] == 12
    assert (tmp_path / "cache" / "smoke_train_8.npz").exists()

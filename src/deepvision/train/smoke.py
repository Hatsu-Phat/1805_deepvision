"""Server smoke tests (3) and (4) of Phase 2 (PROJECT_PLAN "Kế hoạch kiểm thử").

overfit  a tiny CNN trained on `data.n_videos` videos (half real, half fake) of one shard must reach
         100% accuracy on those videos.
resume   run A trains without interruption; run B is the same run in a child process that is killed
         (SIGKILL) at `resume.kill_at_step` and then resumed from ckpt/last.pt. The final metrics of
         A and B are compared.

    python -m deepvision.train.smoke overfit -c configs/smoke.yaml [--set train.epochs=50]
    python -m deepvision.train.smoke resume -c configs/smoke.yaml
The videos are streamed once and cached in <cache_dir>/smoke_<split>_<n>.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from ..data.stream import ShardStream
from ..eval.metrics import binary_metrics
from .loop import Trainer, TrainConfig, predict
from .run import expand, load_run_config, prepare_run, set_seed, setup_device


class TinyVideoNet(nn.Module):
    """Per-frame conv stack, mean over space and frames, one logit. Input uint8 (B, T, 3, H, W)."""

    def __init__(self, width: int = 32):
        super().__init__()
        layers, c_in = [], 3
        for c_out in (width, 2 * width, 4 * width):
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.ReLU()]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        f = self.features(x.flatten(0, 1).float() / 255).mean(dim=(2, 3))  # deterministic, unlike adaptive pooling
        return self.head(f.view(b, t, -1).mean(1)).view(-1)


def load_videos(cfg: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """First n/2 real and n/2 fake videos of data.shard as uint8 (N, T, 3, H, W)."""
    d = cfg["data"]
    n = int(d["n_videos"])
    cache = expand(cfg.get("cache_dir", "~/deepvision_cache")) / f"smoke_{d['split']}_{n}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["frames"], z["labels"], z["video_ids"].tolist()
    stream = ShardStream(d["root"], [d["shard"]], num_frames=int(d.get("num_frames", 8)), streams=1,
                         decode="cpu", rclone=d.get("rclone", "rclone"))
    picked: dict[int, list[dict]] = {0: [], 1: []}
    for video in stream:
        if len(picked[video["label"]]) < n // 2:
            picked[video["label"]].append(video)
        if all(len(v) == n // 2 for v in picked.values()):
            break
    else:
        raise RuntimeError(f"{d['shard']} has fewer than {n // 2} videos of each label")
    videos = picked[0] + picked[1]
    frames = np.stack([v["frames"] for v in videos])
    labels = np.array([v["label"] for v in videos], dtype=np.int64)
    video_ids = [v["video_id"] for v in videos]
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.stem + ".tmp.npz")
    np.savez(tmp, frames=frames, labels=labels, video_ids=np.array(video_ids))
    os.replace(tmp, cache)
    return frames, labels, video_ids


def overfit(cfg: dict, run_id: str, resume: bool = False, kill_at_step: int | None = None) -> dict:
    """Train on the cached videos and return the last evaluation on them."""
    run_dir, log = prepare_run(cfg, run_id, resume)
    device = setup_device(cfg.get("gpu_memory_fraction"))
    tcfg = TrainConfig.from_dict(cfg.get("train", {}))
    set_seed(tcfg.seed, deterministic=True)
    frames, labels, _ = load_videos(cfg)
    data = TensorDataset(torch.from_numpy(frames), torch.from_numpy(labels))
    model = TinyVideoNet(**cfg.get("model", {}))

    def evaluate(m):
        scores = predict(m, data, batch_size=tcfg.batch_size, device=device, amp=tcfg.amp)
        return {"train_eval": binary_metrics(labels, scores)}

    log.info("run %s on %s: %d videos, frames %s", run_id, device, len(labels), frames.shape)
    t0 = time.time()
    progress = Trainer(model, tcfg, run_dir, device, evaluate).fit(data, labels, resume=resume,
                                                                   stop_after=kill_at_step)
    if kill_at_step is not None and progress["global_step"] >= kill_at_step:
        log.info("killing myself at step %d", progress["global_step"])
        for h in log.handlers:
            h.flush()
        if hasattr(signal, "SIGKILL"):
            os.kill(os.getpid(), signal.SIGKILL)
        os._exit(137)
    result = last_eval(run_dir / "metrics.jsonl") | {"seconds": round(time.time() - t0, 1)}
    if torch.cuda.is_available():
        result["max_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    log.info("final: %s", json.dumps(result))
    return result


def read_records(path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def last_eval(path) -> dict:
    return [r for r in read_records(path) if r["set"] == "train_eval"][-1]


def train_losses(path) -> dict[int, float]:
    return {r["step"]: r["loss"] for r in read_records(path) if r["set"] == "train"}


def resume_check(cfg: dict, run_id: str, config_path: str, overrides: list[str]) -> dict:
    kill_at = int(cfg["resume"]["kill_at_step"])
    a = overfit(cfg, f"{run_id}_a")
    cmd = [sys.executable, "-m", "deepvision.train.smoke", "overfit", "-c", config_path, "--run-id", f"{run_id}_b",
           "--kill-at-step", str(kill_at), *[x for o in overrides for x in ("--set", o)]]
    code = subprocess.run(cmd).returncode
    if code == 0:
        raise RuntimeError("child run was not killed")
    b = overfit(cfg, f"{run_id}_b", resume=True)
    runs_root = expand(cfg.get("runs_root", "~/deepvision_runs"))
    la, lb = train_losses(runs_root / f"{run_id}_a" / "metrics.jsonl"), train_losses(runs_root / f"{run_id}_b" / "metrics.jsonl")
    keys = ("auroc", "acc", "f1")
    return {"kill_at_step": kill_at, "child_exit_code": code,
            "final_a": {k: a[k] for k in keys}, "final_b": {k: b[k] for k in keys},
            "max_metric_diff": max(abs(a[k] - b[k]) for k in keys),
            "steps_compared": len(set(la) & set(lb)),
            "max_train_loss_diff": max(abs(la[s] - lb[s]) for s in set(la) & set(lb))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["overfit", "resume"])
    ap.add_argument("-c", "--config", default="configs/smoke.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. train.epochs=50")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--kill-at-step", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    cfg = load_run_config(args.config, args.set)
    run_id = args.run_id or f"smoke_{args.command}_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.command == "overfit":
        result = overfit(cfg, run_id, args.resume, args.kill_at_step)
        result["pass"] = result["acc"] == 1.0
    else:
        result = resume_check(cfg, run_id, args.config, args.set)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

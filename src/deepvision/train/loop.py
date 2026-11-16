"""Generic binary-classification training loop with exact resume.

The dataset is indexed with a list of indices and returns a whole batch (x, label), like
TensorDataset (whose tensors may already sit on the GPU); label 1 = fake; one logit per sample.
Batch order depends only on (seed, epoch), so a run resumed from ckpt/last.pt replays the same
batches with the same RNG state as an uninterrupted run. Every log record goes to
<run_dir>/metrics.jsonl; records written after the checkpoint a run resumes from are dropped.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, SequentialSampler

from .checkpoint import CheckpointManager, rng_state, set_rng_state

LOG = logging.getLogger("deepvision")


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0       # cosine decays to lr * min_lr_ratio
    grad_clip: float = 0.0          # 0 = off
    amp: str = "auto"               # auto: bf16 autocast on a CUDA GPU that supports it | bf16 | off
    seed: int = 0
    sampler: str = "balanced"       # balanced: class-balanced draws with replacement | shuffle
    pos_weight: float | None = None  # BCE weight of the fake class (alternative to the balanced sampler)
    num_workers: int = 0
    ckpt_every: int = 0             # steps between mid-epoch checkpoints; 0 = end of epoch only
    log_every: int = 50
    select: str = "test/auroc"      # metric (higher is better) that picks ckpt/best.pt (user decision 2026-11-05)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown train options: {sorted(unknown)}")
        return cls(**d)


class EpochSampler(Sampler):
    """Deterministic sample order for (seed, epoch); `start` skips samples already seen when resuming."""

    def __init__(self, labels, mode: str = "balanced", seed: int = 0):
        if mode not in ("balanced", "shuffle"):
            raise ValueError(f"unknown sampler {mode!r}")
        self.labels = np.asarray(labels).astype(np.int64)
        self.mode = mode
        self.seed = seed
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = epoch, start

    def order(self) -> np.ndarray:
        rng = np.random.default_rng([self.seed, self.epoch])
        n = len(self.labels)
        if self.mode == "shuffle":
            return rng.permutation(n)
        counts = np.bincount(self.labels, minlength=2).astype(np.float64)
        weights = 1.0 / counts[self.labels]
        return rng.choice(n, n, replace=True, p=weights / weights.sum())

    def __iter__(self):
        return iter(self.order()[self.start:].tolist())

    def __len__(self) -> int:
        return max(0, len(self.labels) - self.start)


def cosine_schedule(total_steps: int, warmup_steps: int = 0, min_ratio: float = 0.0) -> Callable[[int], float]:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    return factor


def autocast_dtype(amp: str, device: torch.device) -> torch.dtype | None:
    if amp == "off" or device.type != "cuda":
        return None
    if amp == "auto" and not torch.cuda.is_bf16_supported():
        return None
    return torch.bfloat16


@torch.no_grad()
def predict(model: torch.nn.Module, dataset: Dataset, batch_size: int = 512, device="cpu",
            amp: str = "auto", num_workers: int = 0) -> np.ndarray:
    """Probability of fake for every sample, in dataset order."""
    device = torch.device(device)
    dtype = autocast_dtype(amp, device)
    model.eval()
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers,
                        sampler=BatchSampler(SequentialSampler(range(len(dataset))), batch_size, drop_last=False),
                        generator=torch.Generator().manual_seed(0))
    out = []
    for x, _ in loader:
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(x.to(device, non_blocking=True))
        out.append(torch.sigmoid(logits.float().view(-1)).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def lookup(metrics: dict, path: str) -> float:
    value = metrics
    for part in path.split("/"):
        value = value[part]
    return float(value)


class Trainer:
    """evaluate_fn(model) -> {set name: {metric: value}} runs after every epoch; the `select`
    metric decides ckpt/best.pt. stop_after (steps) ends fit() without saving, like a kill."""

    def __init__(self, model: torch.nn.Module, cfg: TrainConfig, run_dir: str | Path, device="cpu",
                 evaluate_fn: Callable[[torch.nn.Module], dict] | None = None):
        self.model = model
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        self.evaluate_fn = evaluate_fn
        self.ckpt = CheckpointManager(self.run_dir)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.dtype = autocast_dtype(cfg.amp, self.device)
        pos_weight = None if cfg.pos_weight is None else torch.tensor(cfg.pos_weight, device=self.device)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def log_record(self, record: dict) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")

    def _truncate_metrics(self, global_step: int) -> None:
        if not self.metrics_path.exists():
            return
        kept = []
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line)["step"] <= global_step:
                    kept.append(line)
            except (json.JSONDecodeError, KeyError):
                continue  # partial line from a kill
        self.metrics_path.write_text("".join(l + "\n" for l in kept), encoding="utf-8")

    def _save(self, opt, sched, progress: dict, is_best: bool) -> None:
        self.ckpt.save({"model": self.model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "progress": dict(progress), "rng": rng_state(),
                        "config": dataclasses.asdict(self.cfg)}, is_best)

    def fit(self, dataset: Dataset, labels, resume: bool = True, stop_after: int | None = None) -> dict:
        cfg = self.cfg
        self.model.to(self.device)
        sampler = EpochSampler(labels, cfg.sampler, cfg.seed)
        steps_per_epoch = math.ceil(len(sampler.labels) / cfg.batch_size)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, cosine_schedule(cfg.epochs * steps_per_epoch, cfg.warmup_steps, cfg.min_lr_ratio))
        progress = {"epoch": 0, "step_in_epoch": 0, "global_step": 0, "best": None, "best_step": None}

        saved = self.ckpt.load() if resume else None
        if saved is not None:
            self.model.load_state_dict(saved["model"])
            opt.load_state_dict(saved["optimizer"])
            sched.load_state_dict(saved["scheduler"])
            set_rng_state(saved["rng"])
            progress = saved["progress"]
            self._truncate_metrics(progress["global_step"])
            LOG.info("resumed from step %d (epoch %d, step %d in epoch)", progress["global_step"],
                     progress["epoch"], progress["step_in_epoch"])
        elif self.metrics_path.exists():
            self.metrics_path.unlink()

        for epoch in range(progress["epoch"], cfg.epochs):
            sampler.set_epoch(epoch, progress["step_in_epoch"] * cfg.batch_size)
            loader = DataLoader(dataset, batch_size=None, num_workers=cfg.num_workers,
                                sampler=BatchSampler(sampler, cfg.batch_size, drop_last=False),
                                generator=torch.Generator().manual_seed(cfg.seed))
            self.model.train()
            t0, seen = time.time(), 0
            for x, y in loader:
                loss = self._step(opt, sched, x, y)
                progress["step_in_epoch"] += 1
                progress["global_step"] += 1
                seen += len(y)
                step = progress["global_step"]
                if step % cfg.log_every == 0 or progress["step_in_epoch"] == steps_per_epoch:
                    self.log_record({"step": step, "epoch": epoch, "set": "train", "loss": loss,
                                     "lr": sched.get_last_lr()[0], "samples_per_s": seen / (time.time() - t0)})
                if stop_after is not None and step >= stop_after:
                    return progress
                if cfg.ckpt_every and step % cfg.ckpt_every == 0 and progress["step_in_epoch"] < steps_per_epoch:
                    self._save(opt, sched, progress, False)
            progress["epoch"], progress["step_in_epoch"] = epoch + 1, 0
            is_best = False
            if self.evaluate_fn is not None:
                metrics = self.evaluate_fn(self.model)
                for name, values in metrics.items():
                    self.log_record({"step": progress["global_step"], "epoch": epoch, "set": name} | values)
                score = lookup(metrics, cfg.select)
                is_best = progress["best"] is None or score > progress["best"]
                if is_best:
                    progress["best"], progress["best_step"] = score, progress["global_step"]
                LOG.info("epoch %d: %s = %.4f (best %.4f)", epoch, cfg.select, score, progress["best"])
            self._save(opt, sched, progress, is_best)
        return progress

    def _step(self, opt, sched, x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True).float()
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype is not None):
            logits = self.model(x)
        loss = self.loss_fn(logits.float().view(-1), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        opt.step()
        sched.step()
        return float(loss.detach())

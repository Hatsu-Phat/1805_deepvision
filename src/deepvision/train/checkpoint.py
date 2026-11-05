"""Atomic checkpoints with everything needed to resume a run exactly.

<run_dir>/ckpt/last.pt  written after every checkpoint interval
<run_dir>/ckpt/best.pt  copy of the checkpoint with the best validation score
Files are written to a temporary name and renamed, so a kill never leaves a truncated checkpoint.
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch


def atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointManager:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir) / "ckpt"

    @property
    def last(self) -> Path:
        return self.dir / "last.pt"

    @property
    def best(self) -> Path:
        return self.dir / "best.pt"

    def save(self, state: dict, is_best: bool = False) -> None:
        atomic_save(state, self.last)
        if is_best:
            tmp = self.best.with_name("best.pt.tmp")
            shutil.copyfile(self.last, tmp)
            os.replace(tmp, self.best)

    def load(self, which: str = "last", map_location="cpu") -> dict | None:
        path = self.best if which == "best" else self.last
        if not path.exists():
            return None
        return torch.load(path, map_location=map_location, weights_only=False)

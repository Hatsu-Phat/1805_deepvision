"""Run setup shared by the training scripts: YAML config with overrides, run directory, seeds, GPU limits.

<runs_root>/<run_id>/config.yaml   resolved config (a resumed run must match it)
<runs_root>/<run_id>/env.txt       Python / torch / CUDA / GPU / git commit / pip freeze
<runs_root>/<run_id>/train.log
"""
from __future__ import annotations

import logging
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from ..data.io import setup_logging

REPO = Path(__file__).resolve().parents[3]


def load_run_config(path: str | Path, overrides: list[str] = ()) -> dict:
    """overrides: "train.lr=0.01" style; values are parsed as YAML."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def expand(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_device(gpu_memory_fraction: float | None = None) -> torch.device:
    """CUDA when available, capped at a fraction of the GPU memory (the GPU is shared)."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
    return torch.device("cuda")


def _cmd(args: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_commit() -> str:
    commit = _cmd(["git", "rev-parse", "HEAD"], REPO) or "unknown"
    dirty = _cmd(["git", "status", "--porcelain", "--untracked-files=no"], REPO)
    return commit + ("-dirty" if dirty else "")


def env_text() -> str:
    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}",
             f"torch {torch.__version__} cuda {torch.version.cuda} cudnn {torch.backends.cudnn.version()}",
             f"git {git_commit()}"]
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        lines.append(f"gpu {props.name} {props.total_memory // 2**20} MiB capability {props.major}.{props.minor}")
    lines += ["", "pip freeze:", _cmd([sys.executable, "-m", "pip", "freeze"])]
    return "\n".join(lines) + "\n"


def prepare_run(cfg: dict, run_id: str, resume: bool = False):
    """Create <runs_root>/<run_id>, write config.yaml + env.txt, and start train.log.
    A resumed run must have the same config; a new run must not reuse an existing directory."""
    run_dir = expand(cfg.get("runs_root", "~/deepvision_runs")) / run_id
    config_path = run_dir / "config.yaml"
    text = yaml.safe_dump(cfg, sort_keys=False)
    if config_path.exists():
        if not resume:
            raise FileExistsError(f"{run_dir} exists; pass --resume or choose another run id")
        if yaml.safe_load(config_path.read_text(encoding="utf-8")) != cfg:
            raise ValueError(f"config differs from {config_path}; resume needs the same config")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(text, encoding="utf-8")
    with open(run_dir / "env.txt", "a", encoding="utf-8") as f:
        f.write(env_text())
    log = setup_logging()
    for h in [h for h in log.handlers if isinstance(h, logging.FileHandler)]:  # a previous run in this process
        log.removeHandler(h)
        h.close()
    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    fh.setFormatter(log.handlers[0].formatter)
    log.addHandler(fh)
    return run_dir, log

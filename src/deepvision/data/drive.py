"""Google Drive transfer through rclone (https://rclone.org/drive/).

Upload (Windows/local machine):
    python -m deepvision.data.drive push            # finished shards, indices and manifests
Shards are *moved*: rclone verifies each upload before the local copy is deleted, and the shard
name is recorded in shards/<split>_uploaded.txt so it is never rebuilt. clips/ stays local.

Read on a server without storing the dataset (needs an rclone remote of the same name):
    for key, frames, meta in stream_shard("gdrive:DeepVision/dataset_v1", "shards/train/train-00000.tar"):
        ...
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

from .config import SPLITS, DataConfig, add_config_args, config_from_args
from .io import setup_logging
from .shards import iter_shard


def rclone_binary(cfg: DataConfig | None = None) -> str:
    name = (cfg.drive.get("rclone") if cfg else None) or "rclone"
    found = shutil.which(name)
    if found:
        return found
    winget = Path.home() / "AppData/Local/Microsoft/WinGet/Links/rclone.exe"
    if winget.exists():
        return str(winget)
    raise FileNotFoundError("rclone not found; install it and configure a Drive remote (rclone config)")


def run_rclone(args: list[str], cfg: DataConfig | None = None) -> None:
    subprocess.run([rclone_binary(cfg), *args], check=True)


def stream_shard(remote: str, path: str, rclone: str = "rclone") -> Iterator[tuple[str, list[bytes], dict]]:
    """Yield (key, frames, meta) from a shard on the remote without writing it to disk."""
    proc = subprocess.Popen([shutil.which(rclone) or rclone, "cat", f"{remote.rstrip('/')}/{path}"],
                            stdout=subprocess.PIPE)
    try:
        yield from iter_shard(proc.stdout)
    finally:
        proc.stdout.close()
        if proc.wait() != 0:
            raise RuntimeError(f"rclone cat failed for {path}")


def push(cfg: DataConfig, log, transfers: int = 4) -> None:
    remote = cfg.drive["remote"].rstrip("/")
    common = ["--transfers", str(transfers), "--checkers", "8", "--stats", "60s", "--stats-one-line"]
    for split in SPLITS:
        src = cfg.shards_dir / split
        tars = sorted(p.name for p in src.glob(f"{split}-*.tar")) if src.exists() else []
        if not tars:
            continue
        log.info("push %s: %d shards", split, len(tars))
        files = cfg.logs_dir / f"push_{split}.txt"
        files.write_text("\n".join(tars) + "\n")
        run_rclone(["move", str(src), f"{remote}/shards/{split}", "--files-from", str(files), *common], cfg)
        moved = [t for t in tars if not (src / t).exists()]
        with open(cfg.shards_dir / f"{split}_uploaded.txt", "a") as f:
            f.writelines(t + "\n" for t in moved)
        log.info("push %s: %d shards uploaded", split, len(moved))
    # small files: per-shard sidecars, split indices, manifests
    run_rclone(["copy", str(cfg.shards_dir), f"{remote}/shards", "--include", "*.csv", *common], cfg)
    run_rclone(["copy", str(cfg.manifest_dir), f"{remote}/manifests", "--include", "*_videos.csv",
                "--include", "counts.json", "--include", "splits_*", *common], cfg)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("command", choices=["push"])
    ap.add_argument("--transfers", type=int, default=4)
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    log = setup_logging(cfg.logs_dir / "drive.log")
    if args.command == "push":
        push(cfg, log, args.transfers)


if __name__ == "__main__":
    main()

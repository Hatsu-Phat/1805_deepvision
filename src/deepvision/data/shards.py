"""Step 5: pack clips into ~1 GB shards for upload and streaming.

Shard i of a split holds videos [i * N, (i + 1) * N) of <split>_videos.csv, which is already in
priority order, so any prefix of shards stays balanced. A shard is written only once every video
of its block has a clip or an extraction error, and an existing shard is never rewritten.

Inside a shard each video is a directory of byte-identical copies of its clip members:
    <generator>/<video_id>/000.jpg ... 015.jpg, <generator>/<video_id>/meta.json
meta.json comes last, so a reader can stream a shard sequentially (see iter_shard).

    python -m deepvision.data.shards --split train test
Outputs shards/<split>/<split>-00000.tar, a sidecar .csv per shard and shards/<split>_index.csv.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from pathlib import Path
from typing import IO, Iterator

import pandas as pd

from .clips import clip_path_for, read_clip
from .config import DataConfig, add_config_args, config_from_args
from .io import setup_logging


def shard_options(cfg: DataConfig) -> dict:
    return {"videos_per_shard": 3000} | cfg.shards


def shard_name(split: str, i: int) -> str:
    return f"{split}-{i:05d}.tar"


def write_shard(path: Path, videos: list[tuple[str, list[bytes], dict]]) -> int:
    """videos: (key, frames, meta). Written atomically; returns the file size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tarfile.open(tmp, "w", format=tarfile.USTAR_FORMAT) as tar:
        for key, frames, meta in videos:
            members = [(f"{j:03d}.jpg", b) for j, b in enumerate(frames)]
            members.append(("meta.json", json.dumps(meta).encode("utf-8")))
            for name, data in members:
                info = tarfile.TarInfo(f"{key}/{name}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    os.replace(tmp, path)
    return path.stat().st_size


def iter_shard(fileobj: IO[bytes]) -> Iterator[tuple[str, list[bytes], dict]]:
    """Stream (key, frames, meta) from a shard; works on non-seekable streams (pipes)."""
    frames: dict[str, bytes] = {}
    with tarfile.open(fileobj=fileobj, mode="r|") as tar:
        for m in tar:
            key, name = m.name.rsplit("/", 1)
            data = tar.extractfile(m).read()
            if name == "meta.json":
                yield key, [frames[f"{j:03d}.jpg"] for j in range(len(frames))], json.loads(data)
                frames = {}
            else:
                frames[name] = data


def block_status(cfg: DataConfig, block: pd.DataFrame, errors: set[str]) -> tuple[list[dict], int]:
    """Rows of the block that have a clip, and how many are still pending."""
    ready, pending = [], 0
    for row in block.to_dict("records"):
        if clip_path_for(cfg.clips_dir, row).exists():
            ready.append(row)
        elif row["path"] not in errors:
            pending += 1
    return ready, pending


def build_split(cfg: DataConfig, split: str, log) -> None:
    opts = shard_options(cfg)
    n = int(opts["videos_per_shard"])
    out = cfg.shards_dir / split
    manifest = pd.read_csv(cfg.manifest_dir / f"{split}_videos.csv")
    err_path = cfg.logs_dir / f"extract_errors_{split}.csv"
    errors = set(pd.read_csv(err_path)["path"]) if err_path.exists() and err_path.stat().st_size > 0 else set()
    uploaded = cfg.shards_dir / f"{split}_uploaded.txt"  # shards moved to the remote are listed here
    done_remote = set(uploaded.read_text().split()) if uploaded.exists() else set()

    built = waiting = 0
    for i in range(0, -(-len(manifest) // n)):
        name = shard_name(split, i)
        sidecar = out / name.replace(".tar", ".csv")
        if sidecar.exists() and ((out / name).exists() or name in done_remote):
            continue
        ready, pending = block_status(cfg, manifest.iloc[i * n:(i + 1) * n], errors)
        if pending:
            waiting += 1
            continue
        videos, rows = [], []
        for row in ready:
            frames, meta = read_clip(clip_path_for(cfg.clips_dir, row))
            key = f"{row['generator']}/{row['video_id']}"
            videos.append((key, frames, meta))
            rows.append({k: v for k, v in meta.items() if k not in ("source_indices", "clip_path")}
                        | {"shard": name, "key": key})
        size = write_shard(out / name, videos)
        pd.DataFrame(rows).to_csv(sidecar, index=False)
        built += 1
        log.info("%s: %d videos, %.2f GB", name, len(videos), size / 2**30)

    sidecars = sorted(out.glob(f"{split}-*.csv")) if out.exists() else []
    if sidecars:
        index = pd.concat([pd.read_csv(p) for p in sidecars], ignore_index=True)
        index.to_csv(cfg.shards_dir / f"{split}_index.csv", index=False)
        log.info("%s: built %d new shards, %d shards / %d videos indexed, %d blocks still waiting for extraction",
                 split, built, len(sidecars), len(index), waiting)
    else:
        log.info("%s: no shard ready yet, %d blocks waiting", split, waiting)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--split", choices=["train", "test"], nargs="*", default=["train", "test"])
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    log = setup_logging(cfg.logs_dir / "shards.log")
    for split in args.split:
        build_split(cfg, split, log)


if __name__ == "__main__":
    main()

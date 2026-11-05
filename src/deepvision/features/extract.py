"""Phase 2B: extract frozen features of every encoder in one pass over the shards.

Each video is streamed once (ShardStream, 8 of 16 frames), its JPEGs are decoded on the GPU and
every selected encoder runs on the same frames. Output per encoder and split:
    <out>/<encoder>/<split>/<shard>.npy        float16 (N, 8, D), rows in shard order
    <out>/<encoder>/<split>/<shard>.video.npy  float16 (N, Dv)  (X-CLIP video embedding)
    <out>/<encoder>/<split>/<shard>.csv        video_id, label, generator, key (row order)
    <out>/<encoder>/<split>/index.csv, encoder_card.json   written by `upload`
A shard is written to a temporary file and renamed once all its videos are in, so a killed run
resumes from the first unfinished shard.

    python -m deepvision.features.extract extract --split test [--encoders resnet50 ...] [--shards 1] [--upload]
    python -m deepvision.features.extract upload --split test
    python -m deepvision.features.extract cost
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("deepvision")
FRAME_INDICES = [0, 2, 4, 6, 9, 11, 13, 15]
INDEX_COLUMNS = ["video_id", "label", "generator", "key"]


def shard_stem(shard: str) -> str:
    return shard.rsplit("/", 1)[-1].removesuffix(".tar")


def done_path(out: Path, encoder: str, split: str, shard: str) -> Path:
    return out / encoder / split / f"{shard_stem(shard)}.csv"


def rss_mib() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def cgroup_mib() -> float:
    p = Path("/sys/fs/cgroup/memory.current")
    return int(p.read_text()) / 2**20 if p.exists() else float("nan")


class ShardWriter:
    """Memory-mapped outputs of one shard for the encoders that still need it."""

    def __init__(self, out: Path, split: str, shard: str, rows: pd.DataFrame, encoders: dict):
        self.rows = rows.reset_index(drop=True)
        self.n = len(rows)
        self.filled = np.zeros(self.n, dtype=bool)
        self.files: dict[str, list[tuple[Path, Path, np.memmap]]] = {}
        stem = shard_stem(shard)
        for name, enc in encoders.items():
            d = out / name / split
            d.mkdir(parents=True, exist_ok=True)
            shapes = [("", (self.n, len(FRAME_INDICES), enc.dim))]
            if enc.video_dim:
                shapes.append((".video", (self.n, enc.video_dim)))
            self.files[name] = []
            for suffix, shape in shapes:
                tmp = d / f"{stem}{suffix}.tmp.npy"
                self.files[name].append((tmp, d / f"{stem}{suffix}.npy",
                                         np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=shape)))
        self.csv = {name: out / name / split / f"{stem}.csv" for name in encoders}

    def write(self, pos: int, video_id: str, outputs: dict[str, dict[str, np.ndarray]], i: int) -> bool:
        """Store video i of a batch at row `pos`; True when the shard is complete."""
        if self.rows.at[pos, "video_id"] != video_id:
            raise RuntimeError(f"row {pos}: stream has {video_id}, index has {self.rows.at[pos, 'video_id']}")
        for name, files in self.files.items():
            files[0][2][pos] = outputs[name]["frames"][i]
            if len(files) > 1:
                files[1][2][pos] = outputs[name]["video"][i]
        self.filled[pos] = True
        return bool(self.filled.all())

    def finalize(self) -> None:
        pending = []
        for files in self.files.values():
            for tmp, final, arr in files:
                arr.flush()
                if not np.isfinite(arr).all():
                    raise RuntimeError(f"{final}: NaN or Inf in features")
                pending.append((tmp, final))
        del files, arr
        self.files = {}  # drops the memmaps, which closes the files (needed before renaming on Windows)
        for tmp, final in pending:
            os.replace(tmp, final)
        for csv in self.csv.values():
            self.rows[INDEX_COLUMNS].to_csv(csv, index=False)  # written last = shard done


def load_encoders(names: list[str], device, out: Path) -> dict:
    import torch

    from .encoders import build_encoder

    encoders = {}
    for name in names:
        enc = build_encoder(name).to(device)
        encoders[name] = enc
        info = enc.info | {"dim": enc.dim, "video_dim": enc.video_dim}
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "encoder_info.json").write_text(json.dumps(info, indent=2))
        gc.collect()
        LOG.info("loaded %s: D=%d%s, RSS %.0f MiB, VRAM %.0f MiB", name, enc.dim,
                 f", video D={enc.video_dim}" if enc.video_dim else "", rss_mib(),
                 torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0)
    return encoders


def extract(root: str, split: str, names: list[str], out: Path, batch_videos: int = 32, streams: int = 2,
            n_shards: int | None = None, rclone: str = "rclone", gpu_fraction: float = 0.8,
            device: str | None = None) -> dict:
    import torch

    from ..data.stream import ShardStream, decode_jpeg_batch, read_index
    from ..train.run import setup_device

    if device is None:
        device = setup_device(gpu_fraction)
        if device.type != "cuda":
            raise RuntimeError("feature extraction needs a GPU")
    else:
        device = torch.device(device)
    index = read_index(root, split, rclone)
    shards = sorted(index["shard"].unique())[:n_shards]
    todo = {s: [n for n in names if not done_path(out, n, split, s).exists()] for s in shards}
    todo = {s: ns for s, ns in todo.items() if ns}
    LOG.info("%s: %d shards, %d to do, encoders %s", split, len(shards), len(todo), names)
    if not todo:
        return {"split": split, "shards_done": 0}
    needed = sorted({n for ns in todo.values() for n in ns}, key=names.index)
    encoders = load_encoders(needed, device, out)
    by_shard = {s: g for s, g in index.groupby("shard")}
    writers: dict[str, ShardWriter] = {}
    stream = ShardStream(root, [f"shards/{split}/{s}" for s in todo], streams=streams, decode="none", rclone=rclone)

    t0, seen, finished = time.time(), 0, 0

    def run(batch: list[dict]) -> None:
        nonlocal seen, finished
        frames = decode_jpeg_batch([v["frames"] for v in batch], device)
        outputs = {}
        for name, enc in encoders.items():
            o = enc(frames)
            outputs[name] = {k: v.half().cpu().numpy() for k, v in o.items()}
        for i, v in enumerate(batch):
            shard = v["shard"]
            if shard not in writers:
                writers[shard] = ShardWriter(out, split, shard, by_shard[shard],
                                             {n: encoders[n] for n in todo[shard]})
            if writers[shard].write(v["pos"], v["video_id"], outputs, i):
                writers.pop(shard).finalize()
                finished += 1
                LOG.info("%s done (%d/%d shards)", shard, finished, len(todo))
        seen += len(batch)
        if seen % (batch_videos * 30) < len(batch):
            LOG.info("%d videos, %.1f videos/s, RSS %.0f MiB, cgroup %.0f MiB, max VRAM %.0f MiB", seen,
                     seen / (time.time() - t0), rss_mib(), cgroup_mib(),
                     torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0)

    batch: list[dict] = []
    for video in stream:
        batch.append(video)
        if len(batch) == batch_videos:
            run(batch)
            batch = []
    if batch:
        run(batch)
    if writers:
        raise RuntimeError(f"stream ended with incomplete shards: {sorted(writers)}")
    elapsed = time.time() - t0
    result = {"split": split, "shards_done": finished, "videos": seen, "seconds": round(elapsed, 1),
              "videos_per_s": round(seen / elapsed, 2), "max_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0,
              "rss_mib": round(rss_mib())}
    LOG.info("extract finished: %s", json.dumps(result))
    return result


def encoder_card(name: str, split: str, index: pd.DataFrame, info: dict) -> dict:
    from ..train.run import git_commit

    return {"encoder": name, "split": split, "videos": len(index), "dtype": "float16", "frame_indices": FRAME_INDICES,
            "input": "Phase 1 clips: 16 JPEG frames, DeMamba crop + resize to 224x224, RGB; normalized per encoder",
            "commit": git_commit(), "date": time.strftime("%Y-%m-%d %H:%M:%S")} | info


def upload(split: str, names: list[str], out: Path, remote: str, rclone: str = "rclone") -> dict:
    """Write index.csv + encoder_card.json, rclone copy the split folder and check it (one way)."""
    result = {}
    for name in names:
        d = out / name / split
        parts = sorted(p for p in d.glob("*.csv") if p.name != "index.csv")
        if not parts:
            LOG.warning("%s/%s: nothing to upload", name, split)
            continue
        index = pd.concat([pd.read_csv(p).assign(shard=p.stem + ".tar") for p in parts], ignore_index=True)
        index.to_csv(d / "index.csv", index=False)
        info = json.loads((out / name / "encoder_info.json").read_text())
        card = encoder_card(name, split, index, info)
        (d / "encoder_card.json").write_text(json.dumps(card, indent=2))
        dest = f"{remote.rstrip('/')}/{name}/{split}"
        binary = shutil.which(rclone) or rclone
        flags = ["--exclude", "*.tmp.npy"]
        subprocess.run([binary, "copy", str(d), dest, "--transfers", "4", *flags], check=True)
        check = subprocess.run([binary, "check", str(d), dest, "--one-way", *flags], capture_output=True, text=True)
        result[name] = {"files": len(list(d.glob("*.npy"))), "videos": len(index), "check_ok": check.returncode == 0,
                        "check": check.stderr.strip().splitlines()[-1:] if check.stderr else []}
        LOG.info("upload %s/%s: %s", name, split, result[name])
    return result


def measure_cost(names: list[str], batch_videos: int = 32, iters: int = 10, gpu_fraction: float = 0.8) -> list[dict]:
    """#params, GMACs per frame (FlopCounterMode, GMACs = FLOPs / 2), ms per video and peak VRAM in fp32."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from ..train.run import setup_device
    from .encoders import build_encoder, count_params

    device = setup_device(gpu_fraction)
    rows = []
    for name in names:
        enc = build_encoder(name).to(device)
        one = torch.randint(0, 256, (1, len(FRAME_INDICES), 3, 224, 224), dtype=torch.uint8, device=device)
        counter = FlopCounterMode(display=False)
        with counter:
            enc(one)
        frames = torch.randint(0, 256, (batch_videos, len(FRAME_INDICES), 3, 224, 224), dtype=torch.uint8,
                               device=device)
        for _ in range(3):
            enc(frames)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        for _ in range(iters):
            enc(frames)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters / batch_videos * 1000
        rows.append({"encoder": name, "model_id": enc.info["model_id"], "params_m": round(count_params(enc) / 1e6, 2),
                     "dim": enc.dim, "gmacs_per_frame": round(counter.get_total_flops() / 2 / 1e9 / len(FRAME_INDICES), 3),
                     "ms_per_video": round(ms, 3), "batch_videos": batch_videos,
                     "peak_activation_vram_mib": round((torch.cuda.max_memory_allocated() - base) / 2**20),
                     "weights_vram_mib": round(base / 2**20)})
        LOG.info("cost %s", rows[-1])
        del enc, frames, one
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def main(argv=None) -> None:
    from .encoders import ENCODERS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["extract", "upload", "cost"])
    ap.add_argument("--root", default="gdrive:DeepVision/dataset_v1")
    ap.add_argument("--remote", default="gdrive:DeepVision/features_v1")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))
    ap.add_argument("--out", default="~/deepvision_features")
    ap.add_argument("--shards", type=int, default=None, help="only the first N shards (smoke test)")
    ap.add_argument("--batch", type=int, default=32, help="videos per GPU batch")
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--upload", action="store_true", help="upload each encoder folder after extraction")
    ap.add_argument("--rclone", default="rclone")
    args = ap.parse_args(argv)
    # model weights go to our own cache directory, never to the shared ~/.cache
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/deepvision_cache/hf"))
    os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/deepvision_cache/torch"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(os.path.expanduser(args.out))
    if args.command == "extract":
        result = extract(args.root, args.split, args.encoders, out, args.batch, args.streams, args.shards, args.rclone)
        if args.upload:
            result["upload"] = upload(args.split, args.encoders, out, args.remote, args.rclone)
    elif args.command == "upload":
        result = upload(args.split, args.encoders, out, args.remote, args.rclone)
    else:
        result = measure_cost(args.encoders, args.batch)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result).to_csv(out / "encoder_costs.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

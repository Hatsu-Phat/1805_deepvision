"""Stream videos from many shards in parallel, from a local directory or an rclone remote.

Each DataLoader worker takes every num_workers-th shard and keeps `streams` shards open at once
(one `rclone cat` per shard, read by a thread), so shards read in parallel = workers x streams.
A failed read is reopened up to `retries` times and the videos already yielded from it are skipped.
Only the `num_frames` selected JPEGs of a video are kept (8 of 16 -> 0,2,4,6,9,11,13,15).

Throughput and count check against shards/<split>_index.csv:
    python -m deepvision.data.stream --root gdrive:DeepVision/dataset_v1 --split test --shards 4 --workers 2 --streams 2
"""
from __future__ import annotations

import argparse
import io
import logging
import queue
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

import numpy as np
import pandas as pd

from .config import SPLITS
from .dataset import decode_rgb
from .sampling import uniform_indices
from .shards import iter_shard

try:
    from torch.utils.data import IterableDataset, get_worker_info
except ImportError:  # the stream also works without torch (single process)
    IterableDataset = object

    def get_worker_info():
        return None

LOG = logging.getLogger("deepvision")
_REMOTE = re.compile(r"^[^/\\:]{2,}:")  # "gdrive:..." but not "H:/..."
_DONE = object()


def is_remote(root: str) -> bool:
    return bool(_REMOTE.match(str(root)))


def join_root(root: str, path: str) -> str:
    return f"{str(root).rstrip('/')}/{path}" if is_remote(root) else str(Path(root) / path)


@contextmanager
def open_shard(root: str, path: str, rclone: str = "rclone") -> Iterator[IO[bytes]]:
    """Binary stream of root/path; a remote is read with `rclone cat` and never written to disk."""
    if not is_remote(root):
        with open(Path(root) / path, "rb") as f:
            yield f
        return
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen([shutil.which(rclone) or rclone, "cat", join_root(root, path)],
                                stdout=subprocess.PIPE, stderr=err)
        ok = False
        try:
            yield proc.stdout
            ok = True
            while proc.stdout.read(1 << 16):  # tar padding after the end-of-archive marker
                pass
        finally:
            if not ok:
                proc.kill()
            proc.stdout.close()
            code = proc.wait()
        if code != 0:
            err.seek(0)
            msg = err.read()[-500:].decode("utf-8", "replace").strip()
            raise RuntimeError(f"rclone cat {path} exited with {code}: {msg}")


def read_text(root: str, path: str, rclone: str = "rclone") -> str:
    with open_shard(root, path, rclone) as f:
        return f.read().decode("utf-8")


def read_index(root: str, split: str, rclone: str = "rclone") -> pd.DataFrame:
    return pd.read_csv(io.StringIO(read_text(root, f"shards/{split}_index.csv", rclone)))


def iter_shard_retry(root: str, path: str, rclone: str = "rclone", retries: int = 3,
                     backoff: float = 5.0) -> Iterator[tuple[str, list[bytes], dict]]:
    """iter_shard over root/path that reopens the shard after a read error and resumes after the
    last video it yielded (shards are read in a fixed order)."""
    done = attempt = 0
    while True:
        try:
            with open_shard(root, path, rclone) as f:
                for i, item in enumerate(iter_shard(f)):
                    if i < done:
                        continue
                    yield item
                    done += 1
            return
        except (OSError, RuntimeError, tarfile.TarError, KeyError) as e:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"{path}: giving up after {retries} retries") from e
            LOG.warning("%s: read failed after %d videos (%s), retry %d/%d", path, done, e, attempt, retries)
            time.sleep(backoff * attempt)


def select_frames(frames: list, num_frames: int, indices: list[int] | None = None) -> tuple[list, int]:
    """Evenly spaced frames (or the given clip indices); a video with fewer frames repeats its last one.
    Returns (frames, n_valid)."""
    if indices is not None:
        valid = [j for j in indices if j < len(frames)]
        last = valid[-1] if valid else 0
        return [frames[j if j < len(frames) else last] for j in indices], len(valid)
    idx = uniform_indices(len(frames), num_frames)
    n_valid = len(idx)
    if n_valid < num_frames:
        idx = np.concatenate([idx, np.full(num_frames - n_valid, idx[-1] if n_valid else 0)])
    return [frames[j] for j in idx], n_valid


def decode_frames(frames: list[bytes]) -> np.ndarray:
    """JPEG bytes -> uint8 (T, 3, H, W) RGB, the layout of torchvision.io.decode_jpeg."""
    return np.stack([decode_rgb(np.frombuffer(b, np.uint8)) for b in frames]).transpose(0, 3, 1, 2)


class ShardStream(IterableDataset):
    """Yields one dict per video:
        key, video_id, label, generator, shard, pos (order inside the shard), n_valid,
        frames: list of `num_frames` JPEG bytes (decode="none", e.g. for GPU decoding with
        decode_jpeg_batch) or uint8 (num_frames, 3, H, W) (decode="cpu").

    shards: paths relative to `root`, e.g. "shards/test/test-00000.tar".
    keep:   optional set of video_ids; other videos are dropped.
    Call set_epoch() before each pass when shuffle_shards / shuffle_buffer are used.
    The video order is reproducible only with streams=1; with more streams it depends on read speed.
    """

    def __init__(self, root: str, shards: list[str], num_frames: int = 8, streams: int = 2,
                 decode: str = "none", keep: set[str] | None = None, shuffle_shards: bool = False,
                 shuffle_buffer: int = 0, seed: int = 0, retries: int = 3, backoff: float = 5.0,
                 rclone: str = "rclone", prefetch: int = 64, frame_indices: list[int] | None = None):
        """frame_indices: fixed clip indices to keep instead of `num_frames` evenly spaced ones."""
        if decode not in ("none", "cpu"):
            raise ValueError(f"unknown decode {decode!r}")
        self.root = str(root)
        self.shards = list(shards)
        self.num_frames = num_frames if frame_indices is None else len(frame_indices)
        self.frame_indices = None if frame_indices is None else list(frame_indices)
        self.streams = max(1, streams)
        self.decode = decode
        self.keep = keep
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.retries = retries
        self.backoff = backoff
        self.rclone = rclone
        self.prefetch = prefetch
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def worker_shards(self) -> list[str]:
        shards = list(self.shards)
        if self.shuffle_shards:
            np.random.default_rng([self.seed, self.epoch]).shuffle(shards)
        info = get_worker_info()
        if info is None:
            return shards
        return shards[info.id::info.num_workers]

    def _videos(self, shard: str) -> Iterator[dict]:
        name = shard.rsplit("/", 1)[-1]
        for pos, (key, frames, meta) in enumerate(iter_shard_retry(self.root, shard, self.rclone,
                                                                   self.retries, self.backoff)):
            video_id = meta.get("video_id", key.rsplit("/", 1)[-1])
            if self.keep is not None and video_id not in self.keep:
                continue
            chosen, n_valid = select_frames(frames, self.num_frames, self.frame_indices)
            yield {"key": key, "video_id": video_id, "label": int(meta["label"]),
                   "generator": meta.get("generator", key.split("/", 1)[0]), "shard": name, "pos": pos,
                   "n_valid": n_valid, "frames": chosen}

    def _parallel(self, shards: list[str]) -> Iterator[dict]:
        """Read `streams` shards at a time with threads (subprocess I/O releases the GIL)."""
        if self.streams == 1 or len(shards) <= 1:
            for s in shards:
                yield from self._videos(s)
            return
        todo: queue.Queue = queue.Queue()
        for s in shards:
            todo.put(s)
        out: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def put(item) -> bool:
            while not stop.is_set():
                try:
                    out.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    pass
            return False

        def reader():
            try:
                while not stop.is_set():
                    try:
                        shard = todo.get_nowait()
                    except queue.Empty:
                        break
                    for video in self._videos(shard):
                        if not put(video):
                            return
            except BaseException as e:  # surfaced in the consumer
                put(e)
            finally:
                put(_DONE)

        threads = [threading.Thread(target=reader, daemon=True) for _ in range(min(self.streams, len(shards)))]
        for t in threads:
            t.start()
        finished = 0
        try:
            while finished < len(threads):
                item = out.get()
                if item is _DONE:
                    finished += 1
                elif isinstance(item, BaseException):
                    raise item
                else:
                    yield item
        finally:
            stop.set()

    def _shuffled(self, items: Iterator[dict]) -> Iterator[dict]:
        if self.shuffle_buffer <= 1:
            yield from items
            return
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, self.epoch, info.id if info else 0])
        buf: list[dict] = []
        for item in items:
            if len(buf) < self.shuffle_buffer:
                buf.append(item)
                continue
            j = int(rng.integers(len(buf)))
            buf[j], item = item, buf[j]
            yield item
        rng.shuffle(buf)
        yield from buf

    def __iter__(self) -> Iterator[dict]:
        for video in self._shuffled(self._parallel(self.worker_shards())):
            if self.decode == "cpu":
                video["frames"] = decode_frames(video["frames"])
            yield video


def collate_videos(batch: list[dict]) -> dict:
    """Batch for DataLoader(collate_fn=collate_videos): labels as a tensor, frames stacked when decoded."""
    import torch

    out = {k: [b[k] for b in batch] for k in batch[0]}
    out["label"] = torch.tensor(out["label"], dtype=torch.long)
    if isinstance(batch[0]["frames"], np.ndarray):
        out["frames"] = torch.from_numpy(np.stack(out["frames"]))
    return out


def decode_jpeg_batch(frames: list[list[bytes]], device: str = "cuda"):
    """JPEG bytes of B videos x T frames -> uint8 tensor (B, T, 3, H, W) decoded on `device`."""
    import torch
    from torchvision.io import decode_jpeg

    flat = [torch.frombuffer(bytearray(b), dtype=torch.uint8) for video in frames for b in video]
    images = decode_jpeg(flat, device=device)
    return torch.stack(images).view(len(frames), len(frames[0]), *images[0].shape)


def shard_paths(index: pd.DataFrame, split: str, limit: int | None = None) -> list[str]:
    names = sorted(index["shard"].unique())[:limit]
    return [f"shards/{split}/{n}" for n in names]


def peak_memory_mib() -> dict:
    """Peak RSS of this process and its finished children, and the cgroup peak when available."""
    out = {}
    try:
        import resource

        out["rss_self_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        out["rss_children_mib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    except ImportError:  # Windows
        pass
    peak = Path("/sys/fs/cgroup/memory.peak")
    if peak.exists():
        out["cgroup_peak_mib"] = int(peak.read_text()) / 2**20
    return out


def benchmark(root: str, split: str, n_shards: int, workers: int, streams: int, decode: str,
              limit: int | None, rclone: str) -> dict:
    """Read the first n_shards shards and check per-shard counts and labels against the index."""
    index = read_index(root, split, rclone)
    shards = shard_paths(index, split, n_shards)
    ds = ShardStream(root, shards, streams=streams, decode=decode, rclone=rclone)
    if workers > 0:
        from torch.utils.data import DataLoader

        it = iter(DataLoader(ds, batch_size=None, num_workers=workers, prefetch_factor=4))
    else:
        it = iter(ds)
    seen: list[tuple[str, str, int]] = []
    t0 = time.time()
    for video in it:
        seen.append((video["shard"], video["video_id"], video["label"]))
        if len(seen) % 1000 == 0:
            LOG.info("%d videos, %.1f videos/s", len(seen), len(seen) / (time.time() - t0))
        if limit and len(seen) >= limit:
            break
    elapsed = time.time() - t0
    got = pd.DataFrame(seen, columns=["shard", "video_id", "label"])
    expected = index[index["shard"].isin([s.rsplit("/", 1)[-1] for s in shards])]
    result = {"videos": len(got), "seconds": round(elapsed, 1), "videos_per_s": round(len(got) / elapsed, 2),
              "shards": len(shards), "workers": workers, "streams": streams, "decode": decode,
              "duplicates": int(got["video_id"].duplicated().sum())} | peak_memory_mib()
    if not limit:
        merged = expected.merge(got, on="video_id", how="outer", suffixes=("_index", "_read"), indicator=True)
        result["missing"] = int((merged["_merge"] == "left_only").sum())
        result["unexpected"] = int((merged["_merge"] == "right_only").sum())
        both = merged[merged["_merge"] == "both"]
        result["label_mismatch"] = int((both["label_index"] != both["label_read"]).sum())
        result["shard_mismatch"] = int((both["shard_index"] != both["shard_read"]).sum())
        result["ok"] = not any(result[k] for k in ("missing", "unexpected", "label_mismatch",
                                                   "shard_mismatch", "duplicates"))
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="rclone remote:path or local directory holding shards/")
    ap.add_argument("--split", choices=SPLITS, default="test")
    ap.add_argument("--shards", type=int, default=1, help="read the first N shards")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 = this process)")
    ap.add_argument("--streams", type=int, default=2, help="shards open at once per worker")
    ap.add_argument("--decode", choices=["none", "cpu"], default="none")
    ap.add_argument("--limit", type=int, default=None, help="stop after N videos (skips the count check)")
    ap.add_argument("--rclone", default="rclone")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = benchmark(args.root, args.split, args.shards, args.workers, args.streams, args.decode,
                       args.limit, args.rclone)
    print(result)


if __name__ == "__main__":
    main()

import io
import logging

import pandas as pd

from deepvision.data.clips import clip_path_for, write_clip
from deepvision.data.config import DataConfig
from deepvision.data.shards import build_split, iter_shard, write_shard

LOG = logging.getLogger("test")


class PipeLike(io.RawIOBase):
    """Read-only, non-seekable stream (like rclone cat stdout)."""

    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def readable(self):
        return True

    def readinto(self, b):
        chunk = self._buf.read(min(len(b), 4096))
        b[:len(chunk)] = chunk
        return len(chunk)


def fake_frames(seed: int, n: int = 16) -> list[bytes]:
    return [bytes([seed, j]) * (100 + j) for j in range(n)]


def test_shard_roundtrip_streaming(tmp_path):
    videos = [("A/v.1", fake_frames(1), {"video_id": "v.1"}), ("B/w", fake_frames(2, 3), {"video_id": "w"})]
    size = write_shard(tmp_path / "s.tar", videos)
    assert size == (tmp_path / "s.tar").stat().st_size
    got = list(iter_shard(io.BufferedReader(PipeLike((tmp_path / "s.tar").read_bytes()))))
    assert got == videos


def make_cfg(tmp_path, n_videos=5, per_shard=2):
    cfg = DataConfig(raw_root=tmp_path / "raw", out_root=tmp_path / "out", quarantine_root=tmp_path / "q", sources=[])
    cfg.shards = {"videos_per_shard": per_shard}
    rows = [{"video_id": f"v{i}", "path": f"g/v{i}.mp4", "split": "train", "label": i % 2,
             "generator": "real" if i % 2 == 0 else "G"} for i in range(n_videos)]
    cfg.manifest_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(cfg.manifest_dir / "train_videos.csv", index=False)
    cfg.logs_dir.mkdir(parents=True)
    return cfg, rows


def add_clip(cfg, row, i):
    path = clip_path_for(cfg.clips_dir, row)
    write_clip(path, fake_frames(i), row | {"n_frames": 16, "clip_path": "x", "source_indices": [0]})


def test_build_only_complete_blocks(tmp_path):
    cfg, rows = make_cfg(tmp_path)
    for i in (0, 1, 2):  # block 0 complete, block 1 half done, block 2 (v4) missing
        add_clip(cfg, rows[i], i)
    build_split(cfg, "train", LOG)
    out = cfg.shards_dir / "train"
    assert sorted(p.name for p in out.glob("*.tar")) == ["train-00000.tar"]
    index = pd.read_csv(cfg.shards_dir / "train_index.csv")
    assert index["video_id"].tolist() == ["v0", "v1"] and set(index["shard"]) == {"train-00000.tar"}
    with open(out / "train-00000.tar", "rb") as f:
        keys = [k for k, _, _ in iter_shard(f)]
    assert keys == ["real/v0", "G/v1"]

    # v3 fails extraction, v4 gets a clip -> the remaining blocks are written; block 0 is untouched
    pd.DataFrame([{"path": "g/v3.mp4"}]).to_csv(cfg.logs_dir / "extract_errors_train.csv", index=False)
    add_clip(cfg, rows[4], 4)
    mtime = (out / "train-00000.tar").stat().st_mtime_ns
    build_split(cfg, "train", LOG)
    assert sorted(p.name for p in out.glob("*.tar")) == ["train-00000.tar", "train-00001.tar", "train-00002.tar"]
    assert (out / "train-00000.tar").stat().st_mtime_ns == mtime
    index = pd.read_csv(cfg.shards_dir / "train_index.csv")
    assert index["video_id"].tolist() == ["v0", "v1", "v2", "v4"]


def test_uploaded_shards_are_not_rebuilt(tmp_path):
    cfg, rows = make_cfg(tmp_path, n_videos=2)
    for i in (0, 1):
        add_clip(cfg, rows[i], i)
    build_split(cfg, "train", LOG)
    (cfg.shards_dir / "train" / "train-00000.tar").unlink()  # moved to the remote
    (cfg.shards_dir / "train_uploaded.txt").write_text("train-00000.tar\n")
    build_split(cfg, "train", LOG)
    assert not (cfg.shards_dir / "train" / "train-00000.tar").exists()
    assert len(pd.read_csv(cfg.shards_dir / "train_index.csv")) == 2

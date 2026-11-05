import cv2
import numpy as np
import pytest

from deepvision.data import stream
from deepvision.data.shards import write_shard
from deepvision.data.stream import ShardStream, is_remote, iter_shard_retry, select_frames


def jpeg(value: int, size: int = 16) -> bytes:
    img = np.full((size, size, 3), value, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def make_shards(root, n_shards=4, per_shard=5, n_frames=16):
    """shards/train/train-0000i.tar; frame j of every video is a flat image of value 10 * j."""
    frames = [jpeg(10 * j) for j in range(n_frames)]
    paths, ids = [], []
    for s in range(n_shards):
        videos = []
        for v in range(per_shard):
            vid = f"v{s}_{v}"
            label = v % 2
            videos.append((f"{'G' if label else 'real'}/{vid}", frames,
                           {"video_id": vid, "label": label, "generator": "G" if label else "real"}))
            ids.append(vid)
        name = f"shards/train/train-{s:05d}.tar"
        write_shard(root / name, videos)
        paths.append(name)
    return paths, ids


def test_is_remote():
    assert is_remote("gdrive:DeepVision/dataset_v1")
    assert not is_remote("H:/lab/data") and not is_remote("/home/x") and not is_remote("relative/dir")


def test_select_frames():
    chosen, n = select_frames(list(range(16)), 8)
    assert chosen == [0, 2, 4, 6, 9, 11, 13, 15] and n == 8
    chosen, n = select_frames(list(range(5)), 8)
    assert chosen == [0, 1, 2, 3, 4, 4, 4, 4] and n == 5


@pytest.mark.parametrize("streams", [1, 3])
def test_stream_reads_every_video_once(tmp_path, streams):
    paths, ids = make_shards(tmp_path)
    videos = list(ShardStream(str(tmp_path), paths, streams=streams))
    assert sorted(v["video_id"] for v in videos) == sorted(ids)
    v = videos[0]
    assert len(v["frames"]) == 8 and v["n_valid"] == 8 and v["label"] == int(v["video_id"][-1]) % 2
    assert v["shard"].startswith("train-") and v["key"].endswith(v["video_id"])


def test_stream_keep_decode_and_shuffle(tmp_path):
    paths, ids = make_shards(tmp_path, n_shards=2)
    keep = set(ids[::3])
    ds = ShardStream(str(tmp_path), paths, decode="cpu", keep=keep, shuffle_shards=True, shuffle_buffer=4, seed=1,
                     streams=1)  # with several streams the order depends on thread timing
    first = [v["video_id"] for v in ds]
    assert sorted(first) == sorted(keep)
    v = next(iter(ds))
    assert v["frames"].shape == (8, 3, 16, 16) and v["frames"].dtype == np.uint8
    assert abs(int(v["frames"][1].mean()) - 20) <= 2  # second chosen frame is stored frame 2
    assert [v["video_id"] for v in ds] == first  # same epoch -> same order
    ds.set_epoch(1)
    assert sorted(v["video_id"] for v in ds) == sorted(keep)


def test_retry_resumes_after_failure(tmp_path, monkeypatch):
    paths, _ = make_shards(tmp_path, n_shards=1, per_shard=6)
    real_open = stream.open_shard
    calls = {"n": 0}

    class Broken:
        def __init__(self, f, limit):
            self.f, self.left = f, limit

        def read(self, n=-1):
            if self.left <= 0:
                raise OSError("connection reset")
            data = self.f.read(min(n if n >= 0 else self.left, self.left))
            self.left -= len(data)
            return data

    from contextlib import contextmanager

    @contextmanager
    def flaky(root, path, rclone="rclone"):
        calls["n"] += 1
        with real_open(root, path, rclone) as f:
            yield Broken(f, 30_000) if calls["n"] == 1 else f

    monkeypatch.setattr(stream, "open_shard", flaky)
    keys = [k for k, _, _ in iter_shard_retry(str(tmp_path), paths[0], retries=2, backoff=0)]
    assert calls["n"] == 2
    assert keys == [f"{'G' if v % 2 else 'real'}/v0_{v}" for v in range(6)]


def test_retry_gives_up(tmp_path):
    with pytest.raises(RuntimeError, match="giving up"):
        list(iter_shard_retry(str(tmp_path), "shards/missing.tar", retries=1, backoff=0))


def test_dataloader_workers_split_shards(tmp_path):
    torch = pytest.importorskip("torch")
    from deepvision.data.stream import collate_videos

    paths, ids = make_shards(tmp_path, n_shards=4, per_shard=3)
    ds = ShardStream(str(tmp_path), paths, streams=2, decode="cpu")
    loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2, collate_fn=collate_videos)
    seen = []
    for batch in loader:
        assert batch["frames"].shape[1:] == (8, 3, 16, 16) and batch["label"].dtype == torch.long
        seen += batch["video_id"]
    assert sorted(seen) == sorted(ids)

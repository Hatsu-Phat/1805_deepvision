import json

import numpy as np
import pandas as pd
import pytest

from deepvision.features import extract as fx
from deepvision.features.extract import ShardWriter


class FakeEncoder:
    def __init__(self, dim, video_dim=0):
        self.dim, self.video_dim = dim, video_dim
        self.info = {"model_id": "fake", "mean": [0.5] * 3, "std": [0.5] * 3}


def rows(n=4):
    return pd.DataFrame({"video_id": [f"v{i}" for i in range(n)], "label": [i % 2 for i in range(n)],
                         "generator": ["real", "G"] * (n // 2), "key": [f"k/v{i}" for i in range(n)],
                         "shard": "test-00000.tar"})


def outputs(value, dims=(3, 2)):
    return {"a": {"frames": np.full((1, 8, dims[0]), value, np.float16)},
            "b": {"frames": np.full((1, 8, dims[1]), value, np.float16), "video": np.full((1, 5), value, np.float16)}}


def test_shard_writer_any_order(tmp_path):
    w = ShardWriter(tmp_path, "test", "shards/test/test-00000.tar", rows(),
                    {"a": FakeEncoder(3), "b": FakeEncoder(2, 5)})
    assert not (tmp_path / "a/test/test-00000.csv").exists()
    for pos in (2, 0, 3):
        assert not w.write(pos, f"v{pos}", outputs(pos), 0)
    assert w.write(1, "v1", outputs(1), 0)
    w.finalize()
    a = np.load(tmp_path / "a/test/test-00000.npy")
    assert a.shape == (4, 8, 3) and a.dtype == np.float16 and a[:, 0, 0].tolist() == [0, 1, 2, 3]
    assert np.load(tmp_path / "b/test/test-00000.video.npy").shape == (4, 5)
    assert pd.read_csv(tmp_path / "b/test/test-00000.csv")["video_id"].tolist() == ["v0", "v1", "v2", "v3"]
    assert not list(tmp_path.rglob("*.tmp.npy"))


def test_shard_writer_checks(tmp_path):
    w = ShardWriter(tmp_path, "test", "test-00000.tar", rows(2), {"a": FakeEncoder(3)})
    with pytest.raises(RuntimeError, match="index has v0"):
        w.write(0, "v1", outputs(0), 0)
    w.write(0, "v0", outputs(np.nan), 0)
    w.write(1, "v1", outputs(1), 0)
    with pytest.raises(RuntimeError, match="NaN"):
        w.finalize()
    assert not (tmp_path / "a/test/test-00000.csv").exists()  # not marked done


def test_extract_end_to_end_and_resume(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tests.test_stream import make_shards

    paths, ids = make_shards(tmp_path / "data", n_shards=3, per_shard=4)
    index = pd.concat([pd.DataFrame({"video_id": [f"v{s}_{v}" for v in range(4)], "label": [v % 2 for v in range(4)],
                                     "generator": ["real", "G"] * 2,
                                     "key": [f"{'G' if v % 2 else 'real'}/v{s}_{v}" for v in range(4)],
                                     "shard": f"train-{s:05d}.tar"}) for s in range(3)])
    index.to_csv(tmp_path / "data/shards/train_index.csv", index=False)

    class Mean(torch.nn.Module):  # feature = per-channel mean of each frame
        dim, video_dim, info = 3, 0, {"model_id": "mean"}

        def forward(self, frames):
            return {"frames": frames.float().mean(dim=(3, 4))}

    loaded = []
    monkeypatch.setattr(fx, "load_encoders", lambda names, device, out: loaded.append(names) or {n: Mean() for n in names})
    out = tmp_path / "feat"
    r = fx.extract(str(tmp_path / "data"), "train", ["m1", "m2"], out, batch_videos=3, streams=2, n_shards=2,
                   device="cpu")
    assert r["shards_done"] == 2 and r["videos"] == 8
    feats = np.load(out / "m1/train/train-00001.npy")
    assert feats.shape == (4, 8, 3)
    assert np.allclose(feats[0, :, 0], [10 * j for j in (0, 2, 4, 6, 9, 11, 13, 15)], atol=2)  # frame j = value 10j
    # resume: done shards are skipped, the third one is added
    r = fx.extract(str(tmp_path / "data"), "train", ["m1", "m2"], out, batch_videos=3, device="cpu")
    assert r["shards_done"] == 1 and r["videos"] == 4
    assert fx.extract(str(tmp_path / "data"), "train", ["m1"], out, device="cpu")["shards_done"] == 0
    assert len(loaded) == 2


def test_upload_writes_index_and_card(tmp_path, monkeypatch):
    w = ShardWriter(tmp_path, "test", "test-00000.tar", rows(2), {"a": FakeEncoder(3)})
    w.write(0, "v0", outputs(0), 0)
    w.write(1, "v1", outputs(1), 0)
    w.finalize()
    (tmp_path / "a/encoder_info.json").write_text(json.dumps({"model_id": "fake", "dim": 3, "video_dim": 0}))
    calls = []

    class Done:
        returncode, stderr = 0, "0 differences found"

    monkeypatch.setattr("deepvision.train.run.git_commit", lambda: "test")
    monkeypatch.setattr(fx.subprocess, "run", lambda args, **kw: calls.append(args) or Done())
    r = fx.upload("test", ["a"], tmp_path, "gdrive:X/features_v1")
    assert r["a"]["check_ok"] and r["a"]["videos"] == 2
    assert calls[0][1:4] == ["copy", str(tmp_path / "a/test"), "gdrive:X/features_v1/a/test"]
    card = json.loads((tmp_path / "a/test/encoder_card.json").read_text())
    assert card["frame_indices"] == [0, 2, 4, 6, 9, 11, 13, 15] and card["dim"] == 3 and card["videos"] == 2
    assert pd.read_csv(tmp_path / "a/test/index.csv")["shard"].tolist() == ["test-00000.tar"] * 2


@pytest.mark.parametrize("name", ["resnet50", "effnetv2_b0", "convnext_t", "vit_ti16", "dinov2_s14", "swin_t"])
def test_timm_encoders_random_weights(name):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from deepvision.features.encoders import SPECS, TimmEncoder

    _, model_id, kwargs, _ = SPECS[name]
    enc = TimmEncoder(model_id, pretrained=False, **kwargs)
    out = enc(torch.randint(0, 256, (2, 8, 3, 224, 224), dtype=torch.uint8))
    assert out["frames"].shape == (2, 8, enc.dim) and torch.isfinite(out["frames"]).all()


def test_xclip_encoder_small_random():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from deepvision.features.encoders import XClipEncoder

    cfg = transformers.XCLIPConfig(
        projection_dim=16, prompt_layers=1, mit_hidden_size=16, mit_intermediate_size=32, mit_num_hidden_layers=1,
        mit_num_attention_heads=2, text_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                                                "num_attention_heads": 2},
        vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
                       "num_frames": 8, "image_size": 64, "patch_size": 32})
    enc = XClipEncoder(transformers.XCLIPModel(cfg).eval(), [0.5] * 3, [0.25] * 3)
    out = enc(torch.randint(0, 256, (2, 8, 3, 64, 64), dtype=torch.uint8))
    assert out["frames"].shape == (2, 8, 16) and out["video"].shape == (2, 16)
    with pytest.raises(ValueError, match="8 frames"):
        enc(torch.zeros(1, 4, 3, 64, 64, dtype=torch.uint8))

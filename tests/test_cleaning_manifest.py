import logging
from argparse import Namespace

import pandas as pd
import pytest

from deepvision.data.audit import _probe_one
from deepvision.data.cleaning import classify, cmd_quarantine, cmd_restore, find_duplicates
from deepvision.data.config import DataConfig, Source
from deepvision.data.manifest import build, interleave_priority, make_video_ids

LOG = logging.getLogger("test")


def inv_row(path, **kw):
    base = dict(path=path, source="s", split="train", label=1, generator="G", size_bytes=100, probe_ok=1,
                probe_error="", duration=4.0, fps=8.0, width=320, height=240, nb_frames=32)
    return base | kw


def test_probe_one_flags_broken_and_empty(video_dir):
    ok = _probe_one(video_dir, dict(path="mid_5s_8fps.mp4", size_bytes=1, mtime=0), 30)
    assert ok["probe_ok"] == 1 and ok["width"] == 320 and abs(ok["duration"] - 5.0) < 0.1
    bad = _probe_one(video_dir, dict(path="broken.mp4", size_bytes=1, mtime=0), 30)
    assert bad["probe_ok"] == 0 and bad["probe_error"]
    empty = _probe_one(video_dir, dict(path="empty.mp4", size_bytes=0, mtime=0), 30)
    assert empty["probe_ok"] == 0 and empty["probe_error"] == "empty file"


def test_classify():
    inv = pd.DataFrame([
        inv_row("a.mp4"),
        inv_row("b.mp4", probe_ok=0, probe_error="moov atom not found", width=None, height=None, duration=None),
        inv_row("c.mp4", size_bytes=0, probe_ok=0, probe_error="empty file"),
        inv_row("d.mp4", duration=0.2),
        inv_row("e.mp4", width=32, height=32),
        inv_row("f.mp4", nb_frames=0),
    ])
    st = classify(inv, min_duration=0.5, min_side=64).set_index("path")
    assert st.loc["a.mp4", "status"] == "ok"
    assert st.loc["b.mp4", "status"] == "broken" and "moov" in st.loc["b.mp4", "reason"]
    assert st.loc["c.mp4", "reason"] == "empty file"
    assert st.loc["d.mp4", "status"] == "excluded"
    assert st.loc["e.mp4", "status"] == "excluded"
    assert st.loc["f.mp4", "status"] == "broken"


def make_cfg(tmp_path):
    return DataConfig(raw_root=tmp_path / "raw", out_root=tmp_path / "out", quarantine_root=tmp_path / "q",
                      sources=[Source("s", "s", "train", 1, "G")])


def test_duplicates_prefer_test_copy(tmp_path):
    cfg = make_cfg(tmp_path)
    for name, data in [("train/x.mp4", b"same"), ("test/y.mp4", b"same"), ("train/z.mp4", b"diff")]:
        p = cfg.raw_root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    st = pd.DataFrame([
        dict(path="train/x.mp4", source="s", split="train", label=1, generator="G", size_bytes=4, status="ok", reason=""),
        dict(path="test/y.mp4", source="t", split="test", label=1, generator="H", size_bytes=4, status="ok", reason=""),
        dict(path="train/z.mp4", source="s", split="train", label=1, generator="G", size_bytes=4, status="ok", reason=""),
    ])
    st = find_duplicates(cfg, st, LOG).set_index("path")
    assert st.loc["train/x.mp4", "status"] == "excluded"
    assert st.loc["test/y.mp4", "status"] == "ok"
    assert st.loc["train/z.mp4", "status"] == "ok"


def test_quarantine_and_restore_roundtrip(tmp_path):
    cfg = make_cfg(tmp_path)
    src = cfg.raw_root / "G" / "bad.mp4"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"xx")
    cfg.inventory_dir.mkdir(parents=True)
    pd.DataFrame([dict(path="G/bad.mp4", source="s", split="train", label=1, generator="G", size_bytes=2,
                       status="broken", reason="probe: x")]).to_csv(cfg.inventory_dir / "video_status.csv", index=False)

    cmd_quarantine(cfg, Namespace(apply=False), LOG)
    assert src.exists()  # dry run does not move anything

    cmd_quarantine(cfg, Namespace(apply=True), LOG)
    assert not src.exists() and (cfg.quarantine_root / "G" / "bad.mp4").exists()

    cmd_restore(cfg, Namespace(paths=None), LOG)
    assert src.exists() and not (cfg.quarantine_root / "G" / "bad.mp4").exists()


def test_video_ids_unique():
    paths = pd.Series(["a/1/v.mp4", "a/2/v.mp4", "a/w x.mp4"])
    ids = make_video_ids(paths)
    assert ids.is_unique
    assert ids.iloc[2] == "w_x"


def test_priority_proportional():
    df = pd.DataFrame({"generator": ["A"] * 100 + ["B"] * 10, "label": 1})
    pri = interleave_priority(df, seed=0, balance="proportion")
    assert sorted(pri) == list(range(110))
    first = df.loc[pri[pri < 22].index, "generator"].value_counts()
    # every prefix keeps roughly the generator proportions (10:1)
    assert 1 <= first.get("B", 0) <= 4


def test_priority_class_balanced():
    df = pd.DataFrame({"generator": ["R"] * 50 + ["F1"] * 300 + ["F2"] * 300,
                       "label": [0] * 50 + [1] * 600})
    pri = interleave_priority(df, seed=0, balance="class")
    head = df.loc[pri[pri < 80].index]
    assert 35 <= (head["label"] == 0).sum() <= 45          # ~half real
    fakes = head[head["label"] == 1]["generator"].value_counts()
    assert abs(fakes["F1"] - fakes["F2"]) <= 4              # fake half split evenly
    # after the real videos run out, the rest are still all scheduled
    assert sorted(pri) == list(range(650))


def test_build_manifest_keeps_only_ok():
    inv = pd.DataFrame([inv_row("a.mp4"), inv_row("b.mp4"), inv_row("c.mp4", split="test", generator="T")])
    status = pd.DataFrame({"path": ["a.mp4", "b.mp4", "c.mp4"], "status": ["ok", "broken", "ok"]})
    out = build(inv, status, seed=0)
    assert out["train"]["path"].tolist() == ["a.mp4"]
    assert out["test"]["video_id"].tolist() == ["c"]

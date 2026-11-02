import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from deepvision.data import ffmpeg
from deepvision.data.extract import decode_sampled, extract_one
from deepvision.data.sampling import sampling_rate, window_start

from .conftest import SYNTHETIC

OPTS = dict(num_frames=16, base_fps=2, min_duration_for_base_fps=None, window="center",
            jpeg_quality=95, max_short_side=0, decode_timeout=120)


def demamba_reference(src: Path, rate: int, out: Path) -> list[Path]:
    """What DeMamba's video2frame.py does: ffmpeg -i <video> -r <rate> <dir>/%d.<ext> (png keeps pixels exact)."""
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg.find_binary("ffmpeg"), "-v", "error", "-i", str(src), "-r", str(rate),
                    str(out / "%d.png")], check=True)
    return sorted(out.glob("*.png"), key=lambda p: int(p.stem))


def job_for(src: Path, dst: Path, duration: float) -> dict:
    return dict(src=str(src), dst=str(dst), frames_root=str(dst.parent), opts=OPTS, video_id=dst.name,
                path=src.name, split="test", label=1, generator="synthetic", duration=duration)


@pytest.mark.parametrize("name,duration,fps,size", SYNTHETIC)
def test_pipe_matches_demamba_image2_output(video_dir, tmp_path, name, duration, fps, size):
    src = video_dir / f"{name}.mp4"
    rate = sampling_rate(duration)
    ref = demamba_reference(src, rate, tmp_path / "ref")
    n, frames, _ = decode_sampled(src, rate, None, timeout=60)
    assert n == len(ref)
    for i, p in enumerate(ref):
        mine = cv2.imdecode(np.frombuffer(frames[i], np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(mine, cv2.imread(str(p)))


@pytest.mark.parametrize("name,duration,fps,size", SYNTHETIC)
def test_extract_one(video_dir, tmp_path, name, duration, fps, size):
    src = video_dir / f"{name}.mp4"
    meta = extract_one(job_for(src, tmp_path / name, duration))
    assert "error" not in meta, meta
    rate = sampling_rate(duration)
    n_ref = len(demamba_reference(src, rate, tmp_path / "ref"))
    assert meta["rate"] == rate
    assert meta["n_sampled"] == n_ref
    assert meta["n_frames"] == min(16, n_ref)
    assert meta["window_start"] == window_start(n_ref, 16, "center")
    assert meta["source_indices"] == list(range(meta["window_start"], meta["window_start"] + meta["n_frames"]))
    files = sorted((tmp_path / name).glob("*.jpg"))
    assert [f.name for f in files] == [f"{j:03d}.jpg" for j in range(meta["n_frames"])]
    w, h = map(int, size.split("x"))
    assert (meta["width"], meta["height"]) == (w, h)
    assert json.loads((tmp_path / name / "meta.json").read_text())["n_frames"] == meta["n_frames"]


def test_extract_frames_match_reference_window(video_dir, tmp_path):
    name, duration = "long_12s_25fps", 12.0
    src = video_dir / f"{name}.mp4"
    meta = extract_one(job_for(src, tmp_path / name, duration))
    ref = demamba_reference(src, meta["rate"], tmp_path / "ref")
    ref_imgs = [cv2.imread(str(p)).astype(np.int16) for p in ref]
    for j, i in enumerate(meta["source_indices"]):
        mine = cv2.imread(str(tmp_path / name / f"{j:03d}.jpg")).astype(np.int16)
        errors = [np.abs(mine - r).mean() for r in ref_imgs]
        # the stored JPEG is closest to the reference frame it came from, up to JPEG q95 loss
        assert int(np.argmin(errors)) == i
        assert errors[i] < 4.0


def test_extract_is_resumable(video_dir, tmp_path):
    src = video_dir / "mid_5s_8fps.mp4"
    first = extract_one(job_for(src, tmp_path / "v", 5.0))
    second = extract_one(job_for(src, tmp_path / "v", 5.0))
    assert second.get("skipped") and second["n_frames"] == first["n_frames"]


def test_window_recovered_when_duration_estimate_is_wrong(video_dir, tmp_path):
    # a wrong duration shifts the buffered range; the extractor must re-read the right window
    src = video_dir / "long_12s_25fps.mp4"
    meta = extract_one(job_for(src, tmp_path / "v", 60.0))
    assert "error" not in meta
    assert meta["window_start"] == window_start(meta["n_sampled"], 16, "center")


@pytest.mark.parametrize("name", ["broken.mp4", "empty.mp4"])
def test_extract_reports_broken_video(video_dir, tmp_path, name):
    res = extract_one(job_for(video_dir / name, tmp_path / "x", 3.0))
    assert "error" in res
    assert not (tmp_path / "x").exists()


def test_short_side_cap(video_dir, tmp_path):
    job = job_for(video_dir / "long_12s_25fps.mp4", tmp_path / "v", 12.0)
    job["opts"] = OPTS | {"max_short_side": 120}
    meta = extract_one(job)
    assert min(meta["width"], meta["height"]) == 120

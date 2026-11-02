import shutil
import subprocess
from pathlib import Path

import pytest

from deepvision.data import ffmpeg


def _has_ffmpeg() -> bool:
    try:
        ffmpeg.find_binary("ffmpeg")
        ffmpeg.find_binary("ffprobe")
        return True
    except FileNotFoundError:
        return False


requires_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")


def make_video(path: Path, duration: float, fps: int, size: str = "320x240") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg.find_binary("ffmpeg"), "-v", "error", "-y", "-f", "lavfi",
        "-i", f"testsrc2=size={size}:rate={fps}:duration={duration}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ]
    subprocess.run(cmd, check=True)
    return path


# (name, duration s, fps, size)
SYNTHETIC = [
    ("short_1s_8fps", 1.0, 8, "320x240"),
    ("short_2s_24fps", 2.0, 24, "320x240"),
    ("mid_3p5s_30fps", 3.5, 30, "256x320"),
    ("mid_5s_8fps", 5.0, 8, "320x240"),
    ("long_8s_24fps", 8.0, 24, "320x240"),
    ("long_12s_25fps", 12.0, 25, "426x240"),
]


@pytest.fixture(scope="session")
def video_dir(tmp_path_factory):
    if not _has_ffmpeg():
        pytest.skip("ffmpeg not available")
    root = tmp_path_factory.mktemp("videos")
    for name, dur, fps, size in SYNTHETIC:
        make_video(root / f"{name}.mp4", dur, fps, size)
    (root / "broken.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"garbage" * 100)
    (root / "empty.mp4").write_bytes(b"")
    return root


@pytest.fixture
def clean_dir(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)

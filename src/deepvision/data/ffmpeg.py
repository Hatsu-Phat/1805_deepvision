"""Thin wrappers around ffprobe / ffmpeg."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def find_binary(name: str) -> str:
    env = os.environ.get(f"{name.upper()}_BIN")
    if env:
        return env
    found = shutil.which(name)
    if found:
        return found
    # winget installs a shim here that is not always on PATH for non-login shells
    local = os.environ.get("LOCALAPPDATA")
    if local:
        shim = Path(local) / "Microsoft" / "WinGet" / "Links" / f"{name}.exe"
        if shim.exists():
            return str(shim)
    raise FileNotFoundError(f"{name} not found; install ffmpeg or set {name.upper()}_BIN")


def _fraction(value: str | None) -> float:
    if not value or value in ("0/0", "N/A"):
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(value)


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(float(sd["rotation"]))
    rot = stream.get("tags", {}).get("rotate")
    return int(rot) if rot else 0


def probe(path: str | Path, timeout: float = 60) -> dict:
    """Return basic video metadata. Raises RuntimeError if the file has no readable video stream."""
    cmd = [
        find_binary("ffprobe"), "-v", "error", "-show_format", "-show_streams",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(msg[-1] if msg else f"ffprobe exit {proc.returncode}")
    info = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    videos = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not videos:
        raise RuntimeError("no video stream")
    v = videos[0]
    fmt = info.get("format", {})

    duration = _to_float(fmt.get("duration")) or _to_float(v.get("duration")) or 0.0
    fps = _fraction(v.get("avg_frame_rate")) or _fraction(v.get("r_frame_rate"))
    nb_frames = _to_int(v.get("nb_frames"))
    if duration <= 0 and nb_frames and fps:  # e.g. GIFs report no container duration
        duration = nb_frames / fps
    return {
        "duration": round(duration, 4),
        "fps": round(fps, 4),
        "r_frame_rate": v.get("r_frame_rate", ""),
        "width": _to_int(v.get("width")) or 0,
        "height": _to_int(v.get("height")) or 0,
        "rotation": _rotation(v),
        "codec": v.get("codec_name", ""),
        "pix_fmt": v.get("pix_fmt", ""),
        "nb_frames": nb_frames,
        "bit_rate": _to_int(v.get("bit_rate")) or _to_int(fmt.get("bit_rate")),
        "format_name": fmt.get("format_name", ""),
        "has_audio": any(s.get("codec_type") == "audio" for s in info.get("streams", [])),
    }


def decode_check(path: str | Path, timeout: float = 120) -> tuple[int, str]:
    """Decode the whole video stream and discard the output.

    Returns (number of decoded frames, first error line or "").
    """
    cmd = [
        find_binary("ffmpeg"), "-v", "error", "-nostdin", "-i", str(path),
        "-map", "0:v:0", "-f", "null", "-progress", "pipe:1", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    frames = 0
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        if line.startswith("frame="):
            frames = _to_int(line.split("=", 1)[1]) or frames
    err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
    if proc.returncode != 0 and not err:
        err = [f"ffmpeg exit {proc.returncode}"]
    return frames, (err[0] if err else "")

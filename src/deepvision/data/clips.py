"""Step 4b: compact training copies of the extracted frames.

Each video becomes one uncompressed tar holding 000.jpg ... 015.jpg and meta.json. Frames are
center-cropped as DeMamba does (crop_center_by_percentage) and resized to size x size with
bilinear interpolation (same as albumentations.Resize), so a loader only has to normalize.
One file per video also avoids the cost of millions of tiny files on disk and when uploading.

Every frame takes the same path: native JPEG (q95) -> decode -> crop -> resize -> JPEG. Frames
converted here from frames/ and frames written directly by extract.py therefore share the same
compression history.

    python -m deepvision.data.clips --split train test
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import LABEL_NAMES, DataConfig, add_config_args, config_from_args
from .io import setup_logging
from .transforms import crop_center_by_percentage, demamba_crop_percentage


def clip_options(cfg: DataConfig) -> dict:
    return {"size": 224, "crop": "demamba", "jpeg_quality": 95, "keep_native": ["test"]} | cfg.clips


def clip_path_for(clips_dir: Path, row: dict) -> Path:
    return clips_dir / row["split"] / LABEL_NAMES[int(row["label"])] / row["generator"] / f"{row['video_id']}.tar"


def crop_for(opts: dict, split: str, generator: str, label: int) -> float:
    if opts["crop"] == "demamba":
        return demamba_crop_percentage(split, generator, label)
    return float(opts["crop"] or 0.0)


def shrink_jpeg(data: bytes, crop: float, size: int, quality: int) -> bytes:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cannot decode frame")
    if crop:
        img = crop_center_by_percentage(img, crop)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def write_clip(path: Path, frames: list[bytes], meta: dict) -> int:
    """Write frames + meta.json into an uncompressed tar (atomically). Returns the file size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    members = [(f"{j:03d}.jpg", b) for j, b in enumerate(frames)]
    members.append(("meta.json", json.dumps(meta).encode("utf-8")))
    with tarfile.open(tmp, "w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    os.replace(tmp, path)
    return path.stat().st_size


def read_clip(path: Path) -> tuple[list[bytes], dict]:
    with tarfile.open(path, "r") as tar:
        files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    meta = json.loads(files.pop("meta.json"))
    return [files[f"{j:03d}.jpg"] for j in range(len(files))], meta


def load_clip_meta(path: Path) -> dict:
    return read_clip(path)[1] | {"clip_bytes": path.stat().st_size}


def build_clip(frames: list[bytes], meta: dict, opts: dict, path: Path, clips_root: Path) -> dict:
    """Shrink native JPEG frames and store them as one clip; returns the clip meta."""
    crop = crop_for(opts, meta["split"], meta["generator"], int(meta["label"]))
    size, quality = int(opts["size"]), int(opts["jpeg_quality"])
    small = [shrink_jpeg(b, crop, size, quality) for b in frames]
    meta = meta | {"native_height": meta["height"], "native_width": meta["width"], "native_bytes": meta["bytes"],
                   "height": size, "width": size, "crop": crop, "clip_quality": quality,
                   "clip_path": path.relative_to(clips_root).as_posix(), "bytes": sum(map(len, small))}
    meta["clip_bytes"] = write_clip(path, small, meta)
    got, _ = read_clip(path)
    if len(got) != len(small) or any(a != b for a, b in zip(got, small)):
        raise RuntimeError("clip read-back mismatch")
    return meta


def convert_one(job: dict) -> dict:
    """frames/<...>/ (native JPEGs) -> clips/<...>.tar; optionally remove the native folder."""
    src, dst, opts = Path(job["src"]), Path(job["dst"]), job["opts"]
    base = {k: job[k] for k in ("video_id", "path", "split", "label", "generator")}
    try:
        if dst.exists():
            meta = load_clip_meta(dst)
            if job["delete_native"] and src.exists():
                shutil.rmtree(src)
            return meta | {"skipped": True}
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        frames = [(src / f"{j:03d}.jpg").read_bytes() for j in range(int(meta["n_frames"]))]
        meta = build_clip(frames, meta, opts, dst, Path(job["clips_root"]))
        if job["delete_native"]:
            shutil.rmtree(src)
        return meta
    except Exception as e:  # noqa: BLE001
        return base | {"error": f"{type(e).__name__}: {e}"[:300]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--split", choices=["train", "test"], nargs="*", default=["train", "test"])
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    log = setup_logging(cfg.logs_dir / "clips.log")
    opts = clip_options(cfg)

    for split in args.split:
        manifest = pd.read_csv(cfg.manifest_dir / f"{split}_videos.csv")
        delete = split not in opts["keep_native"]
        jobs = []
        for row in manifest.to_dict("records"):
            src = cfg.frames_dir / row["split"] / LABEL_NAMES[int(row["label"])] / row["generator"] / row["video_id"]
            dst = clip_path_for(cfg.clips_dir, row)
            if dst.exists() or (src / "meta.json").exists():
                jobs.append(row | {"src": str(src), "dst": str(dst), "clips_root": str(cfg.clips_dir),
                                   "opts": opts, "delete_native": delete})
        log.info("clips %s: %d of %d videos have frames, delete native: %s", split, len(jobs), len(manifest), delete)

        t0, done, errors = time.time(), [], []
        with ProcessPoolExecutor(cfg.workers) as pool:
            futures = [pool.submit(convert_one, j) for j in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"clips {split}", mininterval=10):
                res = fut.result()
                (errors if "error" in res else done).append(res)
        pd.DataFrame([{k: v for k, v in d.items() if k != "skipped"} for d in done]).to_csv(
            cfg.manifest_dir / f"{split}_clips.csv", index=False)
        pd.DataFrame(errors, columns=["video_id", "path", "split", "label", "generator", "error"]).to_csv(
            cfg.logs_dir / f"clips_errors_{split}.csv", index=False)
        log.info("clips %s: ok %d, errors %d, %.1f min", split, len(done), len(errors), (time.time() - t0) / 60)


if __name__ == "__main__":
    main()

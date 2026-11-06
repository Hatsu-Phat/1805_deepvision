"""Step 4: extract 16 candidate frames per video.

For every video in manifests/<split>_videos.csv:
  1. pick the sampling rate with the DeMamba rule (see sampling.py),
  2. decode with `ffmpeg -r <rate>` (same muxer semantics as DeMamba's `%d.jpg` output),
  3. keep a centered window of `num_frames` consecutive sampled frames,
  4. write them as JPEG (native resolution unless max_short_side is set) plus meta.json.

A video whose meta.json exists is skipped, so the step can be interrupted and resumed.
Videos that cannot be decoded are listed in logs/extract_errors_<split>.csv.
With `extraction.output: clips` the frames are not kept at native resolution: each video is
written as one resized tar under clips/ (see clips.py) and a clip that exists is skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import ffmpeg
from .clips import build_clip, clip_options, clip_path_for, convert_one, load_clip_meta
from .config import LABEL_NAMES, SPLITS, DataConfig, add_config_args, config_from_args
from .io import setup_logging
from .sampling import sampling_rate, window_start

BUFFER_MARGIN = 4


def _read_exact(stream, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def decode_sampled(src: Path, rate: int, keep: range | None, timeout: float) -> tuple[int, dict[int, bytes], str]:
    """Run ffmpeg -r <rate> and stream BMP frames from a pipe.

    Returns (number of sampled frames, {index: bmp bytes} for indices in `keep` (all if None), stderr).
    """
    cmd = [
        ffmpeg.find_binary("ffmpeg"), "-v", "error", "-nostdin", "-i", str(src),
        "-map", "0:v:0", "-r", str(rate), "-f", "image2pipe", "-vcodec", "bmp", "-",
    ]
    # stderr goes to a temp file so a chatty decoder cannot fill the pipe and block us
    errf = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf)
    frames: dict[int, bytes] = {}
    n = 0
    t0 = time.time()
    try:
        while True:
            header = _read_exact(proc.stdout, 6)
            if len(header) < 6:
                break
            if header[:2] != b"BM":
                raise RuntimeError("unexpected data in ffmpeg output")
            size = int.from_bytes(header[2:6], "little")
            body = _read_exact(proc.stdout, size - 6)
            if len(body) < size - 6:
                break
            if keep is None or n in keep:
                frames[n] = header + body
            n += 1
            if time.time() - t0 > timeout:
                raise TimeoutError(f"decode exceeded {timeout}s")
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        errf.seek(0)
        stderr = errf.read().decode("utf-8", "replace").strip()
        errf.close()
    return n, frames, stderr


def _resize_short_side(img: np.ndarray, max_short: int) -> np.ndarray:
    h, w = img.shape[:2]
    short = min(h, w)
    if not max_short or short <= max_short:
        return img
    scale = max_short / short
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def frames_dir_for(cfg_frames_dir: Path, row: dict) -> Path:
    return cfg_frames_dir / row["split"] / LABEL_NAMES[int(row["label"])] / row["generator"] / row["video_id"]


def extract_one(job: dict) -> dict:
    src, dst, opts = Path(job["src"]), Path(job["dst"]), job["opts"]
    meta_path = dst / "meta.json"
    clip = job.get("clip")
    if clip:
        clip_dst = Path(clip["dst"])
        if clip_dst.exists():
            return load_clip_meta(clip_dst) | {"skipped": True}
        if meta_path.exists():  # extracted earlier at native resolution
            return convert_one(job | {"src": str(dst), "dst": str(clip_dst), "clips_root": clip["root"],
                                      "opts": clip["opts"], "delete_native": clip["delete_native"]})
    elif meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8")) | {"skipped": True}

    base = {"video_id": job["video_id"], "path": job["path"], "split": job["split"],
            "label": int(job["label"]), "generator": job["generator"]}
    num = int(opts["num_frames"])
    try:
        duration = float(job["duration"])
        rate = sampling_rate(duration, num, int(opts["base_fps"]), opts.get("min_duration_for_base_fps"))
        expected = max(1, round(duration * rate))
        s_est = window_start(expected, num, opts["window"])
        keep = range(max(0, s_est - BUFFER_MARGIN), s_est + num + BUFFER_MARGIN)
        n_total, frames, stderr = decode_sampled(src, rate, keep, float(opts["decode_timeout"]))
        if n_total == 0:
            raise RuntimeError(stderr.splitlines()[-1] if stderr else "no frames decoded")

        start = window_start(n_total, num, opts["window"])
        idx = list(range(start, min(start + num, n_total)))
        if any(i not in frames for i in idx):  # the estimate was off; decode again keeping exactly the window
            n_total, frames, stderr = decode_sampled(src, rate, range(idx[0], idx[-1] + 1),
                                                     float(opts["decode_timeout"]))
            if any(i not in frames for i in idx):
                raise RuntimeError("could not re-read the selected window")

        hashes, nbytes, shape, encoded = [], 0, None, []
        for j, i in enumerate(idx):
            img = cv2.imdecode(np.frombuffer(frames[i], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"cannot decode sampled frame {i}")
            hashes.append(hashlib.md5(img.tobytes()).hexdigest())
            img = _resize_short_side(img, int(opts.get("max_short_side") or 0))
            shape = img.shape
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(opts["jpeg_quality"])])
            if not ok:
                raise RuntimeError("jpeg encode failed")
            encoded.append(buf.tobytes())
            nbytes += len(buf)

        meta = base | {
            "frames_dir": dst.relative_to(job["frames_root"]).as_posix(),
            "duration": duration, "rate": rate, "n_sampled": n_total, "window_start": start,
            "source_indices": idx, "n_frames": len(idx), "n_unique": len(set(hashes)),
            "height": int(shape[0]), "width": int(shape[1]), "bytes": nbytes,
            "decode_warning": (stderr.splitlines()[0][:200] if stderr else ""),
        }
        if clip:
            return build_clip(encoded, meta, clip["opts"], Path(clip["dst"]), Path(clip["root"]))
        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for j, data in enumerate(encoded):
            (tmp / f"{j:03d}.jpg").write_bytes(data)
        (tmp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        if dst.exists():
            shutil.rmtree(dst)
        os.replace(tmp, dst)
        return meta
    except Exception as e:  # noqa: BLE001
        return base | {"error": f"{type(e).__name__}: {e}"[:300]}


def build_jobs(cfg: DataConfig, manifest: pd.DataFrame) -> list[dict]:
    opts = {
        "num_frames": 16, "base_fps": 2, "min_duration_for_base_fps": None, "window": "center",
        "jpeg_quality": 95, "max_short_side": 0, "decode_timeout": 300,
    } | cfg.extraction
    copts = clip_options(cfg)
    jobs = []
    for row in manifest.to_dict("records"):
        job = row | {
            "src": str(cfg.raw_root / row["path"]),
            "dst": str(frames_dir_for(cfg.frames_dir, row)),
            "frames_root": str(cfg.frames_dir),
            "opts": opts,
        }
        if opts.get("output", "frames") == "clips":
            job["clip"] = {"dst": str(clip_path_for(cfg.clips_dir, row)), "root": str(cfg.clips_dir),
                           "opts": copts, "delete_native": row["split"] not in copts["keep_native"]}
        jobs.append(job)
    return jobs


def collect_index(cfg: DataConfig, manifest: pd.DataFrame, clips: bool = False) -> pd.DataFrame:
    """Read the meta of every extracted video (or clip) in the manifest."""
    rows = []
    for row in manifest.to_dict("records"):
        if clips:
            path = clip_path_for(cfg.clips_dir, row)
            if path.exists():
                rows.append(load_clip_meta(path))
            continue
        meta = frames_dir_for(cfg.frames_dir, row) / "meta.json"
        if meta.exists():
            rows.append(json.loads(meta.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--split", choices=SPLITS, nargs="*", default=["train", "test"])
    ap.add_argument("--limit-per-generator", type=int, default=0, help="pilot mode: first N videos per generator")
    ap.add_argument("--generators", nargs="*", help="restrict to these generators")
    ap.add_argument("--tag", default="", help="suffix for the error/summary files")
    ap.add_argument("--index-only", action="store_true", help="only rebuild the <split> index from the stored meta")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    log = setup_logging(cfg.logs_dir / "extract.log")
    clips = cfg.extraction.get("output", "frames") == "clips"
    kind = "clips" if clips else "frames"

    for split in args.split:
        manifest = pd.read_csv(cfg.manifest_dir / f"{split}_videos.csv")
        if args.generators:
            manifest = manifest[manifest["generator"].isin(args.generators)]
        if args.limit_per_generator:
            manifest = manifest.groupby("generator", group_keys=False).head(args.limit_per_generator)
        if args.index_only:
            index = collect_index(cfg, manifest, clips)
            index.to_csv(cfg.manifest_dir / f"{split}_{kind}.csv", index=False)
            log.info("index %s: %d of %d videos extracted", split, len(index), len(manifest))
            continue
        jobs = build_jobs(cfg, manifest)
        log.info("extract %s: %d videos, %d workers", split, len(jobs), cfg.workers)

        tag = f"_{args.tag}" if args.tag else ""
        cfg.logs_dir.mkdir(parents=True, exist_ok=True)

        def save(done, errors):
            pd.DataFrame(errors, columns=["video_id", "path", "split", "label", "generator", "error"]).to_csv(
                cfg.logs_dir / f"extract_errors_{split}{tag}.csv", index=False)
            pd.DataFrame([{k: v for k, v in d.items() if k != "skipped"} for d in done]).to_csv(
                cfg.manifest_dir / f"{split}_{kind}{tag}.csv", index=False)

        min_free = float(cfg.extraction.get("min_free_gb", 0)) * 2**30
        t0, done, errors, skipped, stopped = time.time(), [], [], 0, False
        pool = ProcessPoolExecutor(cfg.workers)
        try:
            futures = [pool.submit(extract_one, j) for j in jobs]
            for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc=f"extract {split}",
                                         mininterval=10)):
                res = fut.result()
                if "error" in res:
                    errors.append(res)
                else:
                    skipped += bool(res.get("skipped"))
                    done.append(res)
                if i % 2000 == 1999:
                    save(done, errors)  # keep partial results if the run is killed
                    if min_free and shutil.disk_usage(cfg.frames_dir).free < min_free:
                        log.warning("free disk space below %.0f GB, stopping", min_free / 2**30)
                        stopped = True
                        break
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        elapsed = time.time() - t0

        save(done, errors)
        new = len(done) - skipped
        log.info("extract %s: ok %d (new %d, skipped %d), errors %d, %.1f min, %.2f new videos/s%s",
                 split, len(done), new, skipped, len(errors), elapsed / 60, new / max(elapsed, 1e-9),
                 " [stopped: low disk space]" if stopped else "")


if __name__ == "__main__":
    main()

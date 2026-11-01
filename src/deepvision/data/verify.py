"""Consistency checks on the processed dataset, plus a contact sheet per generator for visual inspection.

Checks, per split:
  * every manifest video either has a frames folder or is listed in the extraction errors
  * each frames folder holds exactly n_frames JPEGs that decode, all with the same size
  * no stray *.tmp folders left by an interrupted run
Writes <out_root>/reports/verify_<split>.json and reports/contact_<split>_<generator>.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import add_config_args, config_from_args
from .io import setup_logging


def check_folder(folder: Path, n_frames: int, deep: bool) -> str:
    files = sorted(folder.glob("*.jpg"))
    if [f.name for f in files] != [f"{j:03d}.jpg" for j in range(n_frames)]:
        return f"expected {n_frames} frames, found {len(files)}"
    if deep:
        shapes = set()
        for f in files:
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return f"unreadable {f.name}"
            shapes.add(img.shape)
        if len(shapes) != 1:
            return f"inconsistent frame sizes {shapes}"
    return ""


def contact_sheet(folders: list[Path], tile: int = 112) -> np.ndarray:
    rows = []
    for folder in folders:
        frames = sorted(folder.glob("*.jpg"))
        row = []
        for f in frames[:16]:
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            row.append(cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA))
        row += [np.zeros((tile, tile, 3), np.uint8)] * (16 - len(row))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--split", nargs="*", default=["train", "test"])
    ap.add_argument("--deep", action="store_true", help="decode every JPEG (slow)")
    ap.add_argument("--sheet-rows", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    log = setup_logging(cfg.logs_dir / "verify.log")
    report_dir = cfg.out_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    for split in args.split:
        manifest = pd.read_csv(cfg.manifest_dir / f"{split}_videos.csv")
        index_path = cfg.manifest_dir / f"{split}_frames.csv"
        index = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame(columns=["video_id", "generator"])
        err_path = cfg.logs_dir / f"extract_errors_{split}.csv"
        errors = pd.read_csv(err_path) if err_path.exists() and err_path.stat().st_size > 0 else pd.DataFrame(columns=["path"])

        key = ["generator", "video_id"]
        merged = manifest.merge(index[key + ["n_frames", "n_unique", "frames_dir"]], on=key, how="left")
        extracted = merged[merged["n_frames"].notna()]
        problems = []
        for row in extracted.itertuples():
            msg = check_folder(cfg.frames_dir / row.frames_dir, int(row.n_frames), args.deep)
            if msg:
                problems.append({"video_id": row.video_id, "generator": row.generator, "problem": msg})
        tmp_dirs = [p.as_posix() for p in (cfg.frames_dir / split).rglob("*.tmp")] if (cfg.frames_dir / split).exists() else []

        per_gen = merged.groupby(["generator", "label"]).agg(
            manifest=("video_id", "size"), extracted=("n_frames", "count"),
            full16=("n_frames", lambda s: int((s == 16).sum())),
            mean_unique=("n_unique", "mean")).reset_index()
        per_gen["errors"] = per_gen["generator"].map(
            manifest[manifest["path"].isin(errors["path"])].groupby("generator").size()).fillna(0).astype(int)
        report = {
            "split": split, "manifest": int(len(manifest)), "extracted": int(len(extracted)),
            "extract_errors": int(manifest["path"].isin(errors["path"]).sum()),
            "not_done": int(len(manifest) - len(extracted) - manifest["path"].isin(errors["path"]).sum()),
            "folder_problems": problems[:50], "n_folder_problems": len(problems), "tmp_dirs": tmp_dirs[:20],
            "n_frames_hist": {int(k): int(v) for k, v in extracted["n_frames"].value_counts().sort_index().items()},
            "per_generator": per_gen.round(2).to_dict("records"),
            "frames_gb": round(float(index["bytes"].sum()) / 2**30, 3) if "bytes" in index else None,
        }
        (report_dir / f"verify_{split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("%s: manifest %d, extracted %d, errors %d, not done %d, folder problems %d, tmp dirs %d",
                 split, report["manifest"], report["extracted"], report["extract_errors"], report["not_done"],
                 len(problems), len(tmp_dirs))

        for gen, g in extracted.groupby("generator"):
            pick = g.sample(min(args.sheet_rows, len(g)), random_state=args.seed)
            sheet = contact_sheet([cfg.frames_dir / d for d in pick["frames_dir"]])
            cv2.imwrite(str(report_dir / f"contact_{split}_{gen}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()

"""Step 1: scan every configured source and probe each video with ffprobe.

Outputs (under <out_root>/inventory/):
    probe_cache.jsonl   resumable raw results
    inventory.csv       one row per video
    summary.json        counts and distributions per source
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from . import ffmpeg
from .config import DataConfig, add_config_args, config_from_args
from .io import JsonlCache, iter_source_files, setup_logging

INVENTORY_COLUMNS = [
    "path", "source", "split", "label", "generator", "size_bytes", "mtime", "probe_ok", "probe_error",
    "duration", "fps", "r_frame_rate", "width", "height", "rotation", "codec", "pix_fmt", "nb_frames",
    "bit_rate", "format_name", "has_audio",
]


def scan(cfg: DataConfig) -> pd.DataFrame:
    rows = []
    for src in cfg.sources:
        for rel, size, mtime in iter_source_files(cfg, src):
            rows.append(dict(path=rel, source=src.name, split=src.split, label=src.label,
                             generator=src.generator, size_bytes=size, mtime=round(mtime, 3)))
    return pd.DataFrame(rows, columns=["path", "source", "split", "label", "generator", "size_bytes", "mtime"])


def _probe_one(raw_root, row: dict, timeout: float) -> dict:
    rec = {"path": row["path"], "size_bytes": row["size_bytes"], "mtime": row["mtime"]}
    if row["size_bytes"] == 0:
        return rec | {"probe_ok": 0, "probe_error": "empty file"}
    try:
        meta = ffmpeg.probe(raw_root / row["path"], timeout=timeout)
        return rec | {"probe_ok": 1, "probe_error": ""} | meta
    except Exception as e:  # noqa: BLE001 - every failure is recorded, not raised
        return rec | {"probe_ok": 0, "probe_error": f"{type(e).__name__}: {e}"[:300]}


def probe_all(cfg: DataConfig, files: pd.DataFrame, log) -> pd.DataFrame:
    cache = JsonlCache(cfg.inventory_dir / "probe_cache.jsonl")
    timeout = float(cfg.validation.get("probe_timeout", 60))

    todo = []
    for row in files.to_dict("records"):
        hit = cache.get(row["path"])
        # re-probe if the file changed since it was cached
        if hit and hit["size_bytes"] == row["size_bytes"] and abs(hit["mtime"] - row["mtime"]) < 1:
            continue
        todo.append(row)
    log.info("probe: %d cached, %d to do", len(files) - len(todo), len(todo))

    with ThreadPoolExecutor(cfg.workers) as pool:
        futures = [pool.submit(_probe_one, cfg.raw_root, r, timeout) for r in todo]
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc="ffprobe", mininterval=5)):
            cache.add(fut.result())
            if i % 500 == 0:
                cache.flush()
    cache.close()

    probed = pd.DataFrame([cache.records[p] for p in files["path"]]).drop(columns=["size_bytes", "mtime"])
    inv = files.merge(probed, on="path", how="left")
    # same fallback as ffmpeg.probe, for entries cached before it existed
    nb = pd.to_numeric(inv["nb_frames"], errors="coerce")
    fix = (inv["duration"].fillna(0) <= 0) & (nb > 0) & (inv["fps"] > 0)
    inv.loc[fix, "duration"] = (nb[fix] / inv.loc[fix, "fps"]).round(4)
    return inv.reindex(columns=INVENTORY_COLUMNS)


def _describe(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {}
    q = s.quantile([0, 0.05, 0.5, 0.95, 1]).round(3).tolist()
    return dict(zip(["min", "p05", "median", "p95", "max"], q)) | {"mean": round(float(s.mean()), 3)}


def summarize(inv: pd.DataFrame) -> dict:
    out = {"total_files": int(len(inv)), "probe_ok": int(inv["probe_ok"].sum()), "sources": {}}
    for name, g in inv.groupby("source", sort=False):
        ok = g[g["probe_ok"] == 1]
        res = (ok["width"].astype(int).astype(str) + "x" + ok["height"].astype(int).astype(str)).value_counts()
        out["sources"][name] = {
            "split": g["split"].iloc[0], "label": int(g["label"].iloc[0]), "generator": g["generator"].iloc[0],
            "files": int(len(g)), "probe_ok": int(len(ok)), "probe_failed": int(len(g) - len(ok)),
            "size_gb": round(g["size_bytes"].sum() / 2**30, 3),
            "duration_s": _describe(ok["duration"]),
            "fps": _describe(ok["fps"]),
            "top_resolutions": {k: int(v) for k, v in res.head(5).items()},
            "codecs": {k: int(v) for k, v in ok["codec"].value_counts().items()},
            "rotated": int((ok["rotation"].fillna(0) != 0).sum()),
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--only", nargs="*", help="restrict to these source names")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    if args.only:
        cfg.sources = [s for s in cfg.sources if s.name in set(args.only)]
    log = setup_logging(cfg.logs_dir / "audit.log")

    t0 = time.time()
    files = scan(cfg)
    log.info("scan: %d files in %.1fs", len(files), time.time() - t0)
    for name, n in files.groupby("source", sort=False).size().items():
        log.info("  %-12s %d", name, n)
    missing = [s.name for s in cfg.sources if not (cfg.raw_root / s.dir).is_dir()]
    if missing:
        log.warning("missing source directories: %s", missing)

    inv = probe_all(cfg, files, log)
    cfg.inventory_dir.mkdir(parents=True, exist_ok=True)
    inv.to_csv(cfg.inventory_dir / "inventory.csv", index=False)
    summary = summarize(inv)
    (cfg.inventory_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("inventory: %d files, %d probed ok, %d failed (%.1f min)",
             summary["total_files"], summary["probe_ok"], summary["total_files"] - summary["probe_ok"],
             (time.time() - t0) / 60)


if __name__ == "__main__":
    main()

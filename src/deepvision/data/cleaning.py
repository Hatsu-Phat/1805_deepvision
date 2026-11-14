"""Step 2: validate videos and move unreadable ones out of the raw dataset.

Subcommands
    check       inventory.csv (+ extraction errors) -> inventory/video_status.csv
    quarantine  move every `broken` video to quarantine_root (dry run unless --apply)
    restore     move quarantined videos back using logs/quarantine_moves.csv

Status values
    ok        usable
    broken    cannot be opened/decoded -> quarantined
    excluded  readable but not used for training (too short, too small, duplicate); the file is left in place

The test split (GenVideo-Val) is a fixed benchmark: only broken files are removed from it; anything
that would be `excluded` in train is kept and described in the `note` column instead.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import time
from pathlib import Path

import pandas as pd

from .config import EVAL_SPLITS, DataConfig, add_config_args, config_from_args
from .io import setup_logging

MOVES_COLUMNS = ["time", "path", "src", "dst", "reason", "action"]


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _quick_hash(path: Path, chunk: int = 1 << 16) -> str:
    """Hash of the first and last 64 KiB; cheap pre-filter before a full MD5."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - chunk))
        h.update(f.read(chunk))
    return h.hexdigest()


def classify(inv: pd.DataFrame, min_duration: float, min_side: int) -> pd.DataFrame:
    st = inv[["path", "source", "split", "label", "generator", "size_bytes"]].copy()
    st["status"], st["reason"], st["note"] = "ok", "", ""
    is_test = st["split"].isin(EVAL_SPLITS)

    def mark(mask, status, reason):
        sel = mask & (st["status"] == "ok")
        if status == "excluded":
            st.loc[sel & is_test, "note"] = reason
            sel = sel & ~is_test
        st.loc[sel, "status"] = status
        st.loc[sel, "reason"] = reason

    nb = pd.to_numeric(inv["nb_frames"], errors="coerce")
    mark(inv["size_bytes"] == 0, "broken", "empty file")
    mark(inv["probe_ok"] != 1, "broken", "probe: " + inv["probe_error"].fillna("").astype(str))
    mark((inv["width"].fillna(0) <= 0) | (inv["height"].fillna(0) <= 0), "broken", "no frame size")
    mark(nb == 0, "broken", "0 frames in stream")
    mark(inv["duration"].fillna(0) <= 0, "broken", "no duration")
    mark(inv["duration"] < min_duration, "excluded", f"duration < {min_duration}s")
    mark(inv[["width", "height"]].min(axis=1) < min_side, "excluded", f"min side < {min_side}px")
    return st


def find_duplicates(cfg: DataConfig, st: pd.DataFrame, log) -> pd.DataFrame:
    """Exact duplicates: same size, then same MD5. Test copies win over train copies."""
    cand = st[(st["status"] == "ok") & st.duplicated("size_bytes", keep=False)].copy()
    log.info("duplicate check: %d files share a size with another file", len(cand))
    if cand.empty:
        return st
    cand["quick"] = [_quick_hash(cfg.raw_root / p) for p in cand["path"]]
    cand = cand[cand.duplicated(["size_bytes", "quick"], keep=False)].copy()
    log.info("duplicate check: %d files share size and head/tail hash", len(cand))
    if cand.empty:
        return st
    cand["md5"] = [_md5(cfg.raw_root / p) for p in cand["path"]]
    # keep test before train before train2 so that the later copy is the one dropped
    rank = {"test": 0, "vif_test": 0, "aigvd_test": 0, "train": 1, "train2": 2, "vif_train": 3}
    cand = cand.sort_values(["md5", "split", "path"], key=lambda s: s.map(rank) if s.name == "split" else s)
    dup = cand[cand.duplicated("md5", keep="first")]
    first = cand.drop_duplicates("md5", keep="first").set_index("md5")["path"]
    n_excluded = 0
    for i, row in dup.iterrows():
        reason = f"duplicate of {first[row['md5']]}"
        if row["split"] in EVAL_SPLITS:  # duplicates inside the benchmark are kept as-is
            st.loc[i, "note"] = reason
            continue
        st.loc[i, "status"] = "excluded"
        st.loc[i, "reason"] = reason
        n_excluded += 1
    log.info("duplicate check: %d duplicates, %d excluded (train copies)", len(dup), n_excluded)
    return st


def add_extract_errors(cfg: DataConfig, st: pd.DataFrame, log) -> pd.DataFrame:
    files = sorted(cfg.logs_dir.glob("extract_errors_*.csv"))
    errs = pd.concat([pd.read_csv(f) for f in files if f.stat().st_size > 0], ignore_index=True) if files else pd.DataFrame()
    if errs.empty:
        return st
    errs = errs.drop_duplicates("path", keep="last").set_index("path")["error"]
    sel = st["path"].isin(errs.index) & (st["status"] == "ok")
    st.loc[sel, "status"] = "broken"
    st.loc[sel, "reason"] = "decode: " + st.loc[sel, "path"].map(errs).astype(str)
    log.info("extraction errors: %d videos marked broken", int(sel.sum()))
    return st


def cmd_check(cfg: DataConfig, args, log) -> None:
    inv = pd.read_csv(cfg.inventory_dir / "inventory.csv")
    v = cfg.validation
    st = classify(inv, float(v.get("min_duration", 0.5)), int(v.get("min_side", 64)))
    if not args.skip_duplicates:
        st = find_duplicates(cfg, st, log)
    if args.with_extract_errors:
        st = add_extract_errors(cfg, st, log)
    # files already quarantined in an earlier run are no longer in the inventory scan,
    # so the quarantine log is the record for them
    st.to_csv(cfg.inventory_dir / "video_status.csv", index=False)

    table = st.pivot_table(index="source", columns="status", values="path", aggfunc="count", fill_value=0)
    log.info("status per source:\n%s", table.to_string())
    groups = st[st["status"] != "ok"]["reason"].str.replace(r"duplicate of .*", "duplicate", regex=True)
    for reason, n in groups.str.slice(0, 80).value_counts().head(15).items():
        log.info("  %6d  %s", n, reason)
    notes = st[st["note"] != ""]["note"].str.replace(r"duplicate of .*", "duplicate", regex=True)
    for note, n in notes.value_counts().items():
        log.info("  kept with note %6d  %s", n, note)


def _moves_log(cfg: DataConfig) -> Path:
    return cfg.logs_dir / "quarantine_moves.csv"


def cmd_quarantine(cfg: DataConfig, args, log) -> None:
    st = pd.read_csv(cfg.inventory_dir / "video_status.csv")
    broken = st[st["status"] == "broken"]
    log.info("quarantine: %d broken videos (%s)", len(broken), "apply" if args.apply else "dry run")
    rows = []
    for row in broken.itertuples():
        src = cfg.raw_root / row.path
        dst = cfg.quarantine_root / row.path
        if not src.exists():
            continue
        rows.append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), path=row.path, src=str(src), dst=str(dst),
                         reason=row.reason, action="move" if args.apply else "dry-run"))
        if args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                raise FileExistsError(dst)
            shutil.move(str(src), str(dst))
            if src.exists() or not dst.exists():
                raise RuntimeError(f"move failed for {src}")
    if not rows:
        log.info("nothing to move")
        return
    out = pd.DataFrame(rows, columns=MOVES_COLUMNS)
    target = _moves_log(cfg) if args.apply else cfg.logs_dir / "quarantine_dry_run.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, mode="a" if (args.apply and target.exists()) else "w",
               header=not (args.apply and target.exists()), index=False)
    log.info("%s %d files, %.2f GB -> %s", "moved" if args.apply else "would move", len(out),
             broken["size_bytes"].sum() / 2**30, target)


def cmd_restore(cfg: DataConfig, args, log) -> None:
    moves = pd.read_csv(_moves_log(cfg))
    moves = moves[moves["action"] == "move"]
    if args.paths:
        moves = moves[moves["path"].isin(args.paths)]
    restored = []
    for row in moves.itertuples():
        dst, src = Path(row.dst), Path(row.src)
        if dst.exists() and not src.exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(src))
            restored.append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), path=row.path, src=str(dst),
                                 dst=str(src), reason="restore", action="restore"))
    if restored:
        pd.DataFrame(restored, columns=MOVES_COLUMNS).to_csv(_moves_log(cfg), mode="a", header=False, index=False)
    log.info("restored %d files", len(restored))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    add_config_args(p)
    p.add_argument("--skip-duplicates", action="store_true")
    p.add_argument("--with-extract-errors", action="store_true", help="also mark videos that failed extraction")
    p = sub.add_parser("quarantine")
    add_config_args(p)
    p.add_argument("--apply", action="store_true", help="actually move files (default is a dry run)")
    p = sub.add_parser("restore")
    add_config_args(p)
    p.add_argument("--paths", nargs="*", help="restore only these relative paths")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    log = setup_logging(cfg.logs_dir / "cleaning.log")
    {"check": cmd_check, "quarantine": cmd_quarantine, "restore": cmd_restore}[args.cmd](cfg, args, log)


if __name__ == "__main__":
    main()

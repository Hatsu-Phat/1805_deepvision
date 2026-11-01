"""Step 3: build per-split video manifests from the inventory and validation status.

Outputs manifests/<split>_videos.csv with one row per usable video:
    video_id, path, split, label, generator, duration, fps, width, height, nb_frames, priority

`priority` interleaves generators in a seeded random order (0 = first). Processing videos by
priority keeps every prefix of the list balanced across generators, so a partial extraction
is still a usable dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import add_config_args, config_from_args
from .io import setup_logging

COLUMNS = ["video_id", "path", "split", "label", "generator", "duration", "fps", "width", "height",
           "nb_frames", "size_bytes", "priority"]


def make_video_ids(paths: pd.Series) -> pd.Series:
    """File stem, made filesystem-safe; a short path hash is appended when stems collide."""
    stems = paths.map(lambda p: re.sub(r"[^A-Za-z0-9._-]+", "_", Path(p).stem))
    dup = stems.duplicated(keep=False)
    suffix = paths[dup].map(lambda p: hashlib.md5(p.encode("utf-8")).hexdigest()[:8])
    stems.loc[dup] = stems[dup] + "_" + suffix
    return stems


def interleave_priority(df: pd.DataFrame, seed: int, balance: str = "class") -> pd.Series:
    """Seeded processing order that keeps every prefix of the list balanced.

    balance="class":      half of each prefix is real, half fake; generators share their class half equally
    balance="proportion": every prefix keeps the generator proportions of the full list
    Once a generator runs out, the remaining ones simply continue.
    """
    rng = np.random.default_rng(seed)
    gens = df.groupby("generator", sort=True)
    labels = gens["label"].first() if "label" in df else pd.Series(1, index=list(gens.groups))
    n_per_label = labels.value_counts()
    pri = pd.Series(0.0, index=df.index)
    for name, g in gens:
        perm = rng.permutation(len(g)) + rng.random()
        if balance == "class":
            weight = 1.0 / (len(n_per_label) * n_per_label[labels[name]])
            pri.loc[g.index] = perm / weight
        else:
            pri.loc[g.index] = perm / len(g)
    return pri.rank(method="first").astype(int) - 1


def build(inv: pd.DataFrame, status: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    ok = status.loc[status["status"] == "ok", ["path"]]
    df = ok.merge(inv, on="path", how="left")
    out = {}
    for split, g in df.groupby("split"):
        g = g.copy()
        g["video_id"] = g.groupby("generator", group_keys=False)["path"].apply(make_video_ids)
        g["priority"] = interleave_priority(g, seed)
        out[split] = g.sort_values("priority").reindex(columns=COLUMNS).reset_index(drop=True)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    log = setup_logging(cfg.logs_dir / "manifest.log")

    inv = pd.read_csv(cfg.inventory_dir / "inventory.csv")
    status = pd.read_csv(cfg.inventory_dir / "video_status.csv")
    manifests = build(inv, status, args.seed)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, m in manifests.items():
        assert not m.duplicated(["generator", "video_id"]).any()
        m.to_csv(cfg.manifest_dir / f"{split}_videos.csv", index=False)
        counts[split] = {f"{g} ({'real' if lab == 0 else 'fake'})": int(n)
                         for (g, lab), n in m.groupby(["generator", "label"]).size().items()}
        log.info("%s: %d videos (real %d, fake %d)", split, len(m), int((m["label"] == 0).sum()),
                 int((m["label"] == 1).sum()))
    (cfg.manifest_dir / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    log.info("counts:\n%s", json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

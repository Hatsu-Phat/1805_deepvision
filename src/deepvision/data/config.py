"""Loading and validating the data pipeline config (configs/data.yaml)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import yaml

LABEL_NAMES = {0: "real", 1: "fake"}
# train2 holds the extra Youku reals added on 2026-11-06 (Youku_1M_10s.split.ab-ae). It is a split of its
# own so that the manifest, shards and features of train (already on Drive) never change; the splits
# manifest and the heads treat train + train2 as one training set.
TRAIN_SPLITS = ("train", "train2")
# vif_train / vif_test: ViF-CoT-4K and ViF-Bench (Skyra, arXiv:2512.15693), added 2026-11-09 for cross-dataset
# tests. They live in their own out_root (configs/vif.yaml) and are never pooled with the GenVideo splits.
SPLITS = ("train", "test", "train2", "vif_train", "vif_test")
# benchmark splits: files are kept even when they would be excluded (too short, duplicates) from training
EVAL_SPLITS = ("test", "vif_test")


@dataclass(frozen=True)
class Source:
    name: str
    dir: str
    split: str
    label: int
    generator: str

    @property
    def label_name(self) -> str:
        return LABEL_NAMES[self.label]


@dataclass
class DataConfig:
    raw_root: Path
    out_root: Path
    quarantine_root: Path
    sources: list[Source]
    video_extensions: tuple[str, ...] = (".mp4",)
    ignore_patterns: tuple[str, ...] = ()
    workers: int = 6
    validation: dict = field(default_factory=dict)
    extraction: dict = field(default_factory=dict)
    clips: dict = field(default_factory=dict)
    shards: dict = field(default_factory=dict)
    drive: dict = field(default_factory=dict)

    @property
    def inventory_dir(self) -> Path:
        return self.out_root / "inventory"

    @property
    def manifest_dir(self) -> Path:
        return self.out_root / "manifests"

    @property
    def frames_dir(self) -> Path:
        return self.out_root / "frames"

    @property
    def clips_dir(self) -> Path:
        return self.out_root / "clips"

    @property
    def shards_dir(self) -> Path:
        return self.out_root / "shards"

    @property
    def logs_dir(self) -> Path:
        return self.out_root / "logs"

    def source(self, name: str) -> Source:
        for s in self.sources:
            if s.name == name:
                return s
        raise KeyError(name)


def load_config(path: str | Path, raw_root: str | None = None, out_root: str | None = None) -> DataConfig:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sources = [Source(**s) for s in cfg["sources"]]
    names = [s.name for s in sources]
    if len(names) != len(set(names)):
        raise ValueError("source names must be unique")
    for s in sources:
        if s.label not in LABEL_NAMES:
            raise ValueError(f"{s.name}: label must be 0 or 1")

    return DataConfig(
        raw_root=Path(raw_root or cfg["raw_root"]),
        out_root=Path(out_root or cfg["out_root"]),
        quarantine_root=Path(cfg["quarantine_root"]),
        sources=sources,
        video_extensions=tuple(e.lower() for e in cfg.get("video_extensions", [".mp4"])),
        ignore_patterns=tuple(cfg.get("ignore_patterns", [])),
        workers=int(cfg.get("workers", 6)),
        validation=cfg.get("validation", {}),
        extraction=cfg.get("extraction", {}),
        clips=cfg.get("clips", {}),
        shards=cfg.get("shards", {}),
        drive=cfg.get("drive", {}),
    )


def add_config_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("-c", "--config", default="configs/data.yaml")
    ap.add_argument("--raw-root", default=None, help="override raw_root from the config")
    ap.add_argument("--out-root", default=None, help="override out_root from the config")
    ap.add_argument("--workers", type=int, default=None)


def config_from_args(args: argparse.Namespace) -> DataConfig:
    cfg = load_config(args.config, args.raw_root, args.out_root)
    if args.workers:
        cfg.workers = args.workers
    return cfg

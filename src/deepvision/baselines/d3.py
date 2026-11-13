"""D3 (Zheng et al., ICCV 2025, arXiv 2508.00701): training-free second-order score, re-implemented on our shards.

Same computation as D3/models/D3_model.py (repo Zig-HS/D3): frames resized to 224x224 (our clips already are),
ImageNet mean/std, XCLIPVisionModel (microsoft/xclip-base-patch16) pooler_output per frame, then
    d1_t = ||f_{t+1} - f_t||_2   (loss "l2")  or cos(f_t, f_{t+1})  (loss "cos")
    d2_t = d1_{t+1} - d1_t,  score = std(d2) (torch.std, unbiased)  ->  real videos score HIGHER (D3 AP uses 1 - y).
We report fake_score = -std so that larger = more fake (AUROC with fake as positive).
Frames are pushed through the vision model in chunks of whole videos, so X-CLIP's 8-frame cross-frame groups
stay aligned with the video boundaries exactly as in D3 (16 frames = 2 groups of 8).
Deviation from D3's preprocessing (documented): D3 crops 3 s at a random start with ffmpeg fps=8 and keeps the
first 16 frames; our clips use a centered 2 s window at 8 fps and the DeMamba 10 % border crop.

    python -m deepvision.baselines.d3 --root gdrive:DeepVision/vif_8fps_v1 --split vif_test --out ~/deepvision_runs/g1/d3_vif_8fps
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("d3")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_ID = "microsoft/xclip-base-patch16"


def d3_scores(feats):
    """feats: torch (B, T, D) float -> dict of (B,) tensors: std/avg of the 2nd-order l2 and cos distances."""
    import torch
    import torch.nn.functional as F

    a, b = feats[:, :-1], feats[:, 1:]
    out = {}
    for loss, d1 in (("l2", torch.norm(a - b, p=2, dim=-1)), ("cos", F.cosine_similarity(a, b, dim=-1))):
        d2 = d1[:, 1:] - d1[:, :-1]
        out[f"std_{loss}"] = torch.std(d2, dim=1)
        out[f"avg_{loss}"] = torch.mean(d2, dim=1)
    return out


def run(root: str, split: str, out: Path, batch_videos: int = 8, num_frames: int = 16, streams: int = 2,
        rclone: str = "rclone", gpu_fraction: float = 0.8, limit: int | None = None) -> dict:
    import torch
    from transformers import XCLIPVisionModel

    from ..data.stream import ShardStream, decode_jpeg_batch, read_index, shard_paths
    from ..train.run import setup_device

    device = setup_device(gpu_fraction)
    out.mkdir(parents=True, exist_ok=True)
    index = read_index(root, split, rclone)
    model = XCLIPVisionModel.from_pretrained(MODEL_ID).to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    stream = ShardStream(root, shard_paths(index, split, limit), num_frames=num_frames, streams=streams,
                         decode="none", rclone=rclone, frame_indices=list(range(num_frames)))
    rows, feats_all, batch, t0 = [], [], [], time.time()

    @torch.no_grad()
    def flush(batch):
        x = decode_jpeg_batch([v["frames"] for v in batch], device)  # (B, T, 3, H, W) uint8
        b, t = x.shape[:2]
        imgs = (x.flatten(0, 1).float() / 255.0 - mean) / std
        f = model(pixel_values=imgs, output_hidden_states=False).pooler_output.view(b, t, -1).float()
        s = d3_scores(f)
        for i, v in enumerate(batch):
            r = {k: v[k] for k in ("key", "video_id", "label", "generator", "shard", "pos")}
            r |= {k: float(val[i]) for k, val in s.items()}
            rows.append(r)
        feats_all.append(f.half().cpu().numpy())

    for v in stream:
        batch.append(v)
        if len(batch) == batch_videos:
            flush(batch)
            batch = []
            if len(rows) % 400 == 0:
                LOG.info("%d videos, %.1f/s, max VRAM %.0f MiB", len(rows), len(rows) / (time.time() - t0),
                         torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0)
    if batch:
        flush(batch)
    df = pd.DataFrame(rows)
    df["fake_score"] = -df["std_l2"]
    df.to_csv(out / "scores.csv", index=False)
    np.save(out / "features_xclip_b16.npy", np.concatenate(feats_all))
    from sklearn.metrics import roc_auc_score

    res = {"root": root, "split": split, "videos": len(df), "seconds": round(time.time() - t0, 1),
           "model": MODEL_ID, "num_frames": num_frames,
           "auroc_fake_neg_std_l2": float(roc_auc_score(df["label"], df["fake_score"])),
           "auroc_fake_neg_std_cos": float(roc_auc_score(df["label"], -df["std_cos"]))}
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    LOG.info("done %s", json.dumps(res))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="vif_test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="first N shards (smoke test)")
    ap.add_argument("--rclone", default="rclone")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.root, args.split, Path(args.out).expanduser(), args.batch, args.frames, rclone=args.rclone,
        limit=args.limit)


if __name__ == "__main__":
    main()

"""Side branch SM (2026-11-13): Scout selection + MIL on one frozen encoder (vit_b16), no heavy second backbone.

    16 clip frames -> vit_b16 (frozen, cached: even store + odd store) -> z (16, 768)
      -> Scout: [LN z ; dz ; d2z] (differences scaled by their clip mean norm) -> Linear + clip context
         + Conv1D k3 d1/d2 -> 16 frame scores
      -> top-K frames (train: perturbed top-K of Cordonnier et al. 2021, as in STTS; test: hard top-K)
      -> MIL on the K selected frames (gated attention, Ilse et al. 2018) -> fake / real logit
The MIL head never sees the scores or the indices. Data: ViF only (16 frames exist only there);
train = ViF-CoT-4K minus a 10 % validation set (whole name groups: a real and every fake sharing its name),
test = ViF-Bench, paired protocol of Skyra. Checkpoints are picked on the validation set, never on test.

Methods (--methods):
    u8_mean     8 fixed frames (0,2,4,6,9,11,13,15), mean -> MLP            (= Phase 2 head)
    a16_mean    16 frames, mean -> MLP
    a16_attn    16 frames, gated-attention MIL                              (MIL without selection)
    topk_inst   per-frame MLP logits, mean of the K largest                 (selection by the instance score)
    rand_attn   K random frames per step (test: K evenly spaced), attention MIL
    scout_attn  Scout top-K, attention MIL                                  (the proposed block, end to end)
    scout_s0s1  same model, trained S0 (head on random K) -> S1 (head frozen, Scout only), pipeline v3 1.3
    scout_xfit  S0 -> S1 with cross-fitting: S0 on one half of the train names, S1 on the other half, so the Scout
                learns from videos the frozen head has never seen
Scout boundary handling (option scout_pad): zeros (first sm runs) | replicate (default since sm2).

    python -m deepvision.features.scout --methods u8_mean a16_attn scout_attn --k 4 8 --seeds 0 1 2
    python -m deepvision.features.scout --summary          # -> <runs_root>/sm_summary.csv + sm_delta.csv
    # sm2 (2026-11-13): replicate padding + the S0 -> S1 schedule
    python -m deepvision.features.scout --methods u8_mean rand_attn scout_attn scout_s0s1 --runs-root ~/deepvision_runs/sm2
Outputs per run <runs_root>/sm_<method>_k<K>_s<seed>/: config.yaml, env.txt, train.log, metrics.jsonl, best.pt,
predictions_{val,test}.csv (with the selected frames), final.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from ..eval.metrics import auroc, binary_metrics, vif_paired_protocol
from ..train.run import expand, prepare_run, set_seed, setup_device
from .extract import FRAME_SETS

LOG = logging.getLogger("deepvision")
T = 16
UNIFORM8 = FRAME_SETS["even"]
METHODS = ("u8_mean", "a16_mean", "a16_attn", "topk_inst", "rand_attn", "scout_attn", "scout_s0s1", "scout_xfit")
SELECTING = ("topk_inst", "rand_attn", "scout_attn", "scout_s0s1", "scout_xfit")
SCOUTS = ("scout_attn", "scout_s0s1", "scout_xfit")
ATTN = ("a16_attn", "rand_attn", "scout_attn", "scout_s0s1", "scout_xfit")  # methods for which K matters
DEFAULTS = {"epochs": 30, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4, "warmup_steps": 50,
            "hidden": 512, "dropout": 0.1, "scout_hidden": 64, "n_samples": 500, "sigma0": 0.1, "sigma_min": 0.01,
            "val_frac": 0.1, "select": "val/auroc",
            # added 2026-11-13 (sm2): boundary handling of the Scout; zeros = the first sm runs
            "scout_pad": "replicate", "s0_epochs": 20, "s1_epochs": 20,
            # added 2026-11-13 (sm3): scout_xfit trains the head on train names with hash < xfit_split, the Scout on the rest
            "xfit_split": 0.55}


# ----------------------------------------------------------------------------------------------- selection
class PerturbedTopK(torch.autograd.Function):
    """Differentiable top-k (Cordonnier et al., CVPR 2021): forward = mean over n noisy draws of the sorted
    one-hot top-k indicator (B, k, T); backward = Monte-Carlo estimate of d E[indicator] / d scores."""

    @staticmethod
    def forward(ctx, s: torch.Tensor, k: int, n: int, sigma: float):
        noise = torch.randn(s.shape[0], n, s.shape[1], device=s.device, dtype=s.dtype)
        idx = torch.topk(s.unsqueeze(1) + sigma * noise, k, dim=-1).indices.sort(-1).values  # (B, n, k)
        ind = F.one_hot(idx, s.shape[1]).to(s.dtype)                                          # (B, n, k, T)
        ctx.save_for_backward(noise, ind)
        ctx.sigma = sigma
        return ind.mean(1)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        noise, ind = ctx.saved_tensors
        w = torch.einsum("bkt,bnkt->bn", g, ind)                       # <g, indicator> of every draw
        return torch.einsum("bn,bnt->bt", w, noise) / (noise.shape[1] * ctx.sigma), None, None, None


def hard_topk(s: torch.Tensor, k: int) -> torch.Tensor:
    idx = torch.topk(s, k, dim=-1).indices.sort(-1).values
    return F.one_hot(idx, s.shape[1]).to(s.dtype)


def even_indices(k: int, t: int = T) -> list[int]:
    return np.linspace(0, t - 1, k).round().astype(int).tolist()


def indicator(idx: torch.Tensor, t: int = T) -> torch.Tensor:
    return F.one_hot(idx, t).float()


def clip_differences(z: torch.Tensor, eps: float = 1e-6, pad: str = "zeros") -> tuple[torch.Tensor, torch.Tensor]:
    """First / second temporal differences (zero for the first frame(s)), each divided by its mean norm over
    the clip, so the Scout sees the shape of the motion and not its amount (which depends on the sampling rate).
    pad="replicate" copies the first defined difference into the undefined boundary frames instead of zeros, so the
    boundary frames carry no constant marker the Scout could use as a position code."""
    d1 = torch.zeros_like(z)
    d1[:, 1:] = z[:, 1:] - z[:, :-1]
    d2 = torch.zeros_like(z)
    d2[:, 2:] = d1[:, 2:] - d1[:, 1:-1]
    n1 = d1[:, 1:].norm(dim=-1).mean(1).clamp_min(eps)
    n2 = d2[:, 2:].norm(dim=-1).mean(1).clamp_min(eps)
    if pad == "replicate":
        d1[:, 0] = d1[:, 1]
        d2[:, :2] = d2[:, 2:3]
    elif pad != "zeros":
        raise ValueError(f"unknown pad {pad!r}")
    return d1 / n1[:, None, None], d2 / n2[:, None, None]


class Scout(nn.Module):
    def __init__(self, dim: int, hidden: int = 64, pad: str = "zeros"):
        super().__init__()
        self.pad = pad
        mode = "replicate" if pad == "replicate" else "zeros"
        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Linear(3 * dim, hidden)
        self.ctx = nn.Linear(hidden, hidden)
        self.c1 = nn.Conv1d(hidden, hidden, 3, padding=1, dilation=1, padding_mode=mode)
        self.c2 = nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2, padding_mode=mode)
        self.out = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        d1, d2 = clip_differences(z, pad=self.pad)
        h = F.gelu(self.proj(torch.cat([self.ln(z), d1, d2], -1)))
        h = h + self.ctx(h.mean(1, keepdim=True))
        h = h.transpose(1, 2)
        h = h + F.gelu(self.c1(h))
        h = h + F.gelu(self.c2(h))
        return self.out(h.transpose(1, 2)).squeeze(-1)                  # (B, T)


# ----------------------------------------------------------------------------------------------- MIL
class GatedAttention(nn.Module):
    def __init__(self, dim: int, att: int = 128):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(dim, att), nn.Linear(dim, att), nn.Linear(att, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.softmax(self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1), dim=1)
        return torch.einsum("bk,bkd->bd", a, h), a


class SMNet(nn.Module):
    """z (B, 16, D) float -> logit (B,). `self.selection` overrides how frames are picked at eval:
    None (the method's own) | "even" (K evenly spaced) | "random" (K random, seeded by the caller)."""

    def __init__(self, method: str, dim: int, mean: torch.Tensor, std: torch.Tensor, k: int = 8,
                 hidden: int = 512, dropout: float = 0.1, scout_hidden: int = 64, n_samples: int = 500,
                 scout_pad: str = "zeros"):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        self.method, self.k, self.n_samples = method, k, n_samples
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(1e-6))
        self.frame = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.attn = GatedAttention(hidden) if method in ATTN else None
        self.cls = nn.Linear(hidden, 1)
        self.scout = Scout(dim, scout_hidden, scout_pad) if method in SCOUTS else None
        self.sigma = 0.1
        self.selection: str | None = None

    def weights(self, z: torch.Tensor) -> torch.Tensor | None:
        """(B, K, T) frame-selection matrix (soft during Scout training), or None = all 16 frames."""
        b, t, dev = z.shape[0], z.shape[1], z.device
        if self.method == "u8_mean":
            return indicator(torch.tensor(UNIFORM8, device=dev), t).expand(b, -1, -1)
        if self.method in ("a16_mean", "a16_attn", "topk_inst"):
            return None
        if self.selection == "random" or (self.method == "rand_attn" and self.training and self.selection is None):
            idx = torch.rand(b, t, device=dev).topk(self.k, -1).indices.sort(-1).values
            return indicator(idx, t)
        if self.selection == "even" or self.method == "rand_attn":
            return indicator(torch.tensor(even_indices(self.k, t), device=dev), t).expand(b, -1, -1)
        s = self.scout(z)
        if self.training:
            return PerturbedTopK.apply(s, self.k, self.n_samples, self.sigma)
        return hard_topk(s, self.k)

    def forward(self, z: torch.Tensor, return_weights: bool = False):
        z = z.float()
        w = self.weights(z)
        x = z if w is None else torch.einsum("bkt,btd->bkd", w, z)
        h = self.frame((x - self.mean) / self.std)                       # (B, K, H)
        if self.method == "topk_inst":
            inst = self.cls(h).squeeze(-1)                                # (B, 16)
            top = inst.topk(self.k, -1)
            logit = top.values.mean(-1)
            w = indicator(top.indices.sort(-1).values, z.shape[1])       # reported selection only
        elif self.attn is not None:
            logit = self.cls(self.attn(h)[0]).squeeze(-1)
        else:
            logit = self.cls(h.mean(1)).squeeze(-1)
        return (logit, w) if return_weights else logit


# ----------------------------------------------------------------------------------------------- data
def name_bucket(video_id: str) -> float:
    """Deterministic value in [0, 1) from the name (real and fakes of the same name share it)."""
    return int(hashlib.sha1(str(video_id).encode()).hexdigest()[:8], 16) / 2**32


def split_val(rows: pd.DataFrame, frac: float) -> np.ndarray:
    """Boolean mask of the validation videos: every video whose name falls in the first `frac` of the hash."""
    return np.array([name_bucket(v) < frac for v in rows["video_id"]])


def xfit_folds(rows: pd.DataFrame, split: float) -> np.ndarray:
    """True for the train videos of the S1 (Scout) fold: name hash >= split. Names stay whole, like split_val."""
    return np.array([name_bucket(v) >= split for v in rows["video_id"]])


def load_vif(even: Path, odd: Path, encoder: str, root: str, rclone: str = "rclone") -> dict:
    from ..data.stream import read_index
    from .structure import load_split

    sets = {}
    for split in ("vif_train", "vif_test"):
        z, rows = load_split(even, odd, encoder, split, read_index(root, split, rclone))
        sets[split] = (torch.from_numpy(z).half(), rows)
    return sets


# ----------------------------------------------------------------------------------------------- metrics
def paired_bacc_boot(df: pd.DataFrame, n_boot: int, seed: int = 0, threshold: float = 0.5) -> np.ndarray:
    """Paired-protocol mean bAcc (as vif_paired_protocol) under a cluster bootstrap over names: each draw
    resamples the real names with replacement and weights every pair by its name's count. (n_boot,) array;
    the same seed gives the same draws for every df with the same reals, so differences are paired."""
    reals = df[df["label"] == 0].drop_duplicates("video_id", keep="last").set_index("video_id")["score"]
    names = np.array(sorted(reals.index))
    pos = {n: i for i, n in enumerate(names)}
    counts = np.random.default_rng(seed).multinomial(len(names), np.full(len(names), 1 / len(names)), size=n_boot)
    out = np.zeros(n_boot)
    gens = 0
    for _, fakes in df[df["label"] == 1].groupby("generator", sort=True):
        fakes = fakes[fakes["video_id"].isin(reals.index)]
        if not len(fakes):
            continue
        j = np.array([pos[v] for v in fakes["video_id"]])
        w = counts[:, j].astype(float)                                   # (n_boot, n_pairs)
        hit = (fakes["score"].to_numpy() > threshold).astype(float)
        rej = (reals.loc[fakes["video_id"]].to_numpy() <= threshold).astype(float)
        tot = w.sum(1).clip(min=1)
        out += 0.5 * ((w * hit).sum(1) + (w * rej).sum(1)) / tot
        gens += 1
    return out / max(gens, 1)


def scores_of(model: SMNet, z: torch.Tensor, device, batch: int = 1024, selection: str | None = None,
              seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities and the selected frames ((N, 16) 0/1, all ones when the method uses every frame)."""
    model.eval()
    model.selection = selection
    gen = torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else [])
    probs, sel = [], []
    with gen, torch.no_grad():
        torch.manual_seed(seed)
        for i in range(0, len(z), batch):
            logit, w = model(z[i:i + batch].to(device), return_weights=True)
            probs.append(torch.sigmoid(logit).cpu())
            sel.append(torch.ones(len(logit), z.shape[1]) if w is None else w.sum(1).clamp(max=1).cpu())
    model.selection = None
    return torch.cat(probs).numpy(), torch.cat(sel).numpy().round().astype(np.int8)


def eval_set(rows: pd.DataFrame, s: np.ndarray, paired: bool) -> dict:
    out = binary_metrics(rows["label"], s)
    if paired:
        v = vif_paired_protocol(rows[["video_id", "label", "generator"]].assign(score=s))
        out |= {f"vif_avg_{x}": v[f"avg_{x}"] for x in ("bacc", "recall", "f1", "real_acc", "auroc")}
        out["vif_per_generator"] = v["per_generator"]
    return out


def selection_stats(sel: np.ndarray, rows: pd.DataFrame, prev: np.ndarray | None = None) -> dict:
    """Frame-position histogram, idx-only probe (AUROC of a logistic fit on the 16-bit selection mask,
    in-sample: an upper bound on the label information carried by the indices), Jaccard to `prev`."""
    from sklearn.linear_model import LogisticRegression

    out = {"position_freq": sel.mean(0).round(4).tolist()}
    y = rows["label"].to_numpy()
    if sel.min(0).sum() < sel.shape[1] and len(np.unique(y)) == 2:
        p = LogisticRegression(max_iter=1000).fit(sel, y).predict_proba(sel)[:, 1]
        out["idx_probe_auroc"] = auroc(y, p)
    if prev is not None:
        inter, union = (sel & prev).sum(1), (sel | prev).sum(1)
        out["jaccard_prev"] = float(np.mean(inter / np.maximum(union, 1)))
    return out


# ----------------------------------------------------------------------------------------------- training
def run_id(method: str, k: int, seed: int) -> str:
    return f"sm_{method}_k{k if method in SELECTING else 0}_s{seed}"


def train_one(cfg: dict, method: str, k: int, seed: int, data: dict, device) -> dict:
    rid = run_id(method, k, seed)
    root = expand(cfg["runs_root"])
    if (root / rid / "final.json").exists():
        return json.loads((root / rid / "final.json").read_text())
    if (root / rid).exists():                         # an interrupted run restarts from scratch (minutes)
        for p in (root / rid).iterdir():
            p.unlink()
    p = cfg["train"]
    run_dir, log = prepare_run(cfg | {"method": method, "k": k, "seed": seed}, rid)
    ztr, ytr = data["train"][0], torch.as_tensor(data["train"][1]["label"].to_numpy(copy=True)).float()
    set_seed(seed)
    zf = ztr.float()
    mean, std = zf.mean((0, 1)), zf.std((0, 1))
    del zf
    model = SMNet(method, ztr.shape[-1], mean, std, k, p["hidden"], p["dropout"], p["scout_hidden"],
                  p["n_samples"], p["scout_pad"]).to(device)
    ztr_d, ytr_d = ztr.to(device), ytr.to(device)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    metrics_f = open(run_dir / "metrics.jsonl", "w", encoding="utf-8")

    def fit(stage: str, params, epochs: int, train_sel: str | None, eval_sel: str | None,
            idx: np.ndarray | None = None) -> dict:
        """Train `params` for `epochs`; return the best state on the validation metric (evaluated with eval_sel)."""
        opt = torch.optim.AdamW(params, lr=p["lr"], weight_decay=p["weight_decay"])
        idx = np.arange(len(ztr)) if idx is None else idx
        steps_per_epoch = math.ceil(len(idx) / p["batch_size"])
        total = epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(p["warmup_steps"], 1))
                                                  * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
        best, best_val, prev_sel, step = None, -1.0, None, 0
        for epoch in range(epochs):
            model.train()
            model.selection = train_sel
            if stage == "s1":
                model.frame.eval()                              # frozen head: no dropout while the Scout learns
            losses = []
            for b in np.array_split(rng.permutation(idx), steps_per_epoch):
                model.sigma = max(p["sigma_min"], p["sigma0"] * (1 - step / max(total - 1, 1)))
                bi = torch.as_tensor(b, device=device)
                loss = F.binary_cross_entropy_with_logits(model(ztr_d[bi]), ytr_d[bi])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                losses.append(loss.item())
                step += 1
            model.selection = None
            rec = {"stage": stage, "epoch": epoch, "step": step, "loss": float(np.mean(losses)), "sigma": model.sigma}
            for name in ("val", "test"):
                s, sel = scores_of(model, data[name][0], device, selection=eval_sel, seed=seed)
                m = eval_set(data[name][1], s, paired=True)
                rec[name] = {x: m[x] for x in ("auroc", "bacc", "vif_avg_bacc", "vif_avg_auroc")}
                if name == "val" and method in SELECTING and eval_sel is None:
                    st = selection_stats(sel, data[name][1], prev_sel)
                    rec["val_jaccard_prev"] = st.get("jaccard_prev")
                    prev_sel = sel
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            key, metric = p["select"].split("/")
            if rec[key][metric] > best_val:
                best_val = rec[key][metric]
                best = {"stage": stage, "epoch": epoch, "model": {n: x.detach().clone() for n, x in model.state_dict().items()}}
            log.info("%s %s ep %d loss %.4f sigma %.3f | val AUROC %.4f pbAcc %.4f | test AUROC %.4f pbAcc %.4f", rid,
                     stage, epoch, rec["loss"], model.sigma, rec["val"]["auroc"], rec["val"]["vif_avg_bacc"],
                     rec["test"]["auroc"], rec["test"]["vif_avg_bacc"])
        model.load_state_dict(best["model"])
        return best

    folds = {}
    if method in ("scout_s0s1", "scout_xfit"):
        # S0: the MIL head on K random frames per step (picked on val with K evenly spaced frames);
        # S1: the head is frozen, only the Scout learns, with perturbed top-K (pipeline v3, section 1.3)
        i0 = i1 = None
        if method == "scout_xfit":
            b = xfit_folds(data["train"][1], p["xfit_split"])
            i0, i1 = np.where(~b)[0], np.where(b)[0]
            folds = {"s0_videos": int(len(i0)), "s1_videos": int(len(i1))}
            log.info("%s: cross-fitting, S0 on %d videos, S1 on %d", rid, len(i0), len(i1))
        head = [x for n, x in model.named_parameters() if not n.startswith("scout.")]
        s0 = fit("s0", head, p["s0_epochs"], "random", "even", i0)
        for x in head:
            x.requires_grad_(False)
        best = fit("s1", list(model.scout.parameters()), p["s1_epochs"], None, None, i1)
        best["s0_epoch"] = s0["epoch"]
    else:
        best = fit("e2e", list(model.parameters()), p["epochs"], None, None)
    metrics_f.close()
    torch.save(best, run_dir / "best.pt")
    final = {"run_id": rid, "method": method, "k": k if method in SELECTING else 0, "seed": seed,
             "best_epoch": best["epoch"], "best_stage": best["stage"], "s0_epoch": best.get("s0_epoch"),
             "select": p["select"], "train_videos": len(ztr), **folds,
             "train_seconds": round(time.time() - t0, 1),
             "params": sum(x.numel() for x in model.parameters()),
             "scout_params": sum(x.numel() for x in model.scout.parameters()) if model.scout is not None else 0}
    for name in ("val", "test"):
        rows = data[name][1]
        s, sel = scores_of(model, data[name][0], device)
        df = rows[["video_id", "generator", "label"]].assign(score=s, frames=["".join(map(str, r)) for r in sel])
        df.to_csv(run_dir / f"predictions_{name}.csv", index=False)
        final[name] = eval_set(rows, s, paired=True)
        if method in SELECTING:
            final[name]["selection"] = selection_stats(sel, rows)
    if method in SCOUTS:                                # same MIL head, other frame choices (mechanism check)
        for sel_mode in ("even", "random"):
            s, _ = scores_of(model, data["test"][0], device, selection=sel_mode, seed=seed)
            df = data["test"][1][["video_id", "generator", "label"]].assign(score=s)
            df.to_csv(run_dir / f"predictions_test_{sel_mode}.csv", index=False)
            final[f"test_{sel_mode}"] = eval_set(data["test"][1], s, paired=True)
    (run_dir / "final.json").write_text(json.dumps(final, indent=2, default=float))
    log.info("%s: best ep %d | test AUROC %.4f paired bAcc %.4f F1 %.4f", rid, best["epoch"], final["test"]["auroc"],
             final["test"]["vif_avg_bacc"], final["test"]["vif_avg_f1"])
    return final


# ----------------------------------------------------------------------------------------------- summary
def summary(root: Path, n_boot: int = 2000, ref: str = "sm_u8_mean_k0") -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for f in sorted(root.glob("sm_*/final.json")):
        r = json.loads(f.read_text())
        for part in [x for x in r if x.startswith("test") or x == "val"]:
            m = r[part]
            rows.append({"run_id": r["run_id"], "method": r["method"], "k": r["k"], "seed": r["seed"], "eval": part,
                         "best_epoch": r["best_epoch"], "auroc": m["auroc"], "bacc": m["bacc"],
                         "paired_bacc": m["vif_avg_bacc"], "paired_f1": m["vif_avg_f1"],
                         "paired_auroc": m["vif_avg_auroc"], "real_acc": m["vif_avg_real_acc"],
                         "idx_probe_auroc": m.get("selection", {}).get("idx_probe_auroc"),
                         "params": r["params"], "scout_params": r["scout_params"]})
    df = pd.DataFrame(rows)
    deltas = []
    refs = {s: pd.read_csv(root / f"{ref}_s{s}" / "predictions_test.csv") for s in df["seed"].unique()
            if (root / f"{ref}_s{s}" / "predictions_test.csv").exists()}
    for (method, k), g in df[df["eval"] == "test"].groupby(["method", "k"]):
        seeds = [s for s in g["seed"] if s in refs]
        if not seeds or f"sm_{method}_k{k}" == ref:
            continue
        d = np.mean([paired_bacc_boot(pd.read_csv(root / f"sm_{method}_k{k}_s{s}" / "predictions_test.csv"), n_boot)
                     - paired_bacc_boot(refs[s], n_boot) for s in seeds], axis=0)
        point = float(g[g["seed"].isin(seeds)]["paired_bacc"].mean()
                      - df[(df["run_id"].str.startswith(ref)) & (df["eval"] == "test") & df["seed"].isin(seeds)]["paired_bacc"].mean())
        deltas.append({"method": method, "k": k, "vs": ref, "seeds": len(seeds), "delta_paired_bacc": point,
                       "ci_lo": float(np.quantile(d, 0.025)), "ci_hi": float(np.quantile(d, 0.975))})
    return df, pd.DataFrame(deltas)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--k", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--encoder", default="vit_b16")
    ap.add_argument("--even", default="~/deepvision_features")
    ap.add_argument("--odd", default="~/deepvision_features_odd")
    ap.add_argument("--root", default="gdrive:DeepVision/vif_v1", help="where shards/<split>_index.csv is")
    ap.add_argument("--runs-root", default="~/deepvision_runs/sm")
    ap.add_argument("--set", action="append", default=[], help="train option override, e.g. epochs=10")
    ap.add_argument("--gpu-fraction", type=float, default=0.5)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = expand(args.runs_root)
    if args.summary:
        df, d = summary(root, args.n_boot)
        df.to_csv(root / "sm_summary.csv", index=False)
        d.to_csv(root / "sm_delta.csv", index=False)
        LOG.info("\n%s", df[df["eval"] == "test"].groupby(["method", "k"])[["auroc", "paired_bacc", "paired_f1"]]
                 .agg(["mean", "std"]).round(4).to_string())
        LOG.info("\n%s", d.round(4).to_string())
        return
    train = dict(DEFAULTS)
    for kv in args.set:
        key, _, val = kv.partition("=")
        if key not in train:
            raise ValueError(f"unknown option {key!r}")
        train[key] = type(train[key])(val)
    device = setup_device(args.gpu_fraction)
    sets = load_vif(Path(os.path.expanduser(args.even)), Path(os.path.expanduser(args.odd)), args.encoder, args.root)
    ztr, rtr = sets["vif_train"]
    val = split_val(rtr, train["val_frac"])
    data = {"train": (ztr[~val], rtr[~val].reset_index(drop=True)), "val": (ztr[val], rtr[val].reset_index(drop=True)),
            "test": sets["vif_test"]}
    LOG.info("train %d (fake %d) | val %d (fake %d) | test %d", (~val).sum(), int(data["train"][1]["label"].sum()),
             val.sum(), int(data["val"][1]["label"].sum()), len(data["test"][1]))
    cfg = {"encoder": args.encoder, "runs_root": str(root), "train": train}
    for seed in args.seeds:
        for method in args.methods:
            for k in (args.k if method in SELECTING else [0]):
                train_one(cfg, method, k, seed, data, device)


if __name__ == "__main__":
    main()

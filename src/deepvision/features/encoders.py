"""The frozen encoders of Phase 2 behind one interface (7 from 2026-11-05, vit_b16 added 2026-11-11).

    enc = build_encoder("resnet50").to("cuda")
    out = enc(frames)   # frames: uint8 (B, T, 3, H, W) RGB in [0, 255], already 224x224 (Phase 1 clips)
    out["frames"]       # (B, T, D) one pooled feature per frame
    out["video"]        # (B, Dv) X-CLIP only: video embedding after its multi-frame integration transformer

Every encoder applies its own input normalization (the clips are stored un-normalized).
Image encoders see each frame independently; X-CLIP sees the T frames together (cross-frame
messages inside its vision encoder), so T must equal its config (8).
Weights come from the Hugging Face Hub (timm and transformers); set HF_HOME to choose the cache.
"""
from __future__ import annotations

import json

import torch
from torch import nn

# name -> (library, model id, extra kwargs, pooling)
SPECS = {
    "resnet50": ("timm", "resnet50.tv_in1k", {}, "global average pool"),
    "effnetv2_b0": ("timm", "tf_efficientnetv2_b0.in1k", {}, "global average pool"),
    "convnext_t": ("timm", "convnext_tiny.fb_in1k", {}, "global average pool + layer norm"),
    "vit_ti16": ("timm", "vit_tiny_patch16_224.augreg_in21k_ft_in1k", {}, "CLS token"),
    # DINOv2 ViT-S/14 without registers; pretrained at 518 px, position embeddings resampled to 224 px by timm
    "dinov2_s14": ("timm", "vit_small_patch14_dinov2.lvd142m", {"img_size": 224}, "CLS token"),
    "xclip_b32": ("xclip", "microsoft/xclip-base-patch32", {}, "per frame: projected CLS; video: MIT output"),
    "swin_t": ("timm", "swin_tiny_patch4_window7_224.ms_in1k", {}, "global average pool"),
    # added 2026-11-11: same AugReg in21k -> in1k recipe as vit_ti16, so the pair isolates model size
    "vit_b16": ("timm", "vit_base_patch16_224.augreg_in21k_ft_in1k", {}, "CLS token"),
}
ENCODERS = tuple(SPECS)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() / 255 - self.mean) / self.std


class TimmEncoder(nn.Module):
    def __init__(self, model_id: str, pretrained: bool = True, **kwargs):
        super().__init__()
        import timm

        self.model = timm.create_model(model_id, pretrained=pretrained, num_classes=0, **kwargs).eval()
        cfg = self.model.pretrained_cfg
        self.norm = Normalize(cfg["mean"], cfg["std"])
        self.dim = self.model.num_features
        self.video_dim = 0
        self.info = {"library": f"timm {timm.__version__}", "model_id": model_id, "kwargs": kwargs,
                     "mean": list(cfg["mean"]), "std": list(cfg["std"]),
                     "pretrained_input_size": list(cfg.get("input_size", ())), "hf_hub_id": cfg.get("hf_hub_id")}

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> dict:
        b, t = frames.shape[:2]
        f = self.model(self.norm(frames.flatten(0, 1)))
        return {"frames": f.view(b, t, -1)}


def _pooled(out) -> torch.Tensor:
    return out.pooler_output if hasattr(out, "pooler_output") else out[1]


class XClipEncoder(nn.Module):
    """Vision side of X-CLIP: per-frame CLS -> visual_projection (frames), then MIT -> video embedding."""

    def __init__(self, model, mean, std, model_id: str = ""):
        super().__init__()
        self.vision = model.vision_model.eval()
        self.proj = model.visual_projection.eval()
        self.mit = model.mit.eval()
        self.num_frames = model.config.vision_config.num_frames
        self.norm = Normalize(mean, std)
        self.dim = self.video_dim = model.config.projection_dim
        import transformers

        self.info = {"library": f"transformers {transformers.__version__}", "model_id": model_id,
                     "mean": list(mean), "std": list(std), "num_frames": self.num_frames}

    @classmethod
    def from_pretrained(cls, model_id: str) -> "XClipEncoder":
        from huggingface_hub import hf_hub_download
        from transformers import XCLIPModel

        with open(hf_hub_download(model_id, "preprocessor_config.json"), encoding="utf-8") as f:
            pre = json.load(f)
        return cls(XCLIPModel.from_pretrained(model_id), pre["image_mean"], pre["image_std"], model_id)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> dict:
        b, t = frames.shape[:2]
        if t != self.num_frames:
            raise ValueError(f"X-CLIP was built for {self.num_frames} frames, got {t}")
        cls_tokens = _pooled(self.vision(pixel_values=self.norm(frames.flatten(0, 1))))
        per_frame = self.proj(cls_tokens).view(b, t, -1)
        return {"frames": per_frame, "video": _pooled(self.mit(per_frame))}


def build_encoder(name: str, pretrained: bool = True) -> nn.Module:
    library, model_id, kwargs, pooling = SPECS[name]
    enc = TimmEncoder(model_id, pretrained, **kwargs) if library == "timm" else XClipEncoder.from_pretrained(model_id)
    enc.info |= {"encoder": name, "pooling": pooling}
    return enc.eval()


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

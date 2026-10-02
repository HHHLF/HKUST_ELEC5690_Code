"""DINOv2 ViT backbone plus a randomly initialized 7-way classification head.

The checkpoint must be the official self-supervised backbone file
``dinov2_vit{b,s}14_pretrain.pth``. Those files contain ``cls_token``, ``pos_embed``,
``patch_embed`` and ``blocks`` only. They do not contain an ImageNet 1000-way head.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from isic_baseline import CLASS_NAMES

VARIANTS = {
    "vitb14": {"embed_dim": 768, "depth": 12, "num_heads": 12, "patch_size": 14},
    "vits14": {"embed_dim": 384, "depth": 12, "num_heads": 6, "patch_size": 14},
}

DEFAULT_WEIGHTS = {
    "vitb14": "weights/dinov2_vitb14_pretrain.pth",
    "vits14": "weights/dinov2_vits14_pretrain.pth",
}


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoV2Backbone(nn.Module):
    """Official DINOv2 ViT, including the positional-embedding interpolation used at non-518 sizes."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant}. Choose from {list(VARIANTS)}.")
        spec = VARIANTS[variant]
        self.variant = variant
        self.embed_dim = spec["embed_dim"]
        self.patch_size = spec["patch_size"]
        self.interpolate_offset = 0.1
        # 518 is the resolution of the released positional embedding (37 x 37 patches).
        num_patches = (518 // self.patch_size) ** 2
        self.patch_embed = PatchEmbed(self.patch_size, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))
        self.blocks = nn.ModuleList(
            [Block(self.embed_dim, spec["num_heads"]) for _ in range(spec["depth"])]
        )
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, x: torch.Tensor, width: int, height: int) -> torch.Tensor:
        num_patches = x.shape[1] - 1
        stored = self.pos_embed.shape[1] - 1
        if num_patches == stored and width == height:
            return self.pos_embed
        class_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        grid_w = width // self.patch_size
        grid_h = height // self.patch_size
        side = int(math.sqrt(stored))
        if side * side != stored:
            raise RuntimeError(f"Stored positional grid is not square: {stored}")
        patch_pos = patch_pos.reshape(1, side, side, dim).permute(0, 3, 1, 2).float()
        scale = (
            (grid_w + self.interpolate_offset) / side,
            (grid_h + self.interpolate_offset) / side,
        )
        patch_pos = F.interpolate(patch_pos, scale_factor=scale, mode="bicubic", antialias=False)
        if patch_pos.shape[-2:] != (grid_h, grid_w):
            # Fall back to an explicit size if the historical 0.1 offset does not land on the grid.
            patch_pos = F.interpolate(
                self.pos_embed[:, 1:].reshape(1, side, side, dim).permute(0, 3, 1, 2).float(),
                size=(grid_h, grid_w),
                mode="bicubic",
                antialias=False,
            )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos, patch_pos.to(dtype=class_pos.dtype)), dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Input {height}x{width} is not divisible by patch size {self.patch_size}."
            )
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self.interpolate_pos_encoding(tokens, width, height)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_features(x)
        return tokens[:, 0]


class ISICClassifier(nn.Module):
    def __init__(self, variant: str = "vitb14", num_classes: int = 7) -> None:
        super().__init__()
        if num_classes != len(CLASS_NAMES):
            raise ValueError(f"This assignment baseline is a {len(CLASS_NAMES)}-class model.")
        self.backbone = DinoV2Backbone(variant)
        # Random head. Nothing from an ImageNet classifier is copied into this layer.
        self.head = nn.Linear(self.backbone.embed_dim, num_classes)
        self.variant = variant
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        """Smaller learning rate on the pretrained backbone, larger rate on the new head."""
        decay, no_decay = [], []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr_backbone, "weight_decay": 0.0},
            {"params": self.head.parameters(), "lr": lr_head, "weight_decay": weight_decay},
        ]


def _flatten_checkpoint(obj) -> dict:
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a state dict, got {type(obj)}")
    for key in ("state_dict", "model", "teacher"):
        inner = obj.get(key)
        if isinstance(inner, dict) and any("patch_embed" in k or "blocks" in k for k in inner):
            obj = inner
            break
    cleaned = {}
    for key, value in obj.items():
        name = key
        for prefix in ("module.", "backbone."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    return cleaned


def load_dinov2_backbone(model: ISICClassifier, weights_path: str | Path) -> None:
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"DINOv2 backbone weights not found: {path}. "
            "Pass the local dinov2_vitb14_pretrain.pth or dinov2_vits14_pretrain.pth path. "
            "The program does not download weights at startup."
        )
    raw = torch.load(path, map_location="cpu")
    state = _flatten_checkpoint(raw)
    forbidden = [
        key
        for key, value in state.items()
        if "classifier" in key
        or key.startswith("head.")
        or (torch.is_tensor(value) and value.ndim == 2 and value.shape[0] == 1000)
    ]
    if forbidden:
        raise RuntimeError(
            "Refusing to load a checkpoint that contains an ImageNet classification head: "
            + ", ".join(forbidden[:8])
        )
    required = {"cls_token", "pos_embed", "patch_embed.proj.weight", "norm.weight"}
    missing_required = required.difference(state)
    if missing_required:
        raise RuntimeError(
            f"{path} is not a DINOv2 backbone checkpoint. Missing keys: {sorted(missing_required)}"
        )
    model.backbone.load_state_dict(state, strict=True)
    # mask_token is part of the released checkpoint but is unused for classification.
    model.backbone.mask_token.requires_grad_(False)
    embed = model.backbone.embed_dim
    if model.head.out_features != len(CLASS_NAMES) or model.head.in_features != embed:
        raise RuntimeError("Classification head does not match the 7-class DINOv2 backbone.")


def build_model(variant: str, weights_path: str | Path | None, load_pretrained: bool = True) -> ISICClassifier:
    model = ISICClassifier(variant=variant, num_classes=len(CLASS_NAMES))
    if load_pretrained:
        if not weights_path:
            raise ValueError("weights_path is required when load_pretrained=True.")
        load_dinov2_backbone(model, weights_path)
    return model

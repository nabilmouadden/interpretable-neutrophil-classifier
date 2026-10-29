"""Shared ViT encoder + cross-attention feature head + class-token category head.

* Encoder: a timm vision transformer (default DinoBloom-B, ViT-B/14). All parameters are
  frozen except the last ``trainable_blocks`` transformer blocks and the final norm.
* Feature head: one learned query per graded feature cross-attends over the encoder
  patch tokens (multi-head attention); each attended vector goes through its own small
  MLP to one logit. ``sigmoid(logit)`` is the graded score in [0, 1]. The attention
  weights of each query are the per-feature attention maps.
* Category head: LayerNorm -> dropout -> linear on the class token, one logit per
  category (independent sigmoids: multi-label).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import timm
import torch
from torch import nn

from neutroclf.utils import get_logger


class ModelOutput(NamedTuple):
    """Forward outputs. ``feature_attention`` has shape (B, F, P) (averaged over heads)."""

    category_logits: torch.Tensor
    feature_logits: torch.Tensor
    feature_attention: torch.Tensor


class Encoder(nn.Module):
    """timm ViT returning the class token and the patch tokens, partially fine-tuned."""

    def __init__(
        self,
        name: str,
        pretrained: bool,
        img_size: int,
        trainable_blocks: int = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.vit = timm.create_model(
            name, pretrained=pretrained, num_classes=0, img_size=img_size, **kwargs
        )
        self.embed_dim: int = int(self.vit.embed_dim)
        self.num_prefix_tokens: int = int(getattr(self.vit, "num_prefix_tokens", 1))
        cfg = getattr(self.vit, "pretrained_cfg", None) or {}
        # architecture name lets a checkpoint be rebuilt without the HF hub
        self.architecture: str = str(cfg.get("architecture") or name)
        self.set_trainable_blocks(trainable_blocks)

    def set_trainable_blocks(self, n: int) -> None:
        """Freeze everything, then unfreeze the last ``n`` blocks (+ final norm); -1 = all."""
        blocks = self.vit.blocks
        if n < 0 or n >= len(blocks):
            for p in self.vit.parameters():
                p.requires_grad_(True)
            return
        for p in self.vit.parameters():
            p.requires_grad_(False)
        if n > 0:
            for blk in blocks[len(blocks) - n :]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            norm = getattr(self.vit, "norm", None)
            if norm is not None:
                for p in norm.parameters():
                    p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.vit.forward_features(x)  # (B, prefix + P, D)
        return tokens[:, 0], tokens[:, self.num_prefix_tokens :]


class FeatureHead(nn.Module):
    """Per-feature learned queries cross-attending over patch tokens, then per-feature MLPs."""

    def __init__(
        self, in_dim: int, num_features: int, dim: int = 128, heads: int = 4, hidden: int = 128
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("feature_dim must be divisible by feature_heads")
        self.proj = nn.Linear(in_dim, dim)
        self.queries = nn.Parameter(torch.randn(num_features, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlps = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
            )
            for _ in range(num_features)
        )

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv = self.proj(patches)
        q = self.queries.unsqueeze(0).expand(patches.shape[0], -1, -1)
        attended, weights = self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        logits = torch.cat([mlp(attended[:, j]) for j, mlp in enumerate(self.mlps)], dim=1)
        return logits, weights


class CategoryHead(nn.Module):
    """Multi-label linear head on the class token."""

    def __init__(self, in_dim: int, num_categories: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, num_categories)
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.net(cls_token)


class NeutrophilClassifier(nn.Module):
    """Joint concept model: encoder + feature head (patch tokens) + category head (class token)."""

    def __init__(
        self,
        encoder: Encoder,
        num_categories: int,
        num_features: int,
        feature_dim: int = 128,
        feature_heads: int = 4,
        feature_mlp_hidden: int = 128,
        category_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_head = FeatureHead(
            encoder.embed_dim, num_features, feature_dim, feature_heads, feature_mlp_hidden
        )
        self.category_head = CategoryHead(encoder.embed_dim, num_categories, category_dropout)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        cls_token, patches = self.encoder(x)
        feature_logits, attention = self.feature_head(patches)
        return ModelOutput(self.category_head(cls_token), feature_logits, attention)

    def param_groups(self, lr_encoder: float, lr_category: float, lr_feature: float) -> list[dict]:
        """Optimiser parameter groups with separate learning rates (trainable params only)."""
        groups = [
            {"params": [p for p in self.encoder.parameters() if p.requires_grad], "lr": lr_encoder},
            {"params": list(self.category_head.parameters()), "lr": lr_category},
            {"params": list(self.feature_head.parameters()), "lr": lr_feature},
        ]
        return [g for g in groups if g["params"]]


def build_model(cfg: dict, pretrained: bool | None = None) -> NeutrophilClassifier:
    """Build the model from a config.

    ``pretrained=False`` rebuilds the architecture without downloading weights (used when
    a checkpoint supplies all weights). For hub encoders, ``model.encoder_architecture`` in
    the config (recorded at training time) names the timm architecture to instantiate.
    """
    mcfg = cfg["model"]
    use_pretrained = bool(mcfg.get("pretrained", True)) if pretrained is None else pretrained
    name = mcfg["encoder"]
    if not use_pretrained and mcfg.get("encoder_architecture"):
        name = mcfg["encoder_architecture"]
    encoder = Encoder(
        name,
        pretrained=use_pretrained,
        img_size=int(cfg["data"]["image_size"]),
        trainable_blocks=int(mcfg.get("trainable_blocks", 6)),
        **(mcfg.get("encoder_kwargs") or {}),
    )
    model = NeutrophilClassifier(
        encoder,
        num_categories=len(cfg["categories"]),
        num_features=len(cfg["feature_map"]["features"]),
        feature_dim=int(mcfg.get("feature_dim", 128)),
        feature_heads=int(mcfg.get("feature_heads", 4)),
        feature_mlp_hidden=int(mcfg.get("feature_mlp_hidden", 128)),
        category_dropout=float(mcfg.get("category_dropout", 0.5)),
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    get_logger().info(
        "model: encoder=%s (%s) trainable %s / %s parameters",
        mcfg["encoder"],
        encoder.architecture,
        f"{n_train:,}",
        f"{n_total:,}",
    )
    return model


def save_checkpoint(
    path, model: NeutrophilClassifier, cfg: dict, extra: dict | None = None
) -> None:
    """Save weights + the resolved config (with the encoder architecture recorded)."""
    cfg = dict(cfg)
    cfg["model"] = dict(cfg["model"], encoder_architecture=model.encoder.architecture)
    payload = {"state_dict": model.state_dict(), "config": cfg, **(extra or {})}
    torch.save(payload, path)


def load_checkpoint(path, device: torch.device | str = "cpu") -> tuple[NeutrophilClassifier, dict]:
    """Rebuild the model from a checkpoint without downloading encoder weights."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, cfg

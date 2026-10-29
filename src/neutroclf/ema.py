"""Exponential moving average (Polyak averaging) of the trainable weights."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


class EMA:
    """EMA of trainable parameters with a warm-up on the decay.

    The effective decay at step ``t`` is ``min(decay, (1 + t) / (10 + t))``, so that short
    runs are not dominated by the random initial weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.steps = 0
        self.shadow = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextmanager
    def applied(self, model: nn.Module):
        """Temporarily swap the EMA weights into ``model``."""
        backup = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    backup[n] = p.detach().clone()
                    p.copy_(self.shadow[n])
        try:
            yield model
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in backup:
                        p.copy_(backup[n])

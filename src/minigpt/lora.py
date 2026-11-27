"""LoRA (Hu et al. 2021): frozen base weights plus a trainable low-rank update B @ A per linear.

`apply_lora` wraps the GPT's attention (and MLP) linears and freezes everything else, including
the embeddings and the tied LM head. `merge_lora` folds the update into the base weights, so the
saved checkpoint has the plain GPT layout and every downstream command works unchanged.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from minigpt.model import GPTModel

ATTN = {"attn": ("c_attn", "c_proj")}  # {block attribute: linear names}
ALL = {**ATTN, "mlp": ("c_fc", "c_proj")}


class LoRALinear(nn.Module):
    """`base(x) + (alpha / rank) * x A^T B^T`; B starts at zero, so the model starts unchanged."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # same as PEFT: U(+-1/sqrt(in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
        return out


def apply_lora(model: GPTModel, rank: int, alpha: float, targets: str) -> int:
    """Freeze `model`, wrap the target linears in place, return the trainable parameter count.

    targets: "attn" = attention `c_attn`, `c_proj`; "all" = those plus the MLP `c_fc`, `c_proj`.
    """
    if targets not in ("all", "attn"):
        raise ValueError(f"lora targets {targets!r}: expected 'all' or 'attn'")
    wanted = ALL if targets == "all" else ATTN
    model.requires_grad_(False)
    for block in model.blocks:
        for attr, names in wanted.items():
            parent = getattr(block, attr)
            for name in names:
                if isinstance(child := getattr(parent, name, None), nn.Linear):
                    setattr(parent, name, LoRALinear(child, rank, alpha))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_lora(model: GPTModel) -> None:
    """Fold every LoRA update into its base weight and unwrap; no-op without LoRA."""
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    child.base.weight += child.scale * child.B @ child.A
                setattr(parent, name, child.base)

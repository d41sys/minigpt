"""Task arithmetic (Ilharco et al., arXiv 2212.04089): add scaled expert deltas to one base.

theta = theta_base + L * sum_i (theta_expert_i - theta_base). L = 1/n averages the deltas;
L = 1 adds them in full. Experts must be fine-tunes of the same base, so every config matches.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch

from minigpt.model import GPTModel


@torch.no_grad()
def task_arithmetic(
    base: GPTModel, experts: Sequence[GPTModel], scale: float | None = None
) -> GPTModel:
    """A new model `base + scale * sum(expert - base)`; `scale=None` means 1/len(experts)."""
    if not experts:
        raise ValueError("need at least one expert")
    for e in experts:
        if e.config != base.config:
            raise ValueError(f"expert config {e.config} differs from the base's {base.config}")
    scale = 1 / len(experts) if scale is None else scale
    b = base.state_dict()
    states = [e.state_dict() for e in experts]
    # Read from the untouched base: the LM head is tied to the embedding, so an in-place add on
    # the shared tensor would apply its delta twice.
    new = {
        k: w + scale * sum(s[k] - w for s in states) if w.is_floating_point() else w
        for k, w in b.items()
    }
    merged = copy.deepcopy(base)
    merged.load_state_dict(new)
    return merged

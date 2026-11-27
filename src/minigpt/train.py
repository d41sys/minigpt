"""Training loop: AdamW or Muon, cosine LR with warmup, grad clipping, eval, checkpointing."""

from __future__ import annotations

import math
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from minigpt.model import GPTConfig, GPTModel


def get_device() -> torch.device:
    """Select best available accelerator in order: mps -> cuda -> cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Hybrid Newton-Schulz coefficients (a, b, c): 8 fast steps, then 2 that pin singular values at 1.
# DeepSeek-V4 report section 2.4, eq. 28.
NS_FAST = (3.4445, -4.7750, 2.0315)
NS_FINAL = (2.0, -1.5, 0.5)


def newton_schulz(m: torch.Tensor, fast_steps: int = 8, final_steps: int = 2) -> torch.Tensor:
    """Approximately orthogonalise `m` to U V^T (its SVD with all singular values set to 1).

    Hybrid Newton-Schulz of DeepSeek-V4 report section 2.4: Frobenius-normalise, then iterate
    M <- a M + b (M M^T) M + c (M M^T)^2 M with `NS_FAST` coefficients, then `NS_FINAL` ones.
    Runs on the short side (transposed if tall) so M M^T is the small Gram matrix.
    """
    x: torch.Tensor = m / (m.norm() + 1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.transpose(0, 1)
    for a, b, c in [NS_FAST] * fast_steps + [NS_FINAL] * final_steps:
        gram = x @ x.transpose(0, 1)
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.transpose(0, 1) if tall else x


class Muon(Optimizer):
    """Muon (Jordan et al. 2024) as in DeepSeek-V4 report section 2.4, Algorithm 1.

    Per 2-D weight W (n x m): Nesterov momentum, hybrid Newton-Schulz, update scaled by
    sqrt(max(n, m)) * rms so its RMS is `rms` (Liu et al. 2025, Moonlight: lets an AdamW learning
    rate be reused; V4 uses 0.18), then decoupled weight decay: W <- W (1 - lr wd) - lr O.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 6e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        rms: float = 0.18,
    ) -> None:
        super().__init__(
            params, {"lr": lr, "momentum": momentum, "weight_decay": weight_decay, "rms": rms}
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("momentum_buffer", torch.zeros_like(p))
                buf.mul_(mu).add_(p.grad)
                update = newton_schulz(p.grad.add(buf, alpha=mu))
                p.mul_(1.0 - lr * group["weight_decay"])
                p.add_(update, alpha=-lr * group["rms"] * max(p.shape) ** 0.5)


class MuonAdamW(Optimizer):
    """Muon on its param groups, AdamW on the rest, behind one optimizer.

    The param groups are shared with the two inner optimizers, so a learning-rate schedule that
    writes `param_groups[i]["lr"]` reaches both. The state dict holds one entry per inner one.
    """

    def __init__(self, muon: Muon, adamw: AdamW) -> None:
        self.inner = {"muon": muon, "adamw": adamw}
        super().__init__([g for o in self.inner.values() for g in o.param_groups], {})

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        for opt in self.inner.values():
            opt.step()

    def state_dict(self) -> dict[str, Any]:
        return {name: opt.state_dict() for name, opt in self.inner.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for name, opt in self.inner.items():
            opt.load_state_dict(state_dict[name])


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    learning_rate: float = 6e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    optim: str = "adamw",
) -> Optimizer:
    """Configure AdamW with decoupled weight decay, or Muon + AdamW with `optim="muon"`.

    AdamW applies weight decay to 2D matrix weights (embeddings, linear layers) and excludes 1D
    biases and layer normalizations. With Muon (DeepSeek-V4 report section 2.4) every 2D weight
    outside the embeddings and any parameter named `*head*`, `*router*` or `*hc_*` (LM and value
    heads, MoE gate, mHC) moves to Muon; the rest stays on AdamW with everything 1D. Muon reuses
    `weight_decay` and `learning_rate`.
    """
    if optim not in ("adamw", "muon"):
        raise ValueError(f"unknown optimizer {optim!r}: expected 'adamw' or 'muon'")
    embeddings = {
        id(p) for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()
    }
    muon_params: list[torch.nn.Parameter] = []
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        hidden = param.dim() == 2 and id(param) not in embeddings
        hidden = hidden and not any(key in name for key in ("head", "router", "hc_"))
        if optim == "muon" and hidden:
            muon_params.append(param)
        elif param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    adamw = AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
    if optim == "adamw":
        return adamw
    return MuonAdamW(Muon(muon_params, lr=learning_rate, weight_decay=weight_decay), adamw)


def get_lr_multiplier(
    step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.1
) -> float:
    """Compute cosine learning rate schedule multiplier with linear warmup."""
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    if step >= total_steps:
        return min_ratio
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_batches: int = 20,
) -> float:
    """Average cross-entropy per window over the first `max_batches` batches."""
    losses = window_losses(model, data_loader, device, max_batches=max_batches)
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def window_losses(
    model: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_batches: int | None = None,
) -> list[float]:
    """Mean cross-entropy of each window (dropout off), in loader order; first batches only."""
    model.eval()
    out: list[float] = []
    for i, (x, y) in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
        out.extend(ce.mean(dim=1).tolist())
    model.train()
    return out


def save_checkpoint(
    path: str | Path,
    model: GPTModel,
    optimizer: Optimizer | None,
    step: int,
    val_loss: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save training checkpoint."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "val_loss": val_loss,
        "config": asdict(model.config),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra is not None:
        payload.update(extra)
    torch.save(payload, ckpt_path)


def load_checkpoint(
    path: str | Path,
    model: GPTModel,
    optimizer: Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load training checkpoint into model and optional optimizer."""
    payload: dict[str, Any] = torch.load(
        path, map_location=device or torch.device("cpu"), weights_only=True
    )
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def load_model(path: str | Path, device: torch.device | None = None) -> GPTModel:
    """Rebuild a GPTModel from a checkpoint (config + weights)."""
    payload: dict[str, Any] = torch.load(
        path, map_location=device or torch.device("cpu"), weights_only=True
    )
    model = GPTModel(GPTConfig(**payload["config"]))
    model.load_state_dict(payload["model_state_dict"])
    return model


def train(
    model: GPTModel,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    max_steps: int = 1000,
    warmup_steps: int = 100,
    learning_rate: float = 6e-4,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    eval_interval: int = 100,
    eval_batches: int = 20,
    checkpoint_dir: str | Path = "runs/minigpt",
    device: torch.device | None = None,
    grad_accum_steps: int = 1,
    bf16: bool = False,
    optim: str = "adamw",
    min_lr_ratio: float = 0.1,
) -> dict[str, Any]:
    """Execute pretraining loop.

    `max_steps` counts optimizer steps; each one accumulates `grad_accum_steps` micro-batches.
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.train()

    optimizer = configure_optimizers(
        model, weight_decay=weight_decay, learning_rate=learning_rate, optim=optim
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "ckpt.pt"

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    saved = False

    data_iter = iter(train_loader)
    initial_val_loss: float | None = None

    if val_loader is not None:
        initial_val_loss = evaluate_loss(model, val_loader, device=device, max_batches=eval_batches)

    for step in range(max_steps):
        # Update learning rate
        lr_mult = get_lr_multiplier(
            step, warmup_steps=warmup_steps, total_steps=max_steps, min_ratio=min_lr_ratio
        )
        current_lr = learning_rate * lr_mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        for _ in range(grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)

            # bf16 keeps fp32's exponent range, so unlike fp16 it needs no GradScaler.
            amp: AbstractContextManager[Any] = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if bf16
                else nullcontext()
            )
            with amp:
                _, micro_loss = model(x, targets=y)
            if micro_loss is None:
                raise RuntimeError("Model forward did not return loss when targets were provided")

            micro_loss = micro_loss / grad_accum_steps
            micro_loss.backward()
            loss += micro_loss.detach()

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        if step % eval_interval == 0 or step == max_steps - 1:
            val_loss = (
                evaluate_loss(model, val_loader, device=device, max_batches=eval_batches)
                if val_loader is not None
                else loss.item()
            )
            step_record = {
                "step": float(step),
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss),
                "lr": float(current_lr),
            }
            history.append(step_record)

            # Selecting on one noisy train batch is meaningless, so only val can pick "best".
            if val_loader is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(best_ckpt_path, model, optimizer, step, val_loss)
                saved = True

    if not saved:  # no val_loader, or val never improved on inf (NaN): keep the final model
        final_loss = history[-1]["val_loss"] if history else float("nan")
        best_val_loss = final_loss
        save_checkpoint(best_ckpt_path, model, optimizer, max_steps - 1, final_loss)

    return {
        "history": history,
        "initial_val_loss": initial_val_loss,
        "final_val_loss": best_val_loss,
        "checkpoint_path": str(best_ckpt_path),
    }

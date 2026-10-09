"""Training loop: AdamW, cosine LR with warmup, grad clipping, eval, checkpointing."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from minigpt.model import GPTConfig, GPTModel


def get_device() -> torch.device:
    """Select best available accelerator in order: mps -> cuda -> cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    learning_rate: float = 6e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """Configure AdamW with decoupled weight decay.

    Applies weight decay to 2D matrix weights (embeddings, linear layers)
    and excludes 1D biases and layer normalizations.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)


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
    """Compute average cross-entropy loss over a dataloader."""
    model.eval()
    total_loss = 0.0
    count = 0

    for i, (x, y) in enumerate(data_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        if loss is not None:
            total_loss += loss.item()
            count += 1

    model.train()
    return total_loss / max(1, count)


def save_checkpoint(
    path: str | Path,
    model: GPTModel,
    optimizer: AdamW | None,
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
    optimizer: AdamW | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load training checkpoint into model and optional optimizer."""
    payload: dict[str, Any] = torch.load(path, map_location=device or torch.device("cpu"))
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def load_model(path: str | Path, device: torch.device | None = None) -> GPTModel:
    """Rebuild a GPTModel from a checkpoint (config + weights)."""
    payload: dict[str, Any] = torch.load(path, map_location=device or torch.device("cpu"))
    model = GPTModel(GPTConfig(**payload["config"]))
    model.load_state_dict(payload["model_state_dict"])
    return model


def train(
    model: GPTModel,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
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
) -> dict[str, Any]:
    """Execute pretraining loop."""
    if device is None:
        device = get_device()

    model.to(device)
    model.train()

    optimizer = configure_optimizers(model, weight_decay=weight_decay, learning_rate=learning_rate)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "ckpt.pt"

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")

    data_iter = iter(train_loader)
    initial_val_loss: float | None = None

    if val_loader is not None:
        initial_val_loss = evaluate_loss(model, val_loader, device=device, max_batches=eval_batches)
        best_val_loss = initial_val_loss

    for step in range(max_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Update learning rate
        lr_mult = get_lr_multiplier(step, warmup_steps=warmup_steps, total_steps=max_steps)
        current_lr = learning_rate * lr_mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        if loss is None:
            raise RuntimeError("Model forward did not return loss when targets were provided")

        loss.backward()

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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(best_ckpt_path, model, optimizer, step, val_loss)

    return {
        "history": history,
        "initial_val_loss": initial_val_loss,
        "final_val_loss": best_val_loss,
        "checkpoint_path": str(best_ckpt_path),
    }

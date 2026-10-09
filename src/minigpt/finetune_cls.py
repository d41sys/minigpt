"""Classifier fine-tuning by head swap on top of a pretrained GPT body."""

from __future__ import annotations

import io
import random
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from minigpt.bpe import BPETokenizer
from minigpt.model import GPTModel
from minigpt.train import get_device

UCI_SPAM_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


class GPTForClassification(nn.Module):
    """GPT body + linear classification head reading the last non-pad token."""

    def __init__(self, gpt_model: GPTModel, num_classes: int = 2, pad_id: int = 0) -> None:
        super().__init__()
        self.gpt = gpt_model
        self.num_classes = num_classes
        self.pad_id = pad_id
        self.cls_head = nn.Linear(gpt_model.config.d_model, num_classes)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.gpt.drop(self.gpt.wte(idx) + self.gpt.wpe(pos))
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.ln_f(x)
        # Right-padded inputs: causal attention means the last real token has seen
        # the whole message and none of the padding.
        last = ((idx != self.pad_id).sum(dim=1) - 1).clamp(min=0)
        pooled = x[torch.arange(x.size(0), device=idx.device), last]
        logits = self.cls_head(pooled)
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss


def freeze_for_finetuning(model: GPTForClassification) -> int:
    """Freeze everything except the last block, final LayerNorm and the head.

    Returns the number of trainable parameters.
    """
    for p in model.parameters():
        p.requires_grad = False
    for module in (model.gpt.blocks[-1], model.gpt.ln_f, model.cls_head):
        for p in module.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SpamTextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Right-padded / truncated token sequences with integer labels."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: BPETokenizer,
        max_length: int = 128,
        pad_id: int = 0,
    ) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for text, label in zip(texts, labels, strict=True):
            tokens = tokenizer.encode(text)[:max_length]
            tokens = tokens + [pad_id] * (max_length - len(tokens))
            self.samples.append(
                (torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long))
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def download_sms_spam(raw_dir: str | Path = "data/raw") -> Path:
    """Download the UCI SMS Spam Collection; raises on network failure."""
    dest_dir = Path(raw_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "SMSSpamCollection"
    if target.exists():
        return target
    req = urllib.request.Request(UCI_SPAM_URL, headers={"User-Agent": "minigpt"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        z.extract("SMSSpamCollection", dest_dir)
    return target


def load_and_balance_dataset(file_path: str | Path, seed: int = 42) -> tuple[list[str], list[int]]:
    """Parse the TSV, undersample ham to match spam 1:1, shuffle. Labels: 0=ham, 1=spam."""
    ham: list[str] = []
    spam: list[str] = []
    with open(file_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            (spam if parts[0].strip().lower() == "spam" else ham).append(parts[1])
    rng = random.Random(seed)
    rng.shuffle(ham)
    n = min(len(ham), len(spam))
    rows = [(t, 0) for t in ham[:n]] + [(t, 1) for t in spam[:n]]
    rng.shuffle(rows)
    return [t for t, _ in rows], [y for _, y in rows]


Split = tuple[list[str], list[int]]


def split_data(
    texts: list[str], labels: list[int], train_ratio: float = 0.7, val_ratio: float = 0.1
) -> tuple[Split, Split, Split]:
    """Contiguous 70/10/20 split (input is already shuffled)."""
    n = len(texts)
    a = int(n * train_ratio)
    b = a + int(n * val_ratio)
    return (texts[:a], labels[:a]), (texts[a:b], labels[a:b]), (texts[b:], labels[b:])


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Accuracy, precision, recall and F1 for the positive (spam) class."""
    model.eval()
    tp = fp = tn = fn = 0
    for x, y in loader:
        logits, _ = model(x.to(device))
        preds = logits.argmax(dim=-1).cpu()
        tp += int(((preds == 1) & (y == 1)).sum())
        fp += int(((preds == 1) & (y == 0)).sum())
        tn += int(((preds == 0) & (y == 0)).sum())
        fn += int(((preds == 0) & (y == 1)).sum())
    total = tp + fp + tn + fn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "accuracy": (tp + tn) / max(1, total),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-8, precision + recall),
        "total": float(total),
    }


def train_classifier(
    model: GPTForClassification,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 3,
    lr: float = 3e-4,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train the unfrozen parameters with AdamW; report val metrics per epoch."""
    device = device or get_device()
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    history: list[dict[str, float]] = []
    val_metrics: dict[str, float] = {}
    for epoch in range(epochs):
        model.train()
        total, batches = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            _, loss = model(x.to(device), targets=y.to(device))
            assert loss is not None
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        val_metrics = evaluate_classifier(model, val_loader, device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": total / max(1, batches),
                "val_accuracy": val_metrics["accuracy"],
            }
        )
    return {"history": history, "final_val_metrics": val_metrics}

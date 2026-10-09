"""Supervised Fine-Tuning (SFT) with Alpaca-format templates and prompt-token loss masking."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from minigpt.bpe import BPETokenizer
from minigpt.model import GPTModel
from minigpt.train import configure_optimizers, get_device

ALPACA_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

ALPACA_TEMPLATE_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

ALPACA_DATA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
)


def format_prompt(instruction: str, input_text: str = "") -> str:
    """Format instruction and optional input into standard Alpaca template."""
    if input_text.strip():
        return ALPACA_TEMPLATE_WITH_INPUT.format(
            instruction=instruction.strip(), input=input_text.strip()
        )
    return ALPACA_TEMPLATE_WITHOUT_INPUT.format(instruction=instruction.strip())


def build_sft_example(
    instruction: str,
    response: str,
    tokenizer: BPETokenizer,
    input_text: str = "",
    max_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize prompt and response with prompt-token loss masking.

    Returns:
        (input_ids, targets): where targets for all prompt tokens are -100.
    """
    prompt = format_prompt(instruction, input_text)
    prompt_ids = tokenizer.encode(prompt, allow_special=True)
    response_ids = tokenizer.encode(response, allow_special=True)

    eos_id = tokenizer.eos_token_id
    full_ids = prompt_ids + response_ids + [eos_id]

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    # Model input: x = full_ids[:-1]
    # Model target: y = full_ids[1:]
    x = full_ids[:-1]
    y = full_ids[1:]

    # Position i of x predicts y[i] = full_ids[i + 1]. Targets with index
    # < len(prompt_ids) - 1 are prompt tokens, so they are masked with -100;
    # the first unmasked target is the first response token.
    targets = list(y)
    mask_len = min(len(prompt_ids) - 1, len(targets))
    for i in range(mask_len):
        targets[i] = -100

    return torch.tensor(x, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class SFTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset of instruction-response pairs with prompt masking and padding."""

    def __init__(
        self,
        records: Sequence[dict[str, str]],
        tokenizer: BPETokenizer,
        max_length: int = 256,
    ) -> None:
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for r in records:
            x, y = build_sft_example(
                instruction=r.get("instruction", ""),
                response=r.get("output", ""),
                input_text=r.get("input", ""),
                tokenizer=tokenizer,
                max_length=max_length,
            )
            self.examples.append((x, y))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]


def sft_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad batch of varying length examples to the maximum length in the batch."""
    max_len = max(len(x) for x, _ in batch)
    padded_x = []
    padded_y = []

    for x, y in batch:
        pad_len = max_len - len(x)
        if pad_len > 0:
            x_pad = torch.cat([x, torch.full((pad_len,), pad_id, dtype=torch.long)])
            y_pad = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])
        else:
            x_pad = x
            y_pad = y
        padded_x.append(x_pad)
        padded_y.append(y_pad)

    return torch.stack(padded_x), torch.stack(padded_y)


def download_alpaca_data(
    dest_path: str | Path = "data/raw/alpaca_1k.json",
    sample_size: int = 1000,
) -> Path:
    """Download Stanford Alpaca and keep the first `sample_size` rows; raises on failure."""
    target = Path(dest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    req = urllib.request.Request(ALPACA_DATA_URL, headers={"User-Agent": "minigpt"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data[:sample_size], f, indent=2)
    return target


def train_sft(
    model: GPTModel,
    dataset: SFTDataset,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train SFT model on instruction-response dataset."""
    if device is None:
        device = get_device()

    model.to(device)
    model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=sft_collate_fn)
    optimizer = configure_optimizers(model, learning_rate=learning_rate, weight_decay=0.01)

    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, targets=y)
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                steps += 1

        avg_loss = total_loss / max(1, steps)
        history.append({"epoch": epoch + 1, "loss": avg_loss})

    return {"history": history, "final_loss": history[-1]["loss"] if history else 0.0}

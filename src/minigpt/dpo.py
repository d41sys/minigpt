"""Direct Preference Optimization (DPO) from scratch.

Computes DPO loss directly from policy and frozen reference model log-probabilities,
bypassing explicit reward modeling and reinforcement learning loops.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from minigpt.bpe import BPETokenizer
from minigpt.model import GPTModel
from minigpt.sft import build_sft_example, format_prompt
from minigpt.train import configure_optimizers, get_device


@dataclass
class PreferenceSample:
    """Pairwise preference sample."""

    prompt: str
    chosen: str
    rejected: str
    input_text: str = ""


def compute_sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute sum of log probabilities for non-masked tokens (labels != -100).

    Args:
        model: Language model forwardable with (B, T) input_ids
        input_ids: Input tokens of shape (B, T)
        labels: Target tokens of shape (B, T), with -100 for masked positions

    Returns:
        Tensor of shape (B,) containing sum of log probabilities per sequence.
    """
    logits, _ = model(input_ids)
    loss_mask = labels != -100

    # Clamp labels so -100 doesn't cause index errors during gather
    labels_clamped = labels.clone()
    labels_clamped[~loss_mask] = 0

    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=2, index=labels_clamped.unsqueeze(2)).squeeze(2)

    # Zero out masked tokens and sum along sequence dimension
    per_token_logps = per_token_logps * loss_mask
    return per_token_logps.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss and implicit reward metrics.

    L_DPO = -E[log sigma(beta * ((log pi_w - log ref_w) - (log pi_l - log ref_l)))]

    Returns:
        (loss, chosen_rewards, rejected_rewards)
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    return loss, chosen_rewards, rejected_rewards


class PreferenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset of paired chosen and rejected token tensors."""

    def __init__(
        self,
        samples: Sequence[PreferenceSample],
        tokenizer: BPETokenizer,
        max_length: int = 256,
    ) -> None:
        self.items: list[dict[str, torch.Tensor]] = []
        for s in samples:
            chosen_x, chosen_y = build_sft_example(
                instruction=s.prompt,
                response=s.chosen,
                tokenizer=tokenizer,
                input_text=s.input_text,
                max_length=max_length,
            )
            rejected_x, rejected_y = build_sft_example(
                instruction=s.prompt,
                response=s.rejected,
                tokenizer=tokenizer,
                input_text=s.input_text,
                max_length=max_length,
            )
            self.items.append(
                {
                    "chosen_input_ids": chosen_x,
                    "chosen_labels": chosen_y,
                    "rejected_input_ids": rejected_x,
                    "rejected_labels": rejected_y,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


def preference_collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Collate preference pairs with sequence padding."""
    keys = ["chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"]
    max_len = max(max(len(b[k]) for k in keys) for b in batch)

    collated: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    for item in batch:
        for k in keys:
            tensor = item[k]
            pad_val = -100 if "labels" in k else pad_id
            pad_len = max_len - len(tensor)
            if pad_len > 0:
                padded = torch.cat([tensor, torch.full((pad_len,), pad_val, dtype=torch.long)])
            else:
                padded = tensor
            collated[k].append(padded)

    return {k: torch.stack(v) for k, v in collated.items()}


def generate_preference_pairs(
    sft_model: GPTModel,
    tokenizer: BPETokenizer,
    alpaca_records: Sequence[dict[str, str]],
    num_pairs: int = 300,
    device: torch.device | None = None,
) -> list[PreferenceSample]:
    """Pair SFT completions (rejected) with dataset reference answers (chosen)."""
    if device is None:
        device = get_device()

    sft_model.eval()
    sft_model.to(device)

    samples: list[PreferenceSample] = []
    for record in alpaca_records[:num_pairs]:
        instruction = record.get("instruction", "")
        input_text = record.get("input", "")
        chosen = record.get("output", "")
        prompt_ids = tokenizer.encode(format_prompt(instruction, input_text), allow_special=True)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            gen_ids = sft_model.generate(
                input_ids,
                max_new_tokens=40,
                temperature=0.8,
                top_k=40,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = [
            t for t in gen_ids[0, len(prompt_ids) :].tolist() if t != tokenizer.eos_token_id
        ]
        rejected = tokenizer.decode(new_tokens).strip()
        if rejected == chosen.strip():
            continue  # identical pair carries no preference signal
        samples.append(
            PreferenceSample(
                prompt=instruction, chosen=chosen, rejected=rejected, input_text=input_text
            )
        )

    return samples


def train_dpo(
    policy_model: GPTModel,
    ref_model: GPTModel,
    dataset: PreferenceDataset,
    beta: float = 0.1,
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 1e-5,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train policy model using DPO against frozen reference model."""
    if device is None:
        device = get_device()

    policy_model.to(device)
    ref_model.to(device)

    # Freeze reference model completely
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Dropout off: otherwise policy != reference at step 0 and the loss starts
    # above log 2 from noise alone. eval() only toggles dropout; grads still flow.
    policy_model.eval()
    optimizer = configure_optimizers(policy_model, learning_rate=lr, weight_decay=0.01)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=preference_collate_fn
    )

    history: list[dict[str, float]] = []
    first_loss: float | None = None
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        seen = 0
        steps = 0
        for batch in loader:
            chosen_x = batch["chosen_input_ids"].to(device)
            chosen_y = batch["chosen_labels"].to(device)
            rejected_x = batch["rejected_input_ids"].to(device)
            rejected_y = batch["rejected_labels"].to(device)

            # Policy model forward
            policy_chosen_logps = compute_sequence_log_probs(policy_model, chosen_x, chosen_y)
            policy_rejected_logps = compute_sequence_log_probs(policy_model, rejected_x, rejected_y)

            # Reference model forward (no grad)
            with torch.no_grad():
                ref_chosen_logps = compute_sequence_log_probs(ref_model, chosen_x, chosen_y)
                ref_rejected_logps = compute_sequence_log_probs(ref_model, rejected_x, rejected_y)

            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=beta,
            )

            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()

            if first_loss is None:
                first_loss = loss.item()
            total_loss += loss.item()
            correct += int((chosen_rewards > rejected_rewards).sum())
            seen += chosen_rewards.numel()
            steps += 1

        history.append(
            {
                "epoch": float(epoch + 1),
                "dpo_loss": total_loss / max(1, steps),
                "reward_accuracy": correct / max(1, seen),
            }
        )

    return {
        "history": history,
        "first_loss": first_loss,
        "final_loss": history[-1]["dpo_loss"] if history else 0.0,
    }

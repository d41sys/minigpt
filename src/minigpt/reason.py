"""Verifiable reasoning tasks and the rollout helpers shared by every RL stage.

The task is multi-digit addition. A completion has the form
`<think>7+5+0=12 4+8+1=13</think><answer>132</answer>`: one step per column, right to left,
with the carry. A program checks the answer, so the reward needs no learned model. This is
"RL with verifiable rewards", the setup behind DeepSeek-R1 and the synthesized, verifiable
tasks of DeepSeek-V4.1's post-training.

Digits are tokenized one byte each (`reasoning_tokenizer`): BPE merges like "12" or "345"
would hide the column structure the model has to learn.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt.bpe import BPETokenizer
from minigpt.model import GPTModel
from minigpt.sft import SFTDataset

ANSWER_RE = re.compile(r"<answer>(\d+)</answer>")
FORMAT_RE = re.compile(r"<think>[^<]*</think><answer>\d+</answer>")


@dataclass(frozen=True)
class Task:
    """One addition problem `a + b`."""

    a: int
    b: int

    @property
    def prompt(self) -> str:
        return f"{self.a}+{self.b}="

    @property
    def answer(self) -> str:
        return str(self.a + self.b)

    def solution(self) -> str:
        """Reference completion: column-wise steps with carries, then the answer."""
        xs, ys = str(self.a)[::-1], str(self.b)[::-1]
        carry, steps = 0, list[str]()
        for i in range(max(len(xs), len(ys))):
            x = int(xs[i]) if i < len(xs) else 0
            y = int(ys[i]) if i < len(ys) else 0
            s = x + y + carry
            steps.append(f"{x}+{y}+{carry}={s}")
            carry = s // 10
        return f"<think>{' '.join(steps)}</think><answer>{self.answer}</answer>"


def make_task(rng: random.Random, digits: int) -> Task:
    """Sample a task whose operands both have exactly `digits` digits (0-9 when digits=1)."""
    lo = 0 if digits == 1 else 10 ** (digits - 1)
    hi = 10**digits - 1
    return Task(rng.randint(lo, hi), rng.randint(lo, hi))


def equation_corpus(rng: random.Random, n: int, max_digits: int) -> str:
    """Pretraining text: bare equations `a+b=c`, one per line, no reasoning traces."""
    lines: list[str] = []
    for _ in range(n):
        t = make_task(rng, rng.randint(1, max_digits))
        lines.append(f"{t.prompt}{t.answer}")
    return "\n".join(lines) + "\n"


def reasoning_tokenizer() -> BPETokenizer:
    """Byte-level tokenizer with no merges: every digit and symbol is its own token."""
    tok = BPETokenizer()
    tok.initialize()
    return tok


def parse_answer(completion: str) -> str | None:
    """The first `<answer>...</answer>` number in a completion, or None."""
    m = ANSWER_RE.search(completion)
    return m.group(1) if m else None


def accuracy_reward(task: Task, completion: str) -> float:
    """1.0 when the parsed answer equals the true sum, else 0.0."""
    return 1.0 if parse_answer(completion) == task.answer else 0.0


def format_reward(completion: str) -> float:
    """1.0 when the whole completion is exactly one think block followed by one answer."""
    return 1.0 if FORMAT_RE.fullmatch(completion.strip()) else 0.0


@torch.no_grad()
def sample_group(
    model: GPTModel,
    prompt_ids: list[int],
    n: int,
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
) -> list[list[int]]:
    """Sample `n` completions of one prompt. Each is cut after its first EOS (kept)."""
    device = next(model.parameters()).device
    x = torch.tensor([prompt_ids] * n, dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens, temperature=temperature, eos_token_id=eos_id)
    completions: list[list[int]] = []
    for row in out[:, len(prompt_ids) :].tolist():
        completions.append(row[: row.index(eos_id) + 1] if eos_id in row else row)
    return completions


def build_batch(
    prompts: list[list[int]], completions: list[list[int]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded `(input_ids, labels)` for prompt+completion pairs.

    `labels[i, t]` is the token predicted at position t; prompt and padding targets are -100,
    so only completion tokens are scored.
    """
    rows = [p + c for p, c in zip(prompts, completions, strict=True)]
    width = max(len(r) for r in rows) - 1
    x = torch.full((len(rows), width), pad_id, dtype=torch.long)
    y = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, (row, p) in enumerate(zip(rows, prompts, strict=True)):
        n = len(row) - 1
        x[i, :n] = torch.tensor(row[:-1])
        y[i, :n] = torch.tensor(row[1:])
        y[i, : len(p) - 1] = -100
    return x, y


def per_token_log_probs(
    model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-prob of every label token, `(B, T)`, and the mask of positions that count."""
    logits, _ = model(input_ids)
    mask = labels != -100
    logps = torch.gather(
        F.log_softmax(logits.float(), dim=-1), 2, labels.clamp(min=0).unsqueeze(2)
    ).squeeze(2)
    return logps * mask, mask


def decode_completion(tok: BPETokenizer, ids: list[int]) -> str:
    """Completion text without the EOS token."""
    return tok.decode([i for i in ids if i != tok.eos_token_id])


def sft_trace_dataset(tasks: list[Task], tok: BPETokenizer) -> SFTDataset:
    """Cold-start SFT data (DeepSeek-R1 style): raw prompt, reference trace + EOS, prompt masked."""
    ds = SFTDataset([], tok)
    for t in tasks:
        x, y = build_batch(
            [tok.encode(t.prompt)], [[*tok.encode(t.solution()), tok.eos_token_id]], pad_id=0
        )
        ds.examples.append((x[0], y[0]))
    return ds


@torch.no_grad()
def evaluate_reasoning(
    model: GPTModel,
    tok: BPETokenizer,
    n_per_digit: int = 50,
    seed: int = 0,
    max_new_tokens: int = 100,
) -> dict[str, Any]:
    """Greedy accuracy, format rate and mean completion tokens (EOS included) on fixed tasks.

    `n_per_digit` tasks for each digit count 1-5, drawn from `Random(seed)`, so every model is
    scored on the same tasks. Returns the overall means plus `per_digit[d]`.
    """
    rng = random.Random(seed)
    device = next(model.parameters()).device
    eos = tok.eos_token_id
    per_digit: dict[int, dict[str, float]] = {}
    for d in range(1, 6):
        tasks = [make_task(rng, d) for _ in range(n_per_digit)]
        # Both operands have d digits, so every prompt of this group has the same length.
        x = torch.tensor([tok.encode(t.prompt) for t in tasks], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens, temperature=0.0, eos_token_id=eos)
        rows = [r[: r.index(eos) + 1] if eos in r else r for r in out[:, x.size(1) :].tolist()]
        texts = [decode_completion(tok, r) for r in rows]
        correct = [accuracy_reward(t, s) for t, s in zip(tasks, texts, strict=True)]
        per_digit[d] = {
            "accuracy": sum(correct) / n_per_digit,
            "format": sum(format_reward(s) for s in texts) / n_per_digit,
            "mean_tokens": sum(len(r) for r in rows) / n_per_digit,
        }
    overall = {k: sum(v[k] for v in per_digit.values()) / len(per_digit) for k in per_digit[1]}
    return {**overall, "per_digit": per_digit}

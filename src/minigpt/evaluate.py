"""Evaluation: perplexity, classification metrics, Judge protocol, ROUGE-L judge."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from minigpt.train import get_device


def compute_perplexity(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | None = None,
    max_batches: int = 50,
) -> float:
    """Compute token perplexity: exp(mean cross-entropy loss)."""
    if device is None:
        device = get_device()

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            # Flatten predictions and ground-truth targets
            v = logits.size(-1)
            targets = y.view(-1)
            active = targets != -100

            if not active.any():
                continue

            loss = nn.functional.cross_entropy(
                logits.view(-1, v),
                targets,
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += active.sum().item()

    if total_tokens == 0:
        return float("nan")

    mean_nll = total_loss / total_tokens
    return math.exp(min(mean_nll, 100.0))


class Judge(Protocol):
    """Protocol for LLM-as-a-judge or automated response evaluation."""

    def judge(self, instruction: str, response: str, reference: str) -> dict[str, float]:
        """Score candidate response against reference."""
        ...


def longest_common_subsequence(seq1: list[str], seq2: list[str]) -> int:
    """Compute length of Longest Common Subsequence between two token sequences."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


class RougeLJudge:
    """Offline default judge computing ROUGE-L token-level LCS precision, recall, and F1."""

    def judge(self, instruction: str, response: str, reference: str) -> dict[str, float]:
        cand_tokens = response.strip().lower().split()
        ref_tokens = reference.strip().lower().split()

        if not cand_tokens or not ref_tokens:
            return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

        lcs = longest_common_subsequence(cand_tokens, ref_tokens)
        prec = lcs / len(cand_tokens)
        rec = lcs / len(ref_tokens)
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        return {
            "rouge_l_precision": prec,
            "rouge_l_recall": rec,
            "rouge_l_f1": f1,
        }


def evaluate_responses(
    judge: Judge,
    samples: Sequence[dict[str, str]],
) -> dict[str, float]:
    """Evaluate candidate responses over a dataset of samples."""
    scores: list[dict[str, float]] = []
    for s in samples:
        sc = judge.judge(
            instruction=s.get("instruction", ""),
            response=s.get("response", ""),
            reference=s.get("reference", ""),
        )
        scores.append(sc)

    if not scores:
        return {}

    keys = scores[0].keys()
    aggregated: dict[str, float] = {}
    for k in keys:
        aggregated[f"mean_{k}"] = sum(s[k] for s in scores) / len(scores)

    return aggregated

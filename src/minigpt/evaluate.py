"""Evaluation: perplexity, EM and token F1, likelihood ranking, Wilson and bootstrap CIs, judges."""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from minigpt.bpe import BPETokenizer
from minigpt.data import hf_rows
from minigpt.dpo import compute_sequence_log_probs
from minigpt.sft import build_sft_example, sft_collate_fn
from minigpt.train import get_device

RANK_DISTRACTOR_OFFSET = 37  # distractor scheme of docs/judge-eval/judge_check.py


@torch.no_grad()
def likelihood_ranking(
    model: nn.Module,
    tokenizer: BPETokenizer,
    rows: list[dict[str, str]],
    device: torch.device,
    max_length: int,
) -> tuple[int, int]:
    """Rows where the model prefers the reference over row (i+37)'s answer for prompt i.

    Returns (wins by summed log-prob, wins by per-token log-prob). The sum favours the shorter
    answer; dividing by response tokens (EOS included) removes that length bias. Ties lose.
    """
    model.eval()
    wins_sum = wins_per_token = 0
    for i, r in enumerate(rows):
        answers = [r["output"], rows[(i + RANK_DISTRACTOR_OFFSET) % len(rows)]["output"]]
        batch = [
            build_sft_example(r["instruction"], a, tokenizer, r.get("input", ""), max_length)
            for a in answers
        ]
        x, y = sft_collate_fn(batch)
        logp = compute_sequence_log_probs(model, x.to(device), y.to(device)).cpu()
        n_tok = (y != -100).sum(dim=1).clamp(min=1)
        wins_sum += int(logp[0] > logp[1])
        wins_per_token += int(logp[0] / n_tok[0] > logp[1] / n_tok[1])
    return wins_sum, wins_per_token


# Multiple-choice benchmarks, zero-shot, scored by summed log-probability of each option given
# the context (the lm-evaluation-harness prompts). Rows come from the Hugging Face datasets
# server: name -> (dataset, config, split).
BENCHMARKS = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "piqa": ("baber/piqa", "default", "validation"),
    "openbookqa": ("allenai/openbookqa", "main", "test"),
    "hellaswag": ("Rowan/hellaswag", "default", "validation"),
}

MCItem = tuple[str, list[str], int]  # (context, options, index of the right option)


def _hellaswag_clean(text: str) -> str:
    """lm-evaluation-harness preprocessing: drop WikiHow "[title]" tags and [bracketed] spans."""
    return re.sub(r"\[.*?\]", "", text.strip().replace(" [title]", ". ")).replace("  ", " ")


def to_mc_item(task: str, row: dict[str, Any]) -> MCItem:
    """One benchmark row as (context, options, label)."""
    if task == "hellaswag":
        ctx = f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}"
        ctx = _hellaswag_clean(ctx)
        return ctx, [_hellaswag_clean(e) for e in row["endings"]], int(row["label"])
    if task == "piqa":
        return f"Question: {row['goal']}\nAnswer:", [row["sol1"], row["sol2"]], int(row["label"])
    options, keys = row["choices"]["text"], row["choices"]["label"]
    ctx = row["question_stem"] if task == "openbookqa" else f"Question: {row['question']}\nAnswer:"
    return ctx, options, keys.index(row["answerKey"])


def load_benchmark(task: str, limit: int = 0, cache_dir: Path | None = None) -> list[MCItem]:
    """The first `limit` rows of the task's test/validation split (0 = all); with `cache_dir`,
    downloaded once to `bench_<task>_<limit>.json` there."""
    cache = cache_dir / f"bench_{task}_{limit}.json" if cache_dir else None
    if cache and cache.exists():
        return [(c, o, label) for c, o, label in json.loads(cache.read_text(encoding="utf-8"))]
    dataset, config, split = BENCHMARKS[task]
    n = limit or 100_000  # hf_rows stops at the end of the split
    items = [to_mc_item(task, r) for r in hf_rows(dataset, config, split, n)]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(items), encoding="utf-8")
    return items


@torch.no_grad()
def score_multiple_choice(
    model: nn.Module,
    tokenizer: BPETokenizer,
    items: list[MCItem],
    device: torch.device,
    max_length: int,
) -> dict[str, Any]:
    """Accuracy of choosing the option with the highest summed log-probability given its context.

    `acc_norm` divides each option's log-probability by its length in characters (the harness's
    acc_norm), which removes the bias towards short options. `stderr` is sqrt(p (1 - p) / n);
    `chance` is the mean of 1 / number of options. Ties go to the first option. Over-long
    sequences keep their last `max_length` input tokens.
    """
    model.eval()
    hits: list[int] = []
    hits_norm: list[int] = []
    for ctx, options, label in items:
        ctx_ids = tokenizer.encode(ctx)
        batch = []
        for option in options:
            cont = tokenizer.encode(" " + option)
            full = (ctx_ids + cont)[-(max_length + 1) :]
            n_cont = min(len(cont), len(full) - 1)
            tgt = [-100] * (len(full) - 1 - n_cont) + full[len(full) - n_cont :]
            batch.append((torch.tensor(full[:-1]), torch.tensor(tgt)))
        x, y = sft_collate_fn(batch)
        logp = compute_sequence_log_probs(model, x.to(device), y.to(device)).cpu()
        hits.append(int(logp.argmax() == label))
        per_char = logp / torch.tensor([float(len(o)) for o in options])
        hits_norm.append(int(per_char.argmax() == label))
    n = len(items)

    def mean_se(h: list[int]) -> tuple[float, float]:
        p = sum(h) / n
        return p, math.sqrt(p * (1 - p) / n)

    acc, stderr = mean_se(hits)
    acc_norm, stderr_norm = mean_se(hits_norm)
    return {
        "n": n,
        "acc": acc,
        "stderr": stderr,
        "acc_norm": acc_norm,
        "stderr_norm": stderr_norm,
        "chance": sum(1 / len(o) for _, o, _ in items) / n,
        "correct": hits,
    }


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
    """LLM-as-a-judge interface: a 1-5 rubric score for one response."""

    def score(self, instruction: str, response: str, reference: str) -> int:
        """Return 1 (unusable) to 5 (matches the reference)."""
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


# --- Exact match, token F1, mock judge, weighted report -----------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation and articles, squash spaces."""
    text = _PUNCT.sub(" ", text.lower())
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if the normalised strings are identical, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Multiset token F1 between normalised prediction and reference."""
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion k / n."""
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def paired_bootstrap_test(
    diff: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float, float]:
    """Mean of paired per-row differences, its 95% percentile CI and a two-sided bootstrap p.

    Rows are resampled i.i.d. with replacement; the same `seed` gives the same row draws for
    every set of differences of equal length. p = 2 * min(P(mean* <= 0), P(mean* >= 0)) with the
    add-one correction, capped at 1, so it is never exactly 0.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_le = (int((means <= 0).sum()) + 1) / (n_boot + 1)
    p_ge = (int((means >= 0).sum()) + 1) / (n_boot + 1)
    return float(diff.mean()), float(lo), float(hi), min(1.0, 2 * min(p_le, p_ge))


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1), in the input order."""
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


class MockJudge:
    """Offline stand-in for an LLM judge: maps ROUGE-L F1 onto a 1-5 rubric score.

    It is deterministic and needs no network, so the eval loop runs anywhere. It is a
    lexical proxy, not a quality judgement; swap in a real judge behind the same method.
    """

    def score(self, instruction: str, response: str, reference: str) -> int:
        """Return 1 (unusable) to 5 (matches the reference)."""
        if not response.strip():
            return 1
        f1 = RougeLJudge().judge(instruction, response, reference)["rouge_l_f1"]
        return 1 + min(4, int(f1 * 5))


JUDGE_RUBRIC = """Grade the RESPONSE to the INSTRUCTION against the REFERENCE answer.

5 = correct and complete; says the same as the reference
4 = correct, minor omissions or extra detail
3 = partly correct or relevant, but incomplete
2 = on topic but mostly wrong
1 = wrong, off topic, repetitive or gibberish

Judge content, not wording or style. Reply with JSON only: {"score": <1-5>}."""

_SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
    "required": ["score"],
}


class OllamaJudge:
    """Reference-guided 1-5 judge on an Ollama server (native `/api/chat`).

    Deterministic as far as Ollama allows: temperature 0, fixed seed, thinking off, and a JSON
    schema constraining the reply. An empty response scores 1 without a call. Any reply that is
    not a score in 1..5 raises instead of being guessed.
    """

    def __init__(
        self, base_url: str, model: str, seed: int = 0, num_ctx: int = 4096, timeout: float = 300
    ) -> None:
        self.url = base_url.rstrip("/") + "/api/chat"
        self.model, self.seed, self.num_ctx, self.timeout = model, seed, num_ctx, timeout

    def score(self, instruction: str, response: str, reference: str) -> int:
        """Return 1 (unusable) to 5 (matches the reference)."""
        if not response.strip():
            return 1
        prompt = (
            f"{JUDGE_RUBRIC}\n\nINSTRUCTION:\n{instruction}\n\n"
            f"REFERENCE:\n{reference}\n\nRESPONSE:\n{response}"
        )
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "format": _SCORE_SCHEMA,
            "options": {"temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx},
        }
        req = urllib.request.Request(
            self.url, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        for attempt in range(3):  # a shared server answers 500 while it (re)loads the model
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    content = json.loads(resp.read())["message"]["content"]
                break
            except urllib.error.HTTPError as err:
                if err.code < 500 or attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        score = json.loads(content).get("score")
        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"judge reply is not a 1-5 score: {content!r}")
        return score


@dataclass(frozen=True)
class EvalReport:
    """Per-model results of the four evals plus a weighted aggregate in [0, 1]."""

    perplexity: float
    vocab_size: int
    exact_match: float
    token_f1: float
    judge_mean: float
    # Reported next to the aggregate, never part of it. Counts of rows (see likelihood_ranking).
    rank_wins_sum: int = 0
    rank_wins_per_token: int = 0
    rank_n: int = 0
    stopped: int = 0
    stopped_n: int = 0

    @property
    def perplexity_score(self) -> float:
        """1 at perplexity 1, 0 at uniform guessing over the vocabulary."""
        if not math.isfinite(self.perplexity):
            return 0.0
        if self.perplexity <= 1.0:
            return 1.0
        return max(0.0, 1.0 - math.log(self.perplexity) / math.log(self.vocab_size))

    def aggregate(self, weights: tuple[float, float, float, float] = (0.25,) * 4) -> float:
        """Weighted mean of (perplexity score, EM, token F1, judge rescaled to [0, 1])."""
        parts = (
            self.perplexity_score,
            self.exact_match,
            self.token_f1,
            (self.judge_mean - 1.0) / 4.0,
        )
        return sum(w * p for w, p in zip(weights, parts, strict=True)) / sum(weights)

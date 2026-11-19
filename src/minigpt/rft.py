"""Offline recipes: rejection-sampling fine-tuning (RFT) and DPO on a model's own samples.

Both start the same way: sample K completions per task from a model, then use the verifier (or
a reward model) to decide what to keep. This is the "rejection sampling" step of Llama 2
(arXiv 2307.09288 section 3.2.3, best of K by reward model, SFT on the winner), of Llama 3
(arXiv 2407.21783 section 4.2, rounds of rejection sampling, SFT and DPO on the model's own
samples) and of DeepSeek-R1 (arXiv 2501.12948 section 2.3.3, SFT on the RL model's correct
samples). Pairs for DPO are (correct, incorrect) samples of the same prompt, as in Tulu 3
(arXiv 2411.15124 section 5, on-policy preference data) with the verifier as the judge.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from minigpt.bpe import BPETokenizer
from minigpt.dpo import PreferenceDataset
from minigpt.finetune_cls import GPTForClassification
from minigpt.model import GPTModel
from minigpt.reason import Task, accuracy_reward, build_batch, decode_completion, format_reward
from minigpt.rlhf import labeler_score, pad_rows, reward_scores, strict_pair
from minigpt.sft import SFTDataset


@dataclass(frozen=True)
class Sampled:
    """K completions of one task; `ids` keep their EOS, `texts` do not."""

    task: Task
    prompt: list[int]
    ids: list[list[int]]
    texts: list[str]


@torch.no_grad()
def sample_tasks(
    model: GPTModel,
    tok: BPETokenizer,
    tasks: list[Task],
    k: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    chunk: int = 512,
) -> list[Sampled]:
    """`k` completions per task, batched across tasks whose prompts have the same length."""
    device = next(model.parameters()).device
    eos = tok.eos_token_id
    prompts = [tok.encode(t.prompt) for t in tasks]
    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)
    ids: list[list[list[int]]] = [[] for _ in tasks]
    per_chunk = max(1, chunk // k)
    for group in by_len.values():
        for s in range(0, len(group), per_chunk):
            part = group[s : s + per_chunk]
            x = torch.tensor([prompts[i] for i in part for _ in range(k)], device=device)
            out = model.generate(x, max_new_tokens, temperature=temperature, eos_token_id=eos)
            rows = out[:, x.size(1) :].tolist()
            for j, i in enumerate(part):
                for r in rows[j * k : (j + 1) * k]:
                    ids[i].append(r[: r.index(eos) + 1] if eos in r else r)
    return [
        Sampled(t, p, c, [decode_completion(tok, r) for r in c])
        for t, p, c in zip(tasks, prompts, ids, strict=True)
    ]


@torch.no_grad()
def rm_scores(
    rm: GPTForClassification, samples: list[Sampled], device: torch.device
) -> list[list[float]]:
    """Reward-model score of every sampled completion, one list per task."""
    out: list[list[float]] = []
    for s in samples:
        rows = [s.prompt + c for c in s.ids]
        out.append(reward_scores(rm, pad_rows(rows, rm.pad_id).to(device)).tolist())
    return out


def select_completions(
    task: Task, texts: list[str], scores: list[float] | None, keep: int
) -> list[int]:
    """Indices of up to `keep` distinct completions to fine-tune on.

    `scores=None` is the verifier: every correct, well-formed sample qualifies, first come
    first kept. With RM `scores` the `keep` highest-scored distinct samples win whether or not
    they are right (Llama 2 best-of-K): what an imperfect reward model costs.
    """
    if scores is None:
        order = [
            i
            for i, t in enumerate(texts)
            if accuracy_reward(task, t) == 1.0 and format_reward(t) == 1.0
        ]
    else:
        order = sorted(range(len(texts)), key=lambda i: -scores[i])
    picked: list[int] = []
    seen: set[str] = set()
    for i in order:
        if texts[i] not in seen and len(picked) < keep:
            seen.add(texts[i])
            picked.append(i)
    return picked


def rft_dataset(
    tok: BPETokenizer, samples: list[Sampled], scores: list[list[float]] | None, keep: int
) -> tuple[SFTDataset, dict[str, float]]:
    """SFT examples (prompt masked, EOS kept) plus selection stats.

    `precision` is the share of kept samples the verifier accepts: 1.0 for verifier selection.
    """
    ds = SFTDataset([], tok)
    right = total = 0
    for n, s in enumerate(samples):
        for i in select_completions(s.task, s.texts, None if scores is None else scores[n], keep):
            x, y = build_batch([s.prompt], [s.ids[i]], pad_id=0)
            ds.examples.append((x[0], y[0]))
            total += 1
            right += labeler_score(s.task, s.texts[i]) == 2.0
    stats = {
        "sample_accuracy": sum(accuracy_reward(s.task, t) for s in samples for t in s.texts)
        / max(1, sum(len(s.texts) for s in samples)),
        "kept": float(total),
        "precision": right / max(1, total),
    }
    return ds, stats


def preference_pairs(tok: BPETokenizer, samples: list[Sampled], seed: int) -> PreferenceDataset:
    """One (chosen, rejected) pair per task that has both a better and a worse sample.

    Chosen has the highest `labeler_score` (right answer, right format), rejected the lowest.
    """
    rng = random.Random(seed)
    ds = PreferenceDataset([], tok)
    for s in samples:
        pick = strict_pair([labeler_score(s.task, t) for t in s.texts], rng)
        if pick is None:
            continue
        cx, cy = build_batch([s.prompt], [s.ids[pick[0]]], pad_id=0)
        rx, ry = build_batch([s.prompt], [s.ids[pick[1]]], pad_id=0)
        ds.items.append(
            {
                "chosen_input_ids": cx[0],
                "chosen_labels": cy[0],
                "rejected_input_ids": rx[0],
                "rejected_labels": ry[0],
            }
        )
    return ds

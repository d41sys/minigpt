"""Process reward model (PRM) vs outcome reward model (ORM), and best-of-N selection.

Toy version of "Let's Verify Step by Step" (Lightman et al. 2023, https://arxiv.org/abs/2305.20050).
A trace `<think>7+5+0=12 4+8+1=13</think><answer>132</answer>` has one step per column plus the
answer, so a program labels every step; the paper's labels came from human annotators (PRM800K).
Following the paper, steps after the first error are not labeled. The ORM sees only whether the
final answer is right. Both are the policy's body plus a scalar head (`GPTForClassification`),
trained on the same samples; the PRM reads its head at the last token of each step, the ORM at
the last token of the sequence. At test time a selector picks one of N samples per task.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from minigpt.bpe import BPETokenizer
from minigpt.finetune_cls import GPTForClassification
from minigpt.model import GPTModel
from minigpt.reason import Task, accuracy_reward, decode_completion, make_task, parse_answer
from minigpt.rft import Sampled, sample_tasks
from minigpt.rlhf import pad_rows, reward_scores
from minigpt.train import configure_optimizers

TRACE_RE = re.compile(r"<think>([^<]*)</think><answer>(\d+)</answer>")
STEP_RE = re.compile(r"[^ ]+")
SELECTORS = ("random", "majority", "orm", "prm_min", "prm_prod", "oracle")


@dataclass(frozen=True)
class Example:
    """One sample: token ids (prompt + completion), step-end positions and labels, outcome."""

    ids: list[int]
    positions: list[int]
    labels: list[int]
    outcome: int


def parse_steps(completion: str) -> list[tuple[int, str]] | None:
    """`(index of last char, text)` of every think step, then of the answer; None if malformed.

    The tokenizer is bytewise, so a character index is a token offset into the completion.
    """
    m = TRACE_RE.fullmatch(completion)
    if m is None:
        return None
    steps = [(m.start(1) + x.end() - 1, x.group()) for x in STEP_RE.finditer(m.group(1))]
    return [*steps, (m.end(2) - 1, m.group(2))]


def step_labels(task: Task, completion: str) -> list[int]:
    """1/0 per step up to and including the first wrong one; [] for a malformed completion.

    A think step is right when it equals the reference step of the same index. The answer step is
    right when every think step was right, none is missing and the answer is the true one.
    """
    steps, ref = parse_steps(completion), parse_steps(task.solution())
    assert ref is not None
    if steps is None:
        return []
    labels: list[int] = []
    for i, (_, text) in enumerate(steps):
        if i == len(steps) - 1:
            ok = len(steps) == len(ref) and text == ref[-1][1]
        else:
            ok = i < len(ref) - 1 and text == ref[i][1]
        labels.append(int(ok))
        if not ok:
            break
    return labels


def build_example(tok: BPETokenizer, s: Sampled, j: int, labeled: bool = True) -> Example:
    """Sample `j` of `s`. With `labeled` the step labels are taken; otherwise only the positions."""
    ids, text, eos = s.ids[j], s.texts[j], tok.eos_token_id
    steps = parse_steps(text) if sum(t != eos for t in ids) == len(text) else None
    labels = step_labels(s.task, text) if labeled and steps is not None else []
    positions = [len(s.prompt) + e for e, _ in steps] if steps is not None else []
    return Example(
        s.prompt + ids,
        positions[: len(labels)] if labeled else positions,
        labels,
        int(accuracy_reward(s.task, text)),
    )


def step_logits(
    rm: GPTForClassification, idx: torch.Tensor, positions: list[list[int]]
) -> list[torch.Tensor]:
    """Head output at the given positions of each row of the right-padded `(B, T)` batch."""
    h = rm.cls_head(rm.gpt.hidden_states(idx)).squeeze(-1)
    return [
        h[i, torch.tensor(p, dtype=torch.long, device=h.device)] for i, p in enumerate(positions)
    ]


def batch_loss(
    rm: GPTForClassification, batch: list[Example], kind: str, device: torch.device
) -> tuple[torch.Tensor, int, int]:
    """Mean BCE over the labels of the batch, the number right and the number of labels."""
    idx = pad_rows([e.ids for e in batch], rm.pad_id).to(device)
    if kind == "orm":
        logits = reward_scores(rm, idx)
        target = torch.tensor([float(e.outcome) for e in batch], device=device)
    else:
        logits = torch.cat(step_logits(rm, idx, [e.positions for e in batch]))
        target = torch.tensor([float(y) for e in batch for y in e.labels], device=device)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    return loss, int(((logits > 0) == (target > 0.5)).sum()), len(target)


def train_verifier(
    rm: GPTForClassification,
    examples: list[Example],
    kind: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float = 0.1,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit `rm` as a `prm` (per-step labels) or an `orm` (outcome); the last `val_frac` is held out.

    The held-out split is by position in `examples`, which are grouped by task.
    """
    if kind not in ("prm", "orm"):
        raise ValueError(f"kind must be prm or orm, got {kind!r}")
    device = device or torch.device("cpu")
    if kind == "prm":
        examples = [e for e in examples if e.labels]
    n_val = max(1, int(len(examples) * val_frac))
    val, train = examples[-n_val:], examples[:-n_val]
    if not train:
        raise ValueError("need at least 2 examples to hold some out")
    rm.to(device)
    rm.eval()  # dropout off, like the reward model; gradients still flow
    optimizer = configure_optimizers(rm, learning_rate=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        order = torch.randperm(len(train), generator=gen).tolist()
        total = right = count = 0.0
        for i in range(0, len(train), batch_size):
            loss, hit, n = batch_loss(
                rm, [train[j] for j in order[i : i + batch_size]], kind, device
            )
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(rm.parameters(), max_norm=1.0)
            optimizer.step()
            total, right, count = total + loss.item() * n, right + hit, count + n
        history.append({"epoch": epoch + 1.0, "loss": total / count, "accuracy": right / count})
        with torch.no_grad():
            vl = va = vn = 0.0
            for i in range(0, len(val), batch_size):
                loss, hit, n = batch_loss(rm, val[i : i + batch_size], kind, device)
                vl, va, vn = vl + loss.item() * n, va + hit, vn + n
        history[-1].update(val_loss=vl / vn, val_accuracy=va / vn)
    return {"history": history, **{k: history[-1][k] for k in ("val_loss", "val_accuracy")}}


@torch.no_grad()
def score_samples(
    prm: GPTForClassification,
    orm: GPTForClassification,
    tok: BPETokenizer,
    samples: list[Sampled],
    device: torch.device,
) -> dict[str, list[list[float]]]:
    """Per task and candidate: `orm` logit, `prm_min` and `prm_prod` (log of the product of step
    probabilities, as in the paper) over all steps; -inf for a malformed completion."""
    out: dict[str, list[list[float]]] = {"orm": [], "prm_min": [], "prm_prod": []}
    prm.eval()
    orm.eval()
    for s in samples:
        ex = [build_example(tok, s, j, labeled=False) for j in range(len(s.ids))]
        idx = pad_rows([e.ids for e in ex], prm.pad_id).to(device)
        out["orm"].append(reward_scores(orm, idx).tolist())
        mins, prods = [], []
        for logits in step_logits(prm, idx, [e.positions for e in ex]):
            ok = len(logits) > 0
            mins.append(logits.min().item() if ok else -math.inf)
            prods.append(F.logsigmoid(logits).sum().item() if ok else -math.inf)
        out["prm_min"].append(mins)
        out["prm_prod"].append(prods)
    return out


def majority_index(texts: list[str]) -> int:
    """First sample holding the most common parsed answer; 0 when none parses."""
    answers = [parse_answer(t) for t in texts]
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return 0
    top = max(counts.values())
    return next(i for i, a in enumerate(answers) if a is not None and counts[a] == top)


def best_of_n(
    samples: list[Sampled], scores: dict[str, list[list[float]]], ns: list[int]
) -> dict[str, dict[int, dict[int, float]]]:
    """`result[selector][n][digits]`: accuracy of the answer each selector picks from the first
    `n` samples. `random` is the exact expectation (share correct among the first `n`);
    `oracle` is pass@n. Ties go to the earliest sample."""
    picks: dict[str, Callable[[int, int], float]] = {}

    def right(i: int, j: int) -> float:
        return accuracy_reward(samples[i].task, samples[i].texts[j])

    picks["random"] = lambda i, n: sum(right(i, j) for j in range(n)) / n
    picks["majority"] = lambda i, n: right(i, majority_index(samples[i].texts[:n]))
    picks["oracle"] = lambda i, n: max(right(i, j) for j in range(n))
    for name, per_task in scores.items():
        picks[name] = lambda i, n, p=per_task: right(i, max(range(n), key=lambda j: p[i][j]))
    digits = sorted({len(str(s.task.a)) for s in samples})
    result: dict[str, dict[int, dict[int, float]]] = {}
    for name, pick in picks.items():
        result[name] = {}
        for n in ns:
            cell = {}
            for d in digits:
                idx = [i for i, s in enumerate(samples) if len(str(s.task.a)) == d]
                cell[d] = sum(pick(i, n) for i in idx) / len(idx)
            result[name][n] = cell
    return result


def save_samples(path: Path, samples: list[Sampled]) -> None:
    rows = [[s.task.a, s.task.b, s.task.op, s.ids] for s in samples]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def load_samples(path: Path, tok: BPETokenizer) -> list[Sampled]:
    out = []
    for a, b, op, ids in json.loads(path.read_text()):
        t = Task(a, b, op)
        out.append(Sampled(t, tok.encode(t.prompt), ids, [decode_completion(tok, c) for c in ids]))
    return out


def cached_samples(
    path: Path,
    model: GPTModel,
    tok: BPETokenizer,
    tasks: list[Task],
    k: int,
    max_new_tokens: int,
    temperature: float,
) -> list[Sampled]:
    """`sample_tasks`, written to `path` on the first call and read back on later ones so that
    every seed and both verifiers see identical samples."""
    if path.exists():
        samples = load_samples(path, tok)
        want = [(t.a, t.b, t.op) for t in tasks]
        if [(s.task.a, s.task.b, s.task.op) for s in samples] != want or any(
            len(s.ids) != k for s in samples
        ):
            raise ValueError(
                f"{path} holds other tasks or another k: delete it or change --samples"
            )
        return samples
    samples = sample_tasks(model, tok, tasks, k, max_new_tokens, temperature)
    save_samples(path, samples)
    return samples


def draw_eval_tasks(rng: random.Random, n_per_digit: int, max_digits: int) -> list[Task]:
    """`n_per_digit` tasks for each digit count 1..`max_digits`."""
    return [make_task(rng, d) for d in range(1, max_digits + 1) for _ in range(n_per_digit)]


def prm_shaped_reward(
    base: Callable[[Task, str], float],
    prm: GPTForClassification,
    tok: BPETokenizer,
    weight: float,
) -> Callable[[Task, str], float]:
    """`base` plus `weight` times the PRM's probability that every step is right (no bonus if
    the completion is malformed)."""
    device = next(prm.parameters()).device

    @torch.no_grad()
    def reward(task: Task, completion: str) -> float:
        steps = parse_steps(completion)
        if steps is None:
            return base(task, completion)
        prompt = tok.encode(task.prompt)
        idx = torch.tensor([[*prompt, *tok.encode(completion), tok.eos_token_id]], device=device)
        (logits,) = step_logits(prm, idx, [[len(prompt) + e for e, _ in steps]])
        return base(task, completion) + weight * F.logsigmoid(logits).sum().exp().item()

    return reward

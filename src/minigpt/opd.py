"""Full-vocabulary on-policy distillation (OPD): the last post-training stage here.

Public source: DeepSeek-V4.1 tech report, section 5.1 ("SFT followed by RL and on-policy
distillation (OPD)") and section 5.2.4, which ends post-training with full-vocabulary OPD from
many teachers. Related: Gu et al. 2024 (MiniLLM, reverse KL for distillation) and Thinking
Machines 2025, "On-policy distillation". This module is a small copy with one teacher per
operator (addition, subtraction); the specifics below (arithmetic tasks, loss details) are this
repo's choices. With several teachers each prompt is scored against the teacher of its operator
and the loss is the token-weighted mean of the per-teacher KLs, like V4.1's merge of per-domain
specialists into one student.

Each step the *student* samples completions, then at every completion position the student's
whole next-token distribution is pulled toward the frozen teacher's under the same prefix:

    reverse KL:  sum_v p_s(v) (log p_s(v) - log p_t(v))     KL(student || teacher)
    forward KL:  sum_v p_t(v) (log p_t(v) - log p_s(v))     KL(teacher || student)

Reverse KL is mode-seeking: the student is punished for putting mass where the teacher puts
little, so a small student commits to one teacher mode. Forward KL is mode-covering: it is
punished for missing any teacher mode, so a student that cannot fit them all spreads mass over
the gaps. The gradient flows through the logits at each sampled prefix only, not through the
sampling of the prefix itself.

`off_policy=True` is the baseline: the *teacher* samples and the student takes plain SFT steps
on those samples (sequence-level distillation). The student then only ever sees teacher
prefixes. At inference it feeds on its own, slightly different, prefixes and has never been
corrected there (exposure bias). On-policy training queries the teacher on exactly the
prefixes the student produces.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt.model import GPTModel
from minigpt.reason import (
    Task,
    accuracy_reward,
    build_batch,
    per_token_log_probs,
    reasoning_tokenizer,
    sample_group,
    sample_task,
)
from minigpt.train import configure_optimizers, get_device

KLMode = Literal["reverse", "forward"]


def token_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    mode: KLMode = "reverse",
) -> torch.Tensor:
    """Mean over masked-in positions of the full-vocab KL between next-token distributions.

    Logits are `(B, T, V)`, `mask` is `(B, T)` bool. Teacher logits take no part in the graph's
    gradient; positions with `mask` False contribute nothing, whatever their logits hold.
    """
    log_s = F.log_softmax(student_logits.float(), dim=-1)
    log_t = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    if mode == "reverse":
        per_pos = (log_s.exp() * (log_s - log_t)).sum(-1)
    elif mode == "forward":
        per_pos = (log_t.exp() * (log_t - log_s)).sum(-1)
    else:
        raise ValueError(f"kl mode must be 'reverse' or 'forward', got {mode!r}")
    return torch.where(mask, per_pos, torch.zeros_like(per_pos)).sum() / mask.sum().clamp(min=1)


def opd_loss(
    student: nn.Module,
    teacher: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    mode: KLMode = "reverse",
) -> torch.Tensor:
    """KL(student, teacher) per completion token; `labels == -100` (prompt, padding) is skipped."""
    student_logits, _ = student(input_ids)
    with torch.no_grad():
        teacher_logits, _ = teacher(input_ids)
    return token_kl(student_logits, teacher_logits, labels != -100, mode)


def routed_opd_loss(
    student: nn.Module,
    teachers: Mapping[str, nn.Module],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    row_ops: Sequence[str],
    mode: KLMode = "reverse",
) -> tuple[torch.Tensor, dict[str, float]]:
    """KL against each row's own teacher, `teachers[row_ops[i]]` for row `i`.

    The loss is the token-weighted mean over teachers: a teacher's weight is its share of the
    batch's completion tokens (so it equals the plain per-token mean over the whole batch).
    Also returns each op's mean KL per token, detached.
    """
    n_total = (labels != -100).sum().clamp(min=1)
    parts: list[torch.Tensor] = []
    by_op: dict[str, float] = {}
    for op in dict.fromkeys(row_ops):
        idx = torch.tensor([i for i, o in enumerate(row_ops) if o == op], device=input_ids.device)
        y = labels[idx]
        op_kl = opd_loss(student, teachers[op], input_ids[idx], y, mode)
        parts.append(op_kl * (y != -100).sum() / n_total)
        by_op[op] = op_kl.item()
    return torch.stack(parts).sum(), by_op


def _rollouts(
    samplers: Mapping[str, GPTModel],
    tasks: list[Task],
    group: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], list[str]]:
    """`samplers[task.op]` completes each task `group` times.

    Returns `(input_ids, labels, completions, row_ops)`; `row_ops[i]` is the op of row `i`.
    """
    tok = reasoning_tokenizer()
    pad_id = tok.special_to_id["<|pad|>"]
    prompts: list[list[int]] = []
    completions: list[list[int]] = []
    row_ops: list[str] = []
    for t in tasks:
        ids = tok.encode(t.prompt)
        for c in sample_group(
            samplers[t.op], ids, group, max_new_tokens, tok.eos_token_id, temperature
        ):
            prompts.append(ids)
            completions.append(c)
            row_ops.append(t.op)
    x, y = build_batch(prompts, completions, pad_id)
    return x.to(device), y.to(device), completions, row_ops


@torch.no_grad()
def evaluate_student(
    student: GPTModel,
    teachers: Mapping[str, GPTModel],
    tasks: list[Task],
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    """Greedy student rollouts: accuracy, mean completion length, and reverse KL per token.

    Each task is scored against its op's teacher. The row also holds `accuracy[op]` and `kl[op]`
    for every op among `tasks`.
    """
    x, y, completions, row_ops = _rollouts(
        {op: student for op in teachers}, tasks, 1, max_new_tokens, 0.0, device
    )
    tok = reasoning_tokenizer()
    eos = tok.eos_token_id
    hits = [
        accuracy_reward(t, tok.decode([i for i in c if i != eos]))
        for t, c in zip(tasks, completions, strict=True)
    ]
    kl, kl_by_op = routed_opd_loss(student, teachers, x, y, row_ops, "reverse")
    row = {"accuracy": sum(hits) / len(tasks), "mean_len": sum(map(len, completions)) / len(tasks)}
    row["kl"] = kl.item()
    for op, op_kl in kl_by_op.items():
        op_hits = [h for h, o in zip(hits, row_ops, strict=True) if o == op]
        row[f"accuracy[{op}]"] = sum(op_hits) / len(op_hits)
        row[f"kl[{op}]"] = op_kl
    return row


def train_opd(
    student: GPTModel,
    teachers: GPTModel | Mapping[str, GPTModel],
    steps: int = 100,
    batch_size: int = 8,
    group: int = 1,
    digits: int = 3,
    max_new_tokens: int = 64,
    lr: float = 3e-4,
    kl: KLMode = "reverse",
    off_policy: bool = False,
    temperature: float = 1.0,
    eval_every: int = 25,
    eval_tasks: int = 64,
    seed: int = 0,
    device: torch.device | None = None,
    ops: Sequence[str] = "+",
) -> dict[str, Any]:
    """Distil `teachers` into `student` on prompts of 1..`digits` digits, one op drawn from `ops`.

    `teachers` maps op -> frozen model (a single model serves every op). Each prompt is scored
    against its op's teacher and the loss is the token-weighted mean of the per-teacher KLs.
    On-policy (default): the student samples at `temperature`, the loss is `kl` against the
    teacher on those samples. `off_policy`: each prompt is sampled from its op's teacher at
    `temperature` (0 = greedy) and the student minimises cross-entropy on them; `kl` is then
    unused for training. Every `eval_every` steps (and at step 0) both modes are scored the same
    way, on a fixed seeded task set with greedy student rollouts, so their reverse-KL curves are
    comparable.
    """
    if isinstance(teachers, GPTModel):
        teachers = {op: teachers for op in ops}
    if missing := [op for op in ops if op not in teachers]:
        raise ValueError(f"no teacher for op(s) {missing}")
    routes = {op: teachers[op] for op in ops}
    frozen = list({id(m): m for m in routes.values()}.values())
    if any(student.config.vocab_size != m.config.vocab_size for m in frozen):
        raise ValueError("student and teacher must share one vocabulary")
    longest = 2 * digits + 2 + max_new_tokens  # prompt "999+999=" plus the completion
    if longest > min(m.config.context_length for m in [student, *frozen]):
        raise ValueError(f"prompt + max_new_tokens = {longest} exceeds a model's context length")
    if device is None:
        device = get_device()

    student.to(device)
    for m in frozen:
        m.to(device).eval()
        m.requires_grad_(False)
    student.eval()  # dropout off, so sampling and scoring see the same distribution
    optimizer = configure_optimizers(student, learning_rate=lr, weight_decay=0.01)

    rng = random.Random(seed)
    eval_rng = random.Random(seed + 10_000)
    held_out = [sample_task(eval_rng, eval_rng.randint(1, digits), ops) for _ in range(eval_tasks)]
    samplers = routes if off_policy else {op: student for op in ops}

    history: list[dict[str, float]] = []
    run_loss, run_n = 0.0, 0
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            row = evaluate_student(student, routes, held_out, max_new_tokens, device)
            history.append({"step": float(step), "loss": run_loss / max(1, run_n), **row})
            run_loss, run_n = 0.0, 0
        if step == steps:
            break

        tasks = [sample_task(rng, rng.randint(1, digits), ops) for _ in range(batch_size)]
        x, y, _, row_ops = _rollouts(samplers, tasks, group, max_new_tokens, temperature, device)
        if off_policy:
            logps, mask = per_token_log_probs(student, x, y)
            loss = -logps.sum() / mask.sum().clamp(min=1)
        else:
            loss, _ = routed_opd_loss(student, routes, x, y, row_ops, kl)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        run_loss += loss.item()
        run_n += 1

    return {"history": history, "first_kl": history[0]["kl"], "final_kl": history[-1]["kl"]}

"""Group Relative Policy Optimization (GRPO) with verifiable rewards, from scratch.

Mirrors the RL stage that DeepSeek-V4.1 (tech report, section 5) describes: SFT, then RL with
GRPO groups on synthesized verifiable tasks, with no algorithmic novelty over earlier GRPO work.
Here the tasks are multi-digit additions (`minigpt.reason`) and the grader is a program.

Per step: sample tasks, draw a group of completions for each, score them, turn each group's
rewards into advantages `(r - mean) / (std + eps)`, then take PPO-clipped policy-gradient steps
with a k3 KL penalty to the frozen SFT reference. Two choices follow the sources:

- Difficulty filtering: a group whose rewards are all equal has zero advantage and teaches
  nothing, so it is dropped (as in DAPO). Tasks come from a digit range so groups are mixed.
  `fill_batch` keeps sampling fresh tasks until `tasks_per_step` groups survive (DAPO dynamic
  sampling), the online form of DeepSeek-V4.1's difficulty calibration.
- `unbiased_kl` multiplies the k3 term by the ratio pi/pi_old, DeepSeek-V3.2's unbiased KL
  estimate (eq. 7). At `mu=1` the ratio is 1 in value but not in gradient.
- Length bias (DeepSeek-V4.1 section 5.2.2: short samples dominate early batches). `seq-mean`
  divides each sequence's loss by its own length, so a token in a short sequence weighs more
  than one in a long sequence. `token-sum` divides by one constant (the max completion length,
  Dr. GRPO), so every token weighs the same. `std_norm=False` is the other Dr. GRPO fix: no
  per-group std division, which over-weights groups with tiny reward spread.
- Controllable effort (DeepSeek-V4.1 section 5.1.4, `efforts`): every task is sampled once per
  effort level b with `e{b}:` in the prompt. Each (task, b) is its own group, so advantages are
  centred within it and levels are never compared; a length penalty on the reasoning tokens,
  whose coefficient k(b) decays with b, is what makes low effort think less.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Sequence
from typing import Any

import torch

from minigpt.bpe import BPETokenizer
from minigpt.model import GPTModel
from minigpt.reason import (
    Task,
    accuracy_reward,
    build_batch,
    decode_completion,
    effort_prefix,
    format_reward,
    per_token_log_probs,
    sample_group,
    sample_task,
    think_tokens,
)
from minigpt.train import configure_optimizers, get_device

RewardFn = Callable[[Task, str], float]
Samples = list[tuple[Task, str]]

# Penalty scale: a 3-digit trace is about 26 tokens, so one such trace at k=1 costs ~0.9.
L_NORM = 30.0
# Cap on the deduction, so a long trace never outweighs a correct answer (reward 1.1).
C_MAX = 1.0
# tau = LAMBDA * mean level spacing; a small tau makes the top level nearly penalty-free.
LAMBDA = 0.25


def effort_coef(effort: int, efforts: tuple[int, ...], k0: float) -> float:
    """k(b) = k0 * exp(-(b - b_min) / tau), tau = LAMBDA * mean spacing of the training levels."""
    levels = sorted(set(efforts))
    spacing = (levels[-1] - levels[0]) / (len(levels) - 1) if len(levels) > 1 else 1.0
    return k0 * math.exp(-(effort - levels[0]) / (LAMBDA * spacing))


def length_penalty(completion: str, k: float) -> float:
    """r_len = -min(C_MAX, k * think_tokens / L_NORM)."""
    return -min(C_MAX, k * think_tokens(completion) / L_NORM)


def default_reward(task: Task, completion: str) -> float:
    """Accuracy plus a small format bonus."""
    return accuracy_reward(task, completion) + 0.1 * format_reward(completion)


def group_advantages(
    rewards: torch.Tensor, std_norm: bool = True, eps: float = 1e-4, mode: str = "group"
) -> torch.Tensor:
    """Advantages of `(groups, G)` rewards, centred per group.

    `group` divides by the group std if `std_norm`. `batch` divides by the std over the whole
    batch instead (Magistral section 2.1: group-centred, then normalised per minibatch).
    `rloo` is the leave-one-out baseline `r_i - mean(r_-i) = G/(G-1) * (r_i - mean)`, no std.
    """
    adv = rewards - rewards.mean(dim=1, keepdim=True)
    if mode == "batch":
        return adv / (adv.std(correction=0) + eps)
    if mode == "rloo":
        g = rewards.shape[1]
        return adv * g / max(g - 1, 1)
    if mode != "group":
        raise ValueError(f"advantage must be 'group', 'batch' or 'rloo', got {mode!r}")
    if std_norm:
        adv = adv / (rewards.std(dim=1, correction=0, keepdim=True) + eps)
    return adv


def kl_k3(logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    """k3 estimator of KL(policy || ref) from tokens sampled by the policy.

    With r = ref/policy it is `r - log r - 1`: non-negative, zero when the log-probs are equal.
    """
    d = ref_logps - logps
    return torch.exp(d) - d - 1


def grpo_loss(
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 0.04,
    loss_agg: str = "seq-mean",
    max_len: int = 1,
    unbiased_kl: bool = False,
    policy_loss: str = "ppo",
    clip_eps_high: float | None = None,
    group_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clipped policy-gradient loss and the mean per-token k3 KL (detached).

    All log-prob tensors are `(B, T)`; `mask` marks completion tokens; `advantages` is `(B,)`.
    `policy_loss` picks the surrogate (all clip the ratio to `[1 - clip_eps, 1 + clip_eps_high]`,
    `clip_eps_high` defaulting to `clip_eps`; a larger one is DAPO's Clip-Higher):
    - `ppo`: GRPO's per-token `min(r A, clip(r) A)`.
    - `gspo`: Qwen's GSPO (arXiv 2507.18071, eq. 5 and 7): one ratio per sequence,
      `s = exp(mean_t log pi/pi_old)`, clipped per sequence; the paper's eps are ~3e-4/4e-4.
    - `cispo`: MiniMax-M1's CISPO (arXiv 2506.13585, eq. 4-5): `sg(clip(r)) A log pi`, so a
      clipped token still passes gradient (scaled by the clipped weight) instead of being dropped.
    `loss_agg`: `seq-mean` (per-sequence token mean), `token-sum` (constant `max_len`),
    `token-mean` (sum over every token / total tokens in the batch, DAPO eq. 8) or `group-mean`
    (per group of `group_size` rows: sum / the group's total length, Magistral section 2.1).
    """
    adv = advantages.unsqueeze(1)
    ratio = torch.exp(logps - old_logps)
    high = clip_eps if clip_eps_high is None else clip_eps_high
    if policy_loss == "ppo":
        surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - clip_eps, 1 + high) * adv)
    elif policy_loss == "gspo":
        s = torch.exp(((logps - old_logps) * mask).sum(1) / mask.sum(1).clamp(min=1)).unsqueeze(1)
        surrogate = torch.minimum(s * adv, s.clamp(1 - clip_eps, 1 + high) * adv)
    elif policy_loss == "cispo":
        surrogate = ratio.detach().clamp(1 - clip_eps, 1 + high) * adv * logps
    else:
        raise ValueError(f"policy_loss must be 'ppo', 'gspo' or 'cispo', got {policy_loss!r}")
    kl = kl_k3(logps, ref_logps)
    if unbiased_kl:
        kl = ratio * kl
    per_token = -(surrogate - beta * kl) * mask
    if loss_agg == "seq-mean":
        loss = (per_token.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
    elif loss_agg == "token-sum":
        loss = (per_token.sum(dim=1) / max_len).mean()
    elif loss_agg == "token-mean":
        loss = per_token.sum() / mask.sum().clamp(min=1)
    elif loss_agg == "group-mean":
        by_group = per_token.sum(dim=1).reshape(-1, group_size).sum(dim=1)
        loss = (by_group / mask.sum(dim=1).reshape(-1, group_size).sum(dim=1).clamp(min=1)).mean()
    else:
        raise ValueError(
            f"loss_agg must be 'seq-mean', 'token-sum', 'token-mean' or 'group-mean', "
            f"got {loss_agg!r}"
        )
    return loss, ((kl * mask).sum() / mask.sum().clamp(min=1)).detach()


def grpo_step(
    policy: GPTModel,
    ref: GPTModel,
    optimizer: torch.optim.Optimizer,
    prompts: list[list[int]],
    completions: list[list[int]],
    rewards: torch.Tensor,
    pad_id: int,
    clip_eps: float = 0.2,
    beta: float = 0.04,
    mu: int = 1,
    std_norm: bool = True,
    loss_agg: str = "seq-mean",
    max_len: int = 80,
    unbiased_kl: bool = False,
    policy_loss: str = "ppo",
    clip_eps_high: float | None = None,
    advantage: str = "group",
) -> float:
    """`mu` optimizer steps on one batch; returns the mean KL to the reference of the last one.

    `prompts` and `completions` are flat, group-major; `rewards` is `(groups, G)`. The old
    log-probs are taken before the first inner step, so the ratio starts at exactly 1.
    """
    device = next(policy.parameters()).device
    adv = group_advantages(rewards, std_norm, mode=advantage).reshape(-1).to(device)
    x, y = build_batch(prompts, completions, pad_id)
    x, y = x.to(device), y.to(device)
    policy.eval()  # dropout off, so the policy at step 0 is exactly the reference
    with torch.no_grad():
        old_logps, _ = per_token_log_probs(policy, x, y)
        ref_logps, _ = per_token_log_probs(ref, x, y)
    kl = 0.0
    for _ in range(mu):
        logps, mask = per_token_log_probs(policy, x, y)
        loss, kl_mean = grpo_loss(
            logps,
            old_logps,
            ref_logps,
            mask,
            adv,
            clip_eps,
            beta,
            loss_agg,
            max_len,
            unbiased_kl,
            policy_loss,
            clip_eps_high,
            rewards.shape[1],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        kl = kl_mean.item()
    return kl


def train_grpo(
    policy: GPTModel,
    ref: GPTModel,
    tok: BPETokenizer,
    steps: int = 200,
    tasks_per_step: int = 8,
    group_size: int = 8,
    digits: tuple[int, int] = (1, 3),
    max_new_tokens: int = 80,
    temperature: float = 1.0,
    lr: float = 5e-5,
    beta: float = 0.04,
    clip_eps: float = 0.2,
    mu: int = 1,
    std_norm: bool = True,
    loss_agg: str = "seq-mean",
    unbiased_kl: bool = False,
    fill_batch: bool = False,
    max_rounds: int = 4,
    efforts: tuple[int, ...] = (),
    effort_k0: float = 0.5,
    shape: GroupShaper | None = None,
    reward_fn: RewardFn = default_reward,
    ops: Sequence[str] = "+",
    observe: Callable[[Samples], dict[str, float]] | None = None,
    on_step: Callable[[dict[str, float]], None] | None = None,
    seed: int = 0,
    device: torch.device | None = None,
    policy_loss: str = "ppo",
    clip_eps_high: float | None = None,
    advantage: str = "group",
    ref_reset_every: int = 0,
    optim: str = "adamw",
) -> dict[str, Any]:
    """Run GRPO; `ref` stays frozen. Returns `{"history": [per-step dict]}`.

    Per step the record holds `reward` (mean of `reward_fn`), `accuracy` (mean of the true
    `accuracy_reward`, whatever `reward_fn` is), `format`, `length` (mean completion tokens),
    `kl` (0 on a skipped step) and `drop_rate` (share of sampled groups dropped as all-equal).
    With `fill_batch`, up to `max_rounds` rounds of `tasks_per_step` tasks are sampled until
    `tasks_per_step` groups survive; `tasks_sampled` counts them. Stats cover
    every sample, dropped groups included. `observe` sees all `(task, completion)` pairs of the
    step and its returned dict is merged into the record.

    With `efforts`, each task yields one group per level, `reward` includes the length penalty
    `length_penalty(., effort_coef(b, efforts, effort_k0))`, `tasks_per_step` still counts tasks
    (so `tasks_per_step * len(efforts)` groups), and the record adds `think_e{b}` (mean think
    tokens) and `acc_e{b}` per level.

    `shape` is a group-level reward hook (`make_shaper`: Kimi K1.5 length reward, DAPO overlong
    penalty, Kimi K3 budget override). It sees each group's completions and rewards, after the
    effort penalty, and returns the rewards the advantages are computed from.

    `policy_loss`, `clip_eps_high` and `advantage` select the loss variants of `grpo_loss` and
    `group_advantages`. With `ref_reset_every=N` (ProRL, arXiv 2505.24864 section 2.3.1), every N
    steps the reference becomes a snapshot of the current policy and the optimizer state is
    re-initialised; `ref` is copied first, so the caller's stays at the start model, and `kl`
    is then measured against the latest snapshot.
    """
    levels: tuple[int | None, ...] = efforts or (None,)
    prefix_len = max(len(effort_prefix(b)) for b in levels)
    if 2 * digits[1] + 2 + prefix_len + max_new_tokens > policy.config.context_length:
        raise ValueError("longest prompt + max_new_tokens exceeds the context length")
    ks = {b: effort_coef(b, efforts, effort_k0) for b in efforts}
    device = device or get_device()
    policy.to(device)
    ref.to(device).eval()
    ref.requires_grad_(False)
    if ref_reset_every:
        ref = copy.deepcopy(ref)
    optimizer = configure_optimizers(policy, weight_decay=0.0, learning_rate=lr, optim=optim)
    rng = random.Random(seed)
    eos, pad = tok.eos_token_id, tok.special_to_id["<|pad|>"]
    history: list[dict[str, float]] = []
    for step in range(steps):
        if ref_reset_every and step and step % ref_reset_every == 0:
            ref.load_state_dict(policy.state_dict())
            optimizer = configure_optimizers(
                policy, weight_decay=0.0, learning_rate=lr, optim=optim
            )
        entries: list[tuple[Task, int | None]] = []  # one per group: (task, effort)
        prompts: list[list[int]] = []
        groups: list[list[list[int]]] = []
        texts: list[list[str]] = []
        for _ in range(max_rounds if fill_batch else 1):
            new_tasks = [sample_task(rng, rng.randint(*digits), ops) for _ in range(tasks_per_step)]
            new = [(t, b) for t in new_tasks for b in levels]
            new_prompts = [tok.encode(effort_prefix(b) + t.prompt) for t, b in new]
            new_groups = [
                sample_group(policy, p, group_size, max_new_tokens, eos, temperature)
                for p in new_prompts
            ]
            entries += new
            prompts += new_prompts
            groups += new_groups
            texts += [[decode_completion(tok, c) for c in g] for g in new_groups]
            rows: list[list[float]] = []
            for (t, b), g in zip(entries, texts, strict=True):
                row = [reward_fn(t, s) for s in g]
                if b is not None:
                    row = [r + length_penalty(s, ks[b]) for r, s in zip(row, g, strict=True)]
                rows.append(shape(step, t, b, g, row) if shape else row)
            rewards = torch.tensor(rows)
            # All-equal rewards give zero advantage: nothing to learn, so drop the group.
            spread = rewards.max(dim=1).values > rewards.min(dim=1).values
            kept = spread.nonzero().flatten().tolist()
            if len(kept) >= tasks_per_step * len(levels):
                break
        kept = kept[: tasks_per_step * len(levels)]
        kl = 0.0
        if kept:
            kl = grpo_step(
                policy,
                ref,
                optimizer,
                [prompts[i] for i in kept for _ in range(group_size)],
                [c for i in kept for c in groups[i]],
                rewards[kept],
                pad,
                clip_eps,
                beta,
                mu,
                std_norm,
                loss_agg,
                max_new_tokens,
                unbiased_kl,
                policy_loss,
                clip_eps_high,
                advantage,
            )
        samples: Samples = [(t, s) for (t, _), g in zip(entries, texts, strict=True) for s in g]
        n = len(samples)
        record = {
            "step": float(step),
            "reward": rewards.mean().item(),
            "accuracy": sum(accuracy_reward(t, s) for t, s in samples) / n,
            "format": sum(format_reward(s) for _, s in samples) / n,
            "length": sum(len(c) for g in groups for c in g) / n,
            "kl": kl,
            "drop_rate": 1 - spread.float().mean().item(),
            "tasks_sampled": float(len(entries) // len(levels)),
        }
        for b in efforts:
            of_b = [(t, s) for (t, e), g in zip(entries, texts, strict=True) for s in g if e == b]
            record[f"think_e{b}"] = sum(think_tokens(s) for _, s in of_b) / len(of_b)
            record[f"acc_e{b}"] = sum(accuracy_reward(t, s) for t, s in of_b) / len(of_b)
        if observe is not None:
            record.update(observe(samples))
        history.append(record)
        if on_step is not None:
            on_step(record)
    return {"history": history}


# --- Length-control reward shaping (Kimi K1.5, Kimi K3, DAPO). Off unless a shaper is passed. ---

# (step, task, effort level or None, completions of one group, their rewards) -> new rewards
GroupShaper = Callable[[int, Task, int | None, list[str], list[float]], list[float]]
Budgets = dict[tuple[int | None, int], float]  # (effort, digits) -> b0, mean think tokens


def kimi_length_rewards(correct: Sequence[bool], lens: Sequence[int]) -> list[float]:
    """Kimi K1.5 section 2.3.3: lambda = 0.5 - (len - min) / (max - min); correct get lambda,
    wrong get min(0, lambda); all zero when the group has one length."""
    lo, hi = min(lens), max(lens)
    if hi == lo:
        return [0.0] * len(lens)
    lams = [0.5 - (n - lo) / (hi - lo) for n in lens]
    return [lam if ok else min(0.0, lam) for lam, ok in zip(lams, correct, strict=True)]


def overlong_penalty(length: int, l_max: int, l_cache: int) -> float:
    """DAPO section 3.4, eq. 13: 0 up to l_max - l_cache, linear down to -1 at l_max, -1 beyond."""
    if length <= l_max - l_cache:
        return 0.0
    return max(-1.0, (l_max - l_cache - length) / l_cache)


def budget_override(rewards: Sequence[float], lens: Sequence[int], limit: float) -> list[float]:
    """Kimi K3 reasoning-effort RL: reward -1 for every trajectory longer than `limit`, correct
    or not."""
    return [-1.0 if n > limit else r for r, n in zip(rewards, lens, strict=True)]


def budget_tau(step: int, steps: int, taus: Sequence[float]) -> float:
    """Stage-wise multiplier: `steps` split into len(taus) equal stages, one tau each."""
    return taus[min(len(taus) - 1, step * len(taus) // max(1, steps))]


@torch.no_grad()
def estimate_budgets(
    model: GPTModel,
    tok: BPETokenizer,
    digits: tuple[int, int],
    efforts: tuple[int, ...],
    ops: Sequence[str],
    n_tasks: int,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> Budgets:
    """b0 per (effort, digit count) = mean think tokens of the cold-start `model`'s samples.

    Kimi K3 has one budget per problem; one per difficulty bucket is the cheap reading here
    (a per-problem budget would double the rollouts).
    """
    rng, eos = random.Random(seed), tok.eos_token_id
    out: Budgets = {}
    for b in efforts or (None,):
        for d in range(digits[0], digits[1] + 1):
            lens = []
            for _ in range(n_tasks):
                prompt = tok.encode(effort_prefix(b) + sample_task(rng, d, ops).prompt)
                for c in sample_group(model, prompt, group_size, max_new_tokens, eos, temperature):
                    lens.append(think_tokens(decode_completion(tok, c)))
            out[b, d] = sum(lens) / len(lens)
    return out


def make_shaper(
    steps: int,
    max_new_tokens: int,
    budgets: Budgets | None = None,
    taus: Sequence[float] = (1.0,),
    length_weight: float = 0.0,
    length_warmup: int = 0,
    overlong_cache: int = 0,
) -> GroupShaper:
    """Compose the three shapings (each off at its default) into one group-level reward hook.

    Order: K1.5 length reward (weight `length_weight`, from step `length_warmup` on) and DAPO
    soft overlong penalty (`overlong_cache` > 0, l_max = `max_new_tokens`) are added; the K3
    budget override (`budgets` given, limit tau(step) * b0) then wins.
    """

    def shape(
        step: int, task: Task, effort: int | None, texts: list[str], rewards: list[float]
    ) -> list[float]:
        lens = [len(s) for s in texts]
        if length_weight and step >= length_warmup:
            extra = kimi_length_rewards([accuracy_reward(task, s) == 1.0 for s in texts], lens)
            rewards = [r + length_weight * x for r, x in zip(rewards, extra, strict=True)]
        if overlong_cache:
            pen = [overlong_penalty(n, max_new_tokens, overlong_cache) for n in lens]
            rewards = [r + p for r, p in zip(rewards, pen, strict=True)]
        if budgets is not None:
            limit = budget_tau(step, steps, taus) * budgets[effort, len(str(task.a))]
            rewards = budget_override(rewards, [think_tokens(s) for s in texts], limit)
        return rewards

    return shape

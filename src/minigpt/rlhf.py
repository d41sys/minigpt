"""Classic RLHF at toy scale: preference pairs, a reward model, then PPO against it.

This is the recipe of InstructGPT (Ouyang et al. 2022, https://arxiv.org/abs/2203.02155) that
DPO (`dpo.py`) and group-relative RL (GRPO) later replaced: fit a Bradley-Terry reward model to
preference pairs, then run PPO on the policy with a per-token KL penalty to the SFT model.

Preferences come from the verifiable addition task of `reason.py`. A program stands in for the
human labeler (the Claude Opus 5.5 system card describes crowd workers doing preference
selection). Because the same program is also an exact *gold* reward, the proxy (learned) reward
can be compared with the truth while PPO pushes on it: the setup of "Scaling Laws for Reward
Model Overoptimization" (Gao et al. 2022, https://arxiv.org/abs/2210.10760). A policy that
climbs the proxy reward without a matching gold gain is exploiting the reward model. The Opus
5.5 card reports a related failure in real training: some snapshots "concealed actions from an
automated grader". Whether this toy run shows overoptimization is read off its history, not
assumed.

Design choice: the value head sits on the policy (`ActorCritic`) instead of a second value model
initialized from the reward model. One body, one optimizer and one forward pass give both the
token log-probs and the values, at the price of the value loss sharing the policy's body
(weighted down by `vf_coef`).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt.bpe import BPETokenizer
from minigpt.finetune_cls import GPTForClassification
from minigpt.model import GPTConfig, GPTModel
from minigpt.reason import (
    Task,
    accuracy_reward,
    build_batch,
    format_reward,
    make_task,
    per_token_log_probs,
    sample_group,
)
from minigpt.train import configure_optimizers, get_device, save_checkpoint


@dataclass(frozen=True)
class RankedPair:
    """One labeled comparison: both completions answer the same prompt."""

    prompt: list[int]
    chosen: list[int]
    rejected: list[int]


def completion_text(tok: BPETokenizer, ids: list[int]) -> str:
    """Decode a sampled completion without its EOS."""
    return tok.decode([t for t in ids if t != tok.eos_token_id])


def labeler_score(task: Task, completion: str) -> float:
    """The simulated human labeler: right answer plus right format, 0 to 2."""
    return accuracy_reward(task, completion) + format_reward(completion)


def strict_pair(scores: list[float], rng: random.Random) -> tuple[int, int] | None:
    """Indices `(chosen, rejected)` of a best and a worst sample; None when all tie."""
    hi, lo = max(scores), min(scores)
    if hi == lo:
        return None
    best = [i for i, s in enumerate(scores) if s == hi]
    worst = [i for i, s in enumerate(scores) if s == lo]
    return rng.choice(best), rng.choice(worst)


def label_pairs(
    policy: GPTModel,
    tok: BPETokenizer,
    tasks: list[Task],
    k: int,
    max_new_tokens: int,
    rng: random.Random,
    temperature: float = 1.0,
) -> list[RankedPair]:
    """Sample `k` completions per task from `policy` and keep one strictly ordered pair each."""
    pairs: list[RankedPair] = []
    for task in tasks:
        prompt = tok.encode(task.prompt)
        group = sample_group(policy, prompt, k, max_new_tokens, tok.eos_token_id, temperature)
        scores = [labeler_score(task, completion_text(tok, c)) for c in group]
        pick = strict_pair(scores, rng)
        if pick is not None:
            pairs.append(RankedPair(prompt, group[pick[0]], group[pick[1]]))
    return pairs


def pad_rows(rows: list[list[int]], pad_id: int) -> torch.Tensor:
    """Right-pad token rows into one `(B, T)` tensor."""
    out = torch.full((len(rows), max(len(r) for r in rows)), pad_id, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.tensor(r)
    return out


def new_reward_model(sft: GPTModel, pad_id: int) -> GPTForClassification:
    """Reward model = SFT body + one scalar head on the last non-pad token."""
    return GPTForClassification(sft, num_classes=1, pad_id=pad_id)


def reward_scores(rm: GPTForClassification, idx: torch.Tensor) -> torch.Tensor:
    """Scalar reward per row of a right-padded `(B, T)` prompt+completion batch."""
    logits, _ = rm(idx)
    scores: torch.Tensor = logits.squeeze(-1)
    return scores


def bradley_terry_loss(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    """`-log sigmoid(r_chosen - r_rejected)`, averaged; equals log 2 when the rewards tie."""
    return -F.logsigmoid(chosen - rejected).mean()


def score_pairs(
    rm: GPTForClassification, pairs: list[RankedPair], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rewards of the chosen and the rejected completions, from a single forward pass."""
    rows = [p.prompt + p.chosen for p in pairs] + [p.prompt + p.rejected for p in pairs]
    scores = reward_scores(rm, pad_rows(rows, rm.pad_id).to(device))
    n = len(pairs)
    return scores[:n], scores[n:]


@torch.no_grad()
def evaluate_pairs(
    rm: GPTForClassification, pairs: list[RankedPair], batch_size: int, device: torch.device
) -> tuple[float, float]:
    """`(Bradley-Terry loss, pairwise accuracy)`; a tie does not count as correct."""
    rm.eval()
    loss = hits = 0.0
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        chosen, rejected = score_pairs(rm, batch, device)
        loss += bradley_terry_loss(chosen, rejected).item() * len(batch)
        hits += (chosen > rejected).sum().item()
    n = max(1, len(pairs))
    return loss / n, hits / n


def train_reward_model(
    rm: GPTForClassification,
    pairs: list[RankedPair],
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 3e-5,
    val_frac: float = 0.2,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit the reward model on the first `1 - val_frac` of `pairs`; the rest is held out.

    Returns the per-epoch history plus the final held-out loss and pairwise accuracy.
    """
    if len(pairs) < 2:
        raise ValueError("need at least 2 pairs to hold some out")
    if device is None:
        device = get_device()
    n_val = max(1, int(len(pairs) * val_frac))
    val, train = pairs[:n_val], pairs[n_val:]
    rm.to(device)
    rm.eval()  # dropout off, like DPO; gradients still flow
    optimizer = configure_optimizers(rm, learning_rate=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)

    history: list[dict[str, float]] = []
    val_loss = val_acc = 0.0
    for epoch in range(epochs):
        order = torch.randperm(len(train), generator=gen).tolist()
        total = hits = 0.0
        for i in range(0, len(train), batch_size):
            batch = [train[j] for j in order[i : i + batch_size]]
            chosen, rejected = score_pairs(rm, batch, device)
            loss = bradley_terry_loss(chosen, rejected)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item() * len(batch)
            hits += (chosen > rejected).sum().item()
        val_loss, val_acc = evaluate_pairs(rm, val, batch_size, device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": total / len(train),
                "train_accuracy": hits / len(train),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
    return {"history": history, "val_loss": val_loss, "val_accuracy": val_acc}


def save_reward_model(
    path: str | Path, rm: GPTForClassification, step: int, val_loss: float, val_accuracy: float
) -> None:
    extra = {
        "head": rm.cls_head.state_dict(),
        "pad_id": rm.pad_id,
        "val_accuracy": val_accuracy,
    }
    save_checkpoint(
        path,
        rm.gpt,
        None,
        step=step,
        val_loss=val_loss,
        extra=extra,
    )


def load_reward_model(path: str | Path) -> GPTForClassification:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    gpt = GPTModel(GPTConfig(**payload["config"]))
    gpt.load_state_dict(payload["model_state_dict"])
    rm = new_reward_model(gpt, payload["pad_id"])
    rm.cls_head.load_state_dict(payload["head"])
    return rm


class ActorCritic(nn.Module):
    """The policy plus a linear value head on the same final hidden states."""

    def __init__(self, gpt: GPTModel) -> None:
        super().__init__()
        self.gpt = gpt
        self.value_head = nn.Linear(gpt.config.d_model, 1)
        nn.init.zeros_(self.value_head.weight)  # start with V = 0
        nn.init.zeros_(self.value_head.bias)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Token logits `(B, T, V)` and state values `(B, T)`."""
        h = self.gpt.hidden_states(idx)
        logits: torch.Tensor = self.gpt.lm_head(h)
        values: torch.Tensor = self.value_head(h).squeeze(-1)
        return logits, values


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1)


def last_action_index(mask: torch.Tensor) -> torch.Tensor:
    """Column of the last scored token in each row of a `(B, T)` completion mask."""
    return (mask.long() * torch.arange(mask.size(1), device=mask.device)).argmax(dim=1)


def shape_rewards(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    mask: torch.Tensor,
    scores: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """InstructGPT's per-token reward: `-beta * (logp - ref_logp)` on every completion token,
    plus the reward-model score `(B,)` added on the last one."""
    rewards = -beta * (logps - ref_logps) * mask
    last = last_action_index(mask).unsqueeze(1)
    return rewards.scatter_add(1, last, scores.unsqueeze(1).to(rewards.dtype))


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation over `(B, T)`; returns `(advantages, returns)`.

    Values and rewards off the mask count as 0, so the episode ends after the last completion
    token with V = 0.
    """
    rewards, values = rewards * mask, values * mask
    adv = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.size(1))):
        nxt = values[:, t + 1] if t + 1 < rewards.size(1) else torch.zeros_like(carry)
        delta = rewards[:, t] + gamma * nxt - values[:, t]
        carry = delta + gamma * lam * carry
        adv[:, t] = carry
    adv = adv * mask
    return adv, adv + values


def whiten(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero mean, unit variance over the masked entries; masked entries become 0."""
    mean = _masked_mean(x, mask)
    var = _masked_mean((x - mean) ** 2, mask)
    return (x - mean) * torch.rsqrt(var + 1e-8) * mask


def ppo_policy_loss(
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    """Clipped surrogate: `-min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)`, token mean."""
    ratio = torch.exp(logps - old_logps)
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)
    return -_masked_mean(surrogate, mask)


def ppo_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    """Clipped value loss: the worse of the plain and the move-limited squared error."""
    clipped = old_values + (values - old_values).clamp(-clip, clip)
    err = torch.maximum((values - returns) ** 2, (clipped - returns) ** 2)
    return 0.5 * _masked_mean(err, mask)


def _logps_values(
    actor: ActorCritic, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probs of labels `y` and values, both zeroed off the completion mask."""
    logits, values = actor(x)
    mask = y != -100
    logps = torch.gather(
        F.log_softmax(logits.float(), dim=-1), 2, y.clamp(min=0).unsqueeze(2)
    ).squeeze(2)
    return logps * mask, values.float() * mask


def train_ppo(
    policy: GPTModel,
    ref: GPTModel,
    rm: GPTForClassification,
    tok: BPETokenizer,
    steps: int = 150,
    tasks_per_step: int = 8,
    samples_per_task: int = 8,
    max_digits: int = 3,
    max_new_tokens: int = 80,
    beta: float = 0.02,
    lr: float = 2e-5,
    ppo_epochs: int = 2,
    minibatch_size: int = 16,
    clip: float = 0.2,
    vf_clip: float = 0.2,
    vf_coef: float = 0.1,
    gamma: float = 1.0,
    lam: float = 0.95,
    seed: int = 0,
    device: torch.device | None = None,
    on_step: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    """PPO on fresh tasks each step, with the frozen reward model as the proxy reward.

    One step: sample `tasks_per_step * samples_per_task` completions, score them with the reward
    model, shape per-token rewards (score at the last token, KL penalty to `ref` on every
    token), run GAE, then `ppo_epochs` passes of minibatch updates. `policy` is trained in place.
    Each history row holds the mean proxy `reward` (reward-model score, before the KL penalty),
    `gold_accuracy` and `gold_format` of the fresh samples, `kl` (sequence-level estimate
    sum(logp - ref_logp), mean over samples), `value_loss` and mean completion `length`.
    """
    if device is None:
        device = get_device()
    policy.to(device)
    ref.to(device)
    rm.to(device)
    for frozen in (ref, rm):
        frozen.eval()
        for p in frozen.parameters():
            p.requires_grad = False
    actor = ActorCritic(policy).to(device)
    actor.eval()  # dropout off, so the first ratio is exactly 1; gradients still flow
    optimizer = configure_optimizers(actor, learning_rate=lr, weight_decay=0.0)
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    eos, pad = tok.eos_token_id, rm.pad_id

    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        # ---- rollout
        prompts: list[list[int]] = []
        completions: list[list[int]] = []
        accuracy: list[float] = []
        fmt: list[float] = []
        for _ in range(tasks_per_step):
            task = make_task(rng, rng.randint(1, max_digits))
            ids = tok.encode(task.prompt)
            for c in sample_group(policy, ids, samples_per_task, max_new_tokens, eos, 1.0):
                text = completion_text(tok, c)
                prompts.append(ids)
                completions.append(c)
                accuracy.append(accuracy_reward(task, text))
                fmt.append(format_reward(text))
        x, y = build_batch(prompts, completions, pad)
        x, y = x.to(device), y.to(device)
        mask = y != -100
        rows = [p + c for p, c in zip(prompts, completions, strict=True)]
        with torch.no_grad():
            scores = reward_scores(rm, pad_rows(rows, pad).to(device))
            old_logps, old_values = _logps_values(actor, x, y)
            ref_logps, _ = per_token_log_probs(ref, x, y)
            rewards = shape_rewards(old_logps, ref_logps, mask, scores, beta)
            advantages, returns = gae(rewards, old_values, mask, gamma, lam)
            advantages = whiten(advantages, mask)

        # ---- update
        value_losses: list[float] = []
        for _ in range(ppo_epochs):
            perm = torch.randperm(x.size(0), generator=gen).to(device)
            for i in range(0, x.size(0), minibatch_size):
                idx = perm[i : i + minibatch_size]
                logps, values = _logps_values(actor, x[idx], y[idx])
                m = mask[idx]
                policy_loss = ppo_policy_loss(logps, old_logps[idx], advantages[idx], m, clip)
                value_loss = ppo_value_loss(values, old_values[idx], returns[idx], m, vf_clip)
                optimizer.zero_grad()
                (policy_loss + vf_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                optimizer.step()
                value_losses.append(value_loss.item())

        row: dict[str, float] = {
            "step": float(step),
            "reward": scores.mean().item(),
            "gold_accuracy": sum(accuracy) / len(accuracy),
            "gold_format": sum(fmt) / len(fmt),
            "kl": ((old_logps - ref_logps) * mask).sum(dim=1).mean().item(),
            "value_loss": sum(value_losses) / len(value_losses),
            "length": mask.sum(dim=1).float().mean().item(),
        }
        history.append(row)
        if on_step is not None:
            on_step(step, row)
    return {"history": history}


def overopt_summary(history: list[dict[str, float]], window: int = 10) -> dict[str, float]:
    """Trailing-mean proxy reward and gold accuracy: first window, last, and the gold peak."""

    def smooth(key: str) -> list[float]:
        vals = [h[key] for h in history]
        out: list[float] = []
        for i in range(len(vals)):
            tail = vals[max(0, i + 1 - window) : i + 1]
            out.append(sum(tail) / len(tail))
        return out

    reward, gold = smooth("reward"), smooth("gold_accuracy")
    peak = max(range(len(gold)), key=gold.__getitem__)
    return {
        "reward_first": reward[min(window, len(reward)) - 1],
        "reward_last": reward[-1],
        "gold_peak": gold[peak],
        "gold_peak_step": float(peak + 1),
        "gold_last": gold[-1],
    }

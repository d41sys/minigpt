"""Agentic RL: multi-turn tool use with a reward at the end of the episode (cf. minimind's
`train_agent.py`: rollout batch -> reward on the whole trajectory -> policy update).

Tasks are chains `a+b-c=` (two binary operations, left to right, no negative intermediate or
result). The only tool is a calculator that evaluates exactly ONE binary operation: the policy
writes `<call>12+7</call>`, generation pauses, the environment appends `<out>19</out>`, and
generation resumes until `<answer>..</answer>` or a length / call cap. Solving a chain therefore
needs two calls, the second one containing the first observation. The observation tokens are
context: they are masked out of the SFT and GRPO losses (`build_batch(keeps=...)`).

The reward is a function of the finished transcript only (`chain_reward`), so the same function
scores a no-tool think trace, which is the control arm.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from minigpt.bpe import BPETokenizer
from minigpt.grpo import grpo_step
from minigpt.model import GPTModel
from minigpt.reason import (
    OPS,
    Task,
    build_batch,
    decode_completion,
    format_reward,
    parse_answer,
)
from minigpt.sft import SFTDataset
from minigpt.train import configure_optimizers, get_device

CALL_RE = re.compile(r"<call>([^<]*)</call>$")
EXPR_RE = re.compile(r"(\d+)([+-])(\d+)")
TOOL_FORMAT_RE = re.compile(r"(<call>\d+[+-]\d+</call><out>-?\d+</out>)*<answer>\d+</answer>")
BAD_OUT = "<out>?</out>"
MAX_CALLS = 4  # a fifth call ends the episode, unfinished
UNFINISHED_PENALTY = 0.2

ChainReward = Callable[["Chain", str], float]


def _apply(a: int, op: str, b: int) -> int:
    return a + b if op == "+" else a - b


@dataclass(frozen=True)
class Chain:
    """`a op1 b op2 c`, evaluated left to right; the intermediate and the result are >= 0.
    With `op2 == ""` it is the easy task `a op1 b` (one call, or none: see `chain_task`)."""

    a: int
    b: int
    c: int = 0
    op1: str = "+"
    op2: str = "+"

    @property
    def prompt(self) -> str:
        return f"{self.a}{self.op1}{self.b}{self.op2}{self.c if self.op2 else ''}="

    @property
    def mid(self) -> int:
        return _apply(self.a, self.op1, self.b)

    @property
    def answer(self) -> str:
        return str(_apply(self.mid, self.op2, self.c) if self.op2 else self.mid)

    def solution(self) -> str:
        """No-tool reference: the column steps of both operations, then the answer."""
        steps = Task(self.a, self.b, self.op1).steps()
        if self.op2:
            steps += Task(self.mid, self.c, self.op2).steps()
        return f"<think>{' '.join(steps)}</think><answer>{self.answer}</answer>"

    def tool_trace(self) -> list[tuple[str, bool]]:
        """Scripted tool episode as `(text, written by the policy)`; observations are False."""
        trace = [
            (f"<call>{self.a}{self.op1}{self.b}</call>", True),
            (f"<out>{self.mid}</out>", False),
        ]
        if self.op2:
            trace += [
                (f"<call>{self.mid}{self.op2}{self.c}</call>", True),
                (f"<out>{self.answer}</out>", False),
            ]
        return [*trace, (f"<answer>{self.answer}</answer>", True)]


def chain_task(
    rng: random.Random, digits: int, exclude: frozenset[str] = frozenset(), easy: bool = False
) -> Chain:
    """Chain whose three operands have `digits` digits (0-9 when 1); prompts in `exclude` are
    redrawn. Rejection keeps the intermediate and the result non-negative. `easy`: a 1-digit
    `a op b` instead (answerable without the tool, since pretraining saw such equations)."""
    lo = 0 if digits == 1 or easy else 10 ** (digits - 1)
    hi = 10 ** (1 if easy else digits) - 1
    while True:
        a, b, c = (rng.randint(lo, hi) for _ in range(3))
        op1, op2 = rng.choice(OPS), "" if easy else rng.choice(OPS)
        if op1 == "-" and a < b:
            a, b = b, a
        if op2 == "-" and c > _apply(a, op1, b):
            continue
        task = Chain(a, b, c, op1, op2)
        if task.prompt not in exclude:
            return task


def calculator(expr: str) -> str | None:
    """The tool: exactly one binary operation `a+b` or `a-b`; None for anything else."""
    m = EXPR_RE.fullmatch(expr)
    return str(_apply(int(m[1]), m[2], int(m[3]))) if m else None


def call_counts(text: str) -> tuple[int, int]:
    """`(calls, malformed calls)` of a transcript: closed `<call>` tags, and the `<out>?</out>`
    the environment answered them with."""
    return text.count("</call>"), text.count(BAD_OUT)


def chain_format(text: str) -> float:
    """1.0 for a clean think trace or a clean tool transcript (calls, each answered, one answer)."""
    return 1.0 if format_reward(text) or TOOL_FORMAT_RE.fullmatch(text.strip()) else 0.0


def correct(task: Chain, text: str) -> float:
    return 1.0 if parse_answer(text) == task.answer else 0.0


def chain_reward(call_cost: float = 0.0, call_bonus: float = 0.0) -> ChainReward:
    """Episode reward: exact answer + 0.1 format - `call_cost` per call - a penalty when no
    answer was given. `call_bonus` pays every well-formed call whatever the answer (the hackable
    variant of the exploratory run)."""

    def reward(task: Chain, text: str) -> float:
        calls, bad = call_counts(text)
        r = correct(task, text) + 0.1 * chain_format(text)
        r += call_bonus * (calls - bad) - call_cost * calls
        return r - UNFINISHED_PENALTY * (parse_answer(text) is None)

    return reward


def draw_task(
    rng: random.Random, digits: tuple[int, int], easy_frac: float, exclude: frozenset[str]
) -> Chain:
    """One training task: easy with probability `easy_frac`, else a chain of `digits` digits
    outside `exclude` (the easy space has 155 prompts, so it is neither held out nor excluded)."""
    if easy_frac > 0 and rng.random() < easy_frac:
        return chain_task(rng, 1, easy=True)
    return chain_task(rng, rng.randint(*digits), exclude)


def chain_trace_dataset(
    tasks: list[Chain], tok: BPETokenizer, mode: str, direct_frac: float = 0.0, seed: int = 0
) -> SFTDataset:
    """Cold-start SFT data. `think`: the no-tool column trace. `tool`: the scripted calls, with
    the observations masked from the loss; a `direct_frac` share of the easy tasks is answered
    directly, with no call."""
    rng = random.Random(seed)
    ds = SFTDataset([], tok)
    for t in tasks:
        if mode == "think":
            pieces = [(t.solution(), True)]
        elif not t.op2 and direct_frac > 0 and rng.random() < direct_frac:
            pieces = [(f"<answer>{t.answer}</answer>", True)]
        else:
            pieces = t.tool_trace()
        ids = [i for text, _ in pieces for i in tok.encode(text)]
        keep = [k for text, k in pieces for _ in tok.encode(text)]
        x, y = build_batch([tok.encode(t.prompt)], [[*ids, tok.eos_token_id]], 0, [[*keep, True]])
        ds.examples.append((x[0], y[0]))
    return ds


@torch.no_grad()
def rollout(
    model: GPTModel,
    tok: BPETokenizer,
    prompts: list[list[int]],
    max_new_tokens: int,
    temperature: float = 1.0,
    tools: bool = True,
) -> tuple[list[list[int]], list[list[bool]]]:
    """Sample one episode per prompt, all in one batch, executing the calculator on the way.

    Every `</call>` pauses the row: the environment appends `<out>..</out>` (`<out>?</out>` for a
    call that is not one binary operation) and the row resumes. A row ends at `</answer>`, EOS,
    `max_new_tokens` (observations count), or a call beyond `MAX_CALLS`. Returns completions and
    `keeps`, False on observation tokens. Rows are right-padded and the model is run without a
    KV cache; causal attention makes the padding harmless. `temperature <= 0` is greedy.
    """
    device = next(model.parameters()).device
    eos, pad = tok.eos_token_id, tok.special_to_id["<|pad|>"]
    call_end, ans_end = tok.encode("</call>"), tok.encode("</answer>")
    rows = [list(p) for p in prompts]
    comps: list[list[int]] = [[] for _ in prompts]
    keeps: list[list[bool]] = [[] for _ in prompts]
    seg = [0] * len(prompts)  # where the policy's current turn starts in comps[i]
    calls = [0] * len(prompts)
    live = list(range(len(prompts)))
    model.eval()
    while live:  # every pass adds at least one token to each live row, so this ends
        x = torch.full((len(live), max(len(rows[i]) for i in live)), pad, dtype=torch.long)
        for r, i in enumerate(live):
            x[r, : len(rows[i])] = torch.tensor(rows[i])
        last = torch.tensor([len(rows[i]) - 1 for i in live], device=device)
        logits = model(x.to(device))[0][torch.arange(len(live), device=device), last].float()
        if temperature > 0:
            nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).squeeze(1)
        else:
            nxt = logits.argmax(dim=-1)
        still = []
        for i, t in zip(live, nxt.tolist(), strict=True):
            for lst in (rows[i], comps[i]):
                lst.append(t)
            keeps[i].append(True)
            done = t == eos or comps[i][-len(ans_end) :] == ans_end
            if tools and not done and comps[i][-len(call_end) :] == call_end:
                calls[i] += 1
                m = CALL_RE.search(tok.decode(comps[i][seg[i] :]))
                result = calculator(m[1]) if m else None
                obs = tok.encode(BAD_OUT if result is None else f"<out>{result}</out>")
                if calls[i] > MAX_CALLS or len(comps[i]) + len(obs) >= max_new_tokens:
                    done = True
                else:
                    for lst in (rows[i], comps[i]):
                        lst.extend(obs)
                    keeps[i].extend([False] * len(obs))
                    seg[i] = len(comps[i])
            if not done and len(comps[i]) < max_new_tokens:
                still.append(i)
        live = still
    return comps, keeps


def episode_stats(task: Chain, text: str) -> dict[str, float]:
    calls, bad = call_counts(text)
    return {
        "accuracy": correct(task, text),
        "calls": float(calls),
        "malformed": float(bad),
        "unfinished": float(parse_answer(text) is None),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    """Means over episodes; `malformed` becomes the share of calls (0 when there were none)."""
    n = len(rows)
    out = {k: sum(r[k] for r in rows) / n for k in ("accuracy", "calls", "unfinished")}
    out["malformed"] = sum(r["malformed"] for r in rows) / max(1.0, sum(r["calls"] for r in rows))
    return out


HELD_OUT_N = 200  # per digit count; training never draws a prompt from the first HELD_OUT_N


def eval_tasks(n_per_digit: int, digits: tuple[int, int]) -> list[Chain]:
    """The fixed evaluation tasks: the first `n_per_digit` of one seeded stream per digit count."""
    tasks: list[Chain] = []
    for d in range(digits[0], digits[1] + 1):
        rng = random.Random(f"chain-eval-{d}")
        tasks += [chain_task(rng, d) for _ in range(n_per_digit)]
    return tasks


def held_out(digits: tuple[int, int]) -> frozenset[str]:
    return frozenset(t.prompt for t in eval_tasks(HELD_OUT_N, digits))


@torch.no_grad()
def evaluate_chain(
    model: GPTModel,
    tok: BPETokenizer,
    n_per_digit: int,
    digits: tuple[int, int],
    max_new_tokens: int = 110,
    tools: bool = True,
    easy_n: int = 0,
) -> dict[str, Any]:
    """Greedy episodes on `eval_tasks`, so every model sees the same held-out tasks. Returns
    accuracy, mean calls, malformed share of calls and unfinished share, overall and `per_digit`;
    with `easy_n`, also `easy` over that many easy tasks (not part of the overall numbers)."""
    tasks = eval_tasks(n_per_digit, digits)
    rng = random.Random("chain-eval-easy")
    easy = [chain_task(rng, 1, easy=True) for _ in range(easy_n)]
    prompts = [tok.encode(t.prompt) for t in tasks + easy]
    comps, _ = rollout(model, tok, prompts, max_new_tokens, 0.0, tools)
    rows = [
        episode_stats(t, decode_completion(tok, c))
        for t, c in zip(tasks + easy, comps, strict=True)
    ]
    per_digit = {
        d: summarize(rows[k * n_per_digit : (k + 1) * n_per_digit])
        for k, d in enumerate(range(digits[0], digits[1] + 1))
    }
    res = {**summarize(rows[: len(tasks)]), "per_digit": per_digit}
    return {**res, "easy": summarize(rows[len(tasks) :])} if easy_n else res


def format_chain(res: dict[str, Any]) -> str:
    def line(label: str, m: dict[str, float]) -> str:
        return (
            f"{label}: accuracy {m['accuracy']:.3f}  calls {m['calls']:.2f}  "
            f"malformed {m['malformed']:.3f}  unfinished {m['unfinished']:.3f}"
        )

    rows = [line(f"{d} digits", m) for d, m in res["per_digit"].items()]
    rows.append(line("all", res))
    return "\n".join([*rows, line("easy", res["easy"])] if "easy" in res else rows)


def train_chain_grpo(
    policy: GPTModel,
    ref: GPTModel,
    tok: BPETokenizer,
    reward_fn: ChainReward,
    tools: bool = True,
    steps: int = 200,
    tasks_per_step: int = 8,
    group_size: int = 8,
    digits: tuple[int, int] = (2, 3),
    max_new_tokens: int = 110,
    temperature: float = 1.0,
    lr: float = 5e-5,
    beta: float = 0.04,
    clip_eps: float = 0.2,
    mu: int = 1,
    std_norm: bool = True,
    loss_agg: str = "seq-mean",
    optim: str = "adamw",
    exclude: frozenset[str] = frozenset(),
    easy_frac: float = 0.0,
    seed: int = 0,
    device: torch.device | None = None,
    on_step: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    """GRPO on whole episodes: per step `tasks_per_step` chains, `group_size` rollouts each, one
    reward per episode, groups with equal rewards dropped, then `grpo_step` on the kept groups
    with observation tokens masked. `exclude` holds the held-out prompts; a share `easy_frac` of
    the tasks are easy (`draw_task`). Returns
    `{"history": [per-step dict]}` with reward, accuracy, calls, malformed, unfinished, kl, clip."""
    if 2 * digits[1] + 3 + max_new_tokens > policy.config.context_length:
        raise ValueError("longest prompt + max_new_tokens exceeds the context length")
    device = device or get_device()
    policy.to(device)
    ref.to(device).eval()
    ref.requires_grad_(False)
    optimizer = configure_optimizers(policy, weight_decay=0.0, learning_rate=lr, optim=optim)
    rng = random.Random(seed)
    pad = tok.special_to_id["<|pad|>"]
    history: list[dict[str, float]] = []
    for step in range(steps):
        tasks = [draw_task(rng, digits, easy_frac, exclude) for _ in range(tasks_per_step)]
        prompts = [tok.encode(t.prompt) for t in tasks for _ in range(group_size)]
        comps, keeps = rollout(policy, tok, prompts, max_new_tokens, temperature, tools)
        texts = [decode_completion(tok, c) for c in comps]
        flat = [t for t in tasks for _ in range(group_size)]
        rewards = torch.tensor([reward_fn(t, s) for t, s in zip(flat, texts, strict=True)])
        rewards = rewards.reshape(tasks_per_step, group_size)
        # All-equal rewards give zero advantage: nothing to learn, so drop the group.
        spread = rewards.max(dim=1).values > rewards.min(dim=1).values
        kept = spread.nonzero().flatten().tolist()
        rows = [i * group_size + j for i in kept for j in range(group_size)]
        kl = clip = 0.0
        if rows:
            kl, clip = grpo_step(
                policy,
                ref,
                optimizer,
                [prompts[r] for r in rows],
                [comps[r] for r in rows],
                rewards[spread],
                pad,
                clip_eps=clip_eps,
                beta=beta,
                mu=mu,
                std_norm=std_norm,
                loss_agg=loss_agg,
                max_len=max_new_tokens,
                keeps=[keeps[r] for r in rows],
            )
        stats = summarize([episode_stats(t, s) for t, s in zip(flat, texts, strict=True)])
        record = {
            "step": float(step),
            "reward": rewards.mean().item(),
            **stats,
            "kl": kl,
            "clip": clip,
            "drop_rate": 1 - spread.float().mean().item(),
        }
        history.append(record)
        if on_step is not None:
            on_step(record)
    return {"history": history}

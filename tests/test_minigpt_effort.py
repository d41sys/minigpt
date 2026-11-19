import copy
import math
import random
from typing import Any

import pytest
import torch

from minigpt import grpo
from minigpt.grpo import C_MAX, L_NORM, effort_coef, length_penalty, train_grpo
from minigpt.model import GPTConfig, GPTModel
from minigpt.reason import Task, reasoning_tokenizer, sft_trace_dataset, think_tokens

TINY = GPTConfig(vocab_size=258, context_length=64, d_model=32, num_layers=2, num_heads=4)


def test_penalty_coefficient_decays_by_e_per_tau_and_the_cap_binds() -> None:
    # Levels 1, 5, 9: spacing 4, tau = LAMBDA * 4. Lowest level pays the full k0.
    levels = (1, 5, 9)
    assert effort_coef(1, levels, 0.5) == pytest.approx(0.5)
    tau = grpo.LAMBDA * 4
    assert effort_coef(1 + 4, levels, 0.5) == pytest.approx(0.5 * math.exp(-4 / tau))
    assert effort_coef(9, levels, 0.5) < effort_coef(5, levels, 0.5) < 0.5
    trace = "<think>" + "x" * 15 + "</think><answer>1</answer>"
    assert think_tokens(trace) == 15
    assert length_penalty(trace, 1.0) == pytest.approx(-15 / L_NORM)
    assert length_penalty(trace, 1e6) == -C_MAX  # the cap
    assert length_penalty("<think></think><answer>1</answer>", 1.0) == 0.0  # non-think is free


def test_effort_sft_prefixes_prompts_and_nothink_is_independent_of_the_level() -> None:
    tok = reasoning_tokenizer()
    tasks = [Task(12, 34)] * 40
    ds = sft_trace_dataset(tasks, tok, efforts=(1, 9), nothink_frac=0.5)
    prefixes, nothink = set[str](), [0, 0]
    for x, y in ds.examples:
        text = tok.decode([t for t in x.tolist() if t != 0])
        prefixes.add(text[:3])
        # y holds the trace after the prompt; the prompt tokens are masked with -100.
        trace = tok.decode([t for t in y.tolist() if t not in (-100, tok.eos_token_id)])
        nothink[text[1] == "9"] += trace.startswith("<think></think>")
    assert prefixes == {"e1:", "e9:"}
    assert all(n > 0 for n in nothink)  # both levels get non-think traces


def test_each_group_has_one_effort_and_levels_are_sampled_for_the_same_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tok = reasoning_tokenizer()
    seen: list[tuple[list[list[int]], torch.Tensor]] = []

    def fake_step(*args: Any, **_kw: Any) -> float:
        seen.append((args[3], args[5]))  # prompts (flat, group-major), rewards (groups, G)
        return 0.0

    monkeypatch.setattr(grpo, "grpo_step", fake_step)
    policy = GPTModel(TINY)
    rnd = random.Random(0)
    out = train_grpo(
        policy,
        copy.deepcopy(policy),
        tok,
        steps=1,
        tasks_per_step=3,
        group_size=4,
        digits=(1, 1),
        max_new_tokens=6,
        efforts=(1, 9),
        reward_fn=lambda _t, _c: rnd.random(),  # always mixed, so no group is dropped
        device=torch.device("cpu"),
    )
    (prompts, rewards), (rec,) = seen[0], out["history"]
    assert rewards.shape == (6, 4)  # 3 tasks x 2 levels groups, G completions each
    blocks = [prompts[i : i + 4] for i in range(0, len(prompts), 4)]
    assert all(all(p == b[0] for p in b) for b in blocks)  # one prompt, so one level, per group
    e1, e9 = tok.encode("e1:"), tok.encode("e9:")
    assert [b[0][:3] for b in blocks] == [e1, e9] * 3
    assert all(a[0][3:] == b[0][3:] for a, b in zip(blocks[0::2], blocks[1::2], strict=True))
    assert {"think_e1", "think_e9", "acc_e1", "acc_e9"} <= rec.keys()

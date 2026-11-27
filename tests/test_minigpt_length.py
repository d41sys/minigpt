import copy
from typing import Any

import pytest
import torch

from minigpt import grpo
from minigpt.grpo import (
    budget_override,
    budget_tau,
    estimate_budgets,
    kimi_length_rewards,
    make_shaper,
    overlong_penalty,
    train_grpo,
)
from minigpt.model import GPTConfig, GPTModel
from minigpt.reason import Task, reasoning_tokenizer

TINY = GPTConfig(vocab_size=258, context_length=64, d_model=32, num_layers=2, num_heads=4)


def test_kimi_length_reward_is_lambda_for_correct_and_clipped_for_wrong() -> None:
    # lambda = 0.5 - (len - min) / (max - min): 0.5, 0.0, -0.5 for lengths 10, 20, 30.
    assert kimi_length_rewards([True, True, True], [10, 20, 30]) == [0.5, 0.0, -0.5]
    # A wrong answer never gains: the shortest wrong one gets min(0, 0.5) = 0.
    assert kimi_length_rewards([False, False, False], [10, 20, 30]) == [0.0, 0.0, -0.5]
    assert kimi_length_rewards([True, False], [7, 7]) == [0.0, 0.0]  # one length: no signal


def test_overlong_penalty_is_zero_linear_then_minus_one() -> None:
    assert overlong_penalty(60, 80, 20) == 0.0
    assert overlong_penalty(70, 80, 20) == pytest.approx(-0.5)
    assert overlong_penalty(80, 80, 20) == -1.0
    assert overlong_penalty(95, 80, 20) == -1.0


def test_budget_override_beats_a_correct_answer_and_tau_anneals_in_stages() -> None:
    assert budget_override([1.1, 1.1, 0.0], [10, 21, 30], limit=20.0) == [1.1, -1.0, -1.0]
    taus = (1.5, 1.0, 0.5)
    assert [budget_tau(s, 9, taus) for s in (0, 2, 3, 5, 6, 8)] == [1.5, 1.5, 1.0, 1.0, 0.5, 0.5]
    assert budget_tau(0, 5, (1.0,)) == 1.0


def test_shaper_composes_length_overlong_and_budget_in_order() -> None:
    task = Task(12, 34)  # answer 46
    good = "<think>2+4+0=6 1+3+0=4</think><answer>46</answer>"  # 15 think tokens, 49 chars
    short = "<think>x</think><answer>46</answer>"  # 1 think token, 35 chars
    texts, ones = [good, short], [1.0, 1.0]
    # K1.5 only: lambda is -0.5 for the longer correct answer and 0.5 for the shorter, weight 2.
    assert make_shaper(10, 80, length_weight=2.0)(0, task, None, texts, ones) == [0.0, 2.0]
    # Warm-up: no length reward before `length_warmup`.
    warm = make_shaper(10, 80, length_weight=2.0, length_warmup=5)
    assert warm(4, task, None, texts, ones) == [1.0, 1.0]
    assert warm(5, task, None, texts, ones) == [0.0, 2.0]
    # DAPO: l_max 49 - cache 10 = 39 free, so the 49-char text pays -1 and the 35-char one 0.
    assert make_shaper(10, 49, overlong_cache=10)(0, task, None, texts, ones) == [0.0, 1.0]
    # K3: limit tau * b0 = 10 overrides the correct 15-token answer, after everything else.
    budgets: grpo.Budgets = {(None, 2): 10.0}
    k3 = make_shaper(10, 80, budgets, (1.0,), length_weight=2.0)
    assert k3(0, task, None, texts, ones) == [-1.0, 2.0]


def test_estimate_budgets_has_one_mean_per_effort_and_digit_count() -> None:
    torch.manual_seed(0)
    out = estimate_budgets(
        GPTModel(TINY), reasoning_tokenizer(), (1, 2), (1, 9), "+", 2, 2, 6, 1.0, 0
    )
    assert set(out) == {(1, 1), (1, 2), (9, 1), (9, 2)}
    assert all(0.0 <= v <= 6.0 for v in out.values())


def test_train_grpo_scores_groups_through_the_shaper(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[torch.Tensor] = []

    def fake_step(*args: Any, **_kw: Any) -> tuple[float, float]:
        seen.append(args[5])
        return 0.0, 0.0

    monkeypatch.setattr(grpo, "grpo_step", fake_step)
    calls: list[tuple[int, int | None, int]] = []

    def shape(step: int, task: Task, effort: int | None, texts: list[str], r: list[float]) -> Any:
        calls.append((step, effort, len(texts)))
        return [float(i) for i in range(len(texts))]  # distinct, so no group is dropped

    policy = GPTModel(TINY)
    train_grpo(
        policy,
        copy.deepcopy(policy),
        reasoning_tokenizer(),
        steps=2,
        tasks_per_step=3,
        group_size=4,
        digits=(1, 1),
        max_new_tokens=6,
        shape=shape,
        device=torch.device("cpu"),
    )
    assert calls == [(0, None, 4)] * 3 + [(1, None, 4)] * 3
    assert all(torch.equal(r, torch.arange(4.0).repeat(3, 1)) for r in seen)

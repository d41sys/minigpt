import copy
import math
from typing import Any

import pytest
import torch

from minigpt.grpo import clip_fraction, group_advantages, grpo_loss, train_grpo
from minigpt.model import GPTConfig, GPTModel
from minigpt.reason import reasoning_tokenizer

TINY = GPTConfig(vocab_size=258, context_length=64, d_model=32, num_layers=2, num_heads=4)


def test_rloo_and_batch_advantages() -> None:
    r = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    rloo = group_advantages(r, mode="rloo")
    # r_i - mean of the other three: 1 - 0 = 1, 0 - 1/3 = -1/3
    assert torch.allclose(rloo[0], torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3]), atol=1e-6)
    assert torch.allclose(rloo[1], torch.tensor([-2 / 3, -2 / 3, 2 / 3, 2 / 3]), atol=1e-6)
    batch = group_advantages(r, mode="batch")
    centred = group_advantages(r, std_norm=False)
    assert torch.allclose(batch.std(correction=0), torch.ones(()), atol=1e-3)
    assert torch.allclose(batch, centred / centred.std(correction=0), atol=1e-3)
    with pytest.raises(ValueError):
        group_advantages(r, mode="nope")


def _grad(log_ratio: float, adv: float, n: int = 3, **kw: Any) -> torch.Tensor:
    """Gradient of the loss w.r.t. the log-probs of one sequence of `n` tokens, old log-prob 0."""
    logps = torch.full((1, n), log_ratio, requires_grad=True)
    zeros, mask = torch.zeros(1, n), torch.ones(1, n, dtype=torch.bool)
    loss, _ = grpo_loss(logps, zeros, zeros, mask, torch.tensor([adv]), **kw)
    loss.backward()
    assert logps.grad is not None
    return logps.grad


def test_clip_higher_widens_only_the_upper_bound() -> None:
    up = math.log(1.25)  # inside 1 + 0.28, outside 1 + 0.2
    assert _grad(up, 1.0, clip_eps=0.2, beta=0.0).abs().sum() == 0.0
    assert _grad(up, 1.0, clip_eps=0.2, clip_eps_high=0.28, beta=0.0).abs().sum() > 0.0
    down = math.log(0.7)  # outside 1 - 0.2 either way
    assert _grad(down, -1.0, clip_eps=0.2, clip_eps_high=0.28, beta=0.0).abs().sum() == 0.0


def test_gspo_clips_whole_sequences_by_their_mean_log_ratio() -> None:
    kw: dict[str, Any] = {"policy_loss": "gspo", "clip_eps": 4e-4, "beta": 0.0}
    assert _grad(0.1, 1.0, **kw).abs().sum() == 0.0  # sequence ratio 1.105 > 1.0004, A > 0
    assert _grad(0.1, -1.0, **kw).abs().sum() > 0.0  # A < 0 above the range keeps its gradient
    # One token far out of range, the rest in: the geometric mean stays inside, so nothing is
    # clipped, whereas PPO's per-token ratio would zero that token's gradient.
    logps = torch.tensor([[0.5, -0.1, -0.1, -0.1]], requires_grad=True)
    mask = torch.ones(1, 4, dtype=torch.bool)
    zeros = torch.zeros(1, 4)
    loss, _ = grpo_loss(
        logps, zeros, zeros, mask, torch.tensor([1.0]), clip_eps=0.2, beta=0.0, policy_loss="gspo"
    )
    loss.backward()
    assert logps.grad is not None
    assert (logps.grad != 0).all()
    # Inside the range the gradient is A * s / |y| per token, shared equally by all tokens.
    g = _grad(1e-5, 1.0, n=4, clip_eps=4e-4, beta=0.0, policy_loss="gspo")
    assert torch.allclose(g, torch.full((1, 4), -math.exp(1e-5) / 4))


def test_cispo_keeps_the_gradient_of_clipped_tokens() -> None:
    kw: dict[str, Any] = {
        "policy_loss": "cispo",
        "loss_agg": "token-mean",
        "clip_eps": 1.0,
        "clip_eps_high": 0.2,
        "beta": 0.0,
    }
    # ratio 1.65 is clipped to 1.2, but unlike PPO the token still pushes: -1.2 * A / 3 each
    g = _grad(0.5, 2.0, **kw)
    assert torch.allclose(g, torch.full((1, 3), -1.2 * 2.0 / 3))
    assert _grad(0.5, 2.0, clip_eps=0.2, beta=0.0).abs().sum() == 0.0  # PPO control
    g = _grad(0.1, 1.0, **kw)  # inside the range the weight is the ratio itself
    assert torch.allclose(g, torch.full((1, 3), -math.exp(0.1) / 3))


def test_clip_fraction_counts_masked_tokens_beyond_the_bounds() -> None:
    old = torch.zeros(2, 3)
    logps = torch.log(torch.tensor([[1.1, 1.5, 0.5], [3.0, 1.0, 9.0]]))
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)  # the 9.0 is padding
    # ppo [0.8, 1.3]: 1.5, 0.5, 3.0 hit
    assert clip_fraction(logps, old, mask, 0.2, 0.3) == pytest.approx(3 / 5)
    # cispo, no lower bound: only 1.5 and 3.0 exceed 1.3
    assert clip_fraction(logps, old, mask, 1.0, 0.3, "cispo") == pytest.approx(2 / 5)
    # per-sequence geometric means: row 0 is 0.94 (inside), row 1 is sqrt(3) (hit, 2 tokens)
    assert clip_fraction(logps, old, mask, 0.2, 0.3, "gspo") == pytest.approx(2 / 5)
    assert clip_fraction(old, old, mask, 0.2) == 0.0


def test_loss_aggregations_weight_tokens_as_documented() -> None:
    # two groups of two rows with 1, 3 | 2, 1 tokens
    mask = torch.tensor([[1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    adv = torch.tensor([2.0, 0.0, 1.0, -1.0])
    z = torch.zeros(4, 3)

    def loss(agg: str) -> float:
        return grpo_loss(z, z, z, mask, adv, beta=0.0, loss_agg=agg, group_size=2)[0].item()

    assert loss("seq-mean") == pytest.approx(-0.5)  # mean of -2, 0, -1, +1
    assert loss("token-mean") == pytest.approx(-3 / 7)  # -(2 + 0 + 2 - 1) over all 7 tokens
    assert loss("group-mean") == pytest.approx(-(2 / 4 + 1 / 3) / 2)  # each over its group length


def test_ref_reset_snapshots_the_policy_and_leaves_the_callers_ref_alone() -> None:
    torch.manual_seed(0)
    policy = GPTModel(TINY)
    ref = copy.deepcopy(policy)
    start = copy.deepcopy(ref.state_dict())

    def run(reset: int) -> list[dict[str, float]]:
        p = copy.deepcopy(policy)
        torch.manual_seed(1)
        out = train_grpo(
            p,
            ref,
            reasoning_tokenizer(),
            steps=2,
            tasks_per_step=3,
            group_size=4,
            digits=(1, 1),
            max_new_tokens=6,
            lr=1e-2,
            reward_fn=lambda _t, c: float(len(c) % 3 == 0),
            ref_reset_every=reset,
            device=torch.device("cpu"),
        )
        history: list[dict[str, float]] = out["history"]
        return history

    plain, reset = run(0), run(1)
    assert plain[1]["kl"] > 0.0  # control: without a reset the policy has drifted from the ref
    assert reset[1]["kl"] == 0.0  # right after the snapshot the policy equals its reference
    assert all(torch.equal(v, ref.state_dict()[k]) for k, v in start.items())

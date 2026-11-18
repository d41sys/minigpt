import copy

import pytest
import torch

from minigpt.grpo import group_advantages, grpo_loss, grpo_step, kl_k3, train_grpo
from minigpt.model import GPTConfig, GPTModel
from minigpt.reason import build_batch, per_token_log_probs, reasoning_tokenizer

TINY = GPTConfig(vocab_size=258, context_length=64, d_model=32, num_layers=2, num_heads=4)


def test_advantages_are_zero_mean_and_unit_std_or_unscaled() -> None:
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.1, 1.1, 0.1, 0.1]])
    adv = group_advantages(r)
    assert torch.allclose(adv.mean(dim=1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(adv.std(dim=1, correction=0), torch.ones(2), atol=1e-3)
    raw = group_advantages(r, std_norm=False)
    assert torch.allclose(raw, torch.tensor([[0.5, -0.5, -0.5, 0.5], [-0.25, 0.75, -0.25, -0.25]]))
    flat = group_advantages(torch.full((1, 4), 0.5))  # all-equal group: zero, not NaN
    assert torch.equal(flat, torch.zeros(1, 4))


def test_no_gradient_when_advantages_are_zero_and_policy_equals_reference() -> None:
    torch.manual_seed(0)
    model = GPTModel(TINY).eval()
    ref = copy.deepcopy(model)
    x, y = build_batch([[1, 2, 3], [4, 5]], [[6, 7, 8], [9, 10]], pad_id=0)
    with torch.no_grad():
        old, _ = per_token_log_probs(model, x, y)
        ref_logps, _ = per_token_log_probs(ref, x, y)

    def max_grad(adv: list[float]) -> float:
        model.zero_grad(set_to_none=True)
        logps, mask = per_token_log_probs(model, x, y)
        loss, kl = grpo_loss(logps, old, ref_logps, mask, torch.tensor(adv))
        loss.backward()
        assert kl.item() == 0.0
        return max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)

    assert max_grad([0.0, 0.0]) < 1e-6
    assert max_grad([1.0, -1.0]) > 1e-4  # control: a real advantage does produce a gradient


def test_clipped_objective_blocks_the_gradient_only_on_the_improving_side() -> None:
    mask = torch.ones(1, 3, dtype=torch.bool)
    old = torch.zeros(1, 3)

    def grad(log_ratio: float, adv: float) -> float:
        logps = torch.full((1, 3), log_ratio, requires_grad=True)
        loss, _ = grpo_loss(logps, old, old, mask, torch.tensor([adv]), clip_eps=0.2, beta=0.0)
        loss.backward()
        assert logps.grad is not None
        return logps.grad.abs().sum().item()

    assert grad(0.5, 1.0) == 0.0  # A > 0, ratio 1.65 > 1.2: already improved enough
    assert grad(-0.5, -1.0) == 0.0  # A < 0, ratio 0.61 < 0.8: already pushed down enough
    assert grad(0.1, 1.0) > 0.0  # inside the trust region
    assert grad(0.5, -1.0) > 0.0  # A < 0 above the range: PPO keeps the unclipped term
    assert grad(-0.5, 1.0) > 0.0  # A > 0 below the range: same


def test_k3_kl_is_nonnegative_zero_at_equality_and_estimates_kl_policy_to_ref() -> None:
    p = torch.tensor([0.8, 0.15, 0.05])
    q = torch.tensor([0.2, 0.3, 0.5])
    lp, lq = p.log(), q.log()
    assert kl_k3(lp, lp).abs().max().item() == 0.0
    assert (kl_k3(lp, lq) >= 0).all()
    # Expectation over tokens drawn from the policy p is KL(p || q), not KL(q || p).
    assert (p * kl_k3(lp, lq)).sum().item() == pytest.approx((p * (lp - lq)).sum().item())
    assert (p * kl_k3(lp, lq)).sum().item() != pytest.approx((q * (lq - lp)).sum().item())


def test_seq_mean_weights_sequences_equally_and_token_sum_weights_tokens_equally() -> None:
    logps = torch.zeros(2, 4)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    adv = torch.ones(2)
    args = (logps, logps, logps, mask, adv)
    seq, _ = grpo_loss(*args, beta=0.0, loss_agg="seq-mean")
    tok, _ = grpo_loss(*args, beta=0.0, loss_agg="token-sum", max_len=4)
    assert seq.item() == pytest.approx(-1.0)  # (-2/2 + -4/4) / 2
    assert tok.item() == pytest.approx(-0.75)  # (-2/4 + -4/4) / 2
    with pytest.raises(ValueError):
        grpo_loss(*args, loss_agg="bogus")


def test_one_grpo_step_raises_the_probability_of_the_rewarded_token() -> None:
    torch.manual_seed(0)
    policy = GPTModel(TINY)
    ref = copy.deepcopy(policy)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    prompt, good, bad = [1, 2, 3], 65, 66

    def logp(token: int) -> float:
        policy.eval()
        with torch.no_grad():
            lp, _ = per_token_log_probs(policy, *build_batch([prompt], [[token]], pad_id=0))
        return lp.sum().item()

    good0, bad0 = logp(good), logp(bad)
    rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    grpo_step(policy, ref, opt, [prompt] * 4, [[good], [bad], [good], [bad]], rewards, pad_id=0)
    assert logp(good) > good0
    assert logp(bad) < bad0


def test_groups_with_equal_rewards_are_dropped_and_leave_the_weights_alone() -> None:
    torch.manual_seed(0)
    policy = GPTModel(TINY)
    ref = copy.deepcopy(policy)
    before = copy.deepcopy(policy.state_dict())
    out = train_grpo(
        policy,
        ref,
        reasoning_tokenizer(),
        steps=2,
        tasks_per_step=3,
        group_size=4,
        digits=(1, 1),
        max_new_tokens=6,
        reward_fn=lambda _t, _c: 0.5,
        observe=lambda samples: {"n": float(len(samples))},
        device=torch.device("cpu"),
    )
    history = out["history"]
    assert [h["drop_rate"] for h in history] == [1.0, 1.0]
    assert [h["n"] for h in history] == [12.0, 12.0]
    after = policy.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in before.items())

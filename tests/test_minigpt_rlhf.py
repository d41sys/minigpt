import copy
import math
import random

import pytest
import torch

from minigpt.model import GPTConfig, GPTModel
from minigpt.reason import reasoning_tokenizer
from minigpt.rlhf import (
    RankedPair,
    bradley_terry_loss,
    gae,
    new_reward_model,
    overopt_summary,
    pad_rows,
    ppo_policy_loss,
    ppo_value_loss,
    reward_scores,
    shape_rewards,
    strict_pair,
    train_ppo,
    train_reward_model,
    whiten,
)

TINY = GPTConfig(vocab_size=258, context_length=64, d_model=32, num_layers=2, num_heads=4)
CPU = torch.device("cpu")


def test_bradley_terry_loss_orders_and_ties() -> None:
    same = torch.tensor([0.3, -1.0])
    assert bradley_terry_loss(same, same).item() == pytest.approx(math.log(2))
    good = bradley_terry_loss(torch.tensor([2.0]), torch.tensor([-1.0])).item()
    bad = bradley_terry_loss(torch.tensor([-1.0]), torch.tensor([2.0])).item()
    assert good == pytest.approx(math.log1p(math.exp(-3.0)))
    assert good < math.log(2) < bad
    chosen = torch.zeros(1, requires_grad=True)
    rejected = torch.zeros(1, requires_grad=True)
    bradley_terry_loss(chosen, rejected).backward()
    assert chosen.grad is not None and rejected.grad is not None
    assert chosen.grad.item() < 0 < rejected.grad.item()  # descent raises chosen, lowers rejected


def test_gae_matches_hand_computation() -> None:
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.5, 0.4, 0.3]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    adv, ret = gae(rewards, values, mask, gamma=1.0, lam=0.95)
    assert torch.allclose(adv, torch.tensor([[0.43675, 0.565, 0.7]]), atol=1e-6)
    assert torch.allclose(ret, torch.tensor([[0.93675, 0.965, 1.0]]), atol=1e-6)
    adv, _ = gae(rewards, values, mask, gamma=0.9, lam=0.95)
    assert torch.allclose(adv, torch.tensor([[0.2605675, 0.4685, 0.7]]), atol=1e-6)


def test_gae_ignores_masked_steps() -> None:
    rewards = torch.tensor([[0.0, 0.0, 1.0, 7.0]])
    values = torch.tensor([[0.5, 0.4, 0.3, 5.0]])
    mask = torch.tensor([[True, True, True, False]])
    adv, ret = gae(rewards, values, mask, gamma=1.0, lam=0.95)
    assert torch.allclose(adv, torch.tensor([[0.43675, 0.565, 0.7, 0.0]]), atol=1e-6)
    assert ret[0, 3].item() == 0.0


def test_whiten_uses_only_masked_entries() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
    mask = torch.tensor([[True, True, True, False]])
    w = whiten(x, mask)
    assert w[0, :3].tolist() == pytest.approx([-1.2247, 0.0, 1.2247], abs=1e-3)
    assert w[0, 3].item() == 0.0


def _policy_loss_and_grad(ratio: float, adv: float) -> tuple[float, float]:
    logp = torch.tensor([math.log(ratio)], requires_grad=True)
    mask = torch.ones(1, dtype=torch.bool)
    loss = ppo_policy_loss(logp, torch.zeros(1), torch.tensor([adv]), mask, clip=0.2)
    loss.backward()
    assert logp.grad is not None
    return loss.item(), logp.grad.item()


def test_ppo_clip_kills_gradient_only_in_the_advantage_direction() -> None:
    # A > 0, ratio above 1 + eps: clipped, the loss is flat at -(1.2 * A)
    assert _policy_loss_and_grad(2.0, 1.0) == pytest.approx((-1.2, 0.0), abs=1e-5)
    # A > 0, ratio below 1 - eps: not clipped, the gradient pushes the ratio back up
    assert _policy_loss_and_grad(0.5, 1.0) == pytest.approx((-0.5, -0.5), abs=1e-5)
    # A < 0, ratio below 1 - eps: clipped
    assert _policy_loss_and_grad(0.5, -1.0) == pytest.approx((0.8, 0.0), abs=1e-5)
    # A < 0, ratio above 1 + eps: not clipped, the gradient pushes the ratio back down
    assert _policy_loss_and_grad(2.0, -1.0) == pytest.approx((2.0, 2.0), abs=1e-5)
    # inside the trust region: plain policy gradient
    assert _policy_loss_and_grad(1.1, 1.0) == pytest.approx((-1.1, -1.1), abs=1e-5)


def _value_loss_and_grad(value: float) -> tuple[float, float]:
    v = torch.tensor([value], requires_grad=True)
    mask = torch.ones(1, dtype=torch.bool)
    loss = ppo_value_loss(v, torch.zeros(1), torch.ones(1), mask, clip=0.2)
    loss.backward()
    assert v.grad is not None
    return loss.item(), v.grad.item()


def test_ppo_value_clip_limits_the_move_per_update() -> None:
    # old value 0, return 1. Moved 0.1 (inside the clip): ordinary squared error
    assert _value_loss_and_grad(0.1) == pytest.approx((0.405, -0.9), abs=1e-5)
    # moved 0.9, past the clip: the clipped (worse) term wins and carries no gradient
    assert _value_loss_and_grad(0.9) == pytest.approx((0.32, 0.0), abs=1e-5)
    # moved away from the return: the unclipped term wins and the gradient stays
    assert _value_loss_and_grad(-0.5) == pytest.approx((1.125, -1.5), abs=1e-5)


def test_kl_penalty_is_per_token_with_the_right_sign() -> None:
    mask = torch.tensor([[False, True, True, False], [False, False, True, True]])
    ref = torch.full((2, 4), -2.0)
    scores = torch.tensor([5.0, 7.0])
    above = shape_rewards(ref + 1.0, ref, mask, scores, beta=0.1)  # policy more likely than ref
    expected = torch.tensor([[0.0, -0.1, 4.9, 0.0], [0.0, 0.0, -0.1, 6.9]])
    assert torch.allclose(above, expected, atol=1e-6)
    below = shape_rewards(ref - 1.0, ref, mask, scores, beta=0.1)  # less likely: a bonus
    assert torch.allclose(below, torch.tensor([[0.0, 0.1, 5.1, 0.0], [0.0, 0.0, 0.1, 7.1]]))
    same = shape_rewards(ref, ref, mask, scores, beta=0.1)
    assert torch.allclose(same, torch.tensor([[0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 7.0]]))


def test_reward_model_pools_the_last_non_pad_token() -> None:
    torch.manual_seed(0)
    pad = 257
    rm = new_reward_model(GPTModel(TINY), pad_id=pad).eval()
    a, b = [10, 11, 12, 13, 14, 15, 16], [20, 21, 22]
    with torch.no_grad():
        alone_a = reward_scores(rm, torch.tensor([a]))
        alone_b = reward_scores(rm, torch.tensor([b]))
        batch = reward_scores(rm, pad_rows([a, b], pad))
        wide = reward_scores(rm, pad_rows([b, list(range(13))], pad)[:1])  # b with 10 pad tokens
    assert not torch.allclose(alone_a, alone_b)
    assert torch.allclose(batch, torch.cat([alone_a, alone_b]), atol=1e-5)
    assert torch.allclose(wide, alone_b, atol=1e-5)


def test_strict_pair_skips_ties_and_picks_the_extremes() -> None:
    rng = random.Random(0)
    assert strict_pair([1.0, 1.0, 1.0], rng) is None
    for _ in range(10):
        pick = strict_pair([2.0, 0.0, 2.0, 1.0, 0.0], rng)
        assert pick is not None
        assert pick[0] in (0, 2) and pick[1] in (1, 4)


def test_overopt_summary_finds_the_gold_peak() -> None:
    history = [
        {"reward": float(i), "gold_accuracy": g} for i, g in enumerate([0.2, 0.4, 0.9, 0.3, 0.1])
    ]
    s = overopt_summary(history, window=2)
    assert s["reward_first"] == pytest.approx(0.5)
    assert s["reward_last"] == pytest.approx(3.5)
    assert s["gold_peak"] == pytest.approx(0.65)
    assert s["gold_peak_step"] == 3.0
    assert s["gold_last"] == pytest.approx(0.2)


def test_train_reward_model_holds_out_a_fifth_of_the_pairs() -> None:
    torch.manual_seed(0)
    rng = random.Random(0)

    def ids(n: int) -> list[int]:
        return [rng.randrange(256) for _ in range(n)]

    pairs = [RankedPair(ids(4), ids(3), ids(5)) for _ in range(10)]
    rm = new_reward_model(GPTModel(TINY), pad_id=257)
    res = train_reward_model(rm, pairs, epochs=2, batch_size=4, device=CPU)
    assert len(res["history"]) == 2
    assert res["val_accuracy"] in (0.0, 0.5, 1.0)  # 2 held-out pairs


def test_train_ppo_starts_at_the_reference_and_updates_the_policy() -> None:
    torch.manual_seed(0)
    tok = reasoning_tokenizer()
    policy = GPTModel(TINY)
    ref = copy.deepcopy(policy)
    rm = new_reward_model(GPTModel(TINY), tok.special_to_id["<|pad|>"])
    res = train_ppo(
        policy,
        ref,
        rm,
        tok,
        steps=2,
        tasks_per_step=2,
        samples_per_task=2,
        max_digits=1,
        max_new_tokens=6,
        minibatch_size=2,
        lr=1e-3,
        device=CPU,
    )
    history = res["history"]
    assert len(history) == 2
    assert history[0]["kl"] == pytest.approx(0.0, abs=1e-5)
    assert all(math.isfinite(v) for row in history for v in row.values())
    assert not torch.equal(policy.wte.weight, ref.wte.weight)
    assert not any(p.requires_grad for p in ref.parameters())

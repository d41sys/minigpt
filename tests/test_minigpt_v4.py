"""DeepSeek-V4 toy options: defaults unchanged, MoE, MTP, mHC, CSA, generate, post-training."""

from dataclasses import asdict
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from minigpt.model import (
    GPTConfig,
    GPTModel,
    HyperConnection,
    MoEFeedForward,
    count_parameters,
    sinkhorn,
)
from minigpt.reason import per_token_log_probs

BASE: dict[str, Any] = dict(
    vocab_size=64, context_length=16, d_model=32, num_layers=2, num_heads=4, d_ff=64
)
MOE: dict[str, Any] = dict(n_experts=4, moe_top_k=2)
MTP: dict[str, Any] = dict(mtp_depth=1)
HC: dict[str, Any] = dict(hc_streams=4)
CSA: dict[str, Any] = dict(attn_type="csa", csa_block=2, csa_topk=2, csa_window=3)
ALL = {**MOE, **MTP, **HC, **CSA, "moe_dense_first": True}


def cfg(**kw: Any) -> GPTConfig:
    return GPTConfig(**{**BASE, **kw})


@pytest.mark.parametrize(
    ("pos_encoding", "total", "last", "mid"),
    [
        # Recorded from the pre-V4 model (seed 0, TINY below): the defaults must not drift.
        ("learned", 19.146499, [0.134765, -0.028528, 0.206829, 0.118018], [0.133016, 0.218384]),
        ("rope", 15.838773, [0.154625, -0.120244, -0.168108, 0.037300], [-0.038776, 0.032641]),
    ],
)
def test_default_config_matches_pre_v4_logits(
    pos_encoding: str, total: float, last: list[float], mid: list[float]
) -> None:
    torch.manual_seed(0)
    tiny = GPTConfig(
        vocab_size=64,
        context_length=16,
        d_model=32,
        num_layers=2,
        num_heads=4,
        pos_encoding=pos_encoding,
    )
    model = GPTModel(tiny).eval()
    logits, loss = model(torch.arange(24).view(2, 12) % 64)
    assert loss is None
    assert logits.double().sum().item() == pytest.approx(total, abs=1e-3)
    assert logits[0, -1, :4].tolist() == pytest.approx(last, abs=1e-4)
    assert logits[1, 3, 10:12].tolist() == pytest.approx(mid, abs=1e-4)
    assert not any(k.split(".")[0] == "mtp" or "route_bias" in k for k in model.state_dict())


def test_old_config_dict_and_state_dict_still_load() -> None:
    old = {k: v for k, v in asdict(cfg()).items() if k in GPTConfig.__dataclass_fields__}
    for field in ("n_experts", "mtp_depth", "hc_streams", "attn_type", "csa_block"):
        old.pop(field)  # a checkpoint written before the V4 options existed
    model = GPTModel(GPTConfig(**old))
    GPTModel(cfg()).load_state_dict(model.state_dict())


def test_moe_picks_top_k_and_bias_moves_load_toward_balance() -> None:
    torch.manual_seed(0)
    moe = MoEFeedForward(cfg(**{**MOE, "n_experts": 8}, moe_bias_rate=0.05)).train()
    x = torch.randn(8, 32, 32)
    flat = x.view(-1, 32)

    moe(x)
    moe.route_bias.zero_()
    affinity = F.softplus(moe.router(flat)).sqrt()
    expected = F.one_hot(affinity.topk(2, dim=-1).indices, 8).sum(dim=(0, 1))
    moe(x)
    assert moe.last_load.long().tolist() == expected.tolist()
    assert moe.last_load.sum() == flat.size(0) * 2  # exactly k experts per token

    moe.route_bias.zero_()
    moe.route_bias[0] = 1.0  # expert 0 wins every token
    moe(x)
    assert moe.last_load[0] == flat.size(0)
    spread = [(moe.last_load.max() - moe.last_load.min()).item()]
    for _ in range(100):
        moe(x)
    spread.append((moe.last_load.max() - moe.last_load.min()).item())
    assert spread[1] < spread[0] / 2

    moe.eval()
    bias = moe.route_bias.clone()
    moe(x)
    assert torch.equal(moe.route_bias, bias)  # no bias update outside training


def test_moe_loss_includes_balance_term_only_in_training() -> None:
    torch.manual_seed(0)
    model = GPTModel(cfg(**MOE)).train()
    idx = torch.randint(0, 64, (2, 10))
    logits, loss = model(idx, idx)
    ce = F.cross_entropy(logits.view(-1, 64), idx.view(-1))
    assert loss is not None
    assert model.aux_losses["balance"] > 0
    assert torch.allclose(loss, ce + model.aux_losses["balance"])
    assert model.blocks[0].mlp.balance_loss.requires_grad
    model.eval()
    logits, loss = model(idx, idx)
    assert loss is not None
    assert torch.allclose(loss, F.cross_entropy(logits.view(-1, 64), idx.view(-1)))


def test_moe_dense_first_layer() -> None:
    model = GPTModel(cfg(**MOE, moe_dense_first=True))
    assert not isinstance(model.blocks[0].mlp, MoEFeedForward)
    assert isinstance(model.blocks[1].mlp, MoEFeedForward)


def test_active_parameters_exclude_idle_experts() -> None:
    dense, moe = GPTModel(cfg()), GPTModel(cfg(**MOE))
    assert count_parameters(dense, active=True) == count_parameters(dense)
    assert count_parameters(moe, active=True) < count_parameters(moe)
    expert = sum(p.numel() for p in moe.blocks[0].mlp.experts[0].parameters())
    assert count_parameters(moe) - count_parameters(moe, active=True) == 2 * 2 * expert


def test_mtp_loss_uses_targets_shifted_by_one_more() -> None:
    torch.manual_seed(0)
    model = GPTModel(cfg(**MTP, mtp_weight=0.5)).train()
    idx = torch.randint(0, 64, (2, 10))
    targets = torch.randint(0, 64, (2, 10))

    logits, loss = model(idx, targets)
    mtp = model.aux_losses["mtp"]
    ce = F.cross_entropy(logits.view(-1, 64), targets.view(-1))
    assert loss is not None
    assert torch.allclose(loss, ce + mtp)

    first = targets.clone()
    first[:, 0] = (first[:, 0] + 1) % 64  # token t+1 of position 0: no MTP target there
    model(idx, first)
    assert torch.equal(model.aux_losses["mtp"], mtp)
    last = targets.clone()
    last[:, -1] = (last[:, -1] + 1) % 64  # token t+2 of position T-2
    model(idx, last)
    assert not torch.equal(model.aux_losses["mtp"], mtp)

    model.eval()  # off outside training
    logits, loss = model(idx, targets)
    assert loss is not None
    assert torch.allclose(loss, F.cross_entropy(logits.view(-1, 64), targets.view(-1)))


def test_sinkhorn_is_doubly_stochastic() -> None:
    torch.manual_seed(0)
    m = sinkhorn(torch.randn(3, 5, 4, 4), 20)
    ones = torch.ones(3, 5, 4)
    assert (m >= 0).all()
    assert torch.allclose(m.sum(dim=-1), ones, atol=1e-5)
    assert torch.allclose(m.sum(dim=-2), ones, atol=1e-3)

    a, b, c = HyperConnection(32, 4, 20).maps(torch.randn(2, 3, 4, 32))
    assert torch.allclose(b.sum(dim=-1), torch.ones(2, 3, 4), atol=1e-3)
    assert torch.allclose(b.sum(dim=-2), torch.ones(2, 3, 4), atol=1e-3)
    assert ((a > 0) & (a < 1)).all() and ((c > 0) & (c < 2)).all()


@pytest.mark.parametrize("pos_encoding", ["learned", "rope"])
def test_csa_is_causal(pos_encoding: str) -> None:
    torch.manual_seed(0)
    model = GPTModel(cfg(**CSA, pos_encoding=pos_encoding)).eval()
    a = torch.randint(0, 64, (1, 14))
    b = a.clone()
    b[0, 9:] = torch.randint(0, 64, (5,))
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[0, :9], lb[0, :9], atol=1e-5)
    assert not torch.allclose(la[0, 9:], lb[0, 9:])


def test_csa_indexer_gets_gradient_from_lm_loss() -> None:
    torch.manual_seed(0)
    model = GPTModel(cfg(**CSA)).train()
    idx = torch.randint(0, 64, (2, 14))
    _, loss = model(idx, idx)
    assert loss is not None
    loss.backward()
    attn = model.blocks[0].attn
    assert attn.index_q.weight.grad is not None and attn.index_q.weight.grad.abs().sum() > 0
    assert attn.comp.kv.weight.grad is not None and attn.comp.kv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "extra",
    [MOE, MTP, HC, CSA, ALL, {**ALL, "pos_encoding": "rope"}],
    ids=["moe", "mtp", "hc", "csa", "all", "all-rope"],
)
def test_generate_with_every_option(extra: dict[str, Any]) -> None:
    torch.manual_seed(0)
    model = GPTModel(cfg(**extra)).train()  # generate() must switch to eval itself
    idx = torch.randint(0, 64, (1, 4))
    cached = model.generate(idx, 16, temperature=0.0)  # 20 tokens > context: slides past 16
    assert cached.shape == (1, 20)
    assert torch.equal(cached, model.generate(idx, 16, temperature=0.0, use_cache=False))


def test_v4_model_works_with_post_training_helpers() -> None:
    torch.manual_seed(0)
    dense, v4 = GPTModel(cfg()), GPTModel(cfg(**ALL))
    x = torch.randint(0, 64, (2, 8))
    y = x.clone()
    y[:, :3] = -100
    for model in (dense, v4):
        logps, mask = per_token_log_probs(model, x, y)
        assert logps.shape == (2, 8) and mask.shape == (2, 8)
        assert model(x)[0].shape == (2, 8, 64)  # MTP never leaks into the logits
        assert model.generate(x, 3).shape == (2, 11)
    assert set(v4(x, y)[1].shape) == set()  # scalar loss

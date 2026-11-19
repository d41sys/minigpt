"""Muon (DeepSeek-V4 report section 2.4): Newton-Schulz, parameter grouping, one step, state."""

from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from minigpt.data import create_dataloader
from minigpt.model import GPTConfig, GPTModel
from minigpt.train import (
    Muon,
    MuonAdamW,
    configure_optimizers,
    load_checkpoint,
    newton_schulz,
    train,
)

CPU = torch.device("cpu")
TINY = GPTConfig(vocab_size=50, context_length=8, d_model=16, num_layers=2, num_heads=2)


@pytest.mark.parametrize("shape", [(16, 32), (32, 16)])
def test_newton_schulz_is_near_orthogonal(shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    s = torch.linalg.svdvals(newton_schulz(torch.randn(shape)))
    assert s.min() > 0.9 and s.max() < 1.1


def test_update_rms_matches_the_target() -> None:
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.zeros(16, 32))
    w.grad = torch.randn(16, 32)
    Muon([w], lr=1.0, weight_decay=0.0, rms=0.18).step()
    assert w.pow(2).mean().sqrt().item() == pytest.approx(0.18, rel=0.1)


def test_grouping_puts_hidden_matrices_on_muon() -> None:
    model = GPTModel(TINY)
    opt = configure_optimizers(model, optim="muon")
    assert isinstance(opt, MuonAdamW)
    names = {id(p): n for n, p in model.named_parameters()}
    muon = {names[id(p)] for g in opt.inner["muon"].param_groups for p in g["params"]}
    adamw = {names[id(p)] for g in opt.inner["adamw"].param_groups for p in g["params"]}
    hidden = {n for n, p in model.named_parameters() if n.startswith("blocks") and p.dim() == 2}
    assert muon == hidden and len(hidden) == 4 * TINY.num_layers
    assert muon.isdisjoint(adamw) and muon | adamw == set(names.values())
    assert {"wte.weight", "wpe.weight", "ln_f.weight", "blocks.0.attn.c_attn.bias"} <= adamw
    assert model.lm_head.weight is model.wte.weight  # the tied head is the embedding: AdamW


def test_router_and_hc_params_stay_on_adamw() -> None:
    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])
            self.router = torch.nn.Linear(4, 2, bias=False)
            self.hc_mix = torch.nn.Parameter(torch.zeros(4, 4))
            self.balance_bias = torch.nn.Parameter(torch.zeros(2))

    model = Toy()
    opt = configure_optimizers(model, optim="muon")
    assert isinstance(opt, MuonAdamW)
    muon = [p for g in opt.inner["muon"].param_groups for p in g["params"]]
    assert {id(p) for p in muon} == {id(e.weight) for e in model.experts}


def test_default_is_plain_adamw() -> None:
    assert type(configure_optimizers(GPTModel(TINY))) is AdamW
    with pytest.raises(ValueError, match="unknown optimizer"):
        configure_optimizers(GPTModel(TINY), optim="sgd")


def test_one_muon_step_lowers_the_loss_on_a_fixed_batch() -> None:
    torch.manual_seed(0)
    model = GPTModel(TINY)
    x = torch.randint(0, 50, (4, 8))
    opt = configure_optimizers(model, learning_rate=1e-2, optim="muon")
    _, before = model(x, targets=x)
    assert before is not None
    before.backward()
    opt.step()
    _, after = model(x, targets=x)
    assert after is not None and after.item() < before.item()


def test_muon_state_round_trips_through_the_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    ids = torch.randint(0, 50, (8 * 8 + 1,)).tolist()
    loader = create_dataloader(ids, 8, batch_size=2, shuffle=False)
    model = GPTModel(TINY)
    res = train(model, loader, None, max_steps=2, checkpoint_dir=tmp_path, device=CPU, optim="muon")
    fresh = GPTModel(TINY)
    opt = configure_optimizers(fresh, optim="muon")
    load_checkpoint(res["checkpoint_path"], fresh, opt)
    bufs = [s["momentum_buffer"] for s in opt.inner["muon"].state.values()]
    assert len(bufs) == 4 * TINY.num_layers and all(b.abs().sum() > 0 for b in bufs)

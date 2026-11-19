import copy

import pytest
import torch
import torch.nn.functional as F

from minigpt.model import GPTConfig, GPTModel
from minigpt.opd import token_pg_loss, train_opd
from minigpt.reason import build_batch

CFG = GPTConfig(vocab_size=258, context_length=64, d_model=32, num_layers=2, num_heads=4)


def _model(seed: int) -> GPTModel:
    torch.manual_seed(seed)
    return GPTModel(CFG).eval()


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    return build_batch([[1, 2, 3], [4, 5]], [[8, 9], [10, 11, 12]], pad_id=257)


def test_token_pg_loss_is_zero_for_an_identical_teacher() -> None:
    student = _model(0)
    x, y = _batch()
    assert token_pg_loss(student, copy.deepcopy(student), x, y).item() == pytest.approx(0.0)


def test_token_pg_loss_matches_clipped_log_ratio_times_logprob() -> None:
    student, teacher = _model(0), _model(1)
    x, y = _batch()
    mask = y != -100
    lp_s = F.log_softmax(student(x)[0], -1).gather(2, y.clamp(min=0).unsqueeze(2)).squeeze(2)
    lp_t = F.log_softmax(teacher(x)[0], -1).gather(2, y.clamp(min=0).unsqueeze(2)).squeeze(2)
    for r_max in (5.0, 0.1):
        reward = (lp_t - lp_s).detach().clamp(-r_max, r_max)
        want = -(reward * lp_s)[mask].sum() / mask.sum()
        assert token_pg_loss(student, teacher, x, y, r_max).item() == pytest.approx(want.item())
    # The tight clip really bites: |reward| <= 0.1 everywhere.
    assert (lp_t - lp_s).detach().abs().max() > 0.1


def test_token_pg_gradient_treats_the_reward_as_a_constant() -> None:
    student, teacher = _model(0), _model(1)
    x, y = _batch()
    mask = y != -100
    token_pg_loss(student, teacher, x, y).backward()  # type: ignore[no-untyped-call]
    assert student.wte.weight.grad is not None
    got = student.wte.weight.grad.clone()
    student.zero_grad()
    lp_s = F.log_softmax(student(x)[0], -1).gather(2, y.clamp(min=0).unsqueeze(2)).squeeze(2)
    lp_t = F.log_softmax(teacher(x)[0], -1).gather(2, y.clamp(min=0).unsqueeze(2)).squeeze(2)
    reward = (lp_t - lp_s).detach().clamp(-5.0, 5.0)
    (-(reward * lp_s)[mask].sum() / mask.sum()).backward()  # type: ignore[no-untyped-call]
    want = student.wte.weight.grad
    assert want is not None and torch.allclose(got, want, atol=1e-6)


def test_train_opd_with_the_token_estimator_runs() -> None:
    res = train_opd(
        _model(2),
        _model(3),
        steps=2,
        batch_size=2,
        digits=1,
        max_new_tokens=8,
        eval_every=1,
        eval_tasks=4,
        device=torch.device("cpu"),
        estimator="token",
    )
    assert len(res["history"]) == 3 and all("kl" in r for r in res["history"])

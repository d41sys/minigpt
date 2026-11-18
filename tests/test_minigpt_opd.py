import copy
import math

import pytest
import torch

from minigpt.model import GPTConfig, GPTModel
from minigpt.opd import KLMode, opd_loss, token_kl, train_opd
from minigpt.reason import build_batch

TINY = GPTConfig(vocab_size=258, context_length=64, d_model=32, num_layers=2, num_heads=4)
SMALL = GPTConfig(vocab_size=258, context_length=64, d_model=16, num_layers=1, num_heads=2)


def _model(cfg: GPTConfig, seed: int) -> GPTModel:
    torch.manual_seed(seed)
    return GPTModel(cfg).eval()


def _perturbed(model: GPTModel, std: float, seed: int) -> GPTModel:
    other = copy.deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in other.parameters():
            p.add_(torch.randn_like(p) * std)
    return other


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    return build_batch([[1, 2, 3], [4, 5]], [[6, 7, 8], [9, 10, 11, 12, 13]], pad_id=257)


def test_reverse_kl_is_zero_for_identical_models_and_positive_otherwise() -> None:
    teacher = _model(TINY, 0)
    x, y = _batch()
    assert opd_loss(copy.deepcopy(teacher), teacher, x, y).abs().item() < 1e-6
    assert opd_loss(_perturbed(teacher, 0.3, 1), teacher, x, y).item() > 1e-3


def test_reverse_and_forward_kl_match_hand_computation() -> None:
    mask = torch.ones(1, 1, dtype=torch.bool)

    def kl(p_s: list[float], p_t: list[float], mode: KLMode) -> float:
        s = torch.tensor(p_s).log().view(1, 1, -1)
        t = torch.tensor(p_t).log().view(1, 1, -1)
        return token_kl(s, t, mask, mode).item()

    # student concentrates on one of the teacher's two equal modes
    p_s, p_t = [0.99, 0.01], [0.5, 0.5]
    rev = 0.99 * math.log(0.99 / 0.5) + 0.01 * math.log(0.01 / 0.5)
    fwd = 0.5 * math.log(0.5 / 0.99) + 0.5 * math.log(0.5 / 0.01)
    assert kl(p_s, p_t, "reverse") == pytest.approx(rev, abs=1e-5)
    assert kl(p_s, p_t, "forward") == pytest.approx(fwd, abs=1e-5)
    # mode-seeking: committing to one mode is cheap under reverse KL, dear under forward KL
    assert rev < fwd
    # swapping the arguments swaps the two directions
    assert kl(p_t, p_s, "forward") == pytest.approx(rev, abs=1e-5)
    with pytest.raises(ValueError):
        token_kl(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), mask, "sideways")  # type: ignore


def test_masked_positions_do_not_affect_the_loss() -> None:
    torch.manual_seed(0)
    s = torch.randn(2, 5, 11, requires_grad=True)
    t = torch.randn(2, 5, 11)
    mask = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 1, 1]], dtype=torch.bool)
    base = token_kl(s, t, mask)
    t_garbage = torch.where(mask.unsqueeze(-1), t, torch.randn(2, 5, 11) * 50)
    assert token_kl(s, t_garbage, mask).item() == pytest.approx(base.item(), abs=1e-6)
    base.backward()
    assert s.grad is not None
    assert s.grad[~mask].abs().max().item() == 0.0

    # model level: a row's loss must not depend on what it is padded or batched with
    teacher, student = _model(TINY, 0), _model(SMALL, 1)
    prompts, comps = [[1, 2, 3], [4, 5]], [[6, 7, 8], [9, 10, 11, 12, 13]]
    x, y = build_batch(prompts, comps, pad_id=257)
    batched = opd_loss(student, teacher, x, y).item()
    singles = [
        (opd_loss(student, teacher, *build_batch([p], [c], 257)).item(), len(c))
        for p, c in zip(prompts, comps, strict=True)
    ]
    expected = sum(v * n for v, n in singles) / sum(n for _, n in singles)
    assert batched == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("mode", ["reverse", "forward"])
def test_gradient_step_lowers_kl_for_a_smaller_student(mode: KLMode) -> None:
    teacher = _perturbed(_model(TINY, 0), 0.3, 1)
    student = _model(SMALL, 2)
    x, y = _batch()
    before = opd_loss(student, teacher, x, y, mode)
    before.backward()
    params = [p for p in student.parameters() if p.grad is not None]
    norm = torch.sqrt(sum((p.grad**2).sum() for p in params))
    with torch.no_grad():
        for p in params:
            p -= 3e-3 * p.grad / norm
    after = opd_loss(student, teacher, x, y, mode)
    assert after.item() < before.item()


@pytest.mark.parametrize("off_policy", [False, True])
def test_train_opd_runs_and_leaves_the_teacher_untouched(off_policy: bool) -> None:
    teacher, student = _model(TINY, 0), _model(SMALL, 1)
    frozen = {k: v.clone() for k, v in teacher.state_dict().items()}
    res = train_opd(
        student,
        teacher,
        steps=2,
        batch_size=2,
        digits=1,
        max_new_tokens=8,
        eval_every=1,
        eval_tasks=3,
        off_policy=off_policy,
        device=torch.device("cpu"),
    )
    assert [row["step"] for row in res["history"]] == [0.0, 1.0, 2.0]
    assert all({"loss", "kl", "accuracy", "mean_len"} <= row.keys() for row in res["history"])
    assert all(1.0 <= row["mean_len"] <= 8.0 for row in res["history"])
    assert all(torch.equal(frozen[k], v) for k, v in teacher.state_dict().items())
    assert not any(p.requires_grad for p in teacher.parameters())

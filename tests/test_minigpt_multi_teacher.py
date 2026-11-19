import copy
import random
import re

import pytest
import torch

from minigpt.model import GPTConfig, GPTModel
from minigpt.opd import opd_loss, routed_opd_loss, train_opd
from minigpt.reason import Task, build_batch, equation_corpus, make_task, sample_task

TINY = GPTConfig(vocab_size=258, context_length=64, d_model=32, num_layers=2, num_heads=4)
SMALL = GPTConfig(vocab_size=258, context_length=64, d_model=16, num_layers=1, num_heads=2)


def _model(cfg: GPTConfig, seed: int) -> GPTModel:
    torch.manual_seed(seed)
    return GPTModel(cfg).eval()


def test_subtraction_trace_borrows_and_answers_correctly() -> None:
    assert Task(52, 38, "-").solution() == "<think>2-8-0=-6 5-3-1=1</think><answer>14</answer>"
    assert Task(52, 38, "-").prompt == "52-38="
    assert Task(1000, 1, "-").solution() == (
        "<think>0-1-0=-1 0-0-1=-1 0-0-1=-1 1-0-1=0</think><answer>999</answer>"
    )
    rng = random.Random(0)
    for _ in range(200):
        t = make_task(rng, rng.randint(1, 5), "-")
        assert t.a >= t.b and int(t.answer) == t.a - t.b
        steps = re.findall(r"(\d)-(\d)-(\d)=(-?\d+)", t.solution())
        borrow, digits = 0, ""
        for x, y, b, s in steps:
            assert int(b) == borrow and int(s) == int(x) - int(y) - borrow
            digits, borrow = str(int(s) % 10) + digits, int(int(s) < 0)
        assert borrow == 0 and int(digits) == t.a - t.b


def test_addition_sampling_is_unchanged_by_the_op_argument() -> None:
    a, b = random.Random(3), random.Random(3)
    assert [make_task(a, 3) for _ in range(5)] == [sample_task(b, 3, "+") for _ in range(5)]
    lines = equation_corpus(random.Random(0), 200, 3, ("+", "-")).splitlines()
    assert {"+", "-"} == {"+" if "+" in line else "-" for line in lines}


def test_routed_loss_scores_each_row_against_its_ops_teacher() -> None:
    student = _model(TINY, 0)
    teachers = {"+": copy.deepcopy(student), "-": _model(TINY, 1)}
    x, y = build_batch([[1, 2, 3], [4, 5], [6, 7]], [[8, 9], [10, 11, 12], [13]], pad_id=257)
    loss, by_op = routed_opd_loss(student, teachers, x, y, ["+", "-", "+"])
    assert by_op["+"] < 1e-6  # the student equals the "+" teacher only
    assert by_op["-"] > 1e-3
    assert by_op["-"] == pytest.approx(opd_loss(student, teachers["-"], x[1:2], y[1:2]).item())
    # Weights are completion-token shares: 3 tokens for "+" (rows 0, 2), 3 for "-" (row 1).
    assert loss.item() == pytest.approx(by_op["-"] * 3 / 6, abs=1e-5)


@pytest.mark.parametrize("off_policy", [False, True])
def test_train_opd_with_two_teachers_reports_per_op_metrics(off_policy: bool) -> None:
    teachers = {"+": _model(TINY, 0), "-": _model(SMALL, 1)}  # teachers may differ in size
    res = train_opd(
        _model(SMALL, 2),
        teachers,
        steps=2,
        batch_size=4,
        digits=1,
        max_new_tokens=8,
        eval_every=1,
        eval_tasks=16,
        off_policy=off_policy,
        device=torch.device("cpu"),
        ops=("+", "-"),
    )
    rows = res["history"]
    assert all({"kl[+]", "kl[-]", "accuracy[+]", "accuracy[-]"} <= r.keys() for r in rows)
    with pytest.raises(ValueError, match="no teacher"):
        train_opd(_model(SMALL, 2), {"+": teachers["+"]}, steps=1, ops=("+", "-"))

import math
from pathlib import Path

import pytest
import torch

from minigpt.model import GPTConfig, GPTModel
from minigpt.prm import (
    best_of_n,
    build_example,
    cached_samples,
    majority_index,
    parse_steps,
    prm_shaped_reward,
    score_samples,
    step_labels,
    train_verifier,
)
from minigpt.reason import Task, reasoning_tokenizer
from minigpt.rft import Sampled
from minigpt.rlhf import new_reward_model

TASK = Task(12, 34)  # answer 46
GOOD = "<think>2+4+0=6 1+3+0=4</think><answer>46</answer>"
BAD_STEP = "<think>2+4+0=7 1+3+0=4</think><answer>46</answer>"  # first step wrong, answer right
BAD_ANSWER = "<think>2+4+0=6 1+3+0=4</think><answer>45</answer>"
SHORT = "<think>2+4+0=6</think><answer>46</answer>"
MALFORMED = "<answer>46</answer>"
TINY = GPTConfig(vocab_size=258, context_length=64, d_model=32, num_layers=2, num_heads=4)


def sampled(texts: list[str], task: Task = TASK) -> Sampled:
    tok = reasoning_tokenizer()
    ids = [[*tok.encode(t), tok.eos_token_id] for t in texts]
    return Sampled(task, tok.encode(task.prompt), ids, texts)


def test_parse_steps_gives_the_last_char_of_each_step() -> None:
    steps = parse_steps(GOOD)
    assert steps is not None
    assert [GOOD[i] for i, _ in steps] == ["6", "4", "6"]
    assert [t for _, t in steps] == ["2+4+0=6", "1+3+0=4", "46"]
    assert parse_steps(MALFORMED) is None


@pytest.mark.parametrize(
    ("text", "labels"),
    [
        (GOOD, [1, 1, 1]),
        (BAD_STEP, [0]),  # labels stop at the first error, even though the answer is right
        (BAD_ANSWER, [1, 1, 0]),
        (SHORT, [1, 0]),  # a missing step makes the answer step wrong
        (MALFORMED, []),
    ],
)
def test_step_labels_stop_at_the_first_error(text: str, labels: list[int]) -> None:
    assert step_labels(TASK, text) == labels


def test_subtraction_steps_are_labeled_against_the_reference() -> None:
    t = Task(52, 17, "-")
    assert step_labels(t, t.solution()) == [1, 1, 1]


def test_example_positions_point_at_step_ends() -> None:
    tok = reasoning_tokenizer()
    s = sampled([GOOD, BAD_STEP])
    good, bad = build_example(tok, s, 0), build_example(tok, s, 1)
    assert [tok.decode([good.ids[p]]) for p in good.positions] == ["6", "4", "6"]
    assert good.labels == [1, 1, 1] and good.outcome == 1
    assert bad.positions == good.positions[:1] and bad.labels == [0] and bad.outcome == 1
    scoring = build_example(tok, s, 1, labeled=False)
    assert len(scoring.positions) == 3 and scoring.labels == []


def test_text_that_is_not_one_char_per_token_has_no_positions() -> None:
    tok = reasoning_tokenizer()
    s = sampled([GOOD.replace("<think>", "<think>é")])  # two byte tokens, one character
    assert parse_steps(s.texts[0]) is not None
    assert build_example(tok, s, 0).positions == []


def test_majority_breaks_ties_toward_the_earliest_answer() -> None:
    assert majority_index([BAD_ANSWER, GOOD, GOOD, BAD_ANSWER]) == 0
    assert majority_index([BAD_ANSWER, GOOD, GOOD]) == 1
    assert majority_index([MALFORMED.replace("46", "x")]) == 0


def test_best_of_n_selectors() -> None:
    s = sampled([BAD_ANSWER, GOOD, BAD_ANSWER, BAD_ANSWER])
    scores = {"orm": [[0.0, 2.0, 1.0, 1.0]], "prm_min": [[3.0, 1.0, 2.0, 2.0]]}
    res = best_of_n([s], scores, [1, 2, 4])
    acc = {k: [v[n][2] for n in (1, 2, 4)] for k, v in res.items()}
    assert acc["orm"] == [0.0, 1.0, 1.0]  # argmax of the first n
    assert acc["prm_min"] == [0.0, 0.0, 0.0]
    assert acc["majority"] == [0.0, 0.0, 0.0]  # three wrong answers outvote one right one
    assert acc["oracle"] == [0.0, 1.0, 1.0]
    assert acc["random"] == [0.0, 0.5, 0.25]


def test_verifiers_learn_their_labels_and_score_malformed_as_minus_inf() -> None:
    tok = reasoning_tokenizer()
    torch.manual_seed(0)
    s = sampled([GOOD, BAD_STEP, BAD_ANSWER, SHORT])
    examples = [build_example(tok, s, j) for j in range(4)] * 3
    pad = tok.special_to_id["<|pad|>"]
    models = {}
    for kind in ("prm", "orm"):
        rm = new_reward_model(GPTModel(TINY), pad)
        res = train_verifier(rm, examples, kind, epochs=40, batch_size=6, lr=3e-3)
        assert res["history"][-1]["loss"] < 0.5 * res["history"][0]["loss"]
        models[kind] = rm
    bad = sampled([GOOD, MALFORMED])
    out = score_samples(models["prm"], models["orm"], tok, [bad], torch.device("cpu"))
    assert out["prm_min"][0][1] == -math.inf and math.isfinite(out["prm_min"][0][0])
    assert math.isfinite(out["orm"][0][1])
    assert out["prm_prod"][0][0] <= 0.0  # log of a product of probabilities


def test_cached_samples_are_reused_and_checked(tmp_path: Path) -> None:
    tok = reasoning_tokenizer()
    torch.manual_seed(0)
    model = GPTModel(TINY)
    tasks = [Task(12, 34), Task(5, 7)]
    path = tmp_path / "s.json"
    first = cached_samples(path, model, tok, tasks, 3, 8, 1.0)
    again = cached_samples(path, model, tok, tasks, 3, 8, 1.0)
    assert [s.ids for s in first] == [s.ids for s in again]
    assert [s.texts for s in first] == [s.texts for s in again]
    with pytest.raises(ValueError, match="another k"):
        cached_samples(path, model, tok, tasks, 4, 8, 1.0)


def test_prm_shaped_reward_adds_a_probability_for_well_formed_completions() -> None:
    tok = reasoning_tokenizer()
    torch.manual_seed(0)
    rm = new_reward_model(GPTModel(TINY), tok.special_to_id["<|pad|>"])
    reward = prm_shaped_reward(lambda _t, _c: 1.0, rm, tok, 2.0)
    assert reward(TASK, MALFORMED) == 1.0
    scores = score_samples(rm, rm, tok, [sampled([GOOD])], torch.device("cpu"))
    p = math.exp(scores["prm_prod"][0][0])
    assert reward(TASK, GOOD) == pytest.approx(1.0 + 2.0 * p, rel=1e-4)

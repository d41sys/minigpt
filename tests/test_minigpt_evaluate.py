import math

import pytest

from minigpt.evaluate import EvalReport, MockJudge, exact_match, normalize_answer, token_f1


def test_normalize_answer_drops_case_punctuation_articles() -> None:
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("an apple, a pear") == "apple pear"


def test_exact_match_after_normalisation() -> None:
    assert exact_match("Paris.", "paris") == 1.0
    assert exact_match("The Paris", "Paris") == 1.0
    assert exact_match("Paris, France", "Paris") == 0.0


def test_token_f1_hand_computed() -> None:
    # pred: cat sat mat (3)   ref: cat sat on mat today (5)   overlap 3
    # P = 1, R = 3/5, F1 = 2 * 0.6 / 1.6 = 0.75
    assert token_f1("the cat sat mat", "the cat sat on mat today") == pytest.approx(0.75)
    assert token_f1("dog", "cat") == 0.0
    assert token_f1("cat cat", "cat") == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_mock_judge_bounds() -> None:
    judge = MockJudge()
    assert judge.score("q", "", "anything") == 1
    assert judge.score("q", "zzz", "the cat sat") == 1
    assert judge.score("q", "the cat sat", "the cat sat") == 5


def _report(**kw: float) -> EvalReport:
    base = dict(
        perplexity=32.0,
        vocab_size=1024,
        exact_match=0.2,
        exact_match_n=10,
        token_f1=0.4,
        judge_mean=3.0,
        open_n=50,
    )
    base.update(kw)
    return EvalReport(**base)  # type: ignore[arg-type]


def test_perplexity_score_endpoints() -> None:
    assert _report(perplexity=1.0).perplexity_score == 1.0
    assert _report(perplexity=1024.0).perplexity_score == pytest.approx(0.0)
    assert _report(perplexity=32.0).perplexity_score == pytest.approx(0.5)  # log 32 / log 1024
    assert _report(perplexity=math.nan).perplexity_score == 0.0


def test_aggregate_is_weighted_mean() -> None:
    r = _report()  # parts: ppl 0.5, EM 0.2, F1 0.4, judge (3-1)/4 = 0.5
    assert r.aggregate() == pytest.approx((0.5 + 0.2 + 0.4 + 0.5) / 4)
    assert r.aggregate((1.0, 0.0, 0.0, 0.0)) == pytest.approx(0.5)
    assert r.aggregate((0.0, 1.0, 3.0, 0.0)) == pytest.approx((0.2 + 3 * 0.4) / 4)

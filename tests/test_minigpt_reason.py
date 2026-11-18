import random

import torch

from minigpt.dpo import compute_sequence_log_probs
from minigpt.model import GPTConfig, GPTModel
from minigpt.reason import (
    Task,
    accuracy_reward,
    build_batch,
    format_reward,
    make_task,
    parse_answer,
    per_token_log_probs,
    reasoning_tokenizer,
    sample_group,
)

TINY = GPTConfig(vocab_size=258, context_length=64, d_model=32, num_layers=2, num_heads=4)


def test_solution_steps_and_answer_are_correct() -> None:
    assert Task(47, 85).solution() == "<think>7+5+0=12 4+8+1=13</think><answer>132</answer>"
    rng = random.Random(0)
    for _ in range(200):
        t = make_task(rng, rng.randint(1, 5))
        sol = t.solution()
        assert parse_answer(sol) == str(t.a + t.b)
        assert accuracy_reward(t, sol) == 1.0
        assert format_reward(sol) == 1.0


def test_make_task_respects_digit_count() -> None:
    rng = random.Random(1)
    for d in (1, 3, 5):
        t = make_task(rng, d)
        assert len(str(t.a)) <= d and len(str(t.b)) <= d
        if d > 1:
            assert len(str(t.a)) == d and len(str(t.b)) == d


def test_rewards_reject_wrong_or_malformed_answers() -> None:
    t = Task(2, 3)
    assert accuracy_reward(t, "<think>x</think><answer>6</answer>") == 0.0
    assert accuracy_reward(t, "the answer is 5") == 0.0
    assert format_reward("<answer>5</answer>") == 0.0
    assert format_reward("<think>a</think><answer>5</answer><answer>5</answer>") == 0.0


def test_tokenizer_keeps_digits_separate() -> None:
    tok = reasoning_tokenizer()
    ids = tok.encode("123+45=")
    assert len(ids) == len("123+45=")
    assert tok.decode(ids) == "123+45="


def test_build_batch_scores_only_completion_tokens() -> None:
    x, y = build_batch([[1, 2, 3], [4, 5]], [[6, 7], [8, 9, 10]], pad_id=0)
    assert x.tolist() == [[1, 2, 3, 6], [4, 5, 8, 9]]
    assert y.tolist() == [[-100, -100, 6, 7], [-100, 8, 9, 10]]


def test_per_token_log_probs_sum_to_sequence_log_probs() -> None:
    torch.manual_seed(0)
    model = GPTModel(TINY).eval()
    x, y = build_batch([[1, 2, 3], [4, 5]], [[6, 7], [8, 9, 10]], pad_id=0)
    logps, mask = per_token_log_probs(model, x, y)
    assert mask.sum().item() == 5
    assert torch.allclose(logps.sum(-1), compute_sequence_log_probs(model, x, y), atol=1e-5)


def test_sample_group_cuts_after_first_eos() -> None:
    torch.manual_seed(0)
    model = GPTModel(TINY)
    tok = reasoning_tokenizer()
    eos = tok.eos_token_id
    with torch.no_grad():  # final hidden state = all ones, so the EOS logit dominates
        model.ln_f.weight.zero_()
        model.ln_f.bias.fill_(1.0)
        model.lm_head.weight[eos] = 10.0
    groups = sample_group(model, tok.encode("1+2="), n=4, max_new_tokens=10, eos_id=eos)
    assert len(groups) == 4
    assert all(g == [eos] for g in groups)

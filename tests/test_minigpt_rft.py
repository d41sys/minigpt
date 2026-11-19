import random

import torch

from minigpt.model import GPTConfig, GPTModel
from minigpt.reason import Task, reasoning_tokenizer
from minigpt.rft import (
    Sampled,
    preference_pairs,
    rft_dataset,
    sample_tasks,
    select_completions,
)

TASK = Task(12, 34)  # answer 46
GOOD = "<think>2+4+0=6 1+3+0=4</think><answer>46</answer>"
WRONG = "<think>2+4+0=6 1+3+0=4</think><answer>45</answer>"
BAD_FORMAT = "<answer>46</answer>"
TINY = GPTConfig(vocab_size=258, context_length=64, d_model=32, num_layers=2, num_heads=4)


def make_sampled(texts: list[str]) -> Sampled:
    tok = reasoning_tokenizer()
    ids = [[*tok.encode(t), tok.eos_token_id] for t in texts]
    return Sampled(TASK, tok.encode(TASK.prompt), ids, texts)


def test_verifier_keeps_distinct_correct_well_formed_samples() -> None:
    spaced = GOOD.replace("<think>", "<think> ")
    texts = [WRONG, GOOD, BAD_FORMAT, GOOD, spaced]
    assert select_completions(TASK, texts, None, keep=5) == [1, 4]  # dup, bad format, wrong out
    assert select_completions(TASK, texts, None, keep=1) == [1]


def test_rm_selection_takes_the_top_score_even_when_wrong() -> None:
    texts = [GOOD, WRONG, BAD_FORMAT]
    assert select_completions(TASK, texts, [0.1, 0.9, 0.5], keep=1) == [1]
    assert select_completions(TASK, texts, [0.1, 0.9, 0.5], keep=2) == [1, 2]


def test_rft_dataset_masks_prompt_and_reports_precision() -> None:
    tok = reasoning_tokenizer()
    s = make_sampled([GOOD, WRONG])
    ds, stats = rft_dataset(tok, [s], None, keep=2)
    assert len(ds) == 1 and stats["precision"] == 1.0 and stats["sample_accuracy"] == 0.5
    x, y = ds.examples[0]
    n_prompt = len(s.prompt)
    assert (y[: n_prompt - 1] == -100).all() and y[n_prompt - 1] != -100
    assert y[-1] == tok.eos_token_id and len(x) == len(y)
    _, rm_stats = rft_dataset(tok, [s], [[0.0, 1.0]], keep=1)  # the RM prefers the wrong one
    assert rm_stats["kept"] == 1 and rm_stats["precision"] == 0.0


def test_preference_pairs_need_a_better_and_a_worse_sample() -> None:
    tok = reasoning_tokenizer()
    mixed, uniform = make_sampled([GOOD, WRONG, GOOD]), make_sampled([GOOD, GOOD])
    ds = preference_pairs(tok, [mixed, uniform], seed=0)
    assert len(ds) == 1
    item = ds.items[0]
    assert item["chosen_input_ids"].tolist() != item["rejected_input_ids"].tolist()
    assert (item["chosen_labels"] != -100).any() and (item["rejected_labels"] != -100).any()
    assert tok.decode([t for t in item["chosen_labels"].tolist() if t >= 0][:-1]) == GOOD


def test_sample_tasks_shapes() -> None:
    tok = reasoning_tokenizer()
    torch.manual_seed(0)
    model = GPTModel(TINY)
    rng = random.Random(0)
    tasks = [Task(rng.randint(10, 99), rng.randint(10, 99)) for _ in range(3)]
    tasks.append(Task(5, 7))  # a different prompt length
    out = sample_tasks(model, tok, tasks, k=3, max_new_tokens=6, chunk=4)
    assert [s.task for s in out] == tasks
    for s in out:
        assert len(s.ids) == len(s.texts) == 3
        assert all(len(c) <= 6 and tok.eos_token_id not in c[:-1] for c in s.ids)

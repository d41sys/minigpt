import json
from pathlib import Path

import numpy as np
import pytest
import torch

from minigpt.cli import build_parser, cmd_pretrain_eval
from minigpt.data import SlidingWindowDataset, create_dataloader, split_val_test
from minigpt.model import GPTConfig, GPTModel
from minigpt.train import evaluate_loss, save_checkpoint, window_losses

TINY = GPTConfig(vocab_size=50, context_length=8, d_model=16, num_layers=1, num_heads=2)


def test_split_val_test_is_contiguous_deterministic_and_lossless() -> None:
    ids = list(range(11))
    val, test = split_val_test(ids)
    assert val == [0, 1, 2, 3, 4]
    assert test == [5, 6, 7, 8, 9, 10]
    assert val + test == ids
    assert split_val_test(ids) == (val, test)


def test_window_losses_match_manual_cross_entropy_and_evaluate_loss() -> None:
    torch.manual_seed(0)
    model = GPTModel(TINY).eval()
    ids = torch.randint(0, 50, (8 * 5 + 1,)).tolist()  # 5 windows; batch 2 -> batches 2, 2, 1
    loader = create_dataloader(ids, 8, batch_size=2, shuffle=False, drop_last=False)
    losses = window_losses(model, loader, torch.device("cpu"))
    assert len(losses) == 5
    x, y = SlidingWindowDataset(ids, 8)[3]
    with torch.no_grad():
        logits, _ = model(x[None])
    manual = torch.nn.functional.cross_entropy(logits[0], y).item()
    assert losses[3] == pytest.approx(manual, abs=1e-5)
    # evaluate_loss weights every window equally even with a short last batch
    assert evaluate_loss(model, loader, torch.device("cpu")) == pytest.approx(
        float(np.mean(losses)), abs=1e-5
    )


def test_stride_one_sampling_reaches_every_offset() -> None:
    ids = list(range(20))
    aligned = SlidingWindowDataset(ids, 4)
    dense = SlidingWindowDataset(ids, 4, stride=1)
    assert [int(aligned[i][0][0]) for i in range(len(aligned))] == [0, 4, 8, 12]
    assert {int(dense[i][0][0]) for i in range(len(dense))} == set(range(16))


def test_pretrain_flag_defaults_follow_the_preregistered_decision() -> None:
    args = build_parser().parse_args(["pretrain"])
    assert (args.dropout, args.sampling, args.results_out) == (0.1, "random", None)
    args = build_parser().parse_args(
        ["pretrain", "--dropout", "0.2", "--sampling", "aligned", "--results-out", "x.json"]
    )
    assert (args.dropout, args.sampling) == (0.2, "aligned")


def test_pretrain_eval_reports_val_and_test_halves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import minigpt.cli as cli

    held = list(range(8 * 6 + 2))  # val half: 25 tokens -> 3 windows, test half: 25 -> 3 windows
    monkeypatch.setattr(cli, "load_tokenizer", lambda: object())
    monkeypatch.setattr(cli, "shakespeare_splits", lambda _tok: ([0], held))
    monkeypatch.setattr(cli, "get_device", lambda: torch.device("cpu"))
    torch.manual_seed(0)
    model = GPTModel(TINY)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, None, step=0, val_loss=0.0)
    cmd_pretrain_eval(build_parser().parse_args(["pretrain-eval", "--checkpoint", str(ckpt)]))
    out = capsys.readouterr().out
    assert "val loss=" in out and "test loss=" in out
    assert out.count("(n=3 windows)") == 2


def test_pretrain_writes_val_selected_test_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    import minigpt.cli as cli

    periodic = [i % 5 for i in range(500)]  # learnable, so a step beats the step-0 val loss
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda: types.SimpleNamespace(vocab_size=50))
    monkeypatch.setattr(
        cli,
        "shakespeare_splits",
        lambda _tok: (periodic[:400], periodic[400:]),
    )
    monkeypatch.setattr(cli, "get_device", lambda: torch.device("cpu"))
    out = tmp_path / "res.json"
    argv = [
        "pretrain", "--name", "t", "--steps", "40", "--eval-interval", "10", "--batch-size", "2",
        "--context-length", "8", "--d-model", "16", "--num-layers", "1", "--num-heads", "2",
        "--sampling", "random", "--dropout", "0.2", "--results-out", str(out),
    ]  # fmt: skip
    args = build_parser().parse_args(argv)
    args.func(args)
    rec = json.loads(out.read_text())
    assert rec["sampling"] == "random" and rec["dropout"] == 0.2
    assert len(rec["val_window_losses"]) == 6 and len(rec["test_window_losses"]) == 6
    assert rec["test_loss"] == pytest.approx(float(np.mean(rec["test_window_losses"])))
    assert rec["selected_step"] in {0, 10, 20, 30, 39}


def test_split_halves_are_disjoint_windows() -> None:
    val, test = split_val_test(list(range(100)))
    val_w = SlidingWindowDataset(val, 8)
    test_w = SlidingWindowDataset(test, 8)
    seen_val = {t for i in range(len(val_w)) for t in val_w[i][0].tolist()}
    seen_test = {t for i in range(len(test_w)) for t in test_w[i][0].tolist()}
    assert not seen_val & seen_test

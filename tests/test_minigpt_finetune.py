import math

import pytest
import torch
from minigpt.bpe import BPETokenizer
from minigpt.dpo import compute_sequence_log_probs, dpo_loss
from minigpt.evaluate import RougeLJudge, longest_common_subsequence
from minigpt.finetune_cls import GPTForClassification, freeze_for_finetuning, split_data
from minigpt.model import GPTConfig, GPTModel
from minigpt.sft import build_sft_example, format_prompt

TINY = GPTConfig(vocab_size=300, context_length=64, d_model=32, num_layers=2, num_heads=4)


def test_dpo_loss_is_log2_when_policy_equals_reference() -> None:
    torch.manual_seed(0)
    model = GPTModel(TINY).eval()
    x = torch.randint(0, 300, (3, 20))
    y_w = torch.randint(0, 300, (3, 20))
    y_l = torch.randint(0, 300, (3, 20))
    y_w[:, :5] = -100
    y_l[:, :5] = -100
    pw = compute_sequence_log_probs(model, x, y_w)
    pl = compute_sequence_log_probs(model, x, y_l)
    loss, rw, rl = dpo_loss(pw, pl, pw.clone(), pl.clone(), beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert torch.all(rw == 0) and torch.all(rl == 0)


def test_dpo_loss_decreases_when_policy_prefers_chosen() -> None:
    ref = torch.tensor([-10.0])
    loss, _, _ = dpo_loss(torch.tensor([-8.0]), torch.tensor([-12.0]), ref, ref, beta=0.1)
    assert loss.item() < math.log(2)


def test_sft_masks_prompt_tokens() -> None:
    tok = BPETokenizer()
    tok.train("Below is an instruction. Write a response. hello world", target_vocab_size=280)
    x, y = build_sft_example("Say hi", "hello", tok)
    prompt_len = len(tok.encode(format_prompt("Say hi"), allow_special=True))
    assert len(x) == len(y)
    assert torch.all(y[: prompt_len - 1] == -100)
    response_targets = y[prompt_len - 1 :].tolist()
    assert response_targets == [*tok.encode("hello"), tok.eos_token_id]


def test_classifier_head_swap_and_freeze() -> None:
    model = GPTForClassification(GPTModel(TINY), num_classes=2, pad_id=0)
    trainable = freeze_for_finetuning(model)
    last = sum(p.numel() for p in model.gpt.blocks[-1].parameters())
    ln = sum(p.numel() for p in model.gpt.ln_f.parameters())
    head = 32 * 2 + 2
    assert trainable == last + ln + head
    assert not model.gpt.wte.weight.requires_grad
    logits, loss = model(torch.tensor([[5, 6, 7, 0, 0]]), targets=torch.tensor([1]))
    assert logits.shape == (1, 2) and loss is not None


def test_classifier_ignores_padding() -> None:
    torch.manual_seed(0)
    model = GPTForClassification(GPTModel(TINY), pad_id=0).eval()
    a, _ = model(torch.tensor([[5, 6, 7, 0, 0]]))
    b, _ = model(torch.tensor([[5, 6, 7, 0, 0, 0, 0]]))
    assert torch.allclose(a, b, atol=1e-5)


def test_split_ratios() -> None:
    tr, va, te = split_data([str(i) for i in range(100)], list(range(100)))
    assert (len(tr[0]), len(va[0]), len(te[0])) == (70, 10, 20)


def test_rouge_l() -> None:
    assert longest_common_subsequence(["a", "b", "c", "d"], ["a", "c", "d", "e"]) == 3
    s = RougeLJudge().judge("", "the cat sat", "the cat sat down")
    assert s["rouge_l_precision"] == 1.0
    assert s["rouge_l_recall"] == pytest.approx(0.75)

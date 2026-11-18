import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from minigpt.bpe import BPETokenizer
from minigpt.dpo import (
    PreferenceDataset,
    PreferenceSample,
    compute_sequence_log_probs,
    dpo_loss,
    train_dpo,
)
from minigpt.evaluate import RougeLJudge, longest_common_subsequence
from minigpt.finetune_cls import (
    GPTForClassification,
    evaluate_classifier,
    freeze_for_finetuning,
    split_data,
)
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


def test_dpo_nll_term_keeps_chosen_likelihood_higher() -> None:
    tok = BPETokenizer()
    tok.train("Below is an instruction. Write a response. good answer bad reply", 280)
    pairs = [PreferenceSample("Say it", "good answer", "bad reply")] * 4
    cfg = GPTConfig(vocab_size=300, context_length=512, d_model=32, num_layers=2, num_heads=4)
    ds = PreferenceDataset(pairs, tok, max_length=cfg.context_length)
    assert (ds[0]["chosen_labels"] != -100).any()  # the response survives truncation
    cpu = torch.device("cpu")
    shift = {}
    for alpha in (0.0, 1.0):
        torch.manual_seed(0)
        ref = GPTModel(cfg)
        policy = GPTModel(cfg)
        policy.load_state_dict(ref.state_dict())
        res = train_dpo(policy, ref, ds, beta=0.1, epochs=5, lr=1e-3, device=cpu, nll_alpha=alpha)
        shift[alpha] = res["history"][-1]["logp_chosen_vs_ref"]
    assert shift[1.0] > shift[0.0]
    assert shift[1.0] > 0  # chosen tokens became more likely than under the reference


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


@pytest.mark.parametrize("pos_encoding", ["learned", "rope"])
def test_classifier_ignores_padding(pos_encoding: str) -> None:
    torch.manual_seed(0)
    gpt = GPTModel(GPTConfig(**{**TINY.__dict__, "pos_encoding": pos_encoding}))
    model = GPTForClassification(gpt, pad_id=0).eval()
    a, _ = model(torch.tensor([[5, 6, 7, 0, 0]]))
    b, _ = model(torch.tensor([[5, 6, 7, 0, 0, 0, 0]]))
    assert torch.allclose(a, b, atol=1e-5)
    # pooling must read the last real token: dropping the padding changes nothing, but changing
    # the last real token (or reading the first token / the first pad) does.
    c, _ = model(torch.tensor([[5, 6, 7]]))
    d, _ = model(torch.tensor([[5, 6, 8, 0, 0]]))
    assert torch.allclose(a, c, atol=1e-5)
    assert not torch.allclose(a, d, atol=1e-5)


class _FixedPreds(nn.Module):
    """Stub classifier: row i predicts class x[i, 0]."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.one_hot(x[:, 0], 2).float(), None


def test_evaluate_classifier_hand_computed_metrics() -> None:
    # preds  1 1 1 1 0 0 / labels 1 1 0 0 1 0  ->  tp 2, fp 2, tn 1, fn 1
    preds = torch.tensor([[1], [1], [1], [1], [0], [0]])
    labels = torch.tensor([1, 1, 0, 0, 1, 0])
    loader = DataLoader(TensorDataset(preds, labels), batch_size=4)  # counts span two batches
    m = evaluate_classifier(_FixedPreds(), loader, torch.device("cpu"))
    assert m["accuracy"] == pytest.approx(3 / 6)
    assert m["precision"] == pytest.approx(2 / 4)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(4 / 7)
    assert m["total"] == 6


def test_evaluate_classifier_all_negative_predictions() -> None:
    preds = torch.zeros(6, 1, dtype=torch.long)
    labels = torch.tensor([1, 1, 0, 0, 1, 0])
    loader = DataLoader(TensorDataset(preds, labels), batch_size=6)
    m = evaluate_classifier(_FixedPreds(), loader, torch.device("cpu"))
    assert (m["precision"], m["recall"], m["f1"]) == (0.0, 0.0, 0.0)
    assert m["accuracy"] == pytest.approx(3 / 6)


def test_sequence_log_probs_match_masked_cross_entropy() -> None:
    torch.manual_seed(0)
    model = GPTModel(TINY).eval()
    x = torch.randint(0, 300, (3, 20))
    y = torch.randint(0, 300, (3, 20))
    y[:, :7] = -100  # masked prompt
    y[1, 15:] = -100  # ragged tail
    y[2, :] = -100  # fully masked row sums to 0
    got = compute_sequence_log_probs(model, x, y)
    logits, _ = model(x)
    expected = -F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction="none")
    assert torch.allclose(got, expected.sum(-1), atol=1e-5)
    assert got[2].item() == 0.0


def test_split_ratios() -> None:
    tr, va, te = split_data([str(i) for i in range(100)], list(range(100)))
    assert (len(tr[0]), len(va[0]), len(te[0])) == (70, 10, 20)


def test_rouge_l() -> None:
    assert longest_common_subsequence(["a", "b", "c", "d"], ["a", "c", "d", "e"]) == 3
    s = RougeLJudge().judge("", "the cat sat", "the cat sat down")
    assert s["rouge_l_precision"] == 1.0
    assert s["rouge_l_recall"] == pytest.approx(0.75)

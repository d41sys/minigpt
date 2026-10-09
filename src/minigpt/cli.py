"""minigpt CLI: tokenizer, pretraining, classifier fine-tune, SFT, DPO, eval, generation.

Pipeline: `pretrain` writes runs/pretrain/ckpt.pt; `finetune-cls` and `sft` start from it;
`dpo` starts from runs/sft/ckpt.pt; `eval` reports on whatever checkpoints exist.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from minigpt.bpe import BPETokenizer
from minigpt.data import create_dataloader
from minigpt.dpo import PreferenceDataset, generate_preference_pairs, train_dpo
from minigpt.evaluate import RougeLJudge, compute_perplexity, evaluate_responses
from minigpt.finetune_cls import (
    GPTForClassification,
    SpamTextDataset,
    download_sms_spam,
    evaluate_classifier,
    freeze_for_finetuning,
    load_and_balance_dataset,
    split_data,
    train_classifier,
)
from minigpt.model import GPTConfig, GPTModel
from minigpt.sft import SFTDataset, download_alpaca_data, format_prompt, train_sft
from minigpt.train import get_device, load_model, save_checkpoint, train

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
RAW_DIR = Path("data/raw")
TOKENIZER_PATH = Path("runs/tokenizer.json")
PRETRAIN_CKPT = Path("runs/pretrain/ckpt.pt")
SFT_CKPT = Path("runs/sft/ckpt.pt")
DPO_CKPT = Path("runs/dpo/ckpt.pt")
SFT_HELDOUT = 100  # last N Alpaca rows are never trained on


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def download_tinyshakespeare(raw_dir: Path = RAW_DIR) -> Path:
    """Download tiny-Shakespeare (~1.1 MB); raises on network failure."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "tinyshakespeare.txt"
    if not path.exists():
        req = urllib.request.Request(TINY_SHAKESPEARE_URL, headers={"User-Agent": "minigpt"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_bytes(resp.read())
    return path


def load_tokenizer() -> BPETokenizer:
    if not TOKENIZER_PATH.exists():
        sys.exit(f"{TOKENIZER_PATH} not found: run `minigpt train-tokenizer` first")
    return BPETokenizer.load(TOKENIZER_PATH)


def require(path: Path, hint: str) -> Path:
    if not path.exists():
        sys.exit(f"{path} not found: run `minigpt {hint}` first")
    return path


def shakespeare_splits(tokenizer: BPETokenizer) -> tuple[list[int], list[int]]:
    ids = tokenizer.encode(download_tinyshakespeare().read_text(encoding="utf-8"))
    n = int(len(ids) * 0.9)
    return ids[:n], ids[n:]


def load_alpaca(n: int = 1000) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = json.loads(
        download_alpaca_data(RAW_DIR / "alpaca_1k.json", sample_size=n).read_text("utf-8")
    )
    return rows


def cmd_data(args: argparse.Namespace) -> None:
    raw = Path(args.dir)
    if args.dataset == "spam":
        print(f"SMS Spam ready at {download_sms_spam(raw)}")
    elif args.dataset == "tinyshakespeare":
        print(f"tiny-Shakespeare ready at {download_tinyshakespeare(raw)}")
    else:
        print(f"Alpaca 1k ready at {download_alpaca_data(raw / 'alpaca_1k.json')}")


def cmd_train_tokenizer(args: argparse.Namespace) -> None:
    corpus = Path(args.corpus) if args.corpus else download_tinyshakespeare()
    text = corpus.read_text(encoding="utf-8", errors="replace")
    tok = BPETokenizer()
    tok.train(text, target_vocab_size=args.vocab_size)
    tok.save(args.out)
    ids = tok.encode(text[:100_000])
    print(
        f"vocab={tok.vocab_size} merges={len(tok.merges)} "
        f"compression={len(text[:100_000].encode()) / len(ids):.2f} bytes/token -> {args.out}"
    )


def cmd_pretrain(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device()
    if args.data != "tinyshakespeare":
        sys.exit("only --data tinyshakespeare is bundled")
    tok = load_tokenizer()
    train_ids, val_ids = shakespeare_splits(tok)
    print(f"device={device} train_tokens={len(train_ids):,} val_tokens={len(val_ids):,}")
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.1,
        use_sdpa=True,
    )
    model = GPTModel(cfg)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    train_loader = create_dataloader(train_ids, cfg.context_length, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_ids, cfg.context_length, args.batch_size, shuffle=False)
    res = train(
        model,
        train_loader,
        val_loader,
        max_steps=args.steps,
        warmup_steps=max(1, args.steps // 20),
        learning_rate=args.lr,
        eval_interval=args.eval_interval,
        checkpoint_dir=Path("runs") / args.name,
        device=device,
    )
    for row in res["history"]:
        print(
            f"step {int(row['step']):5d} train {row['train_loss']:.4f} "
            f"val {row['val_loss']:.4f} lr {row['lr']:.2e}"
        )
    v0, v1 = res["initial_val_loss"], res["final_val_loss"]
    print(f"val loss step0={v0:.4f} best={v1:.4f} drop={(v0 - v1) / v0 * 100:.1f}%")
    print(f"checkpoint -> {res['checkpoint_path']}")


def cmd_finetune_cls(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device()
    tok = load_tokenizer()
    base = load_model(require(Path(args.base), "pretrain"))
    texts, labels = load_and_balance_dataset(download_sms_spam(RAW_DIR))
    tr, va, te = split_data(texts, labels)
    print(f"balanced={len(texts)} train={len(tr[0])} val={len(va[0])} test={len(te[0])}")
    pad = tok.special_to_id.get("<|pad|>", 0)
    max_len = min(args.max_length, base.config.context_length)

    def loader(split: tuple[list[str], list[int]], shuffle: bool) -> DataLoader[Any]:
        ds = SpamTextDataset(split[0], split[1], tok, max_length=max_len, pad_id=pad)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    model = GPTForClassification(base, num_classes=2, pad_id=pad)
    print(f"trainable params={freeze_for_finetuning(model):,}")
    res = train_classifier(
        model, loader(tr, True), loader(va, False), epochs=args.epochs, lr=args.lr, device=device
    )
    for row in res["history"]:
        print(
            f"epoch {int(row['epoch'])} train_loss {row['train_loss']:.4f} "
            f"val_acc {row['val_accuracy']:.4f}"
        )
    m = evaluate_classifier(model, loader(te, False), device)
    print(
        f"test accuracy={m['accuracy']:.4f} precision={m['precision']:.4f} "
        f"recall={m['recall']:.4f} f1={m['f1']:.4f} (n={int(m['total'])})"
    )


def sample_response(
    model: GPTModel, tok: BPETokenizer, row: dict[str, str], device: torch.device, n: int = 60
) -> str:
    ids = tok.encode(format_prompt(row["instruction"], row.get("input", "")), allow_special=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=n, temperature=0.0, eos_token_id=tok.eos_token_id)
    new = [t for t in out[0, len(ids) :].tolist() if t != tok.eos_token_id]
    return tok.decode(new).strip()


def cmd_sft(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device()
    tok = load_tokenizer()
    model = load_model(require(Path(args.base), "pretrain"))
    rows = load_alpaca()
    train_rows = rows[:-SFT_HELDOUT]
    ds = SFTDataset(train_rows, tok, max_length=model.config.context_length)
    print(f"device={device} sft_examples={len(ds)} heldout={SFT_HELDOUT}")
    res = train_sft(
        model,
        ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
    )
    for row in res["history"]:
        print(f"epoch {row['epoch']} loss {row['loss']:.4f}")
    save_checkpoint(SFT_CKPT, model, None, step=args.epochs, val_loss=res["final_loss"])
    held = rows[-SFT_HELDOUT]
    print(f"checkpoint -> {SFT_CKPT}\n\n### Instruction: {held['instruction']}")
    print(f"### Model: {sample_response(model, tok, held, device)}")
    print(f"### Reference: {held['output'][:200]}")


def cmd_dpo(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device()
    tok = load_tokenizer()
    policy = load_model(require(SFT_CKPT, "sft"))
    ref = load_model(SFT_CKPT)
    rows = load_alpaca()[:-SFT_HELDOUT]
    pairs = generate_preference_pairs(ref, tok, rows, num_pairs=args.pairs, device=device)
    ds = PreferenceDataset(pairs, tok, max_length=policy.config.context_length)
    print(f"device={device} preference_pairs={len(ds)} beta={args.beta}")
    res = train_dpo(policy, ref, ds, beta=args.beta, epochs=args.epochs, lr=args.lr, device=device)
    print(f"step 1 dpo_loss {res['first_loss']:.4f} (log 2 = 0.6931)")
    for row in res["history"]:
        print(
            f"epoch {int(row['epoch'])} mean dpo_loss {row['dpo_loss']:.4f} "
            f"reward_accuracy {row['reward_accuracy']:.3f}"
        )
    save_checkpoint(DPO_CKPT, policy, None, step=args.epochs, val_loss=res["final_loss"])
    print(f"checkpoint -> {DPO_CKPT}")


def cmd_eval(args: argparse.Namespace) -> None:
    device = get_device()
    tok = load_tokenizer()
    if PRETRAIN_CKPT.exists():
        model = load_model(PRETRAIN_CKPT).to(device)
        _, val_ids = shakespeare_splits(tok)
        loader = create_dataloader(val_ids, model.config.context_length, 16, shuffle=False)
        print(f"pretrain  val perplexity={compute_perplexity(model, loader, device):.2f}")
    heldout = load_alpaca()[-SFT_HELDOUT:][: args.n]
    judge = RougeLJudge()
    for name, ckpt in (("pretrain", PRETRAIN_CKPT), ("sft", SFT_CKPT), ("dpo", DPO_CKPT)):
        if not ckpt.exists():
            print(f"{name:9s} not run: {ckpt} missing")
            continue
        model = load_model(ckpt).to(device)
        samples = [
            {
                "instruction": r["instruction"],
                "response": sample_response(model, tok, r, device),
                "reference": r["output"],
            }
            for r in heldout
        ]
        score = evaluate_responses(judge, samples)["mean_rouge_l_f1"]
        print(f"{name:9s} heldout ROUGE-L F1={score:.4f} (n={len(samples)})")


def cmd_generate(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device()
    tok = load_tokenizer()
    model = load_model(require(Path(args.checkpoint), "pretrain")).to(device)
    x = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(x, args.max_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="minigpt", description="GPT from scratch.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("data", help="download a dataset into data/raw")
    s.add_argument("dataset", choices=["spam", "tinyshakespeare", "alpaca"])
    s.add_argument("--dir", default=str(RAW_DIR))
    s.set_defaults(func=cmd_data)

    s = sub.add_parser("train-tokenizer", help="train byte-level BPE")
    s.add_argument("--corpus", default=None, help="text file (default: tiny-Shakespeare)")
    s.add_argument("--vocab-size", type=int, default=1024)
    s.add_argument("--out", default=str(TOKENIZER_PATH))
    s.set_defaults(func=cmd_train_tokenizer)

    s = sub.add_parser("pretrain", help="pretrain a small GPT")
    s.add_argument("--data", default="tinyshakespeare")
    s.add_argument("--name", default="pretrain")
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--context-length", type=int, default=256)
    s.add_argument("--d-model", type=int, default=256)
    s.add_argument("--num-layers", type=int, default=6)
    s.add_argument("--num-heads", type=int, default=8)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--eval-interval", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_pretrain)

    s = sub.add_parser("finetune-cls", help="spam classifier via head swap")
    s.add_argument("--base", default=str(PRETRAIN_CKPT))
    s.add_argument("--epochs", type=int, default=5)
    s.add_argument("--lr", type=float, default=5e-4)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--max-length", type=int, default=96)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_finetune_cls)

    s = sub.add_parser("sft", help="instruction-tune on Alpaca 1k")
    s.add_argument("--base", default=str(PRETRAIN_CKPT))
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sft)

    s = sub.add_parser("dpo", help="DPO on SFT-generated preference pairs")
    s.add_argument("--pairs", type=int, default=300)
    s.add_argument("--beta", type=float, default=0.1)
    s.add_argument("--epochs", type=int, default=2)
    s.add_argument("--lr", type=float, default=1e-5)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_dpo)

    s = sub.add_parser("eval", help="perplexity + held-out ROUGE-L")
    s.add_argument("--n", type=int, default=50, help="held-out Alpaca rows to judge")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("generate", help="sample text")
    s.add_argument("--prompt", default="ROMEO:")
    s.add_argument("--checkpoint", default=str(PRETRAIN_CKPT))
    s.add_argument("--max-tokens", type=int, default=80)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_generate)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

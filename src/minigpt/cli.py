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

import numpy as np
import torch
from torch.utils.data import DataLoader

from minigpt.bpe import BPETokenizer
from minigpt.data import create_dataloader, split_val_test
from minigpt.dpo import (
    PreferenceDataset,
    PreferenceSample,
    generate_preference_pairs,
    train_dpo,
)
from minigpt.evaluate import (
    EvalReport,
    Judge,
    MockJudge,
    OllamaJudge,
    compute_perplexity,
    exact_match,
    holm_adjust,
    paired_bootstrap_test,
    token_f1,
    wilson_interval,
)
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
from minigpt.pretrain_compare import compare, format_report, load_records
from minigpt.sft import (
    SFTDataset,
    download_alpaca_data,
    format_prompt,
    sft_collate_fn,
    train_sft,
)
from minigpt.train import (
    get_device,
    load_checkpoint,
    load_model,
    save_checkpoint,
    train,
    window_losses,
)

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


def half_losses(
    model: GPTModel, halves: dict[str, list[int]], batch_size: int, device: torch.device
) -> dict[str, list[float]]:
    """Per-window loss of every non-overlapping window of each token half (dropout off)."""
    model.to(device)
    return {
        name: window_losses(
            model,
            create_dataloader(
                ids, model.config.context_length, batch_size, shuffle=False, drop_last=False
            ),
            device,
        )
        for name, ids in halves.items()
    }


def format_half_losses(losses: dict[str, list[float]]) -> str:
    parts = [
        f"{name} loss={np.mean(v):.4f} ppl={np.exp(np.mean(v)):.2f} (n={len(v)} windows)"
        for name, v in losses.items()
    ]
    return "  ".join(parts)


def cmd_pretrain(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device()
    if args.data != "tinyshakespeare":
        sys.exit("only --data tinyshakespeare is bundled")
    tok = load_tokenizer()
    train_ids, held_ids = shakespeare_splits(tok)
    val_ids, test_ids = split_val_test(held_ids)
    print(
        f"device={device} train_tokens={len(train_ids):,} "
        f"val_tokens={len(val_ids):,} test_tokens={len(test_ids):,}"
    )
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_sdpa=True,
    )
    model = GPTModel(cfg)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    stride = 1 if args.sampling == "random" else None
    train_loader = create_dataloader(
        train_ids, cfg.context_length, args.batch_size, stride=stride, shuffle=True
    )
    val_loader = create_dataloader(
        val_ids, cfg.context_length, args.batch_size, shuffle=False, drop_last=False
    )
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
    ckpt = Path(res["checkpoint_path"])
    if not ckpt.exists():
        print("no checkpoint beat the step-0 val loss; test loss not computed")
        return
    payload = load_checkpoint(ckpt, model, device=device)
    losses = half_losses(model, {"val": val_ids, "test": test_ids}, args.batch_size, device)
    print(f"val-selected checkpoint (step {payload['step']}): {format_half_losses(losses)}")
    if args.results_out:
        out = Path(args.results_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "name": args.name,
            "seed": args.seed,
            "sampling": args.sampling,
            "dropout": args.dropout,
            "steps": args.steps,
            "selected_step": payload["step"],
            "initial_val_loss": v0,
            "history": res["history"],
            "val_window_losses": losses["val"],
            "test_window_losses": losses["test"],
            "val_loss": float(np.mean(losses["val"])),
            "test_loss": float(np.mean(losses["test"])),
        }
        out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(f"results -> {out}")


def cmd_pretrain_eval(args: argparse.Namespace) -> None:
    """Val-half and test-half loss of any pretraining checkpoint, plus bits per byte."""
    device = get_device()
    tok = load_tokenizer()
    _, held_ids = shakespeare_splits(tok)
    halves = dict(zip(("val", "test"), split_val_test(held_ids), strict=True))
    model = load_model(require(Path(args.checkpoint), "pretrain"))
    losses = half_losses(model, halves, args.batch_size, device)
    print(f"{args.checkpoint}: {format_half_losses(losses)}")
    # Tokenizer-independent: nats/token -> bits/byte via the half's own bytes-per-token ratio.
    for name, ids in halves.items():
        n_bytes = len(tok.decode(ids).encode("utf-8"))
        bpb = float(np.mean(losses[name])) / np.log(2) * len(ids) / n_bytes
        print(f"{name}: {n_bytes / len(ids):.3f} bytes/token, {bpb:.4f} bits/byte")


def cmd_pretrain_compare(args: argparse.Namespace) -> None:
    """Apply the pre-registered adoption rule to the per-run JSONs in --dir."""
    report = compare(load_records(args.dir))
    print(format_report(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


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
    pairs_file = Path(args.pairs_file)
    if pairs_file.is_file():
        pairs = [PreferenceSample(**d) for d in json.loads(pairs_file.read_text())]
        print(f"loaded {len(pairs)} fixed preference pairs from {pairs_file}")
    else:
        pairs = generate_preference_pairs(ref, tok, rows, num_pairs=args.pairs, device=device)
        pairs_file.parent.mkdir(parents=True, exist_ok=True)
        pairs_file.write_text(json.dumps([vars(p) for p in pairs]))
        print(f"saved {len(pairs)} preference pairs -> {pairs_file}")
    ds = PreferenceDataset(pairs, tok, max_length=policy.config.context_length)
    print(f"device={device} preference_pairs={len(ds)} beta={args.beta} nll_alpha={args.nll_alpha}")
    res = train_dpo(
        policy,
        ref,
        ds,
        beta=args.beta,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        nll_alpha=args.nll_alpha,
    )
    print(f"step 1 dpo_loss {res['first_loss']:.4f} (log 2 = 0.6931)")
    for row in res["history"]:
        print(
            f"epoch {int(row['epoch'])} mean dpo_loss {row['dpo_loss']:.4f} "
            f"reward_accuracy {row['reward_accuracy']:.3f} "
            f"logp/token chosen {row['logp_chosen']:.3f} ({row['logp_chosen_vs_ref']:+.3f} vs ref) "
            f"rejected {row['logp_rejected']:.3f} ({row['logp_rejected_vs_ref']:+.3f} vs ref)"
        )
    out = Path(args.out)
    save_checkpoint(out, policy, None, step=args.epochs, val_loss=res["final_loss"])
    print(f"checkpoint -> {out}")


SHORT_ANSWER_MAX_WORDS = 5  # held-out rows with references this short form the EM set
DEFAULT_EVAL_N = 50  # default `eval --n`; the paired table also reports F1 on these first rows


def paired_f1_table(
    f1_by_name: dict[str, list[float]], baseline: str, n_boot: int = 10_000, seed: int = 0
) -> list[dict[str, Any]]:
    """Each model minus `baseline` on a per-row metric: paired bootstrap CI, p, Holm over models.

    A model is `different` only if its 95% CI excludes 0 and its Holm-adjusted p is below 0.05.
    """
    base = np.asarray(f1_by_name[baseline])
    names = [name for name in f1_by_name if name != baseline]
    tests = [
        paired_bootstrap_test(np.asarray(f1_by_name[name]) - base, n_boot=n_boot, seed=seed)
        for name in names
    ]
    holm = holm_adjust([t[3] for t in tests])
    rows = []
    for name, (mean, lo, hi, p), p_holm in zip(names, tests, holm, strict=True):
        scores = f1_by_name[name]
        verdict = "no detectable difference"
        if (lo > 0 or hi < 0) and p_holm < 0.05:
            verdict = "different (better)" if mean > 0 else "different (worse)"
        rows.append(
            {
                "model": name,
                "f1": float(np.mean(scores)),
                "f1_first": float(np.mean(scores[:DEFAULT_EVAL_N])),
                "diff": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "p": p,
                "p_holm": p_holm,
                "verdict": verdict,
            }
        )
    return rows


def cmd_eval(args: argparse.Namespace) -> None:
    """Eval report: perplexity, EM, token F1, judge (mock or Ollama), weighted aggregate."""
    device = get_device()
    tok = load_tokenizer()
    _, val_ids = shakespeare_splits(tok)
    heldout = load_alpaca()[-SFT_HELDOUT:]
    short = [r for r in heldout if len(r["output"].split()) <= SHORT_ANSWER_MAX_WORDS]
    open_form = heldout[: args.n]
    judge: Judge = MockJudge()
    judge_label = "Mock judge (ROUGE-L, 1-5)"
    if args.judge == "ollama":
        judge = OllamaJudge(args.judge_url, args.judge_model)
        judge_label = f"Judge ({args.judge_model}, 1-5)"
    w_judge = args.w_judge if args.w_judge is not None else float(args.judge == "ollama")
    weights = (args.w_ppl, args.w_em, args.w_f1, w_judge)
    judge_rows: dict[str, list[int]] = {}
    f1_rows: dict[str, list[float]] = {}
    responses_by_name: dict[str, list[str]] = {}

    print(f"device={device} em_rows={len(short)} open_rows={len(open_form)} weights={weights}")
    print("95% CIs: EM Wilson over rows; token F1 and judge: 10,000-resample row bootstrap")
    print(
        "\n| Model | Alpaca ppl (held-out responses) | Shakespeare ppl | EM [95% CI] "
        f"| Token F1 [95% CI] | {judge_label} [95% CI] | Aggregate |"
    )
    print("|---|---|---|---|---|---|---|")
    checkpoints = [("pretrain", PRETRAIN_CKPT), ("sft", SFT_CKPT), ("dpo", DPO_CKPT)]
    if args.checkpoints:
        checkpoints = [
            (name, Path(path))
            for name, path in (item.split("=", 1) for item in args.checkpoints.split(","))
        ]
    for name, ckpt in checkpoints:
        if not ckpt.exists():
            print(f"| {name} | not run: {ckpt} missing | | | | | |")
            continue
        model = load_model(ckpt).to(device)
        shakespeare = create_dataloader(val_ids, model.config.context_length, 16, shuffle=False)
        alpaca = DataLoader(
            SFTDataset(heldout, tok, max_length=model.config.context_length),
            batch_size=16,
            collate_fn=sft_collate_fn,
        )
        em = [exact_match(sample_response(model, tok, r, device), r["output"]) for r in short]
        responses = [sample_response(model, tok, r, device) for r in open_form]
        f1 = [token_f1(p, r["output"]) for p, r in zip(responses, open_form, strict=True)]
        f1_rows[name] = f1
        responses_by_name[name] = responses
        scores = [
            judge.score(r["instruction"], p, r["output"])
            for p, r in zip(responses, open_form, strict=True)
        ]
        judge_rows[name] = scores
        report = EvalReport(
            perplexity=compute_perplexity(model, alpaca, device),
            vocab_size=model.config.vocab_size,
            exact_match=sum(em) / max(1, len(em)),
            exact_match_n=len(em),
            token_f1=sum(f1) / max(1, len(f1)),
            judge_mean=sum(scores) / max(1, len(scores)),
            open_n=len(open_form),
        )
        shakespeare_ppl = compute_perplexity(model, shakespeare, device)
        em_lo, em_hi = wilson_interval(int(sum(em)), len(em))
        _, f1_lo, f1_hi, _ = paired_bootstrap_test(np.asarray(f1))
        _, j_lo, j_hi, _ = paired_bootstrap_test(np.asarray(scores, dtype=float))
        print(
            f"| {name} | {report.perplexity:.2f} | {shakespeare_ppl:.2f} | "
            f"{report.exact_match:.3f} [{em_lo:.3f}, {em_hi:.3f}] | "
            f"{report.token_f1:.3f} [{f1_lo:.3f}, {f1_hi:.3f}] | "
            f"{report.judge_mean:.2f} [{j_lo:.2f}, {j_hi:.2f}] | {report.aggregate(weights):.3f} |"
        )
    if args.rows_out:
        out = Path(args.rows_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "n": len(open_form),
                    "token_f1": f1_rows,
                    "judge": judge_rows,
                    "judge_name": judge_label,
                    "responses": responses_by_name,
                    "references": [r["output"] for r in open_form],
                }
            )
        )
        print(f"\nper-row token F1, judge scores and responses -> {out}")
    if args.paired_vs:
        if args.paired_vs not in f1_rows:
            raise SystemExit(f"--paired-vs {args.paired_vs!r} is not a scored checkpoint")
        n_boot = 10_000
        metrics: list[tuple[str, dict[str, list[float]]]] = [("token F1", f1_rows)]
        if args.judge == "ollama":
            metrics.append((judge_label, {k: [float(s) for s in v] for k, v in judge_rows.items()}))
        for metric, rows in metrics:
            print(
                f"\nPaired bootstrap on per-row {metric} vs {args.paired_vs} "
                f"(n={len(open_form)} rows, {n_boot} resamples, seed 0, Holm over "
                f"{len(rows) - 1} models):"
            )
            print(
                f"| Model | mean (n={len(open_form)}) | mean (first {DEFAULT_EVAL_N}) | diff vs "
                f"{args.paired_vs} | 95% CI | p | p (Holm) | verdict |"
            )
            print("|---|---|---|---|---|---|---|---|")
            for row in paired_f1_table(rows, args.paired_vs, n_boot=n_boot):
                print(
                    f"| {row['model']} | {row['f1']:.3f} | {row['f1_first']:.3f} | "
                    f"{row['diff']:+.4f} | [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}] | "
                    f"{row['p']:.4f} | {row['p_holm']:.4f} | {row['verdict']} |"
                )


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
    s.add_argument("--dropout", type=float, default=0.1)
    s.add_argument(
        "--sampling",
        choices=["aligned", "random"],
        default="random",
        help="train windows: random offsets (stride 1, default) or aligned (stride = context)",
    )
    s.add_argument("--results-out", default=None, help="write per-run JSON (losses, history)")
    s.set_defaults(func=cmd_pretrain)

    s = sub.add_parser("pretrain-eval", help="val-half and test-half loss of a checkpoint")
    s.add_argument("--checkpoint", default=str(PRETRAIN_CKPT))
    s.add_argument("--batch-size", type=int, default=32)
    s.set_defaults(func=cmd_pretrain_eval)

    s = sub.add_parser("pretrain-compare", help="pre-registered variant comparison on test half")
    s.add_argument("--dir", default="docs/pretrain-regularization")
    s.add_argument("--json-out", default=None)
    s.set_defaults(func=cmd_pretrain_compare)

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
    s.add_argument("--out", default=str(DPO_CKPT))
    s.add_argument(
        "--nll-alpha",
        type=float,
        default=0.0,
        help="weight of the length-normalised NLL on chosen responses (Llama 3 uses 0.2)",
    )
    s.add_argument(
        "--pairs-file",
        default="runs/dpo/pairs.json",
        help="reused if present, so runs differing only in beta train on identical pairs",
    )
    s.set_defaults(func=cmd_dpo)

    s = sub.add_parser("eval", help="perplexity + EM + token F1 + judge (mock or Ollama), weighted")
    s.add_argument(
        "--n",
        type=int,
        default=DEFAULT_EVAL_N,
        help="held-out Alpaca rows for F1 and judge (100 = all held-out rows)",
    )
    s.add_argument("--w-ppl", type=float, default=1.0)
    s.add_argument("--w-em", type=float, default=1.0)
    s.add_argument("--w-f1", type=float, default=1.0)
    s.add_argument(
        "--w-judge",
        type=float,
        default=None,
        help="default 0 for the mock judge (ROUGE-L duplicates token F1), 1 for --judge ollama",
    )
    s.add_argument("--judge", choices=["mock", "ollama"], default="mock")
    s.add_argument("--judge-url", default="http://localhost:11434", help="Ollama server")
    s.add_argument("--judge-model", default="qwen3.6:27b-ctx4k")
    s.add_argument(
        "--checkpoints",
        default=None,
        help="comma-separated name=path list (default: pretrain, sft, dpo)",
    )
    s.add_argument(
        "--paired-vs",
        default=None,
        help="checkpoint name to subtract: paired bootstrap CI of per-row token F1 for the "
        "others, Holm-adjusted across them",
    )
    s.add_argument("--rows-out", default=None, help="write per-row token F1 and responses (JSON)")
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

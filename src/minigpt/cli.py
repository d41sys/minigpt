"""minigpt CLI: tokenizer, pretraining, classifier fine-tune, SFT, DPO, CAI, eval, generation, RL.

Pipeline: `pretrain` writes runs/pretrain/ckpt.pt; `finetune-cls` and `sft` start from it;
`dpo` starts from runs/sft/ckpt.pt; `opd` distils runs/grpo/ckpt.pt into a smaller student;
`reward-model` then `ppo` run RLHF from runs/reason_sft/ckpt.pt; `eval` reports on whatever
checkpoints exist.
`cai-data` / `cai-train` (sl, then rl) turn runs/sft into runs/cai_sl and runs/cai_rl.
Reasoning track (addition with chain of thought): `reason-pretrain` -> `reason-sft` ->
`grpo` or `hack-demo`; `reason-eval` scores any of those checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from minigpt.bpe import BPETokenizer
from minigpt.cai import (
    JudgeError,
    OllamaChat,
    adherence_report,
    label_pairs,
    load_principles,
    revise_rows,
    sample_responses,
    split_prompts,
)
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
    likelihood_ranking,
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
from minigpt.grpo import RewardFn, Samples, default_reward, train_grpo
from minigpt.model import GPTConfig, GPTModel, count_parameters
from minigpt.monitor import lenient_reward, rollout_monitor, summarize_hack
from minigpt.opd import train_opd
from minigpt.pretrain_compare import compare, format_report, load_records
from minigpt.reason import (
    OPS,
    Task,
    accuracy_reward,
    equation_corpus,
    evaluate_reasoning,
    make_task,
    reasoning_tokenizer,
    sample_task,
    sft_trace_dataset,
)
from minigpt.rft import preference_pairs, rft_dataset, rm_scores, sample_tasks
from minigpt.rlhf import (
    label_ranked_pairs,
    load_reward_model,
    new_reward_model,
    overopt_summary,
    save_reward_model,
    train_ppo,
    train_reward_model,
)
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
REASON_PRETRAIN_CKPT = Path("runs/reason_pretrain/ckpt.pt")
REASON_SFT_CKPT = Path("runs/reason_sft/ckpt.pt")
GRPO_CKPT = Path("runs/grpo/ckpt.pt")
HACK_CKPT = Path("runs/hack_demo/ckpt.pt")
OPD_CKPT = Path("runs/opd/ckpt.pt")
OPD_OFF_POLICY_CKPT = Path("runs/opd/off_policy/ckpt.pt")
CAI_DIR = Path("runs/cai")
CAI_SL_CKPT = Path("runs/cai_sl/ckpt.pt")
CAI_RL_CKPT = Path("runs/cai_rl/ckpt.pt")
RM_CKPT = Path("runs/rm/ckpt.pt")
PPO_CKPT = Path("runs/ppo/ckpt.pt")
SFT_HELDOUT = 100  # last N Alpaca rows are never trained on
CAI_HELDOUT = 20  # last N CAI prompts are never trained on


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


def load_alpaca() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = json.loads(
        download_alpaca_data(RAW_DIR / "alpaca_1k.json").read_text("utf-8")
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
        pos_encoding=args.pos_encoding,
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
        grad_accum_steps=args.grad_accum,
        bf16=args.bf16,
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
) -> tuple[str, bool]:
    """Greedy response and whether the model stopped (emitted EOS) within `n` new tokens."""
    ids = tok.encode(format_prompt(row["instruction"], row.get("input", "")))
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=n, temperature=0.0, eos_token_id=tok.eos_token_id)
    new = out[0, len(ids) :].tolist()
    stopped = tok.eos_token_id in new
    return tok.decode([t for t in new if t != tok.eos_token_id]).strip(), stopped


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
    print(f"### Model: {sample_response(model, tok, held, device)[0]}")
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


def parse_ops(text: str) -> tuple[str, ...]:
    """`--ops` value: comma-separated operators, e.g. `+` or `+,-`."""
    ops = tuple(dict.fromkeys(text.split(",")))
    if any(op not in OPS for op in ops):
        raise argparse.ArgumentTypeError(f"--ops: comma list of {','.join(OPS)}, got {text!r}")
    return ops


def add_ops_arg(s: argparse.ArgumentParser) -> None:
    s.add_argument("--ops", type=parse_ops, default=("+",), help="operators to use, e.g. + or +,-")


def load_teachers(specs: list[str], ops: tuple[str, ...]) -> dict[str, GPTModel]:
    """One model per op from `OP=PATH` (that op only) and plain `PATH` (every op) specs."""
    teachers: dict[str, GPTModel] = {}
    for spec in specs:
        op, sep, path = spec.partition("=")
        if not sep or op not in OPS:  # plain PATH (which may itself contain "=")
            op, path = "", spec
        if not Path(path).exists():
            sys.exit(f"{path} not found: pass --teacher")
        model = load_model(path)
        for o in [op] if op else ops:
            teachers[o] = model
    if missing := [o for o in ops if o not in teachers]:
        sys.exit(f"no --teacher for op(s) {','.join(missing)}")
    return {o: teachers[o] for o in ops}


def cmd_opd(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device()
    teachers = load_teachers(args.teacher or [str(GRPO_CKPT)], args.ops)
    ref = next(iter(teachers.values()))
    if args.student_config:
        d_model, num_layers, num_heads = (int(v) for v in args.student_config.split(","))
        student = GPTModel(
            GPTConfig(
                vocab_size=ref.config.vocab_size,
                context_length=ref.config.context_length,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
            )
        )
    else:
        student_path = Path(args.student)
        if not student_path.exists():
            sys.exit(f"{student_path} not found: pass --student or --student-config")
        student = load_model(student_path)
    mode = "off-policy SFT on teacher samples" if args.off_policy else f"on-policy {args.kl} KL"
    print(f"device={device} {mode} ops={','.join(args.ops)} teacher={args.teacher}")
    n_teachers = ", ".join(f"{op} {count_parameters(m):,}" for op, m in teachers.items())
    print(f"params: student {count_parameters(student):,}, teachers {n_teachers}")
    res = train_opd(
        student,
        teachers,
        steps=args.steps,
        batch_size=args.batch_size,
        group=args.group,
        digits=args.digits,
        max_new_tokens=args.max_new_tokens,
        lr=args.lr,
        kl=args.kl,
        off_policy=args.off_policy,
        temperature=args.temperature,
        eval_every=args.eval_every,
        eval_tasks=args.eval_tasks,
        seed=args.seed,
        device=device,
        ops=args.ops,
    )
    for row in res["history"]:
        line = (
            f"step {int(row['step'])} train_loss {row['loss']:.4f} "
            f"reverse_kl/token {row['kl']:.4f} greedy_acc {row['accuracy']:.3f} "
            f"mean_len {row['mean_len']:.1f}"
        )
        if len(args.ops) > 1:
            line += " | " + " ".join(f"{k} {v:.3f}" for k, v in row.items() if "[" in k)
        print(line)
    out = Path(args.out or (OPD_OFF_POLICY_CKPT if args.off_policy else OPD_CKPT))
    save_checkpoint(out, student, None, step=args.steps, val_loss=res["final_kl"])
    print(f"checkpoint -> {out}")


def connect_judge(args: argparse.Namespace) -> OllamaChat:
    """Judge from --judge-url/--judge-model; exits with a clear message if it is unreachable."""
    judge = OllamaChat(args.judge_url, args.judge_model)
    try:
        judge.chat('Reply as JSON: {"ok": true}')
    except JudgeError as err:
        sys.exit(str(err))
    return judge


def cai_prompts(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    """(principles, train prompts, held-out prompts) from --constitution and --prompts."""
    principles = load_principles(Path(args.constitution).read_text(encoding="utf-8"))
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    return (principles, *split_prompts(prompts, args.heldout))


def cmd_cai_data(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device()
    tok = load_tokenizer()
    sl = args.stage == "sl"
    base = require(SFT_CKPT, "sft") if sl else require(CAI_SL_CKPT, "cai-train --stage sl")
    model = load_model(base).to(device)
    principles, prompts, _ = cai_prompts(args)
    judge = connect_judge(args)
    rng = random.Random(args.seed)
    out = Path(args.out or CAI_DIR / ("sl.json" if sl else "pairs.json"))
    out.parent.mkdir(parents=True, exist_ok=True)

    def sample(prompt: str, n: int) -> list[str]:
        return sample_responses(
            model, tok, prompt, n, device, args.temperature, args.max_new_tokens
        )

    print(f"device={device} stage={args.stage} prompts={len(prompts)} principles={len(principles)}")
    if sl:
        answers = [(p, sample(p, 1)[0]) for p in prompts]
        records, stats = revise_rows(judge, principles, answers, rng)
        out.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
        print(f"{stats['n_records']} of {stats['n_rows']} revisions kept (rest: bad judge reply)")
    else:
        rows: list[tuple[str, str, str]] = []
        for p in prompts:
            x, y = sample(p, 2)
            rows.append((p, x, y))
        pairs, pstats = label_pairs(judge, principles, rows, rng)
        out.write_text(json.dumps([vars(s) for s in pairs], indent=1) + "\n", encoding="utf-8")
        stats_out = out.with_name("pairs_stats.json")
        stats_out.write_text(json.dumps(pstats, indent=1) + "\n", encoding="utf-8")
        print(f"{pstats['n_pairs']:.0f} of {pstats['n_rows']:.0f} pairs kept (rest: bad or equal)")
        print(f"position bias: first-shown answer won {pstats['position_bias']:.1%} (ideal 50%)")
        print(f"stats -> {stats_out}")
    print(f"data -> {out}")


def cmd_cai_train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device()
    tok = load_tokenizer()
    sl = args.stage == "sl"
    base = require(SFT_CKPT, "sft") if sl else require(CAI_SL_CKPT, "cai-train --stage sl")
    data = require(CAI_DIR / ("sl.json" if sl else "pairs.json"), f"cai-data --stage {args.stage}")
    items = json.loads(data.read_text(encoding="utf-8"))
    policy = load_model(base)
    ctx = policy.config.context_length
    epochs = args.epochs or (3 if sl else 2)
    lr = args.lr or (1e-4 if sl else 1e-5)
    if sl:
        ds = SFTDataset(items, tok, max_length=ctx)
        print(f"device={device} cai_sl_examples={len(ds)} base={base}")
        res = train_sft(
            policy, ds, epochs=epochs, batch_size=args.batch_size, learning_rate=lr, device=device
        )
        for row in res["history"]:
            print(f"epoch {row['epoch']} loss {row['loss']:.4f}")
        out = CAI_SL_CKPT
    else:
        pds = PreferenceDataset([PreferenceSample(**d) for d in items], tok, max_length=ctx)
        print(f"device={device} cai_rl_pairs={len(pds)} beta={args.beta} base={base}")
        res = train_dpo(
            policy, load_model(base), pds, beta=args.beta, epochs=epochs, lr=lr, device=device
        )
        print(f"step 1 dpo_loss {res['first_loss']:.4f} (log 2 = 0.6931)")
        for row in res["history"]:
            print(
                f"epoch {int(row['epoch'])} mean dpo_loss {row['dpo_loss']:.4f} "
                f"reward_accuracy {row['reward_accuracy']:.3f}"
            )
        out = CAI_RL_CKPT
    save_checkpoint(out, policy, None, step=epochs, val_loss=res["final_loss"])
    print(f"checkpoint -> {out}")


def cmd_cai_eval(args: argparse.Namespace) -> None:
    """Judge's mean 1-5 principle-adherence score of each checkpoint's greedy held-out answers."""
    device = get_device()
    tok = load_tokenizer()
    principles, _, held = cai_prompts(args)
    judge = connect_judge(args)
    checkpoints = [("sft", SFT_CKPT), ("cai_sl", CAI_SL_CKPT), ("cai_rl", CAI_RL_CKPT)]
    if args.checkpoints:
        checkpoints = [
            (name, Path(path))
            for name, path in (item.split("=", 1) for item in args.checkpoints.split(","))
        ]
    print(f"held-out prompts={len(held)} judge={args.judge_model}")
    print("\n| Model | Adherence (1-5) | Scored | Malformed judge replies |\n|---|---|---|---|")
    for name, ckpt in checkpoints:
        if not ckpt.exists():
            print(f"| {name} | not run: {ckpt} missing | | |")
            continue
        model = load_model(ckpt).to(device)
        answers = [sample_responses(model, tok, p, 1, device, temperature=0.0)[0] for p in held]
        rep = adherence_report(judge, principles, held, answers)
        print(f"| {name} | {rep['mean']:.2f} | {rep['n_scored']:.0f} | {rep['n_invalid']:.0f} |")


def add_cai_args(s: argparse.ArgumentParser) -> None:
    """Constitution, prompts and judge options shared by `cai-data` and `cai-eval`."""
    s.add_argument("--constitution", default="docs/cai/constitution.md")
    s.add_argument("--prompts", default="docs/cai/prompts.json")
    s.add_argument("--heldout", type=int, default=CAI_HELDOUT, help="last N prompts: cai-eval set")
    s.add_argument("--judge-url", default="http://localhost:11434", help="Ollama server")
    s.add_argument("--judge-model", default="qwen3.6:27b-ctx4k")


def check_rollout_fits(model: GPTModel, max_digits: int, max_new_tokens: int) -> None:
    longest = 2 * max_digits + 2 + max_new_tokens  # an "a+b=" prompt plus the completion
    if longest > model.config.context_length:
        sys.exit(
            f"prompt + completion can reach {longest} tokens but the context is "
            f"{model.config.context_length}: lower --max-digits or --max-new-tokens"
        )


def cmd_reward_model(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device()
    tok = reasoning_tokenizer()
    sft = load_model(require(Path(args.sft), "reason-sft")).to(device)
    check_rollout_fits(sft, args.max_digits, args.max_new_tokens)
    rng = random.Random(args.seed)
    tasks = [make_task(rng, rng.randint(1, args.max_digits)) for _ in range(args.tasks)]
    print(f"device={device} sampling {args.k} completions for each of {args.tasks} tasks")
    pairs = label_ranked_pairs(sft, tok, tasks, args.k, args.max_new_tokens, rng)
    if len(pairs) < 10:
        sys.exit(f"only {len(pairs)} strictly ordered pairs: the SFT policy is too uniform")
    print(f"{len(pairs)} labeled pairs, the last 20% held out for RM eval")
    rm = new_reward_model(sft, tok.special_to_id["<|pad|>"])
    res = train_reward_model(
        rm,
        pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )
    for row in res["history"]:
        print(
            f"epoch {int(row['epoch'])} bt_loss {row['loss']:.4f} (log 2 = 0.6931) "
            f"train_acc {row['train_accuracy']:.3f} held-out loss {row['val_loss']:.4f} "
            f"acc {row['val_accuracy']:.3f}"
        )
    out = Path(args.out)
    save_reward_model(out, rm, args.epochs, res["val_loss"], res["val_accuracy"])
    print(f"held-out pairwise accuracy {res['val_accuracy']:.3f} (chance 0.5)")
    print(f"checkpoint -> {out}")


def cmd_ppo(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device()
    tok = reasoning_tokenizer()
    sft_path = require(Path(args.sft), "reason-sft")
    policy, ref = load_model(sft_path), load_model(sft_path)
    rm = load_reward_model(require(Path(args.rm), "reward-model"))
    check_rollout_fits(policy, args.max_digits, args.max_new_tokens)
    print(f"device={device} ppo steps={args.steps} beta={args.beta} lr={args.lr}")

    def show(step: int, row: dict[str, float]) -> None:
        if step == 1 or step % args.log_every == 0:
            print(
                f"step {step} reward {row['reward']:+.3f} gold_acc {row['gold_accuracy']:.3f} "
                f"format {row['gold_format']:.3f} kl {row['kl']:.2f} "
                f"value_loss {row['value_loss']:.3f} len {row['length']:.1f}"
            )

    res = train_ppo(
        policy,
        ref,
        rm,
        tok,
        steps=args.steps,
        tasks_per_step=args.tasks_per_step,
        samples_per_task=args.samples_per_task,
        max_digits=args.max_digits,
        max_new_tokens=args.max_new_tokens,
        beta=args.beta,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
        device=device,
        on_step=show,
    )
    out = Path(args.out)
    save_checkpoint(out, policy, None, step=args.steps, val_loss=float("nan"))
    (out.parent / "history.json").write_text(json.dumps(res["history"]))
    s = overopt_summary(res["history"])
    print(f"proxy reward {s['reward_first']:+.3f} -> {s['reward_last']:+.3f} (10-step means)")
    print(f"gold accuracy peak {s['gold_peak']:.3f} at step {int(s['gold_peak_step'])}")
    print(f"gold accuracy last {s['gold_last']:.3f}")
    if s["reward_last"] > s["reward_first"] and s["gold_last"] < s["gold_peak"]:
        print("proxy rose while gold fell below its peak: overoptimization (Gao et al. 2022)")
    print(f"checkpoint + history.json -> {out.parent}")


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
    gen_rows = list({id(r): r for r in [*short, *open_form]}.values())  # generate each row once
    rank_lines: list[str] = []
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
        shakespeare = create_dataloader(
            val_ids, model.config.context_length, 16, shuffle=False, drop_last=False
        )
        alpaca = DataLoader(
            SFTDataset(heldout, tok, max_length=model.config.context_length),
            batch_size=16,
            collate_fn=sft_collate_fn,
        )
        gen = {id(r): sample_response(model, tok, r, device) for r in gen_rows}
        em = [exact_match(gen[id(r)][0], r["output"]) for r in short]
        responses = [gen[id(r)][0] for r in open_form]
        f1 = [token_f1(p, r["output"]) for p, r in zip(responses, open_form, strict=True)]
        f1_rows[name] = f1
        responses_by_name[name] = responses
        scores = [
            judge.score(r["instruction"], p, r["output"])
            for p, r in zip(responses, open_form, strict=True)
        ]
        judge_rows[name] = scores
        wins_sum, wins_tok = likelihood_ranking(
            model, tok, heldout, device, model.config.context_length
        )
        report = EvalReport(
            perplexity=compute_perplexity(model, alpaca, device),
            vocab_size=model.config.vocab_size,
            exact_match=sum(em) / max(1, len(em)),
            token_f1=sum(f1) / max(1, len(f1)),
            judge_mean=sum(scores) / max(1, len(scores)),
            rank_wins_sum=wins_sum,
            rank_wins_per_token=wins_tok,
            rank_n=len(heldout),
            stopped=sum(stopped for _, stopped in gen.values()),
            stopped_n=len(gen),
        )
        rank_cells = []
        for wins in (report.rank_wins_sum, report.rank_wins_per_token):
            lo, hi = wilson_interval(wins, report.rank_n)
            rank_cells.append(f"{wins / report.rank_n:.3f} [{lo:.3f}, {hi:.3f}]")
        rank_lines.append(
            f"| {name} | {rank_cells[0]} | {rank_cells[1]} | {report.stopped}/{report.stopped_n} |"
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
    if rank_lines:
        print(
            f"\nNot in the aggregate. Ranking: share of the {len(heldout)} held-out rows where the "
            "reference beats row (i+37)'s answer for prompt i (chance 0.5; Wilson 95% CI). "
            "Summed log-prob favours short answers; per-token divides by response length. "
            "Stopped: greedy responses that emitted EOS within 60 tokens."
        )
        print("\n| Model | Ranking acc, summed | Ranking acc, per-token | Stopped |")
        print("|---|---|---|---|")
        print("\n".join(rank_lines))
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


def parse_efforts(s: str) -> tuple[int, ...]:
    """`--efforts 1,9` -> (1, 9); the empty string means no effort conditioning."""
    return tuple(int(x) for x in s.split(",")) if s else ()


def format_reasoning(res: dict[str, Any]) -> str:
    rows = [
        f"{d} digits: accuracy {m['accuracy']:.3f}  format {m['format']:.3f}  "
        f"tokens {m['mean_tokens']:.1f}  think {m['think_tokens']:.1f}"
        for d, m in res["per_digit"].items()
    ]
    rows.append(
        f"overall:  accuracy {res['accuracy']:.3f}  format {res['format']:.3f}  "
        f"tokens {res['mean_tokens']:.1f}  think {res['think_tokens']:.1f}"
    )
    if len(res["per_op"]) > 1:
        rows += [f"op {op}: accuracy {acc:.3f}" for op, acc in res["per_op"].items()]
    return "\n".join(rows)


def report_reasoning(
    model: GPTModel,
    tok: BPETokenizer,
    n: int,
    seed: int,
    efforts: tuple[int, ...],
    ops: tuple[str, ...] = ("+",),
) -> str:
    """Greedy report on the fixed tasks, one block per effort level (one plain block if none)."""
    blocks = []
    for b in efforts or (None,):
        res = format_reasoning(evaluate_reasoning(model, tok, n, seed, effort=b, ops=ops))
        blocks.append(res if b is None else f"effort {b}:\n{res}")
    return "\n".join(blocks)


def cmd_reason_pretrain(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device()
    tok = reasoning_tokenizer()
    rng = random.Random(args.seed)
    train_ids = tok.encode(equation_corpus(rng, args.corpus_size, 4, args.ops))
    val_ids = tok.encode(equation_corpus(rng, max(1, args.corpus_size // 25), 4, args.ops))
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        use_sdpa=True,
    )
    model = GPTModel(cfg)
    params = sum(p.numel() for p in model.parameters())
    print(f"device={device} train_tokens={len(train_ids):,} params={params:,}")
    train_loader = create_dataloader(
        train_ids, cfg.context_length, args.batch_size, stride=1, shuffle=True
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
        print(f"step {int(row['step']):5d} train {row['train_loss']:.4f} val {row['val_loss']:.4f}")
    print(f"checkpoint -> {res['checkpoint_path']}")


def cmd_reason_sft(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device()
    tok = reasoning_tokenizer()
    model = load_model(require(Path(args.base), "reason-pretrain"))
    rng = random.Random(args.seed + 1000)  # apart from the fixed tasks of `reason-eval`
    tasks = [sample_task(rng, rng.randint(1, args.max_digits), args.ops) for _ in range(args.n)]
    ops = ",".join(args.ops)
    print(f"device={device} traces={len(tasks)} digits=1-{args.max_digits} ops={ops}")
    res = train_sft(
        model,
        sft_trace_dataset(tasks, tok, args.efforts, args.nothink_frac, args.seed),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
    )
    for row in res["history"]:
        print(f"epoch {row['epoch']} loss {row['loss']:.4f}")
    save_checkpoint(Path(args.out), model, None, step=args.epochs, val_loss=res["final_loss"])
    print(f"checkpoint -> {args.out}")


def cmd_reason_eval(args: argparse.Namespace) -> None:
    device = get_device()
    model = load_model(require(Path(args.checkpoint), "reason-sft")).to(device)
    tok = reasoning_tokenizer()
    print(report_reasoning(model, tok, args.n, args.seed, args.efforts, args.ops))


def run_grpo(
    args: argparse.Namespace,
    reward_fn: RewardFn,
    observe: Callable[[Samples], dict[str, float]] | None,
) -> list[dict[str, float]]:
    """Train from the reasoning SFT checkpoint, save it, print greedy accuracy before and after."""
    seed_everything(args.seed)
    device = get_device()
    tok = reasoning_tokenizer()
    policy = load_model(require(Path(args.base), "reason-sft"))
    ref = load_model(args.base)
    print(
        f"device={device} loss_agg={args.loss_agg} std_norm={not args.no_std_norm} "
        f"unbiased_kl={args.unbiased_kl} fill_batch={args.fill_batch} "
        f"policy_loss={args.policy_loss} advantage={args.advantage} "
        f"clip=[{args.clip_eps},{args.clip_eps_high}] beta={args.beta} mu={args.mu} "
        f"ref_reset_every={args.ref_reset_every}"
    )

    def log(rec: dict[str, float]) -> None:
        if int(rec["step"]) % args.log_every == 0:
            skip = ("step", "tp", "fp", "fn")
            shown = " ".join(f"{k} {v:.3f}" for k, v in rec.items() if k not in skip)
            print(f"step {int(rec['step']):4d} {shown}")

    res = train_grpo(
        policy,
        ref,
        tok,
        steps=args.steps,
        tasks_per_step=args.tasks,
        group_size=args.group,
        digits=(args.min_digits, args.max_digits),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        lr=args.lr,
        beta=args.beta,
        clip_eps=args.clip_eps,
        mu=args.mu,
        std_norm=not args.no_std_norm,
        loss_agg=args.loss_agg,
        unbiased_kl=args.unbiased_kl,
        fill_batch=args.fill_batch,
        efforts=args.efforts,
        effort_k0=args.effort_k0,
        reward_fn=reward_fn,
        ops=args.ops,
        policy_loss=args.policy_loss,
        clip_eps_high=args.clip_eps_high,
        advantage=args.advantage,
        ref_reset_every=args.ref_reset_every,
        observe=observe,
        on_step=log,
        seed=args.seed,
        device=device,
    )
    out = Path(args.out)
    save_checkpoint(out, policy, None, step=args.steps, val_loss=float("nan"))
    (out.parent / "history.json").write_text(json.dumps(res["history"]), encoding="utf-8")
    print("greedy on fixed tasks, SFT start:")
    print(report_reasoning(ref, tok, args.eval_n, 0, args.efforts, args.ops))
    print("greedy on fixed tasks, after RL:")
    print(report_reasoning(policy, tok, args.eval_n, 0, args.efforts, args.ops))
    print(f"checkpoint -> {out}")
    history: list[dict[str, float]] = res["history"]
    return history


def cmd_grpo(args: argparse.Namespace) -> None:
    run_grpo(args, default_reward, None)


def cmd_hack_demo(args: argparse.Namespace) -> None:
    """GRPO against the substring grader, tracking proxy vs true reward and the CoT monitor."""
    summary = summarize_hack(run_grpo(args, lenient_reward, rollout_monitor(lenient_reward)))
    for name, label in (
        ("reward", "proxy reward (substring grader)"),
        ("accuracy", "true accuracy (answer tag)"),
        ("hack_rate", "hack rate (proxy 1, true 0)"),
        ("flag_rate", "monitor flag rate"),
    ):
        first, last = summary[f"{name}_first"], summary[f"{name}_last"]
        print(f"{label:32s} first fifth {first:.3f} -> last fifth {last:.3f}")
    precision, recall = summary["precision"], summary["recall"]
    print(f"monitor vs hack label: precision {precision:.3f}, recall {recall:.3f}")


def add_grpo_args(s: argparse.ArgumentParser, out: Path) -> None:
    s.add_argument("--base", default=str(REASON_SFT_CKPT))
    s.add_argument("--steps", type=int, default=200)
    s.add_argument("--tasks", type=int, default=8, help="tasks sampled per step (B)")
    s.add_argument("--group", type=int, default=8, help="completions per task (G)")
    s.add_argument("--min-digits", type=int, default=1)
    s.add_argument("--max-digits", type=int, default=3)
    s.add_argument("--max-new-tokens", type=int, default=80)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--lr", type=float, default=5e-5)
    s.add_argument("--beta", type=float, default=0.04, help="weight of the KL to the SFT model")
    s.add_argument("--clip-eps", type=float, default=0.2)
    s.add_argument("--mu", type=int, default=1, help="inner optimizer steps per batch")
    s.add_argument(
        "--no-std-norm",
        action="store_true",
        help="advantage = reward - group mean, no std division (Dr. GRPO)",
    )
    s.add_argument(
        "--loss-agg",
        choices=["seq-mean", "token-sum", "token-mean", "group-mean"],
        default="seq-mean",
        help="per-sequence token mean (GRPO), sum over a constant max length (Dr. GRPO), "
        "total-token mean over the batch (DAPO, CISPO) or over each group (Magistral)",
    )
    s.add_argument(
        "--unbiased-kl",
        action="store_true",
        help="k3 KL times pi/pi_old, DeepSeek-V3.2's unbiased estimate",
    )
    s.add_argument(
        "--fill-batch",
        action="store_true",
        help="resample (up to 4 rounds) until --tasks groups have mixed rewards (DAPO)",
    )
    s.add_argument(
        "--efforts",
        type=parse_efforts,
        default=(),
        help="comma list of effort levels, e.g. 1,9: one group per level per task, length "
        "penalty decaying with the level (DeepSeek-V4.1 reasoning effort)",
    )
    s.add_argument("--effort-k0", type=float, default=0.5, help="length penalty at the lowest b")
    s.add_argument("--eval-n", type=int, default=50, help="fixed tasks per digit count")
    s.add_argument("--log-every", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument(
        "--policy-loss",
        choices=["ppo", "gspo", "cispo"],
        default="ppo",
        help="surrogate: per-token PPO clip (GRPO), sequence-level ratio (GSPO) or "
        "stop-gradient clipped IS weight (CISPO)",
    )
    s.add_argument(
        "--clip-eps-high",
        type=float,
        default=None,
        help="upper clip range, default --clip-eps (DAPO/Magistral clip-higher: 0.28)",
    )
    s.add_argument(
        "--advantage",
        choices=["group", "batch", "rloo"],
        default="group",
        help="group: (r - mean)/std (--no-std-norm drops the std); batch: group-centred, "
        "divided by the batch std (Magistral); rloo: r_i - mean of the others, no std",
    )
    s.add_argument(
        "--ref-reset-every",
        type=int,
        default=0,
        help="every N steps set the KL reference to the current policy and reset the "
        "optimizer (ProRL); 0 = never",
    )
    add_ops_arg(s)
    s.add_argument("--out", default=str(out))


def draw_tasks(args: argparse.Namespace, rng: random.Random) -> list[Task]:
    digits = range(args.min_digits, args.max_digits + 1)
    return [sample_task(rng, rng.choice(digits), args.ops) for _ in range(args.tasks)]


def cmd_reason_rft(args: argparse.Namespace) -> None:
    """Per round: sample K per task, select (verifier or RM best-of-K), SFT. Best round seeds on."""
    if args.sampler and args.rounds > 1:
        sys.exit("--sampler is a fixed teacher: use --rounds 1")
    if args.select == "rm" and not args.rm:
        sys.exit("--select rm needs --rm CHECKPOINT")
    seed_everything(args.seed)
    device = get_device()
    tok = reasoning_tokenizer()
    model = load_model(require(Path(args.base), "reason-sft")).to(device)
    teacher = load_model(require(Path(args.sampler), "grpo")).to(device) if args.sampler else None
    rm = None
    if args.select == "rm":
        rm = load_reward_model(require(Path(args.rm), "reward-model")).to(device).eval()
    check_rollout_fits(model, args.max_digits, args.max_new_tokens)
    rng = random.Random(args.seed + 2000)
    best_acc, best_state = -1.0, dict[str, Any]()
    for r in range(1, args.rounds + 1):
        samples = sample_tasks(
            teacher or model,
            tok,
            draw_tasks(args, rng),
            args.k,
            args.max_new_tokens,
            args.temperature,
        )
        scores = rm_scores(rm, samples, device) if rm is not None else None
        data, st = rft_dataset(tok, samples, scores, args.keep)
        if not len(data):
            sys.exit("no sample was selected: the sampler never solves a task")
        print(
            f"round {r}: sample accuracy {st['sample_accuracy']:.3f} kept {int(st['kept'])} "
            f"(verifier-correct {st['precision']:.3f}) select={args.select}"
        )
        res = train_sft(model, data, args.epochs, args.batch_size, args.lr, device=device)
        model.eval()
        acc = evaluate_reasoning(model, tok, args.eval_n, 1, ops=args.ops)["accuracy"]
        print(f"round {r}: sft loss {res['final_loss']:.4f} greedy accuracy (seed 1) {acc:.3f}")
        if acc >= best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
        else:
            model.load_state_dict(best_state)
            print(f"round {r} is worse than the best so far: it does not seed the next round")
    model.load_state_dict(best_state)
    save_checkpoint(Path(args.out), model, None, step=args.rounds, val_loss=float("nan"))
    print("greedy on fixed tasks:")
    print(report_reasoning(model, tok, args.eval_n, 0, (), args.ops))
    print(f"checkpoint -> {args.out}")


def cmd_reason_dpo(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device()
    tok = reasoning_tokenizer()
    policy = load_model(require(Path(args.base), "reason-sft"))
    ref = load_model(args.base)
    check_rollout_fits(policy, args.max_digits, args.max_new_tokens)
    rng = random.Random(args.seed + 3000)
    samples = sample_tasks(
        ref.to(device), tok, draw_tasks(args, rng), args.k, args.max_new_tokens, args.temperature
    )
    ds = preference_pairs(tok, samples, args.seed)
    acc = sum(accuracy_reward(s.task, t) for s in samples for t in s.texts)
    print(
        f"device={device} sample accuracy {acc / (len(samples) * args.k):.3f} "
        f"pairs {len(ds)}/{len(samples)} tasks beta={args.beta} nll_alpha={args.nll_alpha}"
    )
    if not len(ds):
        sys.exit("no task has both a better and a worse sample: nothing to prefer")
    res = train_dpo(
        policy,
        ref,
        ds,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        nll_alpha=args.nll_alpha,
    )
    for row in res["history"]:
        print(
            f"epoch {int(row['epoch'])} dpo_loss {row['dpo_loss']:.4f} "
            f"reward_accuracy {row['reward_accuracy']:.3f} "
            f"chosen {row['logp_chosen_vs_ref']:+.3f} rejected {row['logp_rejected_vs_ref']:+.3f}"
        )
    policy.eval()
    save_checkpoint(Path(args.out), policy, None, step=args.epochs, val_loss=res["final_loss"])
    print("greedy on fixed tasks:")
    print(report_reasoning(policy, tok, args.eval_n, 0, (), args.ops))
    print(f"checkpoint -> {args.out}")


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
        "--pos-encoding",
        choices=["learned", "rope"],
        default="learned",
        help="position encoding: learned table (GPT-2 style, default) or rotary (RoPE)",
    )
    s.add_argument(
        "--sampling",
        choices=["aligned", "random"],
        default="random",
        help="train windows: random offsets (stride 1, default) or aligned (stride = context)",
    )
    s.add_argument("--results-out", default=None, help="write per-run JSON (losses, history)")
    s.add_argument(
        "--grad-accum", type=int, default=1, help="micro-batches per optimizer step (default 1)"
    )
    s.add_argument("--bf16", action="store_true", help="bf16 autocast for forward and loss")
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

    s = sub.add_parser("opd", help="distil a teacher into a smaller student, on-policy")
    s.add_argument(
        "--teacher",
        action="append",
        help="OP=PATH (one teacher per op) or PATH (one teacher for every op); repeatable; "
        f"default {GRPO_CKPT}",
    )
    s.add_argument("--student", default=str(REASON_SFT_CKPT), help="student init checkpoint")
    s.add_argument(
        "--student-config",
        default=None,
        help="d_model,num_layers,num_heads: random-init student of this size instead of --student",
    )
    s.add_argument("--steps", type=int, default=100)
    s.add_argument("--batch-size", type=int, default=8, help="tasks per step")
    s.add_argument("--group", type=int, default=1, help="completions per task")
    s.add_argument("--digits", type=int, default=3, help="operands have 1..digits digits")
    s.add_argument("--max-new-tokens", type=int, default=64)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument(
        "--kl",
        choices=["reverse", "forward"],
        default="reverse",
        help="reverse is mode-seeking, forward is mode-covering",
    )
    s.add_argument(
        "--off-policy",
        action="store_true",
        help="baseline: the teacher samples and the student does SFT on those samples",
    )
    s.add_argument("--temperature", type=float, default=1.0, help="sampler's; 0 = greedy")
    s.add_argument("--eval-every", type=int, default=25)
    s.add_argument("--eval-tasks", type=int, default=64)
    s.add_argument("--seed", type=int, default=0)
    add_ops_arg(s)
    s.add_argument("--out", default=None, help=f"default {OPD_CKPT} ({OPD_OFF_POLICY_CKPT} off)")
    s.set_defaults(func=cmd_opd)

    s = sub.add_parser("cai-data", help="Constitutional AI data: SL revisions or RLAIF pairs")
    s.add_argument("--stage", choices=["sl", "rl"], required=True)
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--max-new-tokens", type=int, default=64)
    s.add_argument("--out", default=None, help="default runs/cai/sl.json or runs/cai/pairs.json")
    s.add_argument("--seed", type=int, default=0)
    add_cai_args(s)
    s.set_defaults(func=cmd_cai_data)

    s = sub.add_parser("cai-train", help="SFT on revisions (sl) or DPO on AI pairs (rl)")
    s.add_argument("--stage", choices=["sl", "rl"], required=True)
    s.add_argument("--epochs", type=int, default=None, help="default 3 (sl) or 2 (rl)")
    s.add_argument("--lr", type=float, default=None, help="default 1e-4 (sl) or 1e-5 (rl)")
    s.add_argument("--batch-size", type=int, default=8, help="sl only; rl uses the DPO default")
    s.add_argument("--beta", type=float, default=0.1, help="rl only")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_cai_train)

    s = sub.add_parser("cai-eval", help="judge scores held-out prompts for principle adherence")
    s.add_argument(
        "--checkpoints",
        default=None,
        help="comma-separated name=path list (default: sft, cai_sl, cai_rl)",
    )
    add_cai_args(s)
    s.set_defaults(func=cmd_cai_eval)

    s = sub.add_parser("reward-model", help="train a Bradley-Terry reward model (RLHF step 1)")
    s.add_argument("--sft", default=str(REASON_SFT_CKPT), help="policy that writes the samples")
    s.add_argument("--tasks", type=int, default=800, help="prompts to sample for")
    s.add_argument("--k", type=int, default=4, help="completions per prompt; one pair per prompt")
    s.add_argument("--max-digits", type=int, default=3, help="operands have 1..max-digits digits")
    s.add_argument("--max-new-tokens", type=int, default=80)
    s.add_argument("--epochs", type=int, default=4)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--lr", type=float, default=3e-5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=str(RM_CKPT))
    s.set_defaults(func=cmd_reward_model)

    s = sub.add_parser("ppo", help="PPO against the reward model (RLHF step 2)")
    s.add_argument("--sft", default=str(REASON_SFT_CKPT), help="initial policy and KL reference")
    s.add_argument("--rm", default=str(RM_CKPT))
    s.add_argument("--steps", type=int, default=150)
    s.add_argument("--tasks-per-step", type=int, default=8)
    s.add_argument("--samples-per-task", type=int, default=8)
    s.add_argument("--max-digits", type=int, default=3, help="operands have 1..max-digits digits")
    s.add_argument("--max-new-tokens", type=int, default=80)
    s.add_argument("--beta", type=float, default=0.02, help="KL penalty to the SFT policy")
    s.add_argument("--lr", type=float, default=2e-5)
    s.add_argument("--ppo-epochs", type=int, default=2, help="passes over each rollout batch")
    s.add_argument("--minibatch-size", type=int, default=16)
    s.add_argument("--log-every", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=str(PPO_CKPT))
    s.set_defaults(func=cmd_ppo)

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

    s = sub.add_parser("reason-pretrain", help="pretrain a tiny GPT on bare equations a+b=c")
    s.add_argument("--name", default="reason_pretrain", help="checkpoint goes to runs/<name>")
    s.add_argument("--steps", type=int, default=1500)
    s.add_argument("--corpus-size", type=int, default=50000, help="equations (digits 1-4)")
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--context-length", type=int, default=128)
    s.add_argument("--d-model", type=int, default=128)
    s.add_argument("--num-layers", type=int, default=4)
    s.add_argument("--num-heads", type=int, default=4)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--eval-interval", type=int, default=250)
    s.add_argument("--seed", type=int, default=0)
    add_ops_arg(s)
    s.set_defaults(func=cmd_reason_pretrain)

    s = sub.add_parser("reason-sft", help="cold-start SFT on reference chain-of-thought traces")
    s.add_argument("--base", default=str(REASON_PRETRAIN_CKPT))
    s.add_argument("--n", type=int, default=20000, help="traces")
    s.add_argument("--max-digits", type=int, default=3)
    s.add_argument(
        "--efforts",
        type=parse_efforts,
        default=(),
        help="comma list of effort levels, e.g. 1,9: each trace gets a random `e<b>:` prefix",
    )
    s.add_argument(
        "--nothink-frac",
        type=float,
        default=0.0,
        help="share of traces with an empty think block, independent of the effort level",
    )
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--lr", type=float, default=5e-4)
    s.add_argument("--seed", type=int, default=0)
    add_ops_arg(s)
    s.add_argument("--out", default=str(REASON_SFT_CKPT))
    s.set_defaults(func=cmd_reason_sft)

    s = sub.add_parser("reason-eval", help="greedy accuracy / format / length, 1-5 digits")
    s.add_argument("--checkpoint", default=str(REASON_SFT_CKPT))
    add_ops_arg(s)
    s.add_argument("--n", type=int, default=50, help="tasks per digit count")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--efforts", type=parse_efforts, default=(), help="comma list, e.g. 1,9")
    s.set_defaults(func=cmd_reason_eval)

    s = sub.add_parser("grpo", help="GRPO on verifiable addition from the reasoning SFT model")
    add_grpo_args(s, GRPO_CKPT)
    s.set_defaults(func=cmd_grpo)

    s = sub.add_parser("hack-demo", help="GRPO against a hackable grader + CoT monitor")
    add_grpo_args(s, HACK_CKPT)
    s.set_defaults(func=cmd_hack_demo)

    s = sub.add_parser("reason-rft", help="rejection-sampling fine-tuning / expert iteration")
    s.add_argument("--base", default=str(REASON_SFT_CKPT), help="the model that is fine-tuned")
    s.add_argument("--sampler", default=None, help="sample from this model instead (1 round)")
    s.add_argument("--select", choices=["verifier", "rm"], default="verifier")
    s.add_argument("--rm", default=None, help="reward model checkpoint for --select rm")
    s.add_argument("--rounds", type=int, default=1, help="the best round seeds the next")
    s.add_argument("--tasks", type=int, default=2000, help="tasks sampled per round")
    s.add_argument("--k", type=int, default=8, help="completions per task")
    s.add_argument("--keep", type=int, default=2, help="distinct completions kept per task")
    s.add_argument("--min-digits", type=int, default=1)
    s.add_argument("--max-digits", type=int, default=3)
    s.add_argument("--max-new-tokens", type=int, default=80)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--epochs", type=int, default=2)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--lr", type=float, default=5e-5)
    s.add_argument("--eval-n", type=int, default=50, help="fixed tasks per digit count")
    s.add_argument("--seed", type=int, default=0)
    add_ops_arg(s)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_reason_rft)

    s = sub.add_parser("reason-dpo", help="DPO on (correct, incorrect) pairs of own samples")
    s.add_argument("--base", default=str(REASON_SFT_CKPT), help="policy and DPO reference")
    s.add_argument("--tasks", type=int, default=2000, help="tasks sampled")
    s.add_argument("--k", type=int, default=8, help="completions per task")
    s.add_argument("--min-digits", type=int, default=1)
    s.add_argument("--max-digits", type=int, default=3)
    s.add_argument("--max-new-tokens", type=int, default=80)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--beta", type=float, default=0.1)
    s.add_argument("--epochs", type=int, default=2)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--lr", type=float, default=1e-5)
    s.add_argument("--nll-alpha", type=float, default=0.2, help="Llama 3 uses 0.2")
    s.add_argument("--eval-n", type=int, default=50, help="fixed tasks per digit count")
    s.add_argument("--seed", type=int, default=0)
    add_ops_arg(s)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_reason_dpo)

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

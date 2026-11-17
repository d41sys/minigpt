#!/usr/bin/env bash
# After run_judge.sh (same clone, same runs/): contamination check, DPO with and without the
# Llama 3 NLL term on the published pairs (identical to runs/dpo/pairs.json in this run), Qwen-judged eval of both.
set -euo pipefail
cd ~/minigpt-judge
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
python docs/judge-eval/contamination.py
minigpt dpo --pairs-file docs/judge-eval/dpo_pairs.json --out runs/dpo_a0/ckpt.pt | tee runs/eval/dpo_a0.log
minigpt dpo --pairs-file docs/judge-eval/dpo_pairs.json --nll-alpha 0.2 --out runs/dpo_nll0.2/ckpt.pt | tee runs/eval/dpo_nll0.2.log
minigpt eval --n 100 --judge ollama --judge-model qwen3.6:27b-ctx4k --paired-vs sft \
  --rows-out runs/eval/rows_nll.json \
  --checkpoints sft=runs/sft/ckpt.pt,dpo_b0.1=runs/dpo_a0/ckpt.pt,dpo_nll0.2=runs/dpo_nll0.2/ckpt.pt \
  | tee runs/eval/eval_nll.log
date -Is
echo DONE

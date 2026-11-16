#!/usr/bin/env bash
# Retrain the README pipeline (run from a clone with a CUDA venv) on one idle GPU, then score every checkpoint with the Qwen judge.
set -euo pipefail
cd ~/minigpt-judge
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
mkdir -p runs/eval
{
  date -Is; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES"
  ollama list | grep -E "^NAME|qwen3.6:27b-ctx4k"
} > runs/eval/env.txt
minigpt train-tokenizer
minigpt pretrain --sampling aligned --dropout 0.1 --steps 2000 --seed 0
minigpt sft
minigpt dpo
minigpt dpo --beta 0.05 --out runs/dpo_b0.05/ckpt.pt
minigpt dpo --beta 0.5 --out runs/dpo_b0.5/ckpt.pt
minigpt eval --n 100 --judge ollama --judge-model qwen3.6:27b-ctx4k --paired-vs sft \
  --rows-out runs/eval/rows.json \
  --checkpoints pretrain=runs/pretrain/ckpt.pt,sft=runs/sft/ckpt.pt,dpo_b0.05=runs/dpo_b0.05/ckpt.pt,dpo_b0.1=runs/dpo/ckpt.pt,dpo_b0.5=runs/dpo_b0.5/ckpt.pt
date -Is
echo DONE

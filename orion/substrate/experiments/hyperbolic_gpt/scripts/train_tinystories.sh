#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
TRAIN="${ROOT}/orion/substrate/experiments/hyperbolic_gpt/train.py"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  "${TRAIN}" \
  --dataset tinystories \
  --max_docs 50000 \
  --max_tokens 5000000 \
  --out_dir ./runs/tinystories_hypgpt_4l_256d \
  --max_steps 10000 \
  --batch_size 16 \
  --block_size 256 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --lr 3e-4 \
  --eval_interval 250 \
  --save_interval 1000 \
  --amp

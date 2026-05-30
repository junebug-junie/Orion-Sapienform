#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  orion/substrate/experiments/hyperbolic_gpt/train_v2.py \
  --dataset tinystories \
  --out_dir ./runs/tinystories_hypgpt_v2_12l_768d \
  --max_steps 50000 \
  --max_docs 1000000 \
  --max_tokens 100000000 \
  --batch_size 4 \
  --grad_accum 4 \
  --block_size 256 \
  --n_layer 12 \
  --n_head 12 \
  --n_embd 768 \
  --lr 3e-4 \
  --semantic_adapter_rank 128 \
  --margin_gap_loss_weight 0.001 \
  --margin_gap_epsilon 0.5 \
  --eval_interval 250 \
  --save_interval 1000 \
  --amp

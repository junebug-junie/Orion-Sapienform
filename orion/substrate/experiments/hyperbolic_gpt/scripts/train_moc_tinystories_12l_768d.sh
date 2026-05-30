#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  orion/substrate/experiments/hyperbolic_gpt/train_moc.py \
  --dataset tinystories \
  --out_dir ./runs/tinystories_hypgpt_moc_12l_768d \
  --max_steps 50000 \
  --max_docs 1000000 \
  --max_tokens 100000000 \
  --batch_size 4 \
  --grad_accum 8 \
  --block_size 512 \
  --n_layer 12 \
  --n_head 12 \
  --n_embd 768 \
  --lr 2e-4 \
  --semantic_adapter_rank 128 \
  --margin_gap_loss_weight 0.001 \
  --margin_gap_epsilon 0.5 \
  --curvature_mode per_head \
  --geo_lambda_mode per_head \
  --moc_curvature_jitter 0.05 \
  --moc_lambda_jitter 0.05 \
  --eval_interval 500 \
  --save_interval 2500 \
  --amp

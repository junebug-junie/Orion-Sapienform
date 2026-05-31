#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  orion/substrate/experiments/hyperbolic_gpt/train_v2.py \
  --dataset fineweb_edu \
  --fineweb_edu_name "${FINEWEB_EDU_NAME:-sample-10BT}" \
  --out_dir "${OUT_DIR:-./runs/fineweb_edu_hypgpt_v2_12l_768d_100m}" \
  --max_steps "${MAX_STEPS:-20000}" \
  --max_docs "${MAX_DOCS:-250000}" \
  --max_tokens "${MAX_TOKENS:-100000000}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --grad_accum "${GRAD_ACCUM:-8}" \
  --block_size "${BLOCK_SIZE:-512}" \
  --n_layer "${N_LAYER:-12}" \
  --n_head "${N_HEAD:-12}" \
  --n_embd "${N_EMBD:-768}" \
  --lr "${LR:-2e-4}" \
  --semantic_adapter_rank "${SEMANTIC_ADAPTER_RANK:-128}" \
  --margin_gap_loss_weight "${MARGIN_GAP_LOSS_WEIGHT:-0.001}" \
  --margin_gap_epsilon "${MARGIN_GAP_EPSILON:-0.5}" \
  --eval_interval "${EVAL_INTERVAL:-500}" \
  --save_interval "${SAVE_INTERVAL:-2500}" \
  --amp

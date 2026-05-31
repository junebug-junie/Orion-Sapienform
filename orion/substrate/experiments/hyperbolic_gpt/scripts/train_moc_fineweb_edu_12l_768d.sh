#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=.

# Real-corpus MoC shakedown.
# Defaults to HuggingFaceFW/fineweb-edu sample-10BT via --fineweb_edu_name.
# Effective tokens/step = batch_size * grad_accum * nproc_per_node * block_size.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  orion/substrate/experiments/hyperbolic_gpt/train_moc.py \
  --dataset fineweb_edu \
  --fineweb_edu_name "${FINEWEB_EDU_NAME:-sample-10BT}" \
  --out_dir "${OUT_DIR:-./runs/fineweb_edu_hypgpt_moc_12l_768d_100m}" \
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
  --curvature_mode "${CURVATURE_MODE:-per_head}" \
  --geo_lambda_mode "${GEO_LAMBDA_MODE:-per_head}" \
  --moc_curvature_jitter "${MOC_CURVATURE_JITTER:-0.05}" \
  --moc_lambda_jitter "${MOC_LAMBDA_JITTER:-0.05}" \
  --eval_interval "${EVAL_INTERVAL:-500}" \
  --save_interval "${SAVE_INTERVAL:-2500}" \
  --amp

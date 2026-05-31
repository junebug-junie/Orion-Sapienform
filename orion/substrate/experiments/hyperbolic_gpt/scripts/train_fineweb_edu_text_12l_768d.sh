#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=.

TEXT_PATH="${TEXT_PATH:-./data/fineweb_edu_sample_10bt.txt}"
if [[ ! -f "${TEXT_PATH}" ]]; then
  echo "Missing ${TEXT_PATH}. Run: bash orion/substrate/experiments/hyperbolic_gpt/scripts/cache_fineweb_edu_sample.sh" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  orion/substrate/experiments/hyperbolic_gpt/train.py \
  --dataset text \
  --text_path "${TEXT_PATH}" \
  --out_dir "${OUT_DIR:-./runs/fineweb_edu_text_hypgpt_12l_768d}" \
  --max_steps "${MAX_STEPS:-20000}" \
  --max_tokens "${MAX_TOKENS:-100000000}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --grad_accum "${GRAD_ACCUM:-8}" \
  --block_size "${BLOCK_SIZE:-512}" \
  --n_layer "${N_LAYER:-12}" \
  --n_head "${N_HEAD:-12}" \
  --n_embd "${N_EMBD:-768}" \
  --lr "${LR:-2e-4}" \
  --eval_interval "${EVAL_INTERVAL:-500}" \
  --save_interval "${SAVE_INTERVAL:-2500}" \
  --amp

#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=.
python orion/substrate/experiments/hyperbolic_gpt/generate_moc.py \
  --checkpoint ./runs/tinystories_hypgpt_moc_12l_768d \
  --prompt "Once upon a time there was a little girl named Lily. She wanted to" \
  --max_new_tokens 160 \
  --temperature 0.6 \
  --top_k 20

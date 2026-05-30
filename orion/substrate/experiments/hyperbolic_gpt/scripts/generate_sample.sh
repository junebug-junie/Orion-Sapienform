#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
python orion/substrate/experiments/hyperbolic_gpt/generate.py \
  --checkpoint ./runs/tinystories_hypgpt_4l_256d \
  --prompt "Once upon a time" \
  --max_new_tokens 120 \
  --temperature 0.9 \
  --top_k 40

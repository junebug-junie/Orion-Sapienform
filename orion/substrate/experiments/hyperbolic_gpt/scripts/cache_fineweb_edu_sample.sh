#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=.

python orion/substrate/experiments/hyperbolic_gpt/cache_fineweb_edu.py \
  --name "${FINEWEB_EDU_NAME:-sample-10BT}" \
  --out "${OUT:-./data/fineweb_edu_sample_10bt.txt}" \
  --max_docs "${MAX_DOCS:-25000}"

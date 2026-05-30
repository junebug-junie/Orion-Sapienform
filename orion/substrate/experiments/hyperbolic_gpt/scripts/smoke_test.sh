#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
# Force CPU if local CUDA arch is unsupported by installed torch wheels
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  python3 orion/substrate/experiments/hyperbolic_gpt/smoke_test.py

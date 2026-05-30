#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=.
python orion/substrate/experiments/hyperbolic_gpt/smoke_test_v2.py

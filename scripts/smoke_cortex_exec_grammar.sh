#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if ! "$PY" -m pytest --version >/dev/null 2>&1; then
  if [ -x "${ROOT}/../.venv/bin/python" ]; then
    PY="${ROOT}/../.venv/bin/python"
  elif [ -x "/mnt/scripts/Orion-Sapienform/.venv/bin/python" ]; then
    PY="/mnt/scripts/Orion-Sapienform/.venv/bin/python"
  fi
fi

echo "1. Run unit tests"
PYTHONPATH=. "$PY" -m pytest services/orion-cortex-exec/tests/test_exec_grammar_emit.py -q
PYTHONPATH=. "$PY" -m pytest services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py -q

echo "2. Optional live bus tap"
echo "In another shell:"
echo 'redis-cli -u "${ORION_BUS_URL:-redis://127.0.0.1:6379/0}" SUBSCRIBE orion:grammar:event'

echo "3. Trigger a normal brain/chat path through existing harness"
echo "python scripts/bus_harness.py brain 'hello from cortex exec grammar smoke'"

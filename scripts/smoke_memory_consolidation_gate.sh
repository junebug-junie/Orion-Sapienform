#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ORION_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$ROOT/orion_dev/bin/python" ]]; then
    PY="$ROOT/orion_dev/bin/python"
  elif [[ -x "$ROOT/../../orion_dev/bin/python" ]]; then
    PY="$ROOT/../../orion_dev/bin/python"
  else
    PY="python3"
  fi
fi

echo "== consolidation gate: greeting window skip =="
PYTHONPATH=. "$PY" -m pytest \
  services/orion-memory-consolidation/tests/test_consolidation_gate.py::test_gate_skips_greeting_window \
  -q

echo "== consolidation gate: topic window propose =="
PYTHONPATH=. "$PY" -m pytest \
  services/orion-memory-consolidation/tests/test_consolidation_gate.py::test_gate_proposes_topic_shift \
  services/orion-memory-consolidation/tests/test_intake_consolidation_window.py::test_builds_proposed_semantic_crystallization \
  -q

echo "== consolidation gate: worker skip/propose branches =="
PYTHONPATH=. "$PY" -m pytest \
  services/orion-memory-consolidation/tests/test_worker_consolidation_gate.py::test_consolidate_window_skips_greeting_without_draft \
  services/orion-memory-consolidation/tests/test_worker_consolidation_gate.py::test_consolidate_window_proposes_crystallization \
  -q

echo "consolidation gate smoke OK"

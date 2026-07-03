#!/usr/bin/env bash
# Spark introspection contract eval: gateway publish policy + introspector enqueue guards.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x ./orion_dev/bin/python ]]; then
  PY="./orion_dev/bin/python"
elif [[ -x ./venv/bin/python ]]; then
  PY="./venv/bin/python"
else
  PY="python3"
fi

echo "== spark introspection contract eval =="
echo "python: $PY"

export PYTHONPATH=.

# Separate pytest processes avoid `app` package collisions between services.
"$PY" -m pytest \
  services/orion-llm-gateway/tests/test_spark_candidate_publish_policy.py \
  -q --tb=short

"$PY" -m pytest \
  services/orion-spark-introspector/tests/test_handle_candidate_enqueue_policy.py \
  -q --tb=short

"$PY" -m pytest \
  tests/integration/test_spark_introspection_feedback_loop.py \
  -q --tb=short

"$PY" - <<'PY'
import ast
from pathlib import Path

files = [
    "services/orion-llm-gateway/app/llm_backend.py",
    "services/orion-llm-gateway/app/main.py",
    "services/orion-spark-introspector/app/worker.py",
    "services/orion-memory-consolidation/app/classify.py",
    "services/orion-cortex-exec/app/executor.py",
]
for rel in files:
    ast.parse(Path(rel).read_text(encoding="utf-8"), filename=rel)
print(f"syntax ok: {len(files)} files")
PY

echo "PASS: spark introspection contract eval"

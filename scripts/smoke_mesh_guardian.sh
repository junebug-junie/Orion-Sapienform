#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MAIN_ROOT="$(cd "${ROOT}/../.." && pwd)"

if [[ -x "${MAIN_ROOT}/venv/bin/python" ]]; then
  PY="${MAIN_ROOT}/venv/bin/python"
elif [[ -x "${ROOT}/venv/bin/python" ]]; then
  PY="${ROOT}/venv/bin/python"
elif [[ -x "${MAIN_ROOT}/orion_dev/bin/python" ]]; then
  PY="${MAIN_ROOT}/orion_dev/bin/python"
elif [[ -x "${ROOT}/orion_dev/bin/python" ]]; then
  PY="${ROOT}/orion_dev/bin/python"
else
  PY="python3"
fi

if command -v curl >/dev/null 2>&1 && command -v jq >/dev/null 2>&1; then
  curl -sf "http://${PROJECT:-orion}-mesh-guardian:7160/health" | jq -e '.ok == true' || true
  curl -sf "http://${PROJECT:-orion}-mesh-guardian:7160/ready" || true
fi

PYTHONPATH=.:services/orion-mesh-guardian "${PY}" -m pytest \
  services/orion-mesh-guardian/tests/test_state_machine.py \
  services/orion-mesh-guardian/tests/test_compose_command.py \
  services/orion-mesh-guardian/tests/test_roster_load.py -q

echo "smoke_mesh_guardian: unit checks passed"

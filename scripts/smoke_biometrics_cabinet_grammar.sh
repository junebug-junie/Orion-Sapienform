#!/usr/bin/env bash
# Smoke: cabinet grammar atoms / unit path (host) + optional bus tap hint.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x /mnt/scripts/Orion-Sapienform/.venv/bin/python ]]; then
    PY=/mnt/scripts/Orion-Sapienform/.venv/bin/python
  elif [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
  else
    PY=python3
  fi
fi

export PYTHONPATH="${ROOT}/services/orion-biometrics:${ROOT}${PYTHONPATH:+:}${PYTHONPATH:-}"

echo "== unit: cabinet grammar + snapshot =="
"${PY}" -m pytest \
  services/orion-biometrics/tests/test_cabinet_grammar.py \
  services/orion-biometrics/tests/test_cabinet_snapshot.py \
  -q

echo "== optional bus tap =="
echo "redis-cli SUBSCRIBE orion:grammar:event"
echo "Expect semantic_role cabinet_*_activity_signal / cabinet_sensor_staleness_signal when Nano is live"

echo "smoke_biometrics_cabinet_grammar: OK"

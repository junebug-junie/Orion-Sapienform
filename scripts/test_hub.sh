#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-orion-athena}"
COMPOSE_FILE="${COMPOSE_FILE:-services/orion-hub/docker-compose.yml}"
RUNNER_MODE="${HUB_TEST_RUNNER_MODE:-container}"

DEFAULT_ARGS=(
  services/orion-hub/tests/test_recall_strategy_profiles_runtime.py
  services/orion-hub/tests/test_substrate_review_runtime_hub_debug.py
  services/orion-hub/tests/test_autonomy_runtime_ui_panel.py
  services/orion-hub/tests/test_recall_canary_profile_dropdown.py
  orion/substrate/tests/test_recall_strategy_readiness.py
  -q
  --tb=short
)

if [[ "$#" -gt 0 ]]; then
  PYTEST_ARGS=("$@")
else
  PYTEST_ARGS=("${DEFAULT_ARGS[@]}")
fi

if [[ "${RUNNER_MODE}" == "local" ]]; then
  "${BASH_SOURCE%/*}/test_service.sh" orion-hub "${PYTEST_ARGS[@]}"
  exit 0
fi

CMD="cd /repo && python3 -m pytest"
for arg in "${PYTEST_ARGS[@]}"; do
  CMD+=" ${arg@Q}"
done

PROJECT="${PROJECT}" docker compose -f "${COMPOSE_FILE}" exec hub-app sh -lc "${CMD}"

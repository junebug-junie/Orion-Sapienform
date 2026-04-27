#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <service-name> [pytest args]"
  exit 1
fi

SERVICE_NAME="$1"
shift || true

SERVICE_DIR="${REPO_ROOT}/services/${SERVICE_NAME}"
if [[ ! -d "${SERVICE_DIR}" ]]; then
  echo "unknown service: ${SERVICE_NAME}"
  echo "expected directory: ${SERVICE_DIR}"
  exit 1
fi

if [[ "$#" -gt 0 ]]; then
  PYTEST_ARGS=("$@")
else
  PYTEST_ARGS=("services/${SERVICE_NAME}/tests" "-q" "--tb=short")
fi

"${SCRIPT_DIR}/bootstrap_test_envs.sh" --service "${SERVICE_NAME}"

choose_python() {
  if [[ -x "${REPO_ROOT}/orion_dev/bin/python" ]]; then
    echo "${REPO_ROOT}/orion_dev/bin/python"
    return
  fi
  if [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
    echo "${REPO_ROOT}/venv/bin/python"
    return
  fi
  echo "python3"
}

PY_BIN="$(choose_python)"
echo "runner_python=${PY_BIN}"
echo "service=${SERVICE_NAME}"
echo "pytest_args=${PYTEST_ARGS[*]}"

cd "${REPO_ROOT}"
"${PY_BIN}" -m pytest "${PYTEST_ARGS[@]}"

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

# Opt-in eval lane. 11 services carry an `evals/` directory (AGENTS.md §11's
# second lane) and NONE of them were reachable through this runner or the
# Makefile -- an eval no tooling invokes is inert. Opt-in rather than default
# so this does not silently start running ten services' evals, some of which
# expect live infrastructure.
#   scripts/test_service.sh orion-hub --with-evals
WITH_EVALS=0
if [[ "${1:-}" == "--with-evals" ]]; then
  WITH_EVALS=1
  shift
fi

if [[ "$#" -gt 0 ]]; then
  PYTEST_ARGS=("$@")
else
  PYTEST_ARGS=("services/${SERVICE_NAME}/tests")
  if [[ "${WITH_EVALS}" -eq 1 ]]; then
    if [[ -d "${SERVICE_DIR}/evals" ]]; then
      PYTEST_ARGS+=("services/${SERVICE_NAME}/evals")
    else
      echo "--with-evals: no evals/ directory for ${SERVICE_NAME}, running tests only" >&2
    fi
  fi
  PYTEST_ARGS+=("-q" "--tb=short")
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

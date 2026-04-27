#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEV_REQS="${REPO_ROOT}/requirements-dev.txt"
SERVICE_REQS=""
SERVICE_NAME=""

if [[ ! -f "${DEV_REQS}" ]]; then
  echo "missing: ${DEV_REQS}"
  exit 1
fi

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --service)
      SERVICE_NAME="${2:-}"
      shift 2
      ;;
    --service-reqs)
      SERVICE_REQS="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1"
      echo "usage: $0 [--service <name>] [--service-reqs <path>]"
      exit 1
      ;;
  esac
done

if [[ -n "${SERVICE_NAME}" && -z "${SERVICE_REQS}" ]]; then
  CANDIDATE_REQS="${REPO_ROOT}/services/${SERVICE_NAME}/requirements.txt"
  if [[ -f "${CANDIDATE_REQS}" ]]; then
    SERVICE_REQS="${CANDIDATE_REQS}"
  fi
fi

bootstrap_env() {
  local env_name="$1"
  local py_bin="${REPO_ROOT}/${env_name}/bin/python"

  if [[ ! -x "${py_bin}" ]]; then
    echo "${env_name}: missing environment at ${REPO_ROOT}/${env_name}"
    echo "${env_name}: create it with: python3 -m venv ${env_name}"
    return 0
  fi

  echo "${env_name}: python=$(${py_bin} -c 'import sys; print(sys.executable)')"
  "${py_bin}" -m pip install --upgrade pip >/dev/null
  if [[ -n "${SERVICE_REQS}" && -f "${SERVICE_REQS}" ]]; then
    if ! "${py_bin}" -m pip install -r "${SERVICE_REQS}" >/dev/null; then
      echo "${env_name}: warning: failed to install service requirements (${SERVICE_REQS}); continuing with dev requirements"
      echo "${env_name}: hint: install missing system libraries if this service needs native wheels"
    fi
  fi
  "${py_bin}" -m pip install -r "${DEV_REQS}" >/dev/null
  echo "${env_name}: pytest_path=$("${py_bin}" -c 'import shutil; print(shutil.which("pytest") or "missing")')"
  echo "${env_name}: pytest_version=$("${py_bin}" -m pytest --version)"
}

echo "repo_root=${REPO_ROOT}"
if [[ -n "${SERVICE_NAME}" ]]; then
  echo "service=${SERVICE_NAME}"
fi
if [[ -n "${SERVICE_REQS}" ]]; then
  echo "service_reqs=${SERVICE_REQS}"
fi
bootstrap_env "venv"
bootstrap_env "orion_dev"

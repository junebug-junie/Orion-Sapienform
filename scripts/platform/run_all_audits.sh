#!/usr/bin/env bash
set -euo pipefail

RUN_ID=${1:-${RUN_ID:-audit_001}}

# Resolve repo root relative to this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

OUT_DIR="${REPO_ROOT}/codex_reviews/${RUN_ID}"
mkdir -p "${OUT_DIR}/reports" "${OUT_DIR}/docs"

echo "[platform] writing audit artifacts to: ${OUT_DIR}"

python "${REPO_ROOT}/scripts/platform/audit_channels.py" "${OUT_DIR}"
python "${REPO_ROOT}/scripts/platform/audit_schemas.py" "${OUT_DIR}"
python "${REPO_ROOT}/scripts/platform/audit_spine.py" "${OUT_DIR}"
python "${REPO_ROOT}/scripts/platform/audit_config_lineage.py" "${OUT_DIR}"
python "${REPO_ROOT}/scripts/platform/audit_antipatterns.py" "${OUT_DIR}"

echo "[platform] done"

#!/usr/bin/env bash
set -euo pipefail

RUN_ID=${1:-${RUN_ID:-audit_001}}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

OUT_DIR="${REPO_ROOT}/codex_reviews/${RUN_ID}"
mkdir -p "${OUT_DIR}/reports" "${OUT_DIR}/docs"

echo "[platform] writing audit artifacts to: ${OUT_DIR}"

# Ensure repo root is importable
export PYTHONPATH="${REPO_ROOT}"

# Run from repo root so -m works cleanly
cd "${REPO_ROOT}"

python -m scripts.platform.audit_channels "${OUT_DIR}"
python -m scripts.platform.audit_schemas "${OUT_DIR}"
python -m scripts.platform.audit_spine "${OUT_DIR}"
python -m scripts.platform.audit_config_lineage "${OUT_DIR}"
python -m scripts.platform.audit_antipatterns "${OUT_DIR}"

echo "[platform] done"

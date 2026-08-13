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

# python3, not python: a bare `python` isn't guaranteed on PATH (found live
# 2026-08-13 running this for real as part of the blast-radius eval for the
# scripts/platform -> scripts/platform_audits rename -- this environment
# has no `python` symlink at all, only `python3`). Pre-existing portability
# gap, unrelated to the rename itself, fixed here since it was blocking the
# new eval from actually exercising this script end-to-end.
python3 -m scripts.platform_audits.audit_channels "${OUT_DIR}"
python3 -m scripts.platform_audits.audit_schemas "${OUT_DIR}"
python3 -m scripts.platform_audits.audit_spine "${OUT_DIR}"
python3 -m scripts.platform_audits.audit_config_lineage "${OUT_DIR}"
python3 -m scripts.platform_audits.audit_antipatterns "${OUT_DIR}"

echo "[platform] done"

#!/usr/bin/env bash
# smoke_vision_persistence_live.sh
#
# Verifies the Orion Vision event persistence path:
#   orion:vision:events -> orion-vision-scribe -> sql-write
#
# The RDF/Fuseki write (and this smoke's RDF enqueue/confirm check) was
# removed 2026-07-23 -- live-verified pure redundancy with Postgres
# vision_events (see services/orion-vision-scribe/README.md). SQL is the
# sole sink now.
#
# MODES (explicit selection required when deps are missing):
#   VISION_PERSISTENCE_SMOKE_MODE=live
#       Requires ORION_BUS_URL and DATABASE_URL (or POSTGRES_URI).
#       Asserts scribe ack and a matching vision_events row.
#
#   VISION_PERSISTENCE_SMOKE_MODE=contract
#       In-process contract validation (no Redis/Postgres). Does NOT claim live persistence.
#
# ENV:
#   ORION_BUS_URL
#   DATABASE_URL (or POSTGRES_URI)
#   VISION_PERSISTENCE_SMOKE_TIMEOUT_SEC (default 30)
#   VISION_PERSISTENCE_SMOKE_MODE
#   VISION_PERSISTENCE_SMOKE_PYTHON (optional interpreter override)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pick_python() {
    local main_root="${REPO_ROOT}"
    if git -C "${REPO_ROOT}" rev-parse --git-common-dir >/dev/null 2>&1; then
        main_root="$(cd "$(git -C "${REPO_ROOT}" rev-parse --path-format=absolute --git-common-dir)/.." && pwd)"
    fi
    if [[ -n "${VISION_PERSISTENCE_SMOKE_PYTHON:-}" ]]; then
        echo "${VISION_PERSISTENCE_SMOKE_PYTHON}"
    elif [[ -x "${REPO_ROOT}/orion_dev/bin/python" ]]; then
        echo "${REPO_ROOT}/orion_dev/bin/python"
    elif [[ -x "${main_root}/orion_dev/bin/python" ]]; then
        echo "${main_root}/orion_dev/bin/python"
    elif [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
        echo "${REPO_ROOT}/venv/bin/python"
    elif [[ -x "${main_root}/venv/bin/python" ]]; then
        echo "${main_root}/venv/bin/python"
    else
        echo "python3"
    fi
}

PY="$(pick_python)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PY}" -m scripts.vision_persistence_smoke "$@"

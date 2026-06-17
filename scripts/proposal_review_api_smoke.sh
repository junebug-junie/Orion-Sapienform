#!/usr/bin/env bash
# Proposal review API smoke — boots context-exec app via TestClient-equivalent pytest path.
# Requires: orion_dev venv, proposal CLI seed-demo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Host-run default; set CONTEXT_EXEC_STORAGE_ROOT=/var/lib/orion/context-exec for container-side smoke.
CONTEXT_EXEC_STORAGE_ROOT="${CONTEXT_EXEC_STORAGE_ROOT:-/mnt/rlm-nvme/context-exec}"
STORE="${PROPOSAL_LEDGER_STORE_PATH:-${CONTEXT_EXEC_STORAGE_ROOT}/ledger/orion-proposals.json}"
REJECT_STORE="${PROPOSAL_LEDGER_REJECT_STORE_PATH:-${CONTEXT_EXEC_STORAGE_ROOT}/ledger/orion-proposals.reject.json}"
PY="${ORION_PY:-orion_dev/bin/python}"
# Resolve venv from repo root when run from a worktree without a local orion_dev.
if [[ ! -x "$PY" ]] && [[ -x "$ROOT/orion_dev/bin/python" ]]; then
  PY="$ROOT/orion_dev/bin/python"
fi

mkdir -p "$(dirname "$STORE")" "$(dirname "$REJECT_STORE")"
rm -f "$STORE" "$REJECT_STORE"

echo "== seed demo ledger =="
PYTHONPATH=. "$PY" scripts/orion_proposal_cli.py seed-demo --store "$STORE"

echo "== proposal review API smoke (pytest harness) =="
PROPOSAL_REVIEW_API_ENABLED=true \
PROPOSAL_LEDGER_STORE_PATH="$STORE" \
PYTHONPATH=. "$PY" -m pytest tests/services/test_proposal_review_api.py::test_context_exec_app_mounts_proposal_review_routes_when_enabled -q

echo "== health block check =="
PROPOSAL_REVIEW_API_ENABLED=true \
PROPOSAL_LEDGER_STORE_PATH="$STORE" \
PYTHONPATH=. "$PY" - <<'PY'
import sys
from pathlib import Path

ROOT = Path.cwd()
SERVICE_DIR = ROOT / "services" / "orion-context-exec"
sys.path[:0] = [str(SERVICE_DIR), str(ROOT)]

for mod in ("app.settings", "app.main", "app.api", "app.proposal_review_api"):
    sys.modules.pop(mod, None)

from starlette.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    health = client.get("/health")
    proposals = client.get("/proposals")
    assert health.status_code == 200, health.text
    block = health.json()["proposal_review_api"]
    assert block["ok"] is True, block
    assert proposals.status_code == 200, proposals.text
    rows = proposals.json()["proposals"]
    assert rows, "expected seeded proposals"
    pid = rows[0]["proposal_id"]
    detail = client.get(f"/proposals/{pid}")
    eligibility = client.get(f"/proposals/{pid}/eligibility")
    assert detail.status_code == 200, detail.text
    assert eligibility.status_code == 200, eligibility.text
print("SMOKE PASS")
PY

echo "proposal_review_api_smoke PASS"

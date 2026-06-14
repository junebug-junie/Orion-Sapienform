#!/usr/bin/env bash
# Proposal review API smoke — boots context-exec app via TestClient-equivalent pytest path.
# Requires: orion_dev venv, proposal CLI seed-demo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STORE="${PROPOSAL_LEDGER_STORE_PATH:-/tmp/orion-proposals.json}"
PY="${ORION_PY:-orion_dev/bin/python}"

rm -f "$STORE"

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

#!/usr/bin/env bash
# Denver memory-correction vertical slice smoke — context-exec → ledger → review API.
# Does not approve, reject, execute, or mutate memory. Read-only verification only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STORE="${PROPOSAL_STORE:-${PROPOSAL_LEDGER_STORE_PATH:-/tmp/orion-denver-proposals.json}}"
PY="${ORION_PY:-orion_dev/bin/python}"
if [[ ! -x "$PY" ]] && [[ -x "$ROOT/orion_dev/bin/python" ]]; then
  PY="$ROOT/orion_dev/bin/python"
fi

rm -f "$STORE"

echo "== Denver vertical slice (context-exec + ledger + auto-triage) =="
PYTHONPATH=. "$PY" - <<PY
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
SERVICE_DIR = ROOT / "services" / "orion-context-exec"
sys.path[:0] = [str(SERVICE_DIR), str(ROOT)]

from tests.fixtures.denver_vertical_slice import (
    DENVER_MEMORY_CORRECTION_PROMPT,
    assert_denver_vertical_slice_safety,
    run_denver_vertical_slice_async,
)
from orion.schemas.context_exec import MemoryCorrectionProposalV1, ProposalEnvelopeV1

store = Path("${STORE}")
result = asyncio.run(run_denver_vertical_slice_async(store))
run = result["run"]
record = result["record"]
envelope = ProposalEnvelopeV1.model_validate(run.artifact)
inner = MemoryCorrectionProposalV1.model_validate(envelope.artifact)

assert run.artifact_type == "ProposalEnvelopeV1", run.artifact_type
assert envelope.proposal_type == "memory_correction_proposal", envelope.proposal_type
assert envelope.artifact_type == "MemoryCorrectionProposalV1", envelope.artifact_type
assert envelope.proposal_id
assert envelope.mutation_allowed is False
assert envelope.requires_human_approval is True
assert run.runtime_debug.get("ledger_status") in {"pending_review", "blocked"}
assert run.runtime_debug.get("ledger_status") == "pending_review", run.runtime_debug
assert_denver_vertical_slice_safety(run, record, envelope, inner=inner)

pending = result["repo"].list_by_status("pending_review")
denver_rows = [
    row for row in pending
    if "denver" in json.dumps(row.envelope.model_dump(mode="json")).lower()
]
assert len(denver_rows) == 1, f"expected one Denver pending_review row, got {len(denver_rows)}"
assert denver_rows[0].envelope.proposal_type == "memory_correction_proposal"
assert denver_rows[0].envelope.artifact_type == "MemoryCorrectionProposalV1"
print(
    f"VERTICAL_SLICE proposal_id={envelope.proposal_id} "
    f"status={record.status} "
    f"type={envelope.proposal_type} "
    f"envelope={run.artifact_type} "
    f"inner={envelope.artifact_type} "
    f"mutation_allowed={envelope.mutation_allowed} "
    f"requires_human_approval={envelope.requires_human_approval}"
)
PY

echo "== proposal review API (pytest harness) =="
PROPOSAL_REVIEW_API_ENABLED=true \
PROPOSAL_LEDGER_STORE_PATH="$STORE" \
PYTHONPATH=. "$PY" -m pytest tests/services/test_proposal_review_api.py::test_proposal_review_api_lists_denver_pending_review -q

echo "== proposal review API inline GET check =="
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
    pending = client.get("/proposals", params={"status": "pending_review"})
    assert pending.status_code == 200, pending.text
    rows = pending.json()["proposals"]
    assert len(rows) == 1, rows
    pid = rows[0]["proposal_id"]
    detail = client.get(f"/proposals/{pid}")
    eligibility = client.get(f"/proposals/{pid}/eligibility")
    assert detail.status_code == 200, detail.text
    assert eligibility.status_code == 200, eligibility.text
    elig_body = eligibility.json()
    assert elig_body["eligible"] is False, elig_body
    assert elig_body.get("execution_requested") is False
    inner = detail.json()["inner_artifact_summary"]
    detail_env = detail.json()["envelope"]
    assert detail_env["proposal_type"] == "memory_correction_proposal"
    assert detail_env["artifact_type"] == "MemoryCorrectionProposalV1"
    assert inner["artifact_type"] == "MemoryCorrectionProposalV1"
    assert "denver" in inner["current_belief"].lower()
    assert inner["correction_type"] in {"mark_uncertain", "mark_contradicted", "replace_belief"}
    assert inner["mutation_allowed"] is False
    assert inner["requires_human_approval"] is True
    assert inner.get("rationale")
    assert inner.get("risk") in {"low", "medium", "high"}
print("API GET PASS eligible=false mutation_allowed=false requires_human_approval=true")
PY

if [[ "${HUB_SMOKE:-false}" == "true" ]] && [[ -n "${HUB_BASE_URL:-}" ]]; then
  echo "== optional Hub pending decisions GET =="
  curl -sf "${HUB_BASE_URL}/api/proposal-review/pending?status=pending_review" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); assert d.get("count",0)>=1'
else
  echo "== Hub live check skipped (set HUB_SMOKE=true and HUB_BASE_URL to enable) =="
fi

echo "denver_memory_correction_vertical_smoke PASS"

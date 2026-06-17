#!/usr/bin/env bash
# Smoke: typed self-experiment → context-exec compile path (dispatch may be disabled).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF_EXPERIMENTS_URL="${SELF_EXPERIMENTS_URL:-http://localhost:7172}"
DISPATCH="${SELF_EXPERIMENTS_DISPATCH_ENABLED:-false}"

echo "== self-experiment context-exec smoke =="
echo "url=$SELF_EXPERIMENTS_URL dispatch=$DISPATCH"

create_payload='{
  "experiment_type": "runtime_drift_check",
  "question": "Check whether transport reducer lag is still the dominant runtime pressure.",
  "source": "daily_metacog_v1",
  "source_ref": "2026-06-17",
  "correlation_id": "smoke-self-experiment-1"
}'

create_resp="$(curl -sf -X POST "$SELF_EXPERIMENTS_URL/v1/experiments" \
  -H 'Content-Type: application/json' \
  -d "$create_payload")"
echo "create: $create_resp"

exp_id="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['experiment_id'])" "$create_resp")"
record="$(curl -sf "$SELF_EXPERIMENTS_URL/v1/experiments/$exp_id")"
echo "record: $record"

python3 - <<'PY' "$record"
import json, sys
record = json.loads(sys.argv[1])
assert record["spec"]["experiment_type"] == "runtime_drift_check"
assert "keyword" not in json.dumps(record).lower() or "keyword_router" not in json.dumps(record).lower()
print("ok: typed experiment record")
PY

dispatch_resp="$(curl -sf -X POST "$SELF_EXPERIMENTS_URL/v1/experiments/$exp_id/dispatch")"
echo "dispatch: $dispatch_resp"

python3 - <<'PY' "$dispatch_resp" "$DISPATCH"
import json, sys
body = json.loads(sys.argv[1])
dispatch_enabled = sys.argv[2].lower() in {"1", "true", "yes"}
if dispatch_enabled:
    assert body.get("context_exec_mode") == "investigation_v2"
    assert body.get("expected_artifact_type") == "InvestigationReportV2"
else:
    assert body.get("status") == "queued"
    assert body.get("message") == "dispatch_disabled"
print("ok: dispatch response")
PY

echo "PASS self_experiment_context_exec_smoke"

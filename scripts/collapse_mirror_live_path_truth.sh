#!/usr/bin/env bash
# Live-path gate for collapse mirror generation.
# Unlike smoke_juniper_collapse_fanout.py (intake fanout only), this checks upstream
# subscriber health (equilibrium metacog trigger, cortex exec) plus substrate truth.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ORION_BUS_URL="${ORION_BUS_URL:-redis://127.0.0.1:6379/0}"
SUBSTRATE_BASE="${SUBSTRATE_BASE:-http://127.0.0.1:8115}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-15}"

CHANNEL_EQUILIBRIUM_METACOG_TRIGGER="${CHANNEL_EQUILIBRIUM_METACOG_TRIGGER:-orion:equilibrium:metacog:trigger}"
CHANNEL_CORTEX_EXEC_REQUEST="${CHANNEL_CORTEX_EXEC_REQUEST:-${CHANNEL_EXEC_REQUEST:-orion:cortex:exec:request}}"
EXEC_REQUEST_PREFIX="${EXEC_REQUEST_PREFIX:-orion:exec:request}"
CHANNEL_COLLAPSE_MIRROR_EXEC="${CHANNEL_COLLAPSE_MIRROR_EXEC:-${EXEC_REQUEST_PREFIX}:CollapseMirrorService}"
CHANNEL_COLLAPSE_INTAKE="${CHANNEL_COLLAPSE_INTAKE:-orion:collapse:intake}"
CHANNEL_COLLAPSE_SQL_WRITE="${CHANNEL_COLLAPSE_SQL_WRITE:-orion:collapse:sql-write}"

fail=0
summary_lines=()

_min_subscribers() {
  local channel="$1"
  local min="$2"
  local count
  count="$(redis-cli -u "${ORION_BUS_URL}" PUBSUB NUMSUB "${channel}" 2>/dev/null | tail -1 || echo -1)"
  if ! [[ "${count}" =~ ^[0-9]+$ ]]; then
    echo "INVALID: ${channel} (redis-cli failed)"
    summary_lines+=("${channel}: invalid")
    fail=1
    return
  fi
  echo "${channel}: ${count} subscriber(s) (min=${min})"
  if (( count < min )); then
    summary_lines+=("${channel}: ${count} < ${min}")
    fail=1
  else
    summary_lines+=("${channel}: ok (${count})")
  fi
}

echo "=== collapse mirror live-path subscriber check ==="
echo "bus_url=${ORION_BUS_URL}"
_min_subscribers "${CHANNEL_EQUILIBRIUM_METACOG_TRIGGER}" 1
_min_subscribers "${CHANNEL_CORTEX_EXEC_REQUEST}" 1
_min_subscribers "${CHANNEL_COLLAPSE_MIRROR_EXEC}" 1
_min_subscribers "${CHANNEL_COLLAPSE_INTAKE}" 1
_min_subscribers "${CHANNEL_COLLAPSE_SQL_WRITE}" 1

echo ""
echo "=== substrate grammar truth (summary) ==="
truth_body=""
if truth_body="$(curl -sf --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" "${SUBSTRATE_BASE}/grammar/truth")"; then
  TRUTH_VALIDATE_BODY="${truth_body}" python3 - <<'PY'
import json, os, sys
body = json.loads(os.environ["TRUTH_VALIDATE_BODY"])
print(json.dumps({
    "ok": body.get("ok"),
    "degraded": body.get("degraded"),
    "degraded_reasons": body.get("degraded_reasons"),
    "pending_backlog_by_reducer": body.get("pending_backlog_by_reducer"),
    "stream_lag_by_reducer": body.get("stream_lag_by_reducer"),
    "reducer_health_by_name": {
        k: {
            "classification": v.get("classification"),
            "unacknowledged_quarantine_count": v.get("unacknowledged_quarantine_count"),
            "pending_backlog": v.get("pending_backlog"),
            "stream_lag_sec": v.get("stream_lag_sec"),
        }
        for k, v in (body.get("reducer_health_by_name") or {}).items()
    },
    "unacknowledged_quarantine_count_by_reducer": body.get("unacknowledged_quarantine_count_by_reducer"),
}, indent=2))
if body.get("degraded"):
    sys.exit(3)
if body.get("ok") is False:
    sys.exit(3)
PY
  truth_rc=$?
  if (( truth_rc != 0 )); then
    summary_lines+=("substrate-truth: degraded or not ok")
    fail=1
  else
    summary_lines+=("substrate-truth: ok")
  fi
else
  echo "UNREACHABLE: substrate (${SUBSTRATE_BASE}/grammar/truth)"
  summary_lines+=("substrate-truth: unreachable")
  fail=1
fi

echo ""
echo "=== aggregate grammar production truth ==="
if ! "${SCRIPT_DIR}/grammar_production_truth.sh"; then
  summary_lines+=("grammar_production_truth: fail")
  fail=1
else
  summary_lines+=("grammar_production_truth: pass")
fi

echo ""
echo "=== summary ==="
for line in "${summary_lines[@]}"; do
  echo "  ${line}"
done

if (( fail != 0 )); then
  echo ""
  echo "collapse_mirror_live_path_truth: FAIL"
  echo "Hint: zero subscribers on metacog trigger or cortex exec matches PR #706 dead-pubsub mode."
  echo "Hint: degraded substrate truth may block live generation even when intake smoke passes."
  exit 1
fi

echo "collapse_mirror_live_path_truth: PASS"
exit 0

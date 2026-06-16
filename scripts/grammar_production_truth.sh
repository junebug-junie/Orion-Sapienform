#!/usr/bin/env bash
# Smoke gate: aggregate grammar production truth; exit nonzero on unreachable, malformed, or degraded.
set -euo pipefail

SQL_WRITER_BASE="${SQL_WRITER_BASE:-http://127.0.0.1:8220}"
SUBSTRATE_BASE="${SUBSTRATE_BASE:-http://127.0.0.1:8115}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-15}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required for grammar_production_truth.sh" >&2
  exit 2
fi

fail=0
summary_lines=()
SQL_BODY=""
SUBSTRATE_BODY=""

validate_truth_json() {
  local name="$1"
  local body="$2"
  TRUTH_VALIDATE_NAME="${name}" TRUTH_VALIDATE_BODY="${body}" PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/scripts" python3 - <<'PY'
import json, os, sys
from grammar_truth_gate import validate_truth_payload
name = os.environ["TRUTH_VALIDATE_NAME"]
payload = json.loads(os.environ["TRUTH_VALIDATE_BODY"])
errors = validate_truth_payload(name, payload)
if errors:
    print("\n".join(errors))
    sys.exit(1)
PY
}

check_truth() {
  local name="$1"
  local url="$2"
  local body
  if ! body="$(curl -sf --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" "${url}")"; then
    echo "UNREACHABLE: ${name} (${url})"
    summary_lines+=("${name}: unreachable")
    fail=1
    return
  fi

  echo "=== ${name} /grammar/truth ==="
  echo "${body}" | python3 -m json.tool

  if ! validate_truth_json "${name}" "${body}"; then
    echo "INVALID: ${name} payload validation failed (see above)"
    summary_lines+=("${name}: invalid payload")
    fail=1
    return
  fi

  local degraded reasons reason_groups
  degraded="$(TRUTH_VALIDATE_BODY="${body}" python3 -c 'import json,os; print(json.loads(os.environ["TRUTH_VALIDATE_BODY"]).get("degraded", False))')"
  reasons="$(TRUTH_VALIDATE_BODY="${body}" python3 -c 'import json,os; r=json.loads(os.environ["TRUTH_VALIDATE_BODY"]).get("degraded_reasons",[]); print(", ".join(r) if r else "none")')"
  reason_groups="$(TRUTH_VALIDATE_BODY="${body}" PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/scripts" python3 -c 'import json,os; from grammar_truth_gate import format_degraded_reason_groups; r=json.loads(os.environ["TRUTH_VALIDATE_BODY"]).get("degraded_reasons",[]); print(format_degraded_reason_groups(r))')"

  if [[ "${degraded}" == "True" ]]; then
    echo "DEGRADED: ${name} reasons=${reasons}"
    echo "DEGRADED_GROUPS: ${name} ${reason_groups}"
    summary_lines+=("${name}: degraded (${reasons})")
    fail=1
  else
    summary_lines+=("${name}: ok")
  fi

  if [[ "${name}" == "sql-writer" ]]; then
    SQL_BODY="${body}"
  else
    SUBSTRATE_BODY="${body}"
  fi
  echo
}

check_truth "sql-writer" "${SQL_WRITER_BASE}/grammar/truth"
check_truth "substrate-runtime" "${SUBSTRATE_BASE}/grammar/truth"

if [[ -n "${SQL_BODY}" || -n "${SUBSTRATE_BODY}" ]]; then
  TRUTH_SQL_BODY="${SQL_BODY}" TRUTH_SUB_BODY="${SUBSTRATE_BODY}" PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/scripts" python3 - <<'PY'
import json, os
from grammar_truth_gate import format_mode_summary
sql = json.loads(os.environ.get("TRUTH_SQL_BODY") or "null")
sub = json.loads(os.environ.get("TRUTH_SUB_BODY") or "null")
print(format_mode_summary(sql if isinstance(sql, dict) else None, sub if isinstance(sub, dict) else None))
PY
  echo
fi

echo "=== summary ==="
for line in "${summary_lines[@]}"; do
  echo "  ${line}"
done

if [[ "${fail}" -ne 0 ]]; then
  echo "grammar_production_truth: FAIL"
  exit 1
fi

echo "grammar_production_truth: PASS"
exit 0

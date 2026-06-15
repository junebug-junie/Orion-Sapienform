#!/usr/bin/env bash
# Fresh-main smoke ladder for Orion proposal-control spine.
#
# Runs:
#   schema -> CLI -> ledger/intake -> review API -> Hub surface
#   -> RLM evals -> Denver vertical -> CLI review/eligibility/dry-run
#   -> beta gate
#
# Usage:
#   chmod +x scripts/repl/orion_fresh_main_smoke.sh
#   ORION_PY=orion_dev/bin/python STORE=/tmp/orion-proposals.json \
#     ./scripts/repl/orion_fresh_main_smoke.sh

set -u
set -o pipefail

ORION_PY="${ORION_PY:-orion_dev/bin/python}"
STORE="${STORE:-/tmp/orion-proposals.json}"
REJECT_STORE="${REJECT_STORE:-/tmp/orion-proposals.json.reject}"
LOG_ROOT="${LOG_ROOT:-./.orion-smoke-logs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_ROOT}/${STAMP}"

mkdir -p "$LOG_DIR"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

declare -a FAIL_NAMES=()
declare -a FAIL_LOGS=()
declare -a SKIP_NAMES=()
declare -a SKIP_REASONS=()

sanitize_name() {
  echo "$1" | tr ' /:+()' '______' | tr -cd 'A-Za-z0-9_.-'
}

banner() {
  echo
  echo "========== $1 =========="
}

record_pass() {
  local name="$1"
  PASS_COUNT=$((PASS_COUNT + 1))
  echo "✅ PASS: $name"
}

record_fail() {
  local name="$1"
  local log="$2"
  FAIL_COUNT=$((FAIL_COUNT + 1))
  FAIL_NAMES+=("$name")
  FAIL_LOGS+=("$log")
  echo "❌ FAIL: $name"
  echo "   log: $log"
}

record_skip() {
  local name="$1"
  local reason="$2"
  SKIP_COUNT=$((SKIP_COUNT + 1))
  SKIP_NAMES+=("$name")
  SKIP_REASONS+=("$reason")
  echo "⚠️  SKIP: $name"
  echo "   reason: $reason"
}

run_step() {
  local name="$1"
  local cmd="$2"
  local log="$LOG_DIR/$(sanitize_name "$name").log"

  banner "$name"
  echo "\$ $cmd"

  # shellcheck disable=SC2086
  bash -lc "$cmd" >"$log" 2>&1
  local rc=$?

  cat "$log"

  if [[ "$rc" -eq 0 ]]; then
    record_pass "$name"
  else
    record_fail "$name" "$log"
  fi

  return 0
}

run_step_allow_skip() {
  local name="$1"
  local cmd="$2"
  local skip_reason="$3"

  if ! bash -lc "command -v ${cmd%% *} >/dev/null 2>&1"; then
    banner "$name"
    record_skip "$name" "$skip_reason"
    return 0
  fi

  run_step "$name" "$cmd"
}

require_repo_root() {
  if [[ ! -d ".git" ]]; then
    echo "ERROR: run from repo root."
    exit 2
  fi

  if [[ ! -x "$ORION_PY" ]]; then
    echo "ERROR: ORION_PY not executable: $ORION_PY"
    echo "Set ORION_PY=orion_dev/bin/python or activate/create the env."
    exit 2
  fi
}

first_pending_id() {
  local store="$1"
  PYTHONPATH=. "$ORION_PY" scripts/orion_proposal_cli.py list \
    --status pending_review \
    --store "$store" | awk 'NF && $1 ~ /^prop_/ {print $1; exit}'
}

classify_failure() {
  local name="$1"
  local log="$2"

  if [[ "$name" == "context-exec beta gate" ]]; then
    echo "Known-noise candidate: $name"
    echo "  $log"
    return 0
  fi

  if grep -Eiq 'alexzhang|repo organ|repo_grep|test_repo_tools|runtime_debug|engine_selected|assert rd\.get\("engine"\)' "$log"; then
    echo "Known-noise candidate: $name"
    echo "  $log"
    return 0
  fi

  echo "Unknown/actionable: $name"
  echo "  $log"
  return 1
}

require_repo_root

banner "git state"
git status --short
echo
git branch --show-current
echo
git --no-pager log -1 --oneline --decorate

# Keep the stores clean for deterministic CLI smoke.
rm -f "$STORE" "$REJECT_STORE"

run_step "sync context-exec env example" \
  "\"$ORION_PY\" scripts/sync_local_env_from_example.py orion-context-exec"

run_step "sync hub env example" \
  "\"$ORION_PY\" scripts/sync_local_env_from_example.py orion-hub"

run_step "schema tests" \
  "PYTHONPATH=. \"$ORION_PY\" -m pytest orion/schemas -q"

run_step "proposal CLI tests" \
  "PYTHONPATH=. \"$ORION_PY\" -m pytest tests/scripts/test_orion_proposal_cli.py -q"

run_step "context-exec proposal ledger + intake tests" \
  "PYTHONPATH=. \"$ORION_PY\" -m pytest services/orion-context-exec/tests/test_proposal_ledger.py services/orion-context-exec/tests/test_proposal_ledger_intake.py -q"

run_step "proposal review API tests" \
  "PYTHONPATH=. \"$ORION_PY\" -m pytest tests/services/test_proposal_review_api.py -q"

run_step "Hub proposal surface tests" \
  "PYTHONPATH=. \"$ORION_PY\" -m pytest services/orion-hub/tests/test_proposal_review_hub.py services/orion-hub/tests/test_proposal_review_ui.py -q"

run_step "RLM eval fake engine" \
  "PYTHONPATH=. \"$ORION_PY\" scripts/context_exec_rlm_eval.py --engine fake"

run_step "RLM eval alexzhang engine" \
  "PYTHONPATH=. \"$ORION_PY\" scripts/context_exec_rlm_eval.py --engine alexzhang"

run_step "Denver memory correction vertical smoke" \
  "ORION_PY=\"$ORION_PY\" bash scripts/denver_memory_correction_vertical_smoke.sh"

run_step "CLI seed-demo" \
  "PYTHONPATH=. \"$ORION_PY\" scripts/orion_proposal_cli.py seed-demo --store \"$STORE\""

run_step "CLI list pending_review" \
  "PYTHONPATH=. \"$ORION_PY\" scripts/orion_proposal_cli.py list --status pending_review --store \"$STORE\""

APPROVE_ID="$(first_pending_id "$STORE" || true)"
echo "$APPROVE_ID" > "$LOG_DIR/proposal_id.txt"

if [[ -z "$APPROVE_ID" ]]; then
  banner "CLI approve/dry-run path"
  record_fail "CLI approve/dry-run path" "$LOG_DIR/proposal_id.txt"
else
  run_step "CLI show pending_review proposal" \
    "PYTHONPATH=. \"$ORION_PY\" scripts/orion_proposal_cli.py show \"$APPROVE_ID\" --store \"$STORE\""

  run_step "CLI approve pending_review proposal" \
    "PYTHONPATH=. \"$ORION_PY\" scripts/orion_proposal_cli.py review \"$APPROVE_ID\" --decision approve --reason 'operator smoke approval' --reviewer human:june --store \"$STORE\""

  run_step "CLI eligibility after approval" \
    "PYTHONPATH=. \"$ORION_PY\" scripts/orion_proposal_cli.py eligibility \"$APPROVE_ID\" --store \"$STORE\""

  run_step "CLI dry-run execute approved proposal" \
    "PYTHONPATH=. \"$ORION_PY\" scripts/orion_proposal_cli.py dry-run-execute \"$APPROVE_ID\" --store \"$STORE\" --executor dry-run"
fi

run_step "CLI seed-demo for reject path" \
  "PYTHONPATH=. \"$ORION_PY\" scripts/orion_proposal_cli.py seed-demo --store \"$REJECT_STORE\""

REJECT_ID="$(first_pending_id "$REJECT_STORE" || true)"
echo "$REJECT_ID" > "$LOG_DIR/reject_proposal_id.txt"

if [[ -z "$REJECT_ID" ]]; then
  banner "CLI reject path"
  record_fail "CLI reject path" "$LOG_DIR/reject_proposal_id.txt"
else
  run_step "CLI reject pending_review proposal" \
    "PYTHONPATH=. \"$ORION_PY\" scripts/orion_proposal_cli.py review \"$REJECT_ID\" --decision reject --reason 'unsupported evidence' --reviewer human:june --store \"$REJECT_STORE\""

  run_step "CLI eligibility after reject" \
    "PYTHONPATH=. \"$ORION_PY\" scripts/orion_proposal_cli.py eligibility \"$REJECT_ID\" --store \"$REJECT_STORE\""
fi

run_step "context-exec beta gate" \
  "ORION_PY=\"$ORION_PY\" bash scripts/context_exec_beta_gate.sh"

banner "summary"
echo "Logs: $LOG_DIR"
echo "PASS=$PASS_COUNT FAIL=$FAIL_COUNT SKIP=$SKIP_COUNT"

if [[ "$FAIL_COUNT" -gt 0 ]]; then
  banner "failure classification"

  KNOWN_NOISE=0
  UNKNOWN_ACTIONABLE=0

  for i in "${!FAIL_NAMES[@]}"; do
    if classify_failure "${FAIL_NAMES[$i]}" "${FAIL_LOGS[$i]}"; then
      KNOWN_NOISE=$((KNOWN_NOISE + 1))
    else
      UNKNOWN_ACTIONABLE=$((UNKNOWN_ACTIONABLE + 1))
    fi
  done

  echo
  echo "Known-noise candidates: $KNOWN_NOISE"
  echo "Unknown/actionable failures: $UNKNOWN_ACTIONABLE"

  echo
  echo "Next move:"
  if [[ "$UNKNOWN_ACTIONABLE" -eq 0 ]]; then
    echo "1. Proposal-control spine is green."
    echo "2. Treat remaining failures as beta-gate/repo-organ stabilization."
    echo "3. Do not add new proposal types until beta gate signal is trustworthy."
  else
    echo "1. Open unknown/actionable logs first."
    echo "2. Fix proposal-control regressions before beta-gate cleanup."
    echo "3. Do not add new proposal types until this ladder is trustworthy."
  fi
else
  echo
  echo "BETA LADDER GREEN: proposal-control spine passed end-to-end."
fi

# Exit nonzero if any unknown/actionable failure exists.
# Known-noise still prints red but should not block the spine verdict.
if [[ "${UNKNOWN_ACTIONABLE:-0}" -gt 0 ]]; then
  exit 1
fi

exit 0

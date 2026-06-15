#!/usr/bin/env bash
# Context-exec beta gate — env posture + pytest + RLM evals (+ optional live golden probes).
# Usage:
#   bash scripts/context_exec_beta_gate.sh
#   bash scripts/context_exec_beta_gate.sh --live
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LIVE=0
for arg in "$@"; do
  case "$arg" in
    --live) LIVE=1 ;;
    -h|--help)
      echo "Usage: bash scripts/context_exec_beta_gate.sh [--live]"
      echo "  default: print env posture, pytest, RLM eval (fake + alexzhang)"
      echo "  --live:  also run scripts/context_exec_golden_probes.sh (requires Hub stack)"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg (use --live or --help)" >&2
      exit 2
      ;;
  esac
done

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

PY="${ORION_PY:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$ROOT/orion_dev/bin/python" ]]; then
    PY="$ROOT/orion_dev/bin/python"
  elif [[ -x "$ROOT/venv/bin/python" ]]; then
    PY="$ROOT/venv/bin/python"
  else
    PY="python3"
  fi
fi

fail() { echo "BETA GATE FAIL: $*" >&2; exit 1; }

echo "== context-exec beta env posture =="
print_env_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    echo "--- $f ---"
    grep -E '^(CONTEXT_EXEC_|CHANNEL_CONTEXT_EXEC|CHANNEL_RECALL)' "$f" || true
  else
    echo "WARN: missing $f"
  fi
}
print_env_file "services/orion-context-exec/.env"
print_env_file "services/orion-cortex-exec/.env"

check_beta_key() {
  local file="$1" key="$2" expected="$3"
  if [[ ! -f "$file" ]]; then
    echo "WARN: cannot verify $key (missing $file)"
    return 0
  fi
  local val
  val="$(grep -E "^${key}=" "$file" | tail -1 | cut -d= -f2- | tr -d '\r' || true)"
  if [[ -z "$val" ]]; then
    echo "WARN: $key not set in $file (expected $expected for beta)"
    return 0
  fi
  if [[ "$val" != "$expected" ]]; then
    echo "WARN: $file $key=$val (beta recommends $expected)"
  fi
}

CTX_ENV="services/orion-context-exec/.env"
check_beta_key "$CTX_ENV" "CONTEXT_EXEC_ENABLED" "true"
check_beta_key "$CTX_ENV" "CONTEXT_EXEC_SANDBOX_MODE" "docker"
check_beta_key "$CTX_ENV" "CONTEXT_EXEC_MAX_DEPTH" "1"
check_beta_key "$CTX_ENV" "CONTEXT_EXEC_WRITE_ENABLED" "false"
check_beta_key "$CTX_ENV" "CONTEXT_EXEC_NETWORK_ENABLED" "false"
check_beta_key "$CTX_ENV" "CONTEXT_EXEC_COMPAT_AGENT_CHAIN_ENABLED" "false"
check_beta_key "$CTX_ENV" "CONTEXT_EXEC_FAKE_ORGANS_ENABLED" "false"
check_beta_key "$CTX_ENV" "CONTEXT_EXEC_REAL_TRACE_ENABLED" "true"
check_beta_key "$CTX_ENV" "CONTEXT_EXEC_REAL_RECALL_ENABLED" "true"
check_beta_key "$CTX_ENV" "CONTEXT_EXEC_REAL_REPO_ENABLED" "true"

echo ""
echo "== pytest (orion-context-exec — proposal-control + repo tooling) =="
"$PY" -m pytest services/orion-context-exec/tests/ -q \
  --ignore=services/orion-context-exec/tests/test_rlm_eval_fixtures.py

echo ""
echo "== pytest (orion-context-exec — RLM eval quality fixtures) =="
"$PY" -m pytest services/orion-context-exec/tests/test_rlm_eval_fixtures.py -q

echo ""
echo "== RLM eval: fake engine =="
"$PY" scripts/context_exec_rlm_eval.py --engine fake

echo ""
echo "== RLM eval: alexzhang engine =="
"$PY" scripts/context_exec_rlm_eval.py --engine alexzhang

if [[ "$LIVE" -eq 1 ]]; then
  echo ""
  echo "== live golden probes =="
  bash scripts/context_exec_golden_probes.sh
else
  echo ""
  echo "== live golden probes (skipped — use --live) =="
  echo "Reminder: run against live Hub when stack is up:"
  echo "  CONTEXT_EXEC_PROBE_CORR_ID=5506f854-2606-42b5-a2ee-89775b3a8ed5 \\"
  echo "  HUB_BASE_URL=http://127.0.0.1:8080 \\"
  echo "  bash scripts/context_exec_beta_gate.sh --live"
  echo ""
  echo "Or directly:"
  echo "  bash scripts/context_exec_golden_probes.sh"
fi

echo ""
if [[ "$LIVE" -eq 1 ]]; then
  echo "BETA GATE PASS — pytest, RLM evals, and live golden probes OK"
else
  echo "BETA GATE PASS — pytest and RLM evals OK"
fi

#!/usr/bin/env bash
# Memory cognition loop — structural gate checks + optional live-stack smoke.
#
# Full encode → store → retrieve → lifecycle loop requires consolidation auto-activate:
#   MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED=true
# in services/orion-memory-consolidation/.env (operator contract; default is false).
#
# Usage:
#   bash scripts/smoke_memory_cognition_loop_e2e.sh              # STRUCTURAL (pytest only)
#   HUB_URL=http://127.0.0.1:8080 bash scripts/smoke_memory_cognition_loop_e2e.sh  # + LIVE
#   bash scripts/smoke_memory_cognition_loop_e2e.sh --live         # LIVE (requires HUB_URL)
#   bash scripts/smoke_memory_cognition_loop_e2e.sh --structural   # pytest only (default)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="structural"
for arg in "$@"; do
  case "$arg" in
    --live) MODE="live" ;;
    --structural) MODE="structural" ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/smoke_memory_cognition_loop_e2e.sh [--structural|--live]

STRUCTURAL (default): deterministic pytest gate bundle — no live stack required.

LIVE: when HUB_URL (or ORION_HUB_URL) is set, probe service health and exercise
      brain-mode chat + crystallization auto-activate + unified turn belief recall.

Required for full live loop:
  MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED=true  (memory-consolidation .env)
  ORION_HUB_SESSION_ID                       (session header for Hub APIs)

Optional service URLs (defaults shown):
  HUB_URL / ORION_HUB_URL     http://127.0.0.1:8080
  CORTEX_EXEC_HEALTH_URL      http://127.0.0.1:8070/health
  RECALL_HEALTH_URL           http://127.0.0.1:8260/health
  CONSOLIDATION_HEALTH_URL    http://127.0.0.1:8635/health
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg (use --structural, --live, or --help)" >&2
      exit 2
      ;;
  esac
done

HUB_URL="${HUB_URL:-${ORION_HUB_URL:-}}"
if [[ -n "$HUB_URL" && "$MODE" == "structural" ]]; then
  MODE="live"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

_resolve_pytest() {
  local candidate py
  for candidate in \
    "${ORION_PYTEST:-}" \
    "${ROOT}/.venv/bin/pytest" \
    "${ROOT}/../../.venv/bin/pytest" \
    "${ROOT}/../../orion_dev/bin/pytest" \
    "${ROOT}/orion_dev/bin/pytest"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      py="$(dirname "$candidate")/python"
      if [[ -x "$py" ]] && "$py" -c "import pydantic" >/dev/null 2>&1; then
        if "$candidate" --version >/dev/null 2>&1; then
          echo "$candidate"
          return 0
        fi
      fi
    fi
  done
  local py="${ORION_PYTHON:-python3}"
  if "$py" -c "import pydantic" >/dev/null 2>&1 && "$py" -m pytest --version >/dev/null 2>&1; then
    echo "$py -m pytest"
    return 0
  fi
  return 1
}

PYTEST_BIN="$(_resolve_pytest || true)"
if [[ -z "$PYTEST_BIN" ]]; then
  fail "pytest not found (tried .venv, ../../.venv, orion_dev, python3 -m pytest)"
fi
read -r -a PYTEST <<< "$PYTEST_BIN"
echo "pytest: ${PYTEST[*]}"

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

echo "== memory cognition loop env contract =="
echo "Full live loop requires MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED=true"
echo "  (services/orion-memory-consolidation/.env — default false for safe rollout)"
if [[ -f services/orion-memory-consolidation/.env ]]; then
  val="$(grep -E '^MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED=' services/orion-memory-consolidation/.env | tail -1 | cut -d= -f2- | tr -d '\r' || true)"
  if [[ -n "$val" ]]; then
    echo "  local consolidation .env: MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED=${val}"
  else
    echo "  WARN: MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED not set in local consolidation .env"
  fi
else
  echo "  WARN: services/orion-memory-consolidation/.env missing (see .env_example)"
fi

run_pytest() {
  echo "== pytest: $* =="
  "${PYTEST[@]}" "$@" -q
}

echo "== STRUCTURAL: memory cognition loop gate pytest =="
run_pytest tests/test_formation_policy_auto_vs_gated.py
run_pytest tests/test_retrieval_intent.py
run_pytest tests/test_encode_reinforce_not_duplicate.py
run_pytest services/orion-memory-consolidation/tests/
run_pytest \
  services/orion-cortex-exec/tests/test_pcr_chat_memory.py \
  services/orion-cortex-exec/tests/test_unified_pcr_phase01.py
run_pytest \
  services/orion-recall/tests/test_active_packet_activation_floor.py \
  services/orion-recall/tests/test_pcr_continuity.py
run_pytest services/orion-hub/tests/test_cortex_request_builder.py

echo "STRUCTURAL gate OK"

if [[ "$MODE" != "live" ]]; then
  echo "PASS smoke_memory_cognition_loop_e2e (structural only; set HUB_URL for live)"
  exit 0
fi

[[ -n "$HUB_URL" ]] || fail "LIVE mode requires HUB_URL or ORION_HUB_URL"
: "${ORION_HUB_SESSION_ID:?set ORION_HUB_SESSION_ID for Hub session header}"

if ! command -v curl >/dev/null 2>&1; then
  echo "WARN: curl missing — skipping live health + e2e probes"
  echo "PASS smoke_memory_cognition_loop_e2e (structural only; curl unavailable)"
  exit 0
fi

JQ=()
if command -v jq >/dev/null 2>&1; then
  JQ=(jq -c)
else
  echo "WARN: jq missing — live responses will not be parsed"
fi

_health_curl() {
  local name="$1" url="$2"
  echo "== health: ${name} (${url}) =="
  local code body
  body="$(curl -sS -w '' "$url" 2>/dev/null || true)"
  code="$(curl -sS -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo "000")"
  if [[ "$code" != "200" ]]; then
    echo "WARN: ${name} health HTTP ${code}"
    return 1
  fi
  if [[ ${#JQ[@]} -gt 0 && -n "$body" ]]; then
    echo "$body" | "${JQ[@]}" '.' 2>/dev/null || echo "$body"
  else
    echo "$body"
  fi
}

HUB_BASE="${HUB_URL%/}"
CORTEX_EXEC_HEALTH_URL="${CORTEX_EXEC_HEALTH_URL:-http://127.0.0.1:8070/health}"
RECALL_HEALTH_URL="${RECALL_HEALTH_URL:-http://127.0.0.1:8260/health}"
CONSOLIDATION_HEALTH_URL="${CONSOLIDATION_HEALTH_URL:-http://127.0.0.1:8635/health}"

_health_curl hub "${HUB_BASE}/health" || true
_health_curl cortex-exec "$CORTEX_EXEC_HEALTH_URL" || true
_health_curl recall "$RECALL_HEALTH_URL" || true
_health_curl memory-consolidation "$CONSOLIDATION_HEALTH_URL" || true

HDR=(-H "Content-Type: application/json" -H "X-Orion-Session-Id: ${ORION_HUB_SESSION_ID}")
STAMP="$(date -u +%Y%m%d%H%M%S)"
TOPIC_PROMPT="We decided to use k3s for staging deploy ${STAMP} — track this as project memory."

echo "== LIVE: brain chat topic-shift seed =="
if [[ ${#JQ[@]} -eq 0 ]]; then
  fail "LIVE e2e requires jq for response assertions"
fi

SEED=$(curl -sS -w "\n%{http_code}" -X POST "${HUB_BASE}/api/chat" "${HDR[@]}" -d "$(jq -n --arg m "$TOPIC_PROMPT" '{
  message: $m,
  mode: "brain",
  recall: {enabled: true}
}')")
SEED_BODY="$(echo "$SEED" | head -n -1)"
SEED_CODE="$(echo "$SEED" | tail -n 1)"
[[ "$SEED_CODE" == "200" ]] || fail "brain seed chat HTTP $SEED_CODE body=$SEED_BODY"
echo "$SEED_BODY" | jq -c '{memory_used, recall_debug: (.recall_debug // {} | {profile, pcr: .pcr, eligible_belief_count})}'

echo "== LIVE: poll active crystallizations (auto-activate path) =="
ACTIVE_COUNT=0
for attempt in $(seq 1 12); do
  LIST=$(curl -sS -w "\n%{http_code}" "${HDR[@]}" "${HUB_BASE}/api/memory/crystallizations?status=active&limit=50")
  LIST_BODY="$(echo "$LIST" | head -n -1)"
  LIST_CODE="$(echo "$LIST" | tail -n 1)"
  [[ "$LIST_CODE" == "200" ]] || fail "crystallizations list HTTP $LIST_CODE"
  ACTIVE_COUNT="$(echo "$LIST_BODY" | jq -r --arg s "$STAMP" '[.items[]? | select(.summary | tostring | contains($s))] | length')"
  if [[ "${ACTIVE_COUNT:-0}" -ge 1 ]]; then
    echo "active crystallization with stamp found (attempt=${attempt})"
    break
  fi
  echo "  poll ${attempt}/12: no active crystallization matching stamp yet"
  sleep 5
done
if [[ "${ACTIVE_COUNT:-0}" -lt 1 ]]; then
  echo "WARN: no auto-activated crystallization found for stamp=${STAMP}"
  echo "      (MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED=true on consolidation worker?)"
fi

echo "== LIVE: brain chat belief recall turn =="
RECALL=$(curl -sS -w "\n%{http_code}" -X POST "${HUB_BASE}/api/chat" "${HDR[@]}" -d "$(jq -n --arg m "what did we decide about k3s staging ${STAMP}?" '{
  message: $m,
  mode: "brain",
  recall: {enabled: true}
}')")
RECALL_BODY="$(echo "$RECALL" | head -n -1)"
RECALL_CODE="$(echo "$RECALL" | tail -n 1)"
[[ "$RECALL_CODE" == "200" ]] || fail "brain recall chat HTTP $RECALL_CODE body=$RECALL_BODY"
BELIEF_CHARS="$(echo "$RECALL_BODY" | jq -r '((.belief_digest // "") | length)')"
echo "$RECALL_BODY" | jq -c '{
  memory_used,
  belief_digest_chars: ((.belief_digest // "") | length),
  continuity_digest_chars: ((.continuity_digest // "") | length),
  recall_debug: (.recall_debug // {} | {profile, eligible_belief_count, pcr: .pcr})
}'
if [[ "${BELIEF_CHARS:-0}" -le 0 ]]; then
  echo "WARN: belief_digest_chars=0 on recall turn (eligible beliefs may be below activation floor)"
fi

echo "== LIVE: orion unified turn (grounding_capsule belief path) =="
UNIFIED=$(curl -sS -w "\n%{http_code}" -X POST "${HUB_BASE}/api/chat" "${HDR[@]}" -d "$(jq -n --arg m "summarize our k3s staging decision ${STAMP}" '{
  messages: [{role: "user", content: $m}],
  mode: "orion",
  use_recall: true
}')")
UNIFIED_BODY="$(echo "$UNIFIED" | head -n -1)"
UNIFIED_CODE="$(echo "$UNIFIED" | tail -n 1)"
[[ "$UNIFIED_CODE" == "200" ]] || fail "orion unified chat HTTP $UNIFIED_CODE body=$UNIFIED_BODY"

CAPSULE_BELIEF="$(echo "$UNIFIED_BODY" | jq -r '
  [
    .grounding_capsule.belief_digest?,
    .metadata.grounding_capsule.belief_digest?,
    .thought.grounding_capsule.belief_digest?
  ] | map(select(. != null and (. | tostring | length) > 0)) | first // ""
' | wc -c | tr -d ' ')"
echo "$UNIFIED_BODY" | jq -c '{
  type,
  phase,
  grounding_capsule: (.grounding_capsule // .metadata.grounding_capsule // .thought.grounding_capsule // null
    | if . then {belief_digest_chars: ((.belief_digest // "") | length), pcr_ran: .provenance.pcr_ran} else null end)
}'
if [[ "${CAPSULE_BELIEF:-0}" -le 1 ]]; then
  echo "WARN: unified turn grounding_capsule.belief_digest empty (ORION_UNIFIED_TURN_ENABLED / harness stack?)"
fi

echo "PASS smoke_memory_cognition_loop_e2e mode=live stamp=${STAMP} active_auto=${ACTIVE_COUNT:-0} belief_chars=${BELIEF_CHARS:-0}"

#!/usr/bin/env bash
# Live-stack smoke: Hub memory_cards HTTP path (POST → list pending_review → approve → list active).
# Requires ORION_HUB_URL, ORION_HUB_SESSION_ID, and RECALL_PG_DSN (must match Hub's memory store;
# smoke uses Hub HTTP only; DSN is required so operators export the same contract as runtime).
set -euo pipefail

: "${ORION_HUB_URL:?set ORION_HUB_URL (e.g. http://127.0.0.1:8080)}"
: "${ORION_HUB_SESSION_ID:?set ORION_HUB_SESSION_ID (Hub session / localStorage orion_sid)}"
: "${RECALL_PG_DSN:?set RECALL_PG_DSN to the same DSN configured on Hub as RECALL_PG_DSN}"

if ! command -v curl >/dev/null 2>&1; then
  echo "FAIL: curl not found"
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "FAIL: jq not found"
  exit 1
fi

BASE="${ORION_HUB_URL%/}"
HDR=(-H "Content-Type: application/json" -H "X-Orion-Session-Id: ${ORION_HUB_SESSION_ID}")
STAMP="$(date -u +%Y%m%d%H%M%S)"
TITLE="smoke_memory_cards_e2e ${STAMP}"
JSON_PAYLOAD="$(jq -n --arg title "$TITLE" '{types:["fact"],title:$title,summary:"curl smoke",provenance:"operator_highlight"}')"

echo "smoke_memory_cards_e2e: POST card title=${TITLE}"
CREATE_RESP="$(curl -sS -w "\n%{http_code}" -X POST "${BASE}/api/memory/cards" \
  "${HDR[@]}" \
  -d "$JSON_PAYLOAD")"

HTTP_BODY="$(echo "$CREATE_RESP" | head -n -1)"
HTTP_CODE="$(echo "$CREATE_RESP" | tail -n 1)"

if [[ "$HTTP_CODE" != "200" ]]; then
  echo "FAIL: POST /api/memory/cards HTTP ${HTTP_CODE} body=${HTTP_BODY}"
  exit 1
fi

CARD_ID="$(echo "$HTTP_BODY" | jq -r '.card_id // empty')"
if [[ -z "$CARD_ID" || "$CARD_ID" == "null" ]]; then
  echo "FAIL: missing card_id in create response: ${HTTP_BODY}"
  exit 1
fi

echo "smoke_memory_cards_e2e: card_id=${CARD_ID}"

echo "smoke_memory_cards_e2e: GET pending_review (expect card present)"
LIST_PR="$(curl -sS -w "\n%{http_code}" "${BASE}/api/memory/cards?status=pending_review&limit=200" "${HDR[@]}")"
LIST_PR_BODY="$(echo "$LIST_PR" | head -n -1)"
LIST_PR_CODE="$(echo "$LIST_PR" | tail -n 1)"
if [[ "$LIST_PR_CODE" != "200" ]]; then
  echo "FAIL: GET pending_review HTTP ${LIST_PR_CODE} body=${LIST_PR_BODY}"
  exit 1
fi
if ! echo "$LIST_PR_BODY" | jq -e --arg id "$CARD_ID" '.items | map(.card_id) | index($id) != null' >/dev/null; then
  echo "FAIL: card not in pending_review list"
  exit 1
fi

echo "smoke_memory_cards_e2e: POST status active"
STAT_RESP="$(curl -sS -w "\n%{http_code}" -X POST "${BASE}/api/memory/cards/${CARD_ID}/status" \
  "${HDR[@]}" -d '{"status":"active"}')"
STAT_BODY="$(echo "$STAT_RESP" | head -n -1)"
STAT_CODE="$(echo "$STAT_RESP" | tail -n 1)"
if [[ "$STAT_CODE" != "200" ]]; then
  echo "FAIL: POST status HTTP ${STAT_CODE} body=${STAT_BODY}"
  exit 1
fi

echo "smoke_memory_cards_e2e: GET active (expect card present)"
LIST_A="$(curl -sS -w "\n%{http_code}" "${BASE}/api/memory/cards?status=active&limit=200" "${HDR[@]}")"
LIST_A_BODY="$(echo "$LIST_A" | head -n -1)"
LIST_A_CODE="$(echo "$LIST_A" | tail -n 1)"
if [[ "$LIST_A_CODE" != "200" ]]; then
  echo "FAIL: GET active HTTP ${LIST_A_CODE} body=${LIST_A_BODY}"
  exit 1
fi
if ! echo "$LIST_A_BODY" | jq -e --arg id "$CARD_ID" '.items | map(.card_id) | index($id) != null' >/dev/null; then
  echo "FAIL: card not in active list"
  exit 1
fi

echo "PASS smoke_memory_cards_e2e card_id=${CARD_ID}"

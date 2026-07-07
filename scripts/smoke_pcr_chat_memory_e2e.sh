#!/usr/bin/env bash
# Live-stack smoke: PCR chat memory (greeting skip + semantic belief path)
set -euo pipefail

: "${ORION_HUB_URL:?set ORION_HUB_URL (e.g. http://127.0.0.1:8080)}"
: "${ORION_HUB_SESSION_ID:?set ORION_HUB_SESSION_ID}"

if ! command -v curl >/dev/null 2>&1; then echo "FAIL: curl missing"; exit 1; fi
if ! command -v jq >/dev/null 2>&1; then echo "FAIL: jq missing"; exit 1; fi

BASE="${ORION_HUB_URL%/}"
HDR=(-H "Content-Type: application/json" -H "X-Orion-Session-Id: ${ORION_HUB_SESSION_ID}")
STAMP="$(date -u +%Y%m%d%H%M%S)"

echo "== health: hub =="
curl -sS "${HDR[@]}" "${BASE}/health" | jq -c '{status, service}'

echo "== PCR smoke: greeting skip =="
GREET=$(curl -sS -w "\n%{http_code}" -X POST "${BASE}/api/chat" "${HDR[@]}" -d "$(jq -n --arg m "hey ${STAMP}" '{
  message: $m,
  mode: "brain",
  recall: {enabled: true}
}')")
GREET_BODY="$(echo "$GREET" | head -n -1)"
GREET_CODE="$(echo "$GREET" | tail -n 1)"
[[ "$GREET_CODE" == "200" ]] || { echo "FAIL greeting chat HTTP $GREET_CODE body=$GREET_BODY"; exit 1; }
echo "$GREET_BODY" | jq -c '{memory_used, recall_debug: (.recall_debug // {} | {pcr_phase, profile, skipped}), debug_pcr: (.debug.pcr // null)}'

echo "== PCR smoke: seed semantic belief placeholder =="
PROPOSE=$(curl -sS -w "\n%{http_code}" -X POST "${BASE}/api/memory/crystallizations/propose" "${HDR[@]}" -d "$(jq -n --arg s "PCR smoke ${STAMP}" '{
  kind: "fact",
  subject: "Move logistics",
  summary: $s,
  scope: ["project:orion"],
  planning_effects: [],
  retrieval_affordances: ["retrieve_when:move"],
  evidence: [{source_kind: "operator_note", source_id: "pcr-smoke-1", excerpt: "smoke"}],
  proposed_by: "smoke"
}')")
PROPOSE_BODY="$(echo "$PROPOSE" | head -n -1)"
PROPOSE_CODE="$(echo "$PROPOSE" | tail -n 1)"
[[ "$PROPOSE_CODE" == "200" ]] || { echo "FAIL propose HTTP $PROPOSE_CODE body=$PROPOSE_BODY"; exit 1; }
CID="$(echo "$PROPOSE_BODY" | jq -r '.crystallization_id // empty')"
[[ -n "$CID" ]] || { echo "FAIL no crystallization_id"; exit 1; }

echo "== validate + approve crystallization =="
curl -sS -X POST "${HDR[@]}" "${BASE}/api/memory/crystallizations/proposals/${CID}/validate" >/dev/null
curl -sS -X POST "${HDR[@]}" "${BASE}/api/memory/crystallizations/proposals/${CID}/approve" -d '{}' >/dev/null

echo "== PCR smoke: semantic belief turn =="
TOPIC=$(curl -sS -w "\n%{http_code}" -X POST "${BASE}/api/chat" "${HDR[@]}" -d "$(jq -n --arg m "what did we decide about move logistics ${STAMP}?" '{
  message: $m,
  mode: "brain",
  recall: {enabled: true}
}')")
TOPIC_BODY="$(echo "$TOPIC" | head -n -1)"
TOPIC_CODE="$(echo "$TOPIC" | tail -n 1)"
[[ "$TOPIC_CODE" == "200" ]] || { echo "FAIL semantic chat HTTP $TOPIC_CODE body=$TOPIC_BODY"; exit 1; }
echo "$TOPIC_BODY" | jq -c '{
  memory_used,
  continuity_digest_chars: ((.continuity_digest // "") | length),
  belief_digest_chars: ((.belief_digest // "") | length),
  memory_digest_chars: ((.memory_digest // "") | length),
  recall_debug: (.recall_debug // {} | {profile, profile_source, pcr: .pcr})
}'

echo "== active-packet probe (semantic seed) =="
PKT=$(curl -sS -w "\n%{http_code}" -X POST "${BASE}/api/memory/active-packet" "${HDR[@]}" -d "{\"query\":\"move logistics ${STAMP}\",\"task_type\":\"architecture\",\"seed_crystallization_id\":\"${CID}\"}")
PKT_BODY="$(echo "$PKT" | head -n -1)"
PKT_CODE="$(echo "$PKT" | tail -n 1)"
[[ "$PKT_CODE" == "200" ]] || { echo "FAIL active-packet HTTP $PKT_CODE body=$PKT_BODY"; exit 1; }
echo "$PKT_BODY" | jq -c '{crystallization_refs, card_refs, chroma_refs, retrieval_trace}'

echo "PASS smoke_pcr_chat_memory_e2e crystallization_id=${CID}"

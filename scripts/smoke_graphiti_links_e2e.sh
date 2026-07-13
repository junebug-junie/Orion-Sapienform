#!/usr/bin/env bash
# Smoke: two crystallizations with supports link → graphiti neighborhood depth 2 returns both
set -euo pipefail
: "${ORION_HUB_URL:?}"
: "${ORION_HUB_SESSION_ID:?}"
BASE="${ORION_HUB_URL%/}"
HDR=(-H "Content-Type: application/json" -H "X-Orion-Session-Id: ${ORION_HUB_SESSION_ID}")
STAMP="$(date -u +%Y%m%d%H%M%S)"

_propose() {
  local subject="$1" summary="$2"
  # planning_effects/retrieval_affordances required for kind=stance since
  # orion/memory/crystallization/validator.py::validate_proposal was tightened -- without
  # them propose() auto-quarantines (status=quarantined) and everything downstream 404s.
  curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/propose" -d "$(jq -n --arg s "$subject" --arg m "$summary" '{
    kind: "stance", subject: $s, summary: $m, scope: ["project:orion"],
    evidence: [{source_kind: "operator_note", source_id: "smoke-link", excerpt: "smoke"}],
    proposed_by: "smoke",
    planning_effects: ["smoke_test_context"],
    retrieval_affordances: ["smoke_test_lookup"]
  }')" | jq -r '.crystallization_id'
}

_approve() {
  local cid="$1"
  curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/proposals/${cid}/validate" >/dev/null
  curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/proposals/${cid}/approve" -d '{}' >/dev/null
}

CID_A="$(_propose "Link smoke A ${STAMP}" "seed A")"
CID_B="$(_propose "Link smoke B ${STAMP}" "target B")"
_approve "$CID_A"
_approve "$CID_B"

curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/${CID_A}/links" \
  -d "$(jq -n --arg t "$CID_B" '{target_crystallization_id: $t, relation: "supports", confidence: 0.9}')" >/dev/null

curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/graphiti/sync/${CID_A}" -d '{}' >/dev/null
curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/graphiti/sync/${CID_B}" -d '{}' >/dev/null

NB=$(curl -sS "${HDR[@]}" "${BASE}/api/memory/graphiti/neighborhood/${CID_A}?depth=2")
CIDS=$(echo "$NB" | jq -r '[.nodes[]?.crystallization_id] | unique | .[]')
echo "$CIDS" | grep -qx "$CID_A"
echo "$CIDS" | grep -qx "$CID_B"
echo "PASS smoke_graphiti_links_e2e seed=${CID_A} linked=${CID_B}"

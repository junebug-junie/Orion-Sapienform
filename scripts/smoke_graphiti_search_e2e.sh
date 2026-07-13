#!/usr/bin/env bash
# Smoke: propose+approve a crystallization with a distinctive subject, then confirm
# POST /v1/search (graphiti_core backend, hybrid vector+graph rail) finds it for real
# against a live FalkorDB — not a mock. Requires GRAPHITI_BACKEND=graphiti_core on the
# adapter (health.backend == "graphiti_core"); otherwise /v1/search returns HTTP 501.
#
# Fixed 2026-07-13 (see docs/superpowers/specs/2026-07-13-graphiti-core-backend-
# activation-spec.md "RELATES_TO schema" follow-up section): ingest_episode() now writes
# graphiti-core's own EntityNode/EntityEdge payloads (uuid-keyed, RELATES_TO-shaped, with a
# self-referential edge per crystallization) instead of a custom (:Entity)-[:HAS_EPISODE]
# /(:Entity)-[:RELATED] shape Graphiti.search() never read. See that spec section for the
# full root cause and field-mapping detail.
set -euo pipefail
: "${ORION_HUB_URL:?}"
: "${ORION_HUB_SESSION_ID:?}"
: "${GRAPHITI_ADAPTER_URL:?}"
BASE="${ORION_HUB_URL%/}"
ADAPTER="${GRAPHITI_ADAPTER_URL%/}"
HDR=(-H "Content-Type: application/json" -H "X-Orion-Session-Id: ${ORION_HUB_SESSION_ID}")
STAMP="$(date -u +%Y%m%d%H%M%S)"
# crystallization duplicate-detection (orion/memory/crystallization/detection.py,
# detect_duplicates) flags a proposal as a duplicate candidate whenever word-level Jaccard
# similarity of "subject summary" against any existing proposed/active crystallization of the
# same kind is >= 0.72; the validate step then sets validation_status=invalid, which blocks
# approve. A fixed template with only the timestamp varying (the original form of this script)
# shares ~6 of ~7 tokens across runs (Jaccard ~0.75) and collides with its own prior runs once
# a few have accumulated. Two independently-varying tokens (STAMP, RAND) on each side, mixed
# with a smaller shared vocabulary, keeps Jaccard well under the threshold run over run.
RAND_A="${RANDOM}${RANDOM}"
RAND_B="${RANDOM}${RANDOM}"
SUBJECT="Graphiti probe ${STAMP} token ${RAND_A}"
SUMMARY="Ephemeral verification note ${STAMP} marker ${RAND_B}"

BACKEND=$(curl -sS "${ADAPTER}/health" | jq -r '.backend')
if [[ "$BACKEND" != "graphiti_core" ]]; then
  echo "FAIL smoke_graphiti_search_e2e: adapter backend=${BACKEND}, expected graphiti_core" >&2
  exit 1
fi

_propose() {
  local subject="$1" summary="$2"
  curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/propose" -d "$(jq -n --arg s "$subject" --arg m "$summary" '{
    kind: "stance", subject: $s, summary: $m, scope: ["project:orion"],
    evidence: [{source_kind: "operator_note", source_id: "smoke-search", excerpt: "smoke"}],
    proposed_by: "smoke",
    planning_effects: ["smoke_test_context"],
    retrieval_affordances: ["smoke_test_lookup"]
  }')" | jq -r '.crystallization_id'
}

_approve() {
  local cid="$1"
  curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/proposals/${cid}/validate" >/dev/null
  local approve_status
  approve_status=$(curl -sS -o /tmp/smoke_graphiti_search_approve.json -w '%{http_code}' "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/proposals/${cid}/approve" -d '{}')
  if [[ "$approve_status" != "200" ]]; then
    echo "FAIL smoke_graphiti_search_e2e: approve cid=${cid} http=${approve_status} response=$(cat /tmp/smoke_graphiti_search_approve.json)" >&2
    exit 1
  fi
}

CID="$(_propose "$SUBJECT" "$SUMMARY")"
_approve "$CID"

curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/graphiti/sync/${CID}" -d '{}' >/dev/null

SR=$(curl -sS -H "Content-Type: application/json" -X POST "${ADAPTER}/v1/search" -d "$(jq -n --arg q "$SUBJECT" '{query: $q, limit: 10}')")
CIDS=$(echo "$SR" | jq -r '.crystallization_ids[]?')
EMBED_USED=$(echo "$SR" | jq -r '.trace.embed_used')

echo "$CIDS" | grep -qx "$CID" || { echo "FAIL smoke_graphiti_search_e2e: crystallization_ids missing seed=${CID} response=${SR}" >&2; exit 1; }
[[ "$EMBED_USED" == "true" ]] || { echo "FAIL smoke_graphiti_search_e2e: trace.embed_used=${EMBED_USED} response=${SR}" >&2; exit 1; }

echo "PASS smoke_graphiti_search_e2e seed=${CID} embed_used=${EMBED_USED}"

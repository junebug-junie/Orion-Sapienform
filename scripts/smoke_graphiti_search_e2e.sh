#!/usr/bin/env bash
# Smoke: propose+approve a crystallization with a distinctive subject, then confirm
# POST /v1/search (graphiti_core backend, hybrid vector+graph rail) finds it for real
# against a live FalkorDB — not a mock. Requires GRAPHITI_BACKEND=graphiti_core on the
# adapter (health.backend == "graphiti_core"); otherwise /v1/search returns HTTP 501.
#
# KNOWN FAILING as of 2026-07-13 activation pass (see PR/spec 2026-07-13-graphiti-core-
# backend-activation): /v1/search runs without crashing and does call the real embed host
# (trace.embed_used=true), but crystallization_ids comes back empty for data written by
# this adapter's own ingest_episode(). Root cause: ingest_episode() writes raw Cypher
# (:Entity)-[:HAS_EPISODE]->(:Episode) and (:Entity)-[:RELATED]->(:Entity) shapes, but
# graphiti-core==0.19.0's Graphiti.search() only queries (:Entity)-[:RELATES_TO {uuid,
# fact, fact_embedding, group_id}]->(:Entity) edges (confirmed via
# graphiti_core.search.search_utils.edge_fulltext_search source) plus a fulltext index
# this adapter never creates. The write path and graphiti-core's own read path are two
# different, incompatible graph schemas. Populating the RELATES_TO schema natively means
# either wiring a real LLM client for graphiti-core's add_episode()/add_triplet() (which
# internally call resolve_extracted_edge(self.llm_client, ...) even for the no-LLM-needed
# add_triplet path), or hand-writing RELATES_TO-shaped edges with a synthesized fact +
# fact_embedding — both larger, separate-decision-required patches, not this activation's
# scope. Left failing intentionally rather than deleted or weakened: this is the correct,
# honest signal until a human decides how to close the gap. Neighborhood/BFS (backend-
# agnostic, scripts/smoke_graphiti_links_e2e.sh) is unaffected and still finds linked
# crystallizations; so does the graphiti_neighborhood rail inside retrieve_active_packet().
set -euo pipefail
: "${ORION_HUB_URL:?}"
: "${ORION_HUB_SESSION_ID:?}"
: "${GRAPHITI_ADAPTER_URL:?}"
BASE="${ORION_HUB_URL%/}"
ADAPTER="${GRAPHITI_ADAPTER_URL%/}"
HDR=(-H "Content-Type: application/json" -H "X-Orion-Session-Id: ${ORION_HUB_SESSION_ID}")
STAMP="$(date -u +%Y%m%d%H%M%S)"
SUBJECT="Graphiti search smoke subject zz${STAMP}"

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
  curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/proposals/${cid}/approve" -d '{}' >/dev/null
}

CID="$(_propose "$SUBJECT" "smoke summary for graphiti search")"
_approve "$CID"

curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/graphiti/sync/${CID}" -d '{}' >/dev/null

SR=$(curl -sS -H "Content-Type: application/json" -X POST "${ADAPTER}/v1/search" -d "$(jq -n --arg q "$SUBJECT" '{query: $q, limit: 10}')")
CIDS=$(echo "$SR" | jq -r '.crystallization_ids[]?')
EMBED_USED=$(echo "$SR" | jq -r '.trace.embed_used')

echo "$CIDS" | grep -qx "$CID" || { echo "FAIL smoke_graphiti_search_e2e: crystallization_ids missing seed=${CID} response=${SR}" >&2; exit 1; }
[[ "$EMBED_USED" == "true" ]] || { echo "FAIL smoke_graphiti_search_e2e: trace.embed_used=${EMBED_USED} response=${SR}" >&2; exit 1; }

echo "PASS smoke_graphiti_search_e2e seed=${CID} embed_used=${EMBED_USED}"

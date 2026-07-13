#!/usr/bin/env bash
# Smoke: prove the /v1/search rail (not neighborhood/links) reaches a real chat-facing
# API response — POST /api/memory/active-packet on Hub, the same endpoint a live chat
# turn's recall path consumes.
#
# Design: propose+approve two UNLINKED crystallizations A and B with distinctive,
# unrelated subject text. Call active-packet seeded on A but querying with B's subject.
# A and B have no CrystallizationLinkV1 link between them, so graphiti_neighborhood
# (depth=2 from seed A) cannot explain B showing up — retrieve_active_packet() only
# reaches B via graphiti_adapter.search(query, seed_crystallization_id=A) matching B's
# own self-referential RELATES_TO fact edge (orion/memory/crystallization/retriever.py
# ~line 79). This isolates proof of the search-rail fix specifically, distinct from
# scripts/smoke_graphiti_links_e2e.sh (neighborhood, backend-agnostic, was never broken)
# and scripts/smoke_graphiti_search_e2e.sh (adapter API directly, not the chat-facing
# Hub response a live turn actually consumes).
set -euo pipefail
: "${ORION_HUB_URL:?}"
: "${ORION_HUB_SESSION_ID:?}"
: "${GRAPHITI_ADAPTER_URL:?}"
BASE="${ORION_HUB_URL%/}"
ADAPTER="${GRAPHITI_ADAPTER_URL%/}"
HDR=(-H "Content-Type: application/json" -H "X-Orion-Session-Id: ${ORION_HUB_SESSION_ID}")
# orion/memory/crystallization/detection.py::detect_duplicates scores word-level Jaccard
# (regex [a-z0-9]{3,}, threshold 0.72) over "subject summary" against EVERY existing
# non-rejected/archived/quarantined crystallization of the same kind+scope — not just this
# script's own prior runs. A shared timestamp/RANDOM-digit template collided in practice:
# generic verification vocabulary ("probe", "note", "marker", "target", "seed") recurs
# across the OTHER graphiti smoke scripts too (they run in the same session, same scope,
# same kind=stance), so the real existing pool this compares against is large and
# vocabulary-heavy well before this script's own repeat-run count would predict a collision.
# UUIDs split into hex segments by _token_set's regex give ~5 effectively-unique tokens per
# entry with zero realistic collision probability, keeping fixed/shared vocabulary to just 2
# short anchor words per entry (well under threshold regardless of pool size or history).
UUID_A="$(cat /proc/sys/kernel/random/uuid 2>/dev/null || uuidgen | tr 'A-Z' 'a-z')"
UUID_B="$(cat /proc/sys/kernel/random/uuid 2>/dev/null || uuidgen | tr 'A-Z' 'a-z')"
SUBJECT_A="Probe alpha ${UUID_A}"
SUBJECT_B="Probe bravo ${UUID_B}"
SUMMARY_A="Seed note ${UUID_A}"
SUMMARY_B="Target note ${UUID_B}"

BACKEND=$(curl -sS "${ADAPTER}/health" | jq -r '.backend')
if [[ "$BACKEND" != "graphiti_core" ]]; then
  echo "FAIL smoke_graphiti_active_packet_search_e2e: adapter backend=${BACKEND}, expected graphiti_core" >&2
  exit 1
fi

_propose() {
  local subject="$1" summary="$2"
  curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/propose" -d "$(jq -n --arg s "$subject" --arg m "$summary" '{
    kind: "stance", subject: $s, summary: $m, scope: ["project:orion"],
    evidence: [{source_kind: "operator_note", source_id: "smoke-active-packet", excerpt: "smoke"}],
    proposed_by: "smoke",
    planning_effects: ["smoke_test_context"],
    retrieval_affordances: ["smoke_test_lookup"]
  }')" | jq -r '.crystallization_id'
}

_approve() {
  local cid="$1"
  curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/proposals/${cid}/validate" >/dev/null
  local approve_status
  approve_status=$(curl -sS -o /tmp/smoke_graphiti_active_packet_approve.json -w '%{http_code}' "${HDR[@]}" -X POST "${BASE}/api/memory/crystallizations/proposals/${cid}/approve" -d '{}')
  if [[ "$approve_status" != "200" ]]; then
    echo "FAIL smoke_graphiti_active_packet_search_e2e: approve cid=${cid} http=${approve_status} response=$(cat /tmp/smoke_graphiti_active_packet_approve.json)" >&2
    exit 1
  fi
}

CID_A="$(_propose "$SUBJECT_A" "$SUMMARY_A")"
CID_B="$(_propose "$SUBJECT_B" "$SUMMARY_B")"
_approve "$CID_A"
_approve "$CID_B"

# No /api/memory/crystallizations/{id}/links call for either — the whole point is that
# they are NOT linked, so neighborhood cannot explain what this script asserts.
curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/graphiti/sync/${CID_A}" -d '{}' >/dev/null
curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/graphiti/sync/${CID_B}" -d '{}' >/dev/null

AP=$(curl -sS "${HDR[@]}" -X POST "${BASE}/api/memory/active-packet" -d "$(jq -n --arg q "$SUBJECT_B" --arg seed "$CID_A" '{query: $q, seed_crystallization_id: $seed}')")
REFS=$(echo "$AP" | jq -r '.graphiti_refs[]?')

echo "$REFS" | grep -qx "$CID_B" || {
  echo "FAIL smoke_graphiti_active_packet_search_e2e: graphiti_refs missing search-only target=${CID_B} seed=${CID_A} response=${AP}" >&2
  exit 1
}

echo "PASS smoke_graphiti_active_packet_search_e2e seed=${CID_A} search_reached=${CID_B}"

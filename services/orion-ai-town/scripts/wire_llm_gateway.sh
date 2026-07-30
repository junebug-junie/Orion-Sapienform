#!/usr/bin/env bash
# Wire AI Town Convex LLM env to Orion LLM gateway (→ llamacpp route table).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${ROOT}/upstream"

if [[ ! -d "${UPSTREAM}/convex" ]]; then
  echo "missing ${UPSTREAM}; clone upstream first (see README.md)" >&2
  exit 1
fi

# quick_background (not plain quick): AI Town shares atlas-worker-fast-1 with
# orion-mind/orion-hub/orion-embodiment's own quick traffic, and this route
# waits for upstream /slots slack before dispatching so AI Town's NPC
# dialogue never makes those snappier consumers wait behind it. See
# services/orion-llm-gateway/README.md's "Background-priority routes".
GATEWAY_URL="${AITOWN_LLM_GATEWAY_URL:-${ORION_LLM_GATEWAY_URL:-http://127.0.0.1:8210}}"
CHAT_ROUTE="${AITOWN_LLM_CHAT_ROUTE:-quick_background}"
EMBED_MODEL="${AITOWN_LLM_EMBEDDING_MODEL:-orion-vector-host}"
EMBED_DIM="${AITOWN_EMBEDDING_DIMENSION:-1024}"

# Convex backend runs in Docker — reach host gateway via mesh/LAN IP, not 127.0.0.1.
if [[ "${GATEWAY_URL}" == *127.0.0.1* ]] || [[ "${GATEWAY_URL}" == *localhost* ]]; then
  MESH_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
  if [[ -n "${MESH_IP}" ]]; then
    GATEWAY_URL="${GATEWAY_URL//127.0.0.1/${MESH_IP}}"
    GATEWAY_URL="${GATEWAY_URL//localhost/${MESH_IP}}"
    echo "rewrote gateway URL for convex container: ${GATEWAY_URL}"
  fi
fi

cd "${UPSTREAM}"
echo "Setting Convex LLM env:"
echo "  LLM_API_URL=${GATEWAY_URL}"
echo "  LLM_MODEL=${CHAT_ROUTE}"
echo "  LLM_EMBEDDING_MODEL=${EMBED_MODEL}"
echo "  EMBEDDING_DIMENSION=${EMBED_DIM} (compiled in convex/util/llm.ts for Orion mesh)"

npx convex env set LLM_API_URL "${GATEWAY_URL}"
npx convex env set LLM_MODEL "${CHAT_ROUTE}"
npx convex env set LLM_EMBEDDING_MODEL "${EMBED_MODEL}"

echo "Redeploying Convex functions (embedding dimension ${EMBED_DIM})..."
npx convex dev --once

# Circe is reserved for Juniper's direct deep/FCC turns -- AI Town must never
# land on it. Verify what was JUST written, not just what we intended to
# write: confirmed live 2026-07-30 that this exact drift (a stale LLM_MODEL
# left pointed at circe) can silently persist for weeks with nothing
# checking the live value. See check_llm_route_not_circe.py's docstring.
python3 "${ROOT}/scripts/check_llm_route_not_circe.py"

echo "Done. Chat → gateway ${CHAT_ROUTE} lane (llamacpp). Embeddings → gateway → orion-vector-host (${EMBED_DIM} dims)."

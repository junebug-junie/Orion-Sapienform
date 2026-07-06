#!/usr/bin/env bash
# Wire AI Town Convex LLM env to Orion LLM gateway (→ llamacpp route table).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${ROOT}/upstream"

if [[ ! -d "${UPSTREAM}/convex" ]]; then
  echo "missing ${UPSTREAM}; clone upstream first (see README.md)" >&2
  exit 1
fi

GATEWAY_URL="${AITOWN_LLM_GATEWAY_URL:-${ORION_LLM_GATEWAY_URL:-http://127.0.0.1:8210}}"
CHAT_ROUTE="${AITOWN_LLM_CHAT_ROUTE:-chat}"
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

echo "Done. Chat → gateway ${CHAT_ROUTE} lane (llamacpp). Embeddings → gateway → orion-vector-host (${EMBED_DIM} dims)."

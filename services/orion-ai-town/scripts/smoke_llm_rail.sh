#!/usr/bin/env bash
# Smoke AI Town LLM rail: gateway chat (llamacpp) + embeddings (vector-host).
set -euo pipefail

GATEWAY_URL="${AITOWN_LLM_GATEWAY_URL:-http://127.0.0.1:8210}"
MESH_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
CONVEX_GATEWAY="${GATEWAY_URL}"
if [[ -n "${MESH_IP}" ]]; then
  CONVEX_GATEWAY="${GATEWAY_URL//127.0.0.1/${MESH_IP}}"
  CONVEX_GATEWAY="${CONVEX_GATEWAY//localhost/${MESH_IP}}"
fi

fail() { echo "smoke_llm_rail FAIL: $*" >&2; exit 1; }

echo "== gateway health =="
curl -fsS "${GATEWAY_URL}/health" >/dev/null || fail "gateway unreachable at ${GATEWAY_URL}"

echo "== gateway chat (chat lane -> llamacpp) =="
CHAT="$(curl -fsS -X POST "${GATEWAY_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"chat","messages":[{"role":"user","content":"Reply exactly: RAIL-OK"}],"max_tokens":16,"temperature":0}')"
echo "${CHAT}" | python3 -c "import sys,json; d=json.load(sys.stdin); c=d['choices'][0]['message']['content']; assert 'RAIL' in c, c; print('chat:', c.strip())" \
  || fail "chat completion"

echo "== gateway embeddings (-> vector-host) =="
EMBED="$(curl -fsS -X POST "${GATEWAY_URL}/v1/embeddings" \
  -H 'Content-Type: application/json' \
  -d '{"model":"orion-vector-host","input":"smoke probe"}')"
echo "${EMBED}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
vec = d['data'][0]['embedding']
model = d.get('model', '')
assert len(vec) == 1024, len(vec)
assert 'bge-large' in model, model
print('embed:', model, 'dim', len(vec))
" || fail "embeddings"

if docker ps --format '{{.Names}}' | rg -q '^orion-ai-town-backend-1$'; then
  echo "== convex backend container -> gateway =="
  docker exec orion-ai-town-backend-1 curl -fsS "${CONVEX_GATEWAY}/health" >/dev/null \
    || fail "convex backend cannot reach gateway at ${CONVEX_GATEWAY}"
  docker exec orion-ai-town-backend-1 curl -fsS -X POST "${CONVEX_GATEWAY}/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d '{"input":"from convex"}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert len(d['data'][0]['embedding']) == 1024
print('convex-container embed ok')
" || fail "convex container embeddings"
fi

echo "smoke_llm_rail PASS"

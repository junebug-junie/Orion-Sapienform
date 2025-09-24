#!/usr/bin/env bash
set -euo pipefail

# --- knobs ---
NET=${NET:-app-net}
BUS_DIR=${BUS_DIR:-/mnt/services/Orion-Sapienform/services/orion-bus}
BRAIN_DIR=${BRAIN_DIR:-/mnt/services/Orion-Sapienform/services/orion-brain}
RAG_DIR=${RAG_DIR:-/mnt/services/Orion-Sapienform/services/orion-rag}
HUB_DIR=${HUB_DIR:-/mnt/services/Orion-Sapienform/services/orion-hub}
MIRROR_DIR=${MIRROR_DIR:-/mnt/services/Orion-Sapienform/services/orion-collapse-mirror}
CLIENT_DIR=${CLIENT_DIR:-/mnt/services/Orion-Sapienform/services/orion-llm-client}

PORT=${PORT:-8088}
RAG_PORT=${RAG_PORT:-8001}
HUB_PORT=${HUB_PORT:-8080}
MIRROR_PORT=${MIRROR_PORT:-8087}

BACKENDS=${BACKENDS:-http://llm-brain:11434}
MODEL=${MODEL:-mistral:instruct}
USERS=${USERS:-2}

MANAGED_REDIS_URL=${MANAGED_REDIS_URL:-}
SKIP_BUS=${SKIP_BUS:-0}
PRUNE=${PRUNE:-0}

# --- helpers ---
compose_file() {
  if [[ -f "$1/docker-compose.yml" ]]; then echo "$1/docker-compose.yml"
  elif [[ -f "$1/docker-compose.yaml" ]]; then echo "$1/docker-compose.yaml"
  elif [[ -f "$1/compose.yml" ]]; then echo "$1/compose.yml"
  elif [[ -f "$1/compose.yaml" ]]; then echo "$1/compose.yaml"
  else die "No compose file found in $1"
  fi
}

die() { echo "✖ $*" >&2; exit 1; }

wait_for_http() {
  local url="$1" tries="${2:-60}" sleep_s="${3:-0.5}"
  for i in $(seq 1 "$tries"); do
    if curl -fsS "$url" >/dev/null; then return 0; fi
    sleep "$sleep_s"
  done
  return 1
}

# managed redis override
if [[ -n "$MANAGED_REDIS_URL" ]]; then
  export REDIS_URL="$MANAGED_REDIS_URL"
  SKIP_BUS=1
  echo "ℹ Using managed Redis: $MANAGED_REDIS_URL"
fi

echo "== Teardown =="
( cd "$BRAIN_DIR" && docker compose down --remove-orphans || true )
( cd "$BRAIN_DIR" && docker rm -f orion-llm-brain >/dev/null 2>&1 || true )
( cd "$RAG_DIR" && docker compose down --remove-orphans || true )
( cd "$HUB_DIR" && docker compose down --remove-orphans || true )
( cd "$MIRROR_DIR" && docker compose down --remove-orphans || true )
if [[ "$SKIP_BUS" == "0" ]]; then
  ( cd "$BUS_DIR" && docker compose down --remove-orphans || true )
fi
if [[ "$PRUNE" == "1" ]]; then
  docker system prune -af || true
  docker volume prune -f || true
fi

echo "== Network =="
docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET"
docker network ls | awk '/'"$NET"'$/{print "✔ network:",$2}'

# bus
if [[ "$SKIP_BUS" == "0" ]]; then
  echo "== Start Redis bus =="
  ( cd "$BUS_DIR" && docker compose -f "$(compose_file "$BUS_DIR")" up -d )
  RID="$(docker compose -f "$(compose_file "$BUS_DIR")" ps -q orion-redis || true)"
  [[ -n "$RID" ]] || die "Redis container not found; check your orion-bus compose."
  docker exec -it "$RID" redis-cli PING || true
else
  echo "ℹ SKIP_BUS=1: assuming Redis is already up."
fi

# brain
echo "== Build & start orion-brain stack =="
(
  cd "$BRAIN_DIR" && \
  BACKENDS="$BACKENDS" PORT="$PORT" \
  REDIS_URL="${REDIS_URL:-redis://orion-redis:6379/0}" SERVICE_NAME="orion-brain" \
  EVENTS_ENABLE="${EVENTS_ENABLE:-true}" EVENTS_STREAM="${EVENTS_STREAM:-orion:evt:gateway}" \
  BUS_OUT_ENABLE="${BUS_OUT_ENABLE:-true}" BUS_OUT_STREAM="${BUS_OUT_STREAM:-orion:bus:out}" \
  docker compose -f "$(compose_file "$BRAIN_DIR")" up -d --build
)

LLM=$(docker compose -f "$(compose_file "$BRAIN_DIR")" ps -q llm-brain || true)
[[ -n "$LLM" ]] || die "llm-brain container not found after compose up."

echo "→ ensuring model: $MODEL in $LLM"
docker exec -it "$LLM" ollama pull "$MODEL" || true

echo "→ waiting for orion-brain on :$PORT ..."
wait_for_http "http://localhost:$PORT/health" 80 0.25 || die "orion-brain did not become healthy"
curl -s "http://localhost:$PORT/models" | jq . || true

# rag
echo "== Build & start orion-rag stack =="
(
  cd "$RAG_DIR" && \
  PORT="$RAG_PORT" BRAIN_URL="http://orion-brain:$PORT" \
  REDIS_URL="${REDIS_URL:-redis://orion-redis:6379/0}" SERVICE_NAME="orion-rag" \
  docker compose -f "$(compose_file "$RAG_DIR")" up -d --build
)
echo "→ waiting for orion-rag on :$RAG_PORT ..."
wait_for_http "http://localhost:$RAG_PORT/health" 80 0.25 || die "orion-rag did not become healthy"

# hub
echo "== Build & start orion-hub stack =="
(
  cd "$HUB_DIR" && \
  PORT="$HUB_PORT" BRAIN_URL="http://orion-brain:$PORT" \
  REDIS_URL="${REDIS_URL:-redis://orion-redis:6379/0}" SERVICE_NAME="orion-hub" \
  docker compose -f "$(compose_file "$HUB_DIR")" up -d --build
)
echo "→ waiting for orion-hub on :$HUB_PORT ..."
for i in {1..80}; do
  if curl -sf "http://localhost:$HUB_PORT/docs" >/dev/null; then
    echo "✔ orion-hub is healthy on :$HUB_PORT"
    break
  fi
  sleep 0.25
done || {
  echo "✖ orion-hub did not become healthy"
  exit 1
}

# collapse mirror
echo "== Build & start orion-collapse-mirror stack =="
(
  cd "$MIRROR_DIR" && \
  PORT="$MIRROR_PORT" REDIS_URL="${REDIS_URL:-redis://orion-redis:6379/0}" SERVICE_NAME="orion-collapse-mirror" \
  docker compose -f "$(compose_file "$MIRROR_DIR")" up -d --build
)
echo "→ waiting for orion-collapse-mirror on :$MIRROR_PORT ..."
wait_for_http "http://localhost:$MIRROR_PORT/health" 80 0.25 || die "orion-collapse-mirror did not become healthy"

# client
echo "== Build client image =="
docker build -t orion-llm-client:0.1 "$CLIENT_DIR"

echo "== Smoke (two parallel clients) =="
if [[ "$USERS" -gt 0 ]]; then
  docker run --rm --network "$NET" \
    -e ORION_BRAIN_URL="http://orion-brain:$PORT" \
    -e ORION_MODEL="$MODEL" \
    orion-llm-client:0.1 \
    python -m orion_llm_client.examples.simple_generate "Smoke: hello 1" &
  sleep 0.5
  docker run --rm --network "$NET" \
    -e ORION_BRAIN_URL="http://orion-brain:$PORT" \
    -e ORION_MODEL="$MODEL" \
    orion-llm-client:0.1 \
    python -m orion_llm_client.examples.simple_generate "Smoke: hello 2" &
  wait
else
  echo "ℹ USERS=0: skipping client smoke"
fi

echo "== Clean JSON test =="
curl -sS "http://localhost:$PORT/generate" -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"Say hi, then stop.\",\"options\":{\"num_predict\":32},\"stream\":false,\"return_json\":true,\"user_id\":\"u1\",\"session_id\":\"s1\"}" | jq .

if [[ "$SKIP_BUS" == "0" ]]; then
  echo "== Tail bus (latest 5 each) =="
  docker exec -it "$RID" redis-cli XREVRANGE orion:evt:gateway + - COUNT 5 || true
  docker exec -it "$RID" redis-cli XREVRANGE orion:bus:out + - COUNT 5 || true
fi

echo "✅ Rebuild+test completed."

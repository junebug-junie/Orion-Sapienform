#!/usr/bin/env bash
set -euo pipefail

# --- knobs (override via env: VAR=value ./rebuild_orion.sh) ---
NET=${NET:-app-net}
BUS_DIR=${BUS_DIR:-/mnt/services/Orion-Sapienform/services/orion-bus}
BRAIN_DIR=${BRAIN_DIR:-/mnt/services/Orion-Sapienform/services/orion-brain}
CLIENT_DIR=${CLIENT_DIR:-/mnt/services/Orion-Sapienform/services/orion-llm-client}
PORT=${PORT:-8088}
BACKENDS=${BACKENDS:-http://llm-brain:11434}
MODEL=${MODEL:-mistral:instruct}
USERS=${USERS:-2}

MANAGED_REDIS_URL=${MANAGED_REDIS_URL:-}
SKIP_BUS=${SKIP_BUS:-0}
PRUNE=${PRUNE:-0}

if [[ -n "$MANAGED_REDIS_URL" ]]; then
  export REDIS_URL="$MANAGED_REDIS_URL"
  SKIP_BUS=1
  echo "ℹ Using managed Redis: $MANAGED_REDIS_URL"
fi

die() { echo "✖ $*" >&2; exit 1; }

wait_for_http() {
  local url="$1" tries="${2:-60}" sleep_s="${3:-0.5}"
  for i in $(seq 1 "$tries"); do
    if curl -fsS "$url" >/dev/null; then return 0; fi
    sleep "$sleep_s"
  done
  return 1
}

echo "== Teardown =="
( cd "$BRAIN_DIR" && docker compose down --remove-orphans || true )
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

if [[ "$SKIP_BUS" == "0" ]]; then
  echo "== Start Redis bus =="
  ( cd "$BUS_DIR" && docker compose up -d )
  RID="$(docker compose -f "$BUS_DIR/docker-compose.yml" ps -q orion-redis || true)"
  [[ -n "$RID" ]] || die "Redis container not found; check your orion-bus compose."
  docker exec -it "$RID" redis-cli PING || true
else
  echo "ℹ SKIP_BUS=1: assuming Redis is already up."
fi

echo "== Build & start brain stack =="
(
  cd "$BRAIN_DIR" && \
  BACKENDS="$BACKENDS" PORT="$PORT" \
  REDIS_URL="${REDIS_URL:-redis://orion-redis:6379/0}" SERVICE_NAME="orion-brain" \
  EVENTS_ENABLE="${EVENTS_ENABLE:-true}" EVENTS_STREAM="${EVENTS_STREAM:-orion:evt:gateway}" \
  BUS_OUT_ENABLE="${BUS_OUT_ENABLE:-true}" BUS_OUT_STREAM="${BUS_OUT_STREAM:-orion:bus:out}" \
  docker compose up -d --build
)

# resolve llm-brain container created by compose
LLM=$(docker compose -f "$BRAIN_DIR/docker-compose.yml" ps -q llm-brain || true)
[[ -n "$LLM" ]] || die "llm-brain container not found after compose up."

echo "→ ensuring model: $MODEL in $LLM"
docker exec -it "$LLM" ollama pull "$MODEL" || true

echo "→ waiting for brain-service on :$PORT ..."
wait_for_http "http://localhost:$PORT/health" 80 0.25 || die "brain-service did not become healthy"
curl -s "http://localhost:$PORT/models" | jq . || true

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

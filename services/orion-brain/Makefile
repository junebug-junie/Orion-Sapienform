# ===== Defaults (override: make start-prod PORT=8090 USERS=2) =====
PORT        ?= 8088
NET         ?= app-net
BACKENDS    ?= http://llm-brain:11434
MODEL       ?= mistral:instruct
CLIENT_DIR  ?= ../orion-llm-client
CLIENT_IMG  ?= orion-llm-client:0.1
USERS       ?= 0
PROMPT      ?= Hello from Orion mesh!
STAGGER_MS  ?= 500

# Redis / events (points at your shared orion-bus by default)
REDIS_URL      ?= redis://orion-redis:6379/0
SERVICE_NAME   ?= orion-brain
EVENTS_ENABLE  ?= true
EVENTS_STREAM  ?= orion:evt:gateway
BUS_OUT_ENABLE ?= true
BUS_OUT_STREAM ?= orion:bus:out

# Dev toggles: auto-start helpers if missing
AUTO_DEV_REDIS  ?= 0   # 0 = warn only; 1 = auto-start local 'orion-redis'
AUTO_DEV_OLLAMA ?= 1   # 0 = skip; 1 = auto-start local 'llm-brain' if BACKENDS uses it

.PHONY: start-prod stop-prod status help \
        _net _ensure_redis _ensure_llm _brain_up _wait_ready _health _models \
        _maybe_clients _client_img env.print json-smoke bus.tail bus.out.tail

help:
	@echo "Targets:"
	@echo "  start-prod     -> one-shot bring-up (net, redis check, llm, brain, readiness, health)"
	@echo "  stop-prod      -> stop brain-service container"
	@echo "  status         -> /health and /models"
	@echo "  json-smoke     -> single clean-JSON generate call"
	@echo "  bus.tail       -> tail internal event stream"
	@echo "  bus.out.tail   -> tail external bus stream"
	@echo "  env.print      -> show effective env passed to compose"
	@echo ""
	@echo "Vars: PORT NET BACKENDS MODEL REDIS_URL USERS PROMPT AUTO_DEV_REDIS AUTO_DEV_OLLAMA"

start-prod: _net _ensure_redis _ensure_llm _brain_up _wait_ready _health _models _maybe_clients
	@echo "✔ start-prod complete: http://localhost:$(PORT)"

stop-prod:
	@docker compose down --remove-orphans || true

status:
	@$(MAKE) -s _health
	@$(MAKE) -s _models

env.print:
	@echo "PORT=$(PORT)"
	@echo "NET=$(NET)"
	@echo "BACKENDS=$(BACKENDS)"
	@echo "MODEL=$(MODEL)"
	@echo "REDIS_URL=$(REDIS_URL)"
	@echo "EVENTS_ENABLE=$(EVENTS_ENABLE)  EVENTS_STREAM=$(EVENTS_STREAM)"
	@echo "BUS_OUT_ENABLE=$(BUS_OUT_ENABLE) BUS_OUT_STREAM=$(BUS_OUT_STREAM)"

# --- Internals ---

_net:
	@docker network inspect $(NET) >/dev/null 2>&1 || docker network create $(NET)
	@docker network ls | awk '/$(NET)$$/{print "✔ network:",$$2}'

_ensure_redis:
	@if ! docker ps --format '{{.Names}}' | grep -qx orion-redis; then \
		if [ "$(AUTO_DEV_REDIS)" = "1" ]; then \
			echo "→ starting local Redis 'orion-redis' on $(NET)"; \
			docker rm -f orion-redis >/dev/null 2>&1 || true; \
			docker run -d --name orion-redis --network $(NET) -p 6379:6379 \
			  redis:7-alpine redis-server --appendonly yes >/dev/null; \
		else \
			echo "⚠ orion-redis not running on $(NET). Start your bus: cd /mnt/services/Orion-Sapienform/services/orion-bus && docker compose up -d"; \
		fi; \
	else \
		echo "✔ using Redis at $(REDIS_URL)"; \
	fi

_ensure_llm:
	@if echo "$(BACKENDS)" | grep -q "llm-brain"; then \
		if [ "$(AUTO_DEV_OLLAMA)" = "1" ]; then \
			if ! docker ps --format '{{.Names}} {{.Image}}' | awk '$$1=="llm-brain"{ok=1} END{exit ok?0:1}'; then \
				echo "→ starting local Ollama backend 'llm-brain' on $(NET)"; \
				docker rm -f llm-brain >/dev/null 2>&1 || true; \
				docker run -d --name llm-brain --network $(NET) -p 11434:11434 ollama/ollama:latest >/dev/null; \
			else \
				echo "✔ llm-brain already running"; \
			fi; \
			docker network connect --alias llm-brain $(NET) llm-brain 2>/dev/null || true; \
			echo "→ ensuring model $(MODEL) is present"; \
			docker exec -it llm-brain ollama pull $(MODEL) >/dev/null || true; \
		else \
			echo "ℹ BACKENDS uses llm-brain; AUTO_DEV_OLLAMA=0 so not auto-starting"; \
		fi; \
	else \
		echo "ℹ BACKENDS does not reference 'llm-brain'; skipping local backend setup"; \
	fi

_brain_up:
	@echo "→ starting orion-brain on :$(PORT) (BACKENDS=$(BACKENDS))"
	@BACKENDS="$(BACKENDS)" PORT="$(PORT)" \
	REDIS_URL="$(REDIS_URL)" SERVICE_NAME="$(SERVICE_NAME)" \
	EVENTS_ENABLE="$(EVENTS_ENABLE)" EVENTS_STREAM="$(EVENTS_STREAM)" \
	BUS_OUT_ENABLE="$(BUS_OUT_ENABLE)" BUS_OUT_STREAM="$(BUS_OUT_STREAM)" \
	docker compose up -d --build
	@docker ps --format 'table {{.Names}}\t{{.Ports}}' | awk 'NR==1||/orion-brain/'

_wait_ready:
	@echo "→ waiting for orion-brain to be ready on :$(PORT) ..."
	@for i in $$(seq 1 40); do \
	  if curl -fsS "http://localhost:$(PORT)/health" >/dev/null; then \
	    echo "✔ brain-service ready"; exit 0; \
	  fi; \
	  sleep 0.25; \
	done; \
	echo "✖ timed out waiting for orion-brian"; \
	docker compose logs --no-color orion-brain | tail -n 80; \
	exit 1

_health:
	@curl -sS http://localhost:$(PORT)/health | jq

_models:
	@curl -sS http://localhost:$(PORT)/models | jq

_maybe_clients:
	@if [ "$(USERS)" -gt 0 ]; then \
		$(MAKE) -s _client_img; \
		i=1; delay=$$(awk 'BEGIN{printf "%.3f", $(STAGGER_MS)/1000.0}'); \
		echo "→ launching $(USERS) client(s) on $(NET) → orion-brain:$(PORT)"; \
		while [ $$i -le $(USERS) ]; do \
			docker run --rm --network $(NET) \
			  -e ORION_BRAIN_URL=http://orion-brain:$(PORT) \
			  -e ORION_MODEL=$(MODEL) \
			  $(CLIENT_IMG) \
			  python -m orion_llm_client.examples.simple_generate "[client $$i] $(PROMPT)" & \
			i=$$(( $$i + 1 )); \
			sleep $$delay; \
		done; \
		wait || true; \
		echo "✔ clients finished"; \
	fi

_client_img:
	@if ! docker image inspect $(CLIENT_IMG) >/dev/null 2>&1; then \
		echo "→ building client image $(CLIENT_IMG) from $(CLIENT_DIR)"; \
		cd $(CLIENT_DIR) && docker build -t $(CLIENT_IMG) .; \
	fi

# --- Convenience ---

json-smoke:
	@curl -sS http://localhost:$(PORT)/generate \
	  -H 'content-type: application/json' \
	  -d '{"model":"$(MODEL)","prompt":"Say hi, then stop.","options":{"num_predict":32},"stream":false,"return_json":true,"user_id":"u1","session_id":"s1"}' | jq

BUS_SERVICE ?= orion-redis

bus.tail:
	@docker compose -f ../orion-bus/compose.yml exec -T $(BUS_SERVICE) \
	  redis-cli XREVRANGE $(EVENTS_STREAM) + - COUNT 10

bus.out.tail:
	@docker compose -f ../orion-bus/compose.yml exec -T $(BUS_SERVICE) \
	  redis-cli XREVRANGE $(BUS_OUT_STREAM) + - COUNT 10


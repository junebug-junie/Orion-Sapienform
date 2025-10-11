# =============================================================
# 🧠 Orion Brain | Mesh Cognitive Core
# =============================================================
# Defaults (override: make start-prod PORT=8090 USERS=2)
PORT        ?= 8088
NET         ?= app-net
PROJECT     ?= orion-janus
BACKENDS    ?= http://llm-brain:11434
MODEL       ?= mistral:instruct
CLIENT_DIR  ?= ../orion-llm-client
CLIENT_IMG  ?= orion-llm-client:0.1
USERS       ?= 0
PROMPT      ?= Hello from Orion mesh!
STAGGER_MS  ?= 500

# Redis / Orion Bus connection (shared across Mesh)
REDIS_URL      ?= redis://$(PROJECT)-bus-core:6379/0
SERVICE_NAME   ?= $(PROJECT)-brain
EVENTS_ENABLE  ?= true
EVENTS_STREAM  ?= orion:evt:gateway
BUS_OUT_ENABLE ?= true
BUS_OUT_STREAM ?= orion:bus:out

# Dev toggles: auto-start helpers if missing
AUTO_DEV_REDIS  ?= 0   # 1 = auto-start local 'bus-core'
AUTO_DEV_OLLAMA ?= 1   # 1 = auto-start local 'llm-brain'

.PHONY: start-prod stop-prod restart status help \
        _net _ensure_bus _ensure_llm _brain_up _wait_ready _health _models \
        _pull_model _maybe_clients _client_img env.print bus.tail bus.out.tail

# =============================================================
# HELP
# =============================================================
help:
	@echo "Orion Brain Makefile — Mesh Core Controller"
	@echo ""
	@echo "Targets:"
	@echo "  make start-prod      → Bring up Orion Brain + dependencies"
	@echo "  make stop-prod       → Stop all containers for Brain"
	@echo "  make restart         → Rebuild + restart Brain"
	@echo "  make status          → Show health + models info"
	@echo "  make env.print       → Show effective environment vars"
	@echo "  make bus.tail        → Tail internal event stream"
	@echo "  make bus.out.tail    → Tail outbound bus stream"
	@echo ""
	@echo "Env Vars:"
	@echo "  PORT NET PROJECT BACKENDS MODEL REDIS_URL USERS PROMPT AUTO_DEV_REDIS AUTO_DEV_OLLAMA"

# =============================================================
# ENTRY TARGETS
# =============================================================
start-prod: _net _ensure_bus _ensure_llm _brain_up _wait_ready _health _models _pull_model _maybe_clients
	@echo "✔ Orion Brain started: http://localhost:$(PORT)"

stop-prod:
	@docker compose down --remove-orphans || true

restart: stop-prod start-prod

status:
	@$(MAKE) -s _health
	@$(MAKE) -s _models

env.print:
	@echo "PROJECT=$(PROJECT)"
	@echo "PORT=$(PORT)"
	@echo "NET=$(NET)"
	@echo "BACKENDS=$(BACKENDS)"
	@echo "MODEL=$(MODEL)"
	@echo "REDIS_URL=$(REDIS_URL)"
	@echo "EVENTS_ENABLE=$(EVENTS_ENABLE)  EVENTS_STREAM=$(EVENTS_STREAM)"
	@echo "BUS_OUT_ENABLE=$(BUS_OUT_ENABLE) BUS_OUT_STREAM=$(BUS_OUT_STREAM)"

# =============================================================
# INTERNALS
# =============================================================
_net:
	@docker network inspect $(NET) >/dev/null 2>&1 || docker network create $(NET)
	@docker network ls | awk '/$(NET)$$/{print "✔ network:",$$2}'

_ensure_bus:
	@if ! docker ps --format '{{.Names}}' | grep -qx $(PROJECT)-bus-core; then \
		if [ "$(AUTO_DEV_REDIS)" = "1" ]; then \
			echo "→ starting temporary local Bus '$(PROJECT)-bus-core'"; \
			docker rm -f $(PROJECT)-bus-core >/dev/null 2>&1 || true; \
			docker run -d --name $(PROJECT)-bus-core --network $(NET) -p 6379:6379 \
			  redis:7-alpine redis-server --appendonly yes >/dev/null; \
		else \
			echo "⚠ Orion Bus not detected on $(NET). Start it manually:"; \
			echo "   cd ../orion-bus && docker compose up -d"; \
		fi; \
	else \
		echo "✔ Using Orion Bus at $(REDIS_URL)"; \
	fi

_ensure_llm:
	@if echo "$(BACKENDS)" | grep -q "llm-brain"; then \
		if [ "$(AUTO_DEV_OLLAMA)" = "1" ]; then \
			if ! docker ps --format '{{.Names}}' | grep -qx $(PROJECT)-brain-llm; then \
				echo "→ starting local Ollama backend '$(PROJECT)-brain-llm'"; \
				docker rm -f $(PROJECT)-brain-llm >/dev/null 2>&1 || true; \
				docker run -d --name $(PROJECT)-brain-llm --network $(NET) -p 11434:11434 ollama/ollama:latest >/dev/null; \
			else \
				echo "✔ Ollama backend already running"; \
			fi; \
			docker network connect --alias llm-brain $(NET) $(PROJECT)-brain-llm 2>/dev/null || true; \
			echo "→ ensuring model $(MODEL) is present"; \
			docker exec -it $(PROJECT)-brain-llm ollama pull $(MODEL) >/dev/null || true; \
		else \
			echo "ℹ BACKENDS references llm-brain; AUTO_DEV_OLLAMA=0 so skipping autostart"; \
		fi; \
	else \
		echo "ℹ BACKENDS does not reference llm-brain; skipping local backend setup"; \
	fi

_brain_up:
	@echo "→ starting Brain service ($(PROJECT)-brain) on port :$(PORT)"
	@PROJECT=$(PROJECT) BACKENDS="$(BACKENDS)" PORT="$(PORT)" \
	REDIS_URL="$(REDIS_URL)" SERVICE_NAME="$(SERVICE_NAME)" \
	EVENTS_ENABLE="$(EVENTS_ENABLE)" EVENTS_STREAM="$(EVENTS_STREAM)" \
	BUS_OUT_ENABLE="$(BUS_OUT_ENABLE)" BUS_OUT_STREAM="$(BUS_OUT_STREAM)" \
	docker compose up -d --build
	@docker ps --format 'table {{.Names}}\t{{.Ports}}' | awk 'NR==1||/brain/'

_wait_ready:
	@echo "→ waiting for brain-service on :$(PORT) ..."
	@for i in $$(seq 1 40); do \
	  if curl -fsS "http://localhost:$(PORT)/health" >/dev/null; then \
	    echo "✔ brain-service ready"; exit 0; \
	  fi; \
	  sleep 0.25; \
	done; \
	echo "✖ timed out waiting for orion-brain"; \
	docker compose logs --no-color $(PROJECT)-brain | tail -n 80; \
	exit 1

_health:
	@curl -sS http://localhost:$(PORT)/health | jq || echo "⚠ No /health endpoint"

_models:
	@echo "→ Listing available LLM backends and models"
	@curl -sS http://localhost:$(PORT)/chat \
	  -H "content-type: application/json" \
	  -d '{"prompt":"list models","session_id":"system","user_id":"system"}' | jq || echo "⚠ No /models endpoint (using /chat fallback)"

_pull_model:
	@echo "→ ensuring model $(MODEL) is available in llm-brain"
	@if docker ps --format '{{.Names}}' | grep -qx $(PROJECT)-brain-llm; then \
		docker exec -it $(PROJECT)-brain-llm ollama pull $(MODEL) || true; \
	else \
		echo "⚠ llm-brain container not running, skipping model pull"; \
	fi

# =============================================================
# CLIENTS + BUS
# =============================================================
_maybe_clients:
	@if [ "$(USERS)" -gt 0 ]; then \
		$(MAKE) -s _client_img; \
		i=1; delay=$$(awk 'BEGIN{printf "%.3f", $(STAGGER_MS)/1000.0}'); \
		echo "→ launching $(USERS) simulated client(s) on $(NET)"; \
		while [ $$i -le $(USERS) ]; do \
			docker run --rm --network $(NET) \
			  -e ORION_BRAIN_URL=http://brain:$(PORT) \
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

# =============================================================
# UTILITIES
# =============================================================
json-smoke:
	@curl -sS http://localhost:$(PORT)/chat \
	  -H 'content-type: application/json' \
	  -d '{"model":"mistral:instruct","messages":[{"role":"user","content":"Say hi, then stop."}],"user_id":"u1","session_id":"s1"}' | jq

bus.tail:
	@docker compose -f ../orion-bus/docker-compose.yml exec -T $(PROJECT)-bus-core \
	  redis-cli XREVRANGE $(EVENTS_STREAM) + - COUNT 10

bus.out.tail:
	@docker compose -f ../orion-bus/docker-compose.yml exec -T $(PROJECT)-bus-core \
	  redis-cli XREVRANGE $(BUS_OUT_STREAM) + - COUNT 10

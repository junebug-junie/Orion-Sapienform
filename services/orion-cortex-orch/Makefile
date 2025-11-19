# Orion Cortex Orchestrator Makefile
#
# This Makefile is meant to be run from:
#   /mnt/scripts/Orion-Sapienform/services/orion-cortex-orch
#
# But docker compose is executed from the repo root:
#   /mnt/scripts/Orion-Sapienform
#
# matching how you currently run:
#   docker compose --env-file .env \
#                  --env-file services/orion-cortex-orch/.env \
#                  -f services/orion-cortex-orch/docker-compose.yml up

# Path to repo root (from this service dir)
ROOT_DIR := $(abspath ../..)

COMPOSE_FILE_REL   := services/orion-cortex-orch/docker-compose.yml
ENV_ROOT_REL       := .env
ENV_LOCAL_REL      := services/orion-cortex-orch/.env

SERVICE_NAME ?= orion-cortex-orchestrator

# Name of the service in the service-specific compose file
# Change if your service is named differently in that yaml.
SERVICE_NAME ?= orion-athena-cortex-orch

.PHONY: help build up down restart logs smoke-worker smoke-http smoke-clean

help:
	@echo ""
	@echo "Orion Cortex Orchestrator – Make targets"
	@echo "========================================="
	@echo "  make build                 Build the $(SERVICE_NAME) image"
	@echo "  make up                    Start stack via services/orion-cortex-orch/docker-compose.yml"
	@echo "  make down                  Stop stack"
	@echo "  make restart               Restart stack"
	@echo "  make logs                  Tail logs for $(SERVICE_NAME)"
	@echo ""
	@echo "  make smoke-worker          Run the bus smoke-test worker INSIDE the service image"
	@echo "  make smoke-http            Fire a one-step HTTP /orchestrate smoke test"
	@echo "  make smoke-clean           Stop stack and prune dangling containers"
	@echo ""

build:
	cd $(ROOT_DIR) && \
	docker compose --env-file $(ENV_ROOT_REL) --env-file $(ENV_LOCAL_REL) \
	  -f $(COMPOSE_FILE_REL) build $(SERVICE_NAME)

up:
	cd $(ROOT_DIR) && \
	docker compose --env-file $(ENV_ROOT_REL) --env-file $(ENV_LOCAL_REL) \
	  -f $(COMPOSE_FILE_REL) up -d

down:
	cd $(ROOT_DIR) && \
	docker compose --env-file $(ENV_ROOT_REL) --env-file $(ENV_LOCAL_REL) \
	  -f $(COMPOSE_FILE_REL) down

restart: down up

logs:
	cd $(ROOT_DIR) && \
	docker compose --env-file $(ENV_ROOT_REL) --env-file $(ENV_LOCAL_REL) \
	  -f $(COMPOSE_FILE_REL) logs -f $(SERVICE_NAME)

# --------------------------------------------------------------------
# Smoke testing
# --------------------------------------------------------------------

smoke-worker:
	cd $(ROOT_DIR) && \
	docker compose --env-file $(ENV_ROOT_REL) --env-file $(ENV_LOCAL_REL) \
	  -f $(COMPOSE_FILE_REL) run --rm \
	    --entrypoint python \
	    $(SERVICE_NAME) \
	    app/tests/smoke_bus_worker.py

smoke-http:
	curl -X POST http://localhost:8072/orchestrate \
	  -H "Content-Type: application/json" \
	  -d '{"verb_name":"introspect","origin_node":"athena-smoke","context":{"foo":"bar","test":"cortex-orch smoke"},"steps":[{"verb_name":"introspect","step_name":"echo-test","description":"Simple echo test step.","order":0,"services":["TestService"],"prompt_template":"This is a smoke test of the Cortex Orchestrator and bus wiring.","requires_gpu":false,"requires_memory":false}],"timeout_ms":8000}' \
	  | jq .

smoke-clean: down
	cd $(ROOT_DIR) && docker container prune -f >/dev/null 2>&1 || true

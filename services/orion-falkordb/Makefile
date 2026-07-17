SERVICE ?= orion-falkordb
COMPOSE_FILE ?= docker-compose.yml

.PHONY: help up down logs logs-falkordb rebuild config preflight health-probe

help:
	@echo "Orion FalkorDB (shared property-graph operator stack)"
	@echo "Targets:"
	@echo "  help            -> this message"
	@echo "  preflight       -> mkdir FALKORDB_DATA_DIR on the host"
	@echo "  config          -> docker compose config sanity-check"
	@echo "  up              -> start FalkorDB"
	@echo "  down            -> stop and remove"
	@echo "  logs            -> tail logs"
	@echo "  logs-falkordb   -> tail falkordb service logs"
	@echo "  rebuild         -> pull image and restart"
	@echo "  health-probe    -> Redis PING probe (host port or container exec)"

preflight:
	mkdir -p "$${FALKORDB_DATA_DIR:-/mnt/graphdb/falkordb}"

config:
	docker compose --env-file .env_example -f $(COMPOSE_FILE) config

up: preflight
	docker compose --env-file .env -f $(COMPOSE_FILE) up -d

down:
	docker compose --env-file .env -f $(COMPOSE_FILE) down --remove-orphans

logs:
	docker compose --env-file .env -f $(COMPOSE_FILE) logs -f

logs-falkordb:
	docker compose --env-file .env -f $(COMPOSE_FILE) logs -f falkordb

rebuild:
	docker compose --env-file .env -f $(COMPOSE_FILE) pull
	docker compose --env-file .env -f $(COMPOSE_FILE) up -d

health-probe:
	@chmod +x scripts/falkordb_health_probe.sh
	@./scripts/falkordb_health_probe.sh

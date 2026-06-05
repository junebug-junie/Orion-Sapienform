SERVICE ?= orion-rdf-store
COMPOSE_FILE ?= docker-compose.yml

.PHONY: help up down logs logs-fuseki rebuild config preflight health-probe recover

help:
	@echo "Orion RDF Store (Fuseki operator stack)"
	@echo "Targets:"
	@echo "  help         -> this message"
	@echo "  preflight    -> mkdir durable data dirs (see RDF_STORE_DATA_ROOT)"
	@echo "  config       -> docker compose config with .env_example (sanity-check compose + env)"
	@echo "  up           -> start Fuseki (orion-athena-fuseki)"
	@echo "  down         -> stop and remove"
	@echo "  logs         -> tail all service logs"
	@echo "  logs-fuseki  -> tail Fuseki only"
	@echo "  rebuild      -> pull fresh image and restart"
	@echo "  health-probe -> ping + query + graph-store write probe (host-side)"
	@echo "  recover      -> restart Fuseki when write probe fails (TDB lock exhaustion)"

preflight:
	mkdir -p "$${RDF_STORE_DATA_ROOT:-/mnt/storage-lukewarm/rdf-store}/fuseki"
	mkdir -p "$${RDF_STORE_DATA_ROOT:-/mnt/storage-lukewarm/rdf-store}/fuseki-backups"

config:
	docker compose --env-file .env_example -f $(COMPOSE_FILE) config

up: preflight
	docker compose -f $(COMPOSE_FILE) up -d

down:
	docker compose -f $(COMPOSE_FILE) down --remove-orphans

logs:
	docker compose -f $(COMPOSE_FILE) logs -f

logs-fuseki:
	docker compose -f $(COMPOSE_FILE) logs -f orion-athena-fuseki

rebuild:
	docker compose -f $(COMPOSE_FILE) pull
	docker compose -f $(COMPOSE_FILE) up -d

health-probe:
	@chmod +x scripts/fuseki_health_probe.sh
	@./scripts/fuseki_health_probe.sh

recover:
	@chmod +x scripts/fuseki_health_probe.sh scripts/fuseki_recover.sh
	@./scripts/fuseki_recover.sh

SERVICE ?= orion-gdb-client
CONTAINER ?= $(SERVICE)
COMPOSE_FILE ?= docker-compose.yml

.PHONY: up down logs rebuild

help:
	@echo "Targets:"
	@echo "  up       -> build and start $(SERVICE) (includes graphdb)"
	@echo "  down     -> stop and remove $(SERVICE) and graphdb"
	@echo "  logs     -> tail logs for $(SERVICE)"
	@echo "  rebuild  -> force rebuild and restart"

up:
	docker compose -f $(COMPOSE_FILE) up -d --build

down:
	docker compose -f $(COMPOSE_FILE) down --remove-orphans

logs:
	docker compose -f $(COMPOSE_FILE) logs -f

logs-client:
	docker compose -f $(COMPOSE_FILE) logs -f orion-gdb-client

logs-graphdb:
	docker compose -f $(COMPOSE_FILE) logs -f graphdb

rebuild:
	docker compose -f $(COMPOSE_FILE) build --no-cache
	docker compose -f $(COMPOSE_FILE) up -d

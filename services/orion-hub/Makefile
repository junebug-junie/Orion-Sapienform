# Orion Hub Makefile
# ==================
# Convenience targets for local dev and prod deployments of orion-hub

NET  ?= app-net
MODE ?= dev
LOG_TAIL ?= 200

.PHONY: up down restart build status logs log shell env

## Start Orion Hub in the selected MODE (dev|prod)
up:
	@MODE=$(MODE) docker compose up -d --build
	@echo "✔ Orion Hub started in $(MODE) mode"

## Stop Orion Hub and remove orphan containers
down:
	@docker compose down --remove-orphans

## Restart Orion Hub
restart: down up

## Build images without starting containers
build:
	@docker compose build

## Show running containers for this service
status:
	@docker compose ps

## Follow logs from all services
logs:
	@docker compose logs -f --tail=$(LOG_TAIL)

## Follow logs from a specific service: make log SVC=voice-app
SVC ?= voice-app
log:
	@docker compose logs -f --tail=$(LOG_TAIL) $(SVC)

## Open a shell inside the main voice-app container
shell:
	@docker compose exec -it voice-app /bin/bash || docker compose exec -it voice-app /bin/sh

## Show effective env values
env:
	@echo "MODE=$(MODE)"
	@echo "NET=$(NET)"
	@echo "LOG_TAIL=$(LOG_TAIL)"

# Makefile for Orion Hub

# Default env
MODE ?= dev
COMPOSE = MODE=$(MODE) docker compose

# =====================
# Lifecycle Targets
# =====================

start: ## Start services (MODE defaults to dev)
	@$(COMPOSE) up -d

start-build: ## Start with fresh build
	@$(COMPOSE) up -d --build

stop: ## Stop services (no volume removal)
	@$(COMPOSE) down

down: ## Stop services & remove volumes/orphans
	@$(COMPOSE) down --volumes --remove-orphans

nuke: ## Remove containers, volumes, networks (CAREFUL!)
	@$(COMPOSE) down --volumes --remove-orphans
	@docker network prune -f

# =====================
# Logs & Status
# =====================

logs: ## Tail all logs (200 lines)
	@$(COMPOSE) logs -f --tail=200

log-voice: ## Tail logs for voice-app
	@$(COMPOSE) logs -f voice-app

log-tts: ## Tail logs for coqui-tts
	@$(COMPOSE) logs -f coqui-tts

log-caddy: ## Tail logs for caddy
	@$(COMPOSE) logs -f caddy

status: ## Show container status
	@docker compose ps

# =====================
# Dev Utils
# =====================

tunnel: ## SSH tunnel to remote dev host (edit host as needed)
	ssh -N -L 18080:127.0.0.1:80 janus@192.168.1.7

shell: ## Exec into voice-app container
	@docker exec -it orion-voice-app /bin/bash

rebuild: ## Rebuild without cache
	@$(COMPOSE) build --no-cache

# =====================
# Help Target
# =====================

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'


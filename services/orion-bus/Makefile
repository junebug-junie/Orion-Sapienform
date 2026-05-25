# ───────────────────────────────────────────────
# 🧠 Orion Bus Makefile — Generic Message Bus Layer
# ───────────────────────────────────────────────

PROJECT ?= $(shell grep ^PROJECT= /mnt/services/Orion-Sapienform/.env | cut -d= -f2)
NET ?= app-net
CORE ?= $(PROJECT)-bus-core
EXPORTER ?= $(PROJECT)-bus-exporter
OBSERVER ?= orion-$(PROJECT)-bus-observer

.PHONY: up down ps status logs cli tail stream metrics observer-logs observer-ps

# Start or rebuild the bus stack
up:
	@docker network inspect $(NET) >/dev/null 2>&1 || docker network create $(NET)
	@docker compose up -d --build
	@echo "✔ Bus Core: tcp://localhost:6379"
	@echo "✔ Metrics:  http://localhost:9121/metrics"

# Stop and remove containers
down:
	@docker compose down --remove-orphans

# Show running containers
ps status:
	@docker compose ps
	@echo
	@docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}' \
	  | awk 'NR==1 || /bus-core|bus-exporter|bus-observer/'

observer-logs:
	@docker logs --tail=200 $(OBSERVER) 2>&1

observer-ps:
	@docker ps --format 'table {{.Names}}\t{{.Image}}' | awk 'NR==1 || /bus/'

# Follow logs from all services
logs:
	@docker compose logs -f --tail=100

# Drop into bus-core CLI (temporary redis backend)
cli:
	@docker exec -it $(CORE) redis-cli

# Stream raw event data (use carefully — very noisy)
tail:
	@docker exec -it $(CORE) redis-cli MONITOR

# Inspect any stream key (default = orion:evt:gateway)
STREAM ?= orion:evt:gateway
stream:
	@docker exec -it $(CORE) redis-cli XREVRANGE $(STREAM) + - COUNT 20

# Quick metric peek
metrics:
	@curl -s http://localhost:9121/metrics | head -n 25

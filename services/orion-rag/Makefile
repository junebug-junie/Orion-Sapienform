# Orion RAG Service Makefile
NET ?= app-net
PORT ?= 8001
SERVICE ?= orion-rag-service

.PHONY: up down status logs smoke rebuild

help:
	@echo "Targets:"
	@echo "  up         -> build and start RAG service"
	@echo "  down       -> stop and remove RAG service"
	@echo "  status     -> show container status and mapped ports"
	@echo "  logs       -> tail logs for the RAG service"
	@echo "  smoke      -> run a health check on /health endpoint"
	@echo "  rebuild    -> teardown, rebuild, restart service"

up:
	@docker compose up -d --build
	@echo "✔ Orion RAG service running at http://localhost:$(PORT)"

down:
	@docker compose down --remove-orphans

status:
	@docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}' | awk 'NR==1||/$(SERVICE)/||/orion-qdrant/'

logs:
	@docker compose logs -f --tail=200 $(SERVICE)

smoke:
	@echo "→ Running health check..."
	@curl -s http://localhost:$(PORT)/health || echo "Health check failed."

rebuild: down up

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
	@echo "  health     -> show health
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

health:
	@curl -sf http://localhost:$(PORT)/health | jq .

logs:
	@docker compose logs -f --tail=200 $(SERVICE)


ingest:
	@curl -s http://localhost:$(PORT)/ingest \
	  -H "Content-Type: application/json" \
	  -d '{"path":"$(SRC_FILE)"}' | jq

ask:
	@curl -s http://localhost:$(PORT)/ask \
	  -H "Content-Type: application/json" \
	  -d '{"question":"$(QUESTION)"}' | jq

smoke: health ingest ask
	@echo "✔ Smoke test finished against $(SRC_FILE)"

rebuild: down up

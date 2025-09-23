NET ?= app-net

.PHONY: up down status logs smoke

up:
	@docker compose up -d --build
	@echo "✔ Collapse Mirror running on http://localhost:8085"

down:
	@docker compose down --remove-orphans

status:
	@docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}' \
	  | awk 'NR==1||/orion-collapse-mirror/'

logs:
	@docker compose logs -f orion-collapse-mirror

smoke:
	@echo "Running health check..."
	@curl -sf http://localhost:8085/health || echo "✖ Health check failed."

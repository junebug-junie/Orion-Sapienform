NET ?= orion-net

.PHONY: up down status logs smoke

up:
	docker compose up -d --build
	@echo "✔ Collapse Mirrors running on http://localhost:8085"

down:
	docker compose down --remove-orphans

status:
	docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}' | awk 'NR==1||/collapse-mirrors/'

logs:
	docker compose logs -f collapse-mirrors

smoke:
	@echo "Running health check..."
	@curl -s http://localhost:8085/health || echo "Health check failed."
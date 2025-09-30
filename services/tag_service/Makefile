SERVICE ?= tag-service

.PHONY: up down logs

up:
	docker compose up -d --build
	@echo "✔ Tag Service running on http://localhost:8001/docs"

down:
	docker compose down --remove-orphans

logs:
	docker compose logs -f $(SERVICE)

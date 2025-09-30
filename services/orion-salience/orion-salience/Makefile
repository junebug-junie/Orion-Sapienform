SERVICE ?= orion-salience
PORT ?= 8091

.PHONY: up down logs health check-dream dream-now

up:
	@docker compose up -d --build
	@echo "✔ Orion Salience running on http://localhost:$(PORT)"

down:
	@docker compose down --remove-orphans

logs:
	@docker compose logs -f $(SERVICE)

health:
	@curl -s http://localhost:$(PORT)/health | jq

check-dream:
	@curl -s http://localhost:$(PORT)/health/cron | jq

dream-now:
	@echo "🌙 Triggering manual dream cycle..."
	@docker exec -it $(SERVICE) python offline_worker.py full
	@echo "→ Dream cycle complete. Check status with 'make check-dream'."

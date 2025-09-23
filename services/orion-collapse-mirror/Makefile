# Orion Collapse Mirror Makefile
SERVICE ?= orion-collapse-mirror
CONTAINER ?= orion-collapse-mirror-orion-collapse-mirror-1
DB_PATH ?= /mnt/storage/collapse-mirrors/collapse.db

.PHONY: up down logs sqlite init-db dump-db

help:
	@echo "Targets:"
	@echo "  up        -> build and start Collapse Mirror"
	@echo "  down      -> stop and remove Collapse Mirror"
	@echo "  logs      -> tail logs from Collapse Mirror"
	@echo "  sqlite    -> open sqlite3 REPL inside container"
	@echo "  init-db   -> install sqlite3 and run schema/table introspection"
	@echo "  dump-db   -> export collapse_mirror table to CSV on host"

up:
	@docker compose up -d --build
	@echo "✔ Collapse Mirror running on http://localhost:8087"

down:
	@docker compose down --remove-orphans

logs:
	@docker compose logs -f $(SERVICE)

sqlite:
	@docker exec -it $(CONTAINER) sqlite3 $(DB_PATH)

init-db:
	@docker exec -it $(CONTAINER) bash -c "\
		apt-get update && apt-get install -y sqlite3 && \
		sqlite3 $(DB_PATH) '.tables' && \
		sqlite3 $(DB_PATH) '.schema collapse_mirror' \
	"

dump-db:
	@docker exec -it $(CONTAINER) bash -c "\
		apt-get update && apt-get install -y sqlite3 && \
		sqlite3 $(DB_PATH) '.headers on' '.mode csv' \
			'.output /mnt/storage/collapse-mirrors/collapse_export.csv' \
			'SELECT * FROM collapse_mirror;' '.output stdout' \
	"
	@echo "✔ Exported collapse_mirror table → /mnt/storage/collapse-mirrors/collapse_export.csv"

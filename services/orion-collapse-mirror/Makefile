# Orion Collapse Mirror Makefile
SERVICE ?= orion-collapse-mirror
CONTAINER ?= orion-collapse-mirror-orion-collapse-mirror-1
DB_PATH ?= /mnt/storage/collapse-mirrors/collapse.db
GRAPHDB_URL ?= http://graphdb:7200
GRAPHDB_REPO ?= collapse

.PHONY: up down logs sqlite sql-db-limit-5 init-db migrate-db dump-db inspect

help:
	@echo "Targets:"
	@echo "  up             -> build and start Collapse Mirror"
	@echo "  down           -> stop and remove Collapse Mirror"
	@echo "  inspect        -> see across all memory storage events"
	@echo "  logs           -> tail logs from Collapse Mirror"
	@echo "  sqlite         -> open sqlite3 REPL inside container"
	@echo "  sql-db-limit-5 -> preview last 5 collapse entries"
	@echo "  init-db        -> install sqlite3 and introspect schema"
	@echo "  migrate-db     -> patch schema (make timestamp + environment nullable)"
	@echo "  dump-db        -> export collapse_mirror table to CSV on host"

up:
	@docker compose up -d --build
	@echo "✔ Collapse Mirror running on http://localhost:8087"

down:
	@docker compose down --remove-orphans

logs:
	@docker compose logs -f $(SERVICE)

inspect:
	@if [ -z "$(ID)" ]; then \
		echo "❌ Usage: make inspect ID=<collapse_id>"; \
	else \
		docker exec -e GRAPHDB_URL=$(GRAPHDB_URL) -e GRAPHDB_REPO=$(GRAPHDB_REPO) -it $(CONTAINER) \
			python tools/inspect_collapse.py $(ID); \
	fi

sql-db-limit-5:
	@docker exec -it $(CONTAINER) \
		sqlite3 $(DB_PATH) \
		"SELECT id, observer, timestamp, summary FROM collapse_mirror ORDER BY timestamp DESC LIMIT 5;"

sqlite:
	@docker exec -it $(CONTAINER) sqlite3 $(DB_PATH)

init-db:
	@docker exec -it $(CONTAINER) bash -c "\
		apt-get update && apt-get install -y sqlite3 && \
		sqlite3 $(DB_PATH) '.tables' && \
		sqlite3 $(DB_PATH) '.schema collapse_mirror' \
	"

migrate-db:
	@docker exec -it $(CONTAINER) bash -c "\
		apt-get update && apt-get install -y sqlite3 && \
		sqlite3 $(DB_PATH) 'PRAGMA foreign_keys=off; \
			BEGIN TRANSACTION; \
			CREATE TABLE IF NOT EXISTS collapse_mirror_new ( \
				id TEXT PRIMARY KEY, \
				observer TEXT, \
				trigger TEXT, \
				observer_state TEXT, \
				field_resonance TEXT, \
				type TEXT, \
				emergent_entity TEXT, \
				summary TEXT, \
				mantra TEXT, \
				causal_echo TEXT, \
				timestamp TEXT NULL, \
				environment TEXT NULL \
			); \
			INSERT INTO collapse_mirror_new SELECT * FROM collapse_mirror; \
			DROP TABLE collapse_mirror; \
			ALTER TABLE collapse_mirror_new RENAME TO collapse_mirror; \
			COMMIT;' \
	"
	@echo "✔ Schema migrated: timestamp + environment now nullable"

dump-db:
	@docker exec -it $(CONTAINER) bash -c "\
		apt-get update && apt-get install -y sqlite3 && \
		sqlite3 $(DB_PATH) '.headers on' '.mode csv' \
			'.output /mnt/storage/collapse-mirrors/collapse_export.csv' \
			'SELECT * FROM collapse_mirror;' '.output stdout' \
	"
	@echo "✔ Exported collapse_mirror table → /mnt/storage/collapse-mirrors/collapse_export.csv"

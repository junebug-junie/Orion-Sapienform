# Global hydration Makefile

# --- Config (tweak if needed)
PG_CONTAINER ?= orion-postgres-1
PG_USER ?= postgres
PG_DB ?= conjourney

SQL_DIR := services/orion-enrichment-writer/examples

.PHONY: hydrate-db
hydrate-db:
	@echo "⚡ Applying hydration views into $(PG_DB) via $(PG_CONTAINER)"
	docker exec -i $(PG_CONTAINER) psql -U $(PG_USER) -d $(PG_DB) < $(SQL_DIR)/hydrate_latest.sql
	docker exec -i $(PG_CONTAINER) psql -U $(PG_USER) -d $(PG_DB) < $(SQL_DIR)/hydrate_all.sql
	@echo "✅ Hydration views installed"

.PHONY: list-views
list-views:
	docker exec -it $(PG_CONTAINER) psql -U $(PG_USER) -d $(PG_DB) -c "\dv"

.PHONY: sample-query
sample-query:
	@echo "🔍 Showing 5 most recent collapse + enrichment entries"
	docker exec -it $(PG_CONTAINER) psql -U $(PG_USER) -d $(PG_DB) -c "\
		SELECT id, summary, tags, entities, enrichment_ts \
		FROM collapse_with_latest_enrichment \
		ORDER BY timestamp DESC \
		LIMIT 5;"

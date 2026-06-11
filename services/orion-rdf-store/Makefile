SERVICE ?= orion-rdf-store
COMPOSE_FILE ?= docker-compose.yml

.PHONY: help up down logs logs-fuseki rebuild config preflight health-probe recover \
	disk-guard storage-status delete-stale-copy restore-mount compact prune prune-dry-run

help:
	@echo "Orion RDF Store (Fuseki operator stack)"
	@echo "Targets:"
	@echo "  help               -> this message"
	@echo "  preflight          -> mkdir data dirs + soft disk guard"
	@echo "  disk-guard         -> fail if FUSEKI_DATA_DIR filesystem is low on space"
	@echo "  storage-status     -> show active mount vs stale duplicate sizes"
	@echo "  delete-stale-copy  -> remove non-active copy (CONFIRM=1; frees ~1.3T after migration)"
	@echo "  restore-mount      -> remount Fuseki on the copy that still has a full dataset"
	@echo "  compact            -> offline TDB2 compact (SOURCE/DEST env; stops Fuseki)"
	@echo "  prune              -> SPARQL retention prune (RDF_RETENTION_* env)"
	@echo "  prune-dry-run      -> retention dry-run (no UPDATE executed)"
	@echo "  config             -> docker compose config sanity-check"
	@echo "  up                 -> start Fuseki (blocked when disk guard fails)"
	@echo "  down               -> stop and remove"
	@echo "  logs / logs-fuseki -> tail logs"
	@echo "  rebuild            -> pull image and restart"
	@echo "  health-probe       -> ping + query + graph-store write probe"
	@echo "  recover            -> restart when write probe fails"

preflight:
	mkdir -p "$${RDF_STORE_DATA_ROOT:-/mnt/graphdb/rdf-store}/fuseki"
	mkdir -p "$${RDF_STORE_DATA_ROOT:-/mnt/graphdb/rdf-store}/fuseki-backups"
	@chmod +x scripts/fuseki_disk_guard.sh
	@./scripts/fuseki_disk_guard.sh || true

disk-guard:
	@chmod +x scripts/fuseki_disk_guard.sh
	@./scripts/fuseki_disk_guard.sh

storage-status:
	@chmod +x scripts/fuseki_storage_status.sh
	@./scripts/fuseki_storage_status.sh

delete-stale-copy:
	@chmod +x scripts/fuseki_delete_stale_copy.sh
	@./scripts/fuseki_delete_stale_copy.sh

restore-mount:
	@chmod +x scripts/fuseki_restore_active_mount.sh scripts/fuseki_storage_status.sh
	@./scripts/fuseki_restore_active_mount.sh

compact:
	@chmod +x scripts/fuseki_tdb_compact.sh
	@./scripts/fuseki_tdb_compact.sh

prune:
	@chmod +x scripts/fuseki_prune_retention.sh
	@./scripts/fuseki_prune_retention.sh

prune-dry-run:
	@chmod +x scripts/fuseki_prune_retention.sh
	@./scripts/fuseki_prune_retention.sh --dry-run

config:
	docker compose --env-file .env_example -f $(COMPOSE_FILE) config

up: preflight disk-guard
	docker compose -f $(COMPOSE_FILE) up -d

down:
	docker compose -f $(COMPOSE_FILE) down --remove-orphans

logs:
	docker compose -f $(COMPOSE_FILE) logs -f

logs-fuseki:
	docker compose -f $(COMPOSE_FILE) logs -f orion-athena-fuseki

rebuild:
	docker compose -f $(COMPOSE_FILE) pull
	docker compose -f $(COMPOSE_FILE) up -d

health-probe:
	@chmod +x scripts/fuseki_health_probe.sh
	@./scripts/fuseki_health_probe.sh

recover:
	@chmod +x scripts/fuseki_health_probe.sh scripts/fuseki_recover.sh
	@./scripts/fuseki_recover.sh

-- ORION-MIGRATION-NOT-A-MIGRATION: pg_dump output, never applied as a migration
-- Kept as a schema capture. Its objects are not expected to exist as written -- live
-- collapse_enrichment has only its primary key and 0 rows.
-- Marker read by scripts/check_sql_migrations_applied.py.

PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE collapse_mirror (
	id VARCHAR NOT NULL, 
	observer VARCHAR, 
	"trigger" TEXT, 
	observer_state VARCHAR, 
	field_resonance TEXT, 
	type VARCHAR, 
	emergent_entity TEXT, 
	summary TEXT, 
	mantra TEXT, 
	causal_echo TEXT, 
	timestamp VARCHAR, 
	environment VARCHAR, 
	PRIMARY KEY (id)
);
-- Row data removed 2026-10-14: these were private journal entries (repo is public).
CREATE TABLE collapse_enrichment (
	id VARCHAR NOT NULL, 
	collapse_id VARCHAR NOT NULL, 
	service_name VARCHAR NOT NULL, 
	service_version VARCHAR NOT NULL, 
	enrichment_type VARCHAR NOT NULL, 
	tags TEXT, 
	entities TEXT, 
	salience FLOAT, 
	ts DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(collapse_id) REFERENCES collapse_mirror (id)
);
CREATE INDEX ix_collapse_mirror_id ON collapse_mirror (id);
CREATE INDEX ix_collapse_enrichment_id ON collapse_enrichment (id);
COMMIT;

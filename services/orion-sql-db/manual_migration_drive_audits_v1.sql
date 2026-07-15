-- Drive audit durable measurement store (bus event -> sql-writer -> SQL read).
-- Producer: orion-spark-concept-induction (memory.drives.audit.v1 on orion:memory:drives:audit)
-- Consumer: autonomy measurement gate (windowed range scans over created_at, reads active_count)
--
-- On boot, sql-writer creates the table via Base.metadata.create_all (from the
-- DriveAuditSQL model) and creates the index via the equivalent DDL in
-- app/main.py lifespan (CREATE INDEX IF NOT EXISTS). This file is the standalone
-- equivalent for manual application against a running Postgres; the CREATE TABLE
-- here is a harmless no-op once the model has already created it.
--
-- Indexes are owned here (not via model index=True) to keep a single source of
-- truth. The gate windows on COALESCE(observed_at, created_at) — and since the
-- artifact's ts always populates observed_at, a bare created_at index would
-- never serve that predicate; the expression index below matches it exactly.
--
-- active_count is derived by sql-writer as len(active_drives) (0 when
-- malformed/absent); it is not on the wire payload. evidence_items /
-- source_event_refs / summary / tick_attribution are intentionally NOT stored:
-- this is a slim measurement table, not an archive.

CREATE TABLE IF NOT EXISTS drive_audits (
    artifact_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    active_count INTEGER NOT NULL,
    active_drives JSONB,
    dominant_drive TEXT NULL,
    drive_pressures JSONB,
    correlation_id TEXT NULL,
    observed_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drive_audits_window ON drive_audits ((COALESCE(observed_at, created_at)) DESC);

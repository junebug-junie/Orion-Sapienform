-- orion-hardware-watch: one row per incident (cooling / cpu_heat / gpu_heat), so a restart never
-- re-fires an alert or a second urgent run for an incident that is already open.
-- Plan: docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md
--
-- Apply BEFORE starting orion-hardware-watch (it refuses to start without the table):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < services/orion-sql-db/manual_migration_hardware_watch_v1.sql
--
-- Additive only: a new table, two indexes, and a read-only grant for Orion's sandbox role.

BEGIN;

CREATE TABLE IF NOT EXISTS hardware_watch_incident (
    incident_id         text PRIMARY KEY,
    rule                text NOT NULL CHECK (rule IN ('cooling', 'cpu_heat', 'gpu_heat')),
    subject             text NOT NULL,
    status              text NOT NULL CHECK (status IN ('open', 'resolved')),
    open_reason         text NOT NULL,
    opened_at           timestamptz NOT NULL,
    resolved_at         timestamptz,
    resolve_reason      text,
    resolved_by         text,
    -- after an operator resolve, the same (rule, subject) does not re-open before this
    snooze_until        timestamptz,
    shed_requested      boolean NOT NULL DEFAULT false,
    shed_reason         text,
    shed_requested_at   timestamptz,
    alert_sent_at       timestamptz,
    alert_attempts      integer NOT NULL DEFAULT 0,
    alert_error         text,
    urgent_requested_at timestamptz,
    urgent_error        text,
    evidence            jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at          timestamptz NOT NULL DEFAULT now()
);

-- The feedback guard: at most one open incident per (rule, subject), enforced by the database.
CREATE UNIQUE INDEX IF NOT EXISTS hardware_watch_incident_one_open
    ON hardware_watch_incident (rule, subject) WHERE status = 'open';

CREATE INDEX IF NOT EXISTS hardware_watch_incident_opened_at
    ON hardware_watch_incident (opened_at DESC);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'orion_readonly') THEN
        GRANT SELECT ON public.hardware_watch_incident TO orion_readonly;
    END IF;
END $$;

COMMIT;

-- Undo:
--   DROP TABLE IF EXISTS hardware_watch_incident;
--
-- Verify:
--   SELECT count(*) FROM hardware_watch_incident;
--   SELECT indexname FROM pg_indexes WHERE tablename = 'hardware_watch_incident';

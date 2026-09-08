-- endogenous_outreach_decisions v2: add sustained_load_pressure_channel /
-- sustained_load_pressure_node_id.
--
-- WHY THIS EXISTS (2026-09-07, root-caused live): Orion sent Juniper an
-- unprompted message naming a specific internal channel
-- ("harness_closure prediction error") that was never in the context it was
-- given and had read 0.0/NULL for the prior 24h. The real driver that tick
-- WAS `sustained_load_pressure` (`node:athena`, a real 7-reading run) --
-- confirmed by tracing this very table, row 6e6887fe-5789-412a-a76c-
-- f40cd8a727dc, 2026-09-07 23:35:36 UTC. `target_id`/`run_length`/
-- `peak_deviation_pressure`/`sustained_load_pressure` were all already
-- columns here and made that trace possible; the two new ones close the
-- one remaining gap this table had for that same trace -- there was no
-- durable record of WHICH channel/node produced the sustained-load
-- reading, only the scalar. `orion.field.significance.
-- SustainedLoadReading` (orion/field/significance.py) now carries that
-- identity end to end; this column pair is where it lands for durable,
-- post-hoc forensic tracing, same role the four pre-existing structured
-- columns already play for their own fields.
--
-- Both NULL in exactly the same two cases the pre-existing
-- `sustained_load_pressure` column already allows for: a genuine
-- "nothing loaded_steady right now" reading, or (for rows written before
-- this migration) a decision cycle that predates identity being tracked at
-- all. Never fabricated.
--
-- On boot, sql-writer does NOT auto-apply this file (this table's v1
-- migration is itself "manual", not auto-run) -- apply by hand:
--   psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_endogenous_outreach_decisions_v2.sql
-- Harmless no-op once applied (ADD COLUMN IF NOT EXISTS), same convention
-- manual_migration_drive_audits_v2.sql already uses.

ALTER TABLE endogenous_outreach_decisions
    ADD COLUMN IF NOT EXISTS sustained_load_pressure_channel TEXT;
ALTER TABLE endogenous_outreach_decisions
    ADD COLUMN IF NOT EXISTS sustained_load_pressure_node_id TEXT;

-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Body economy Patch A3 (docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, Organ 1 Body).
--
-- physiology_state: one row per Body-organ tick (~120 s). Writer: orion-sql-writer off bus
-- orion:physiology:state (PhysiologyStateV1; producer: orion-durable-runs' `physiology` node in the
-- regulation thread). constraints holds every reading verbatim: heat, energy, lane:<role>,
-- attention:juniper, claude, each with level/unit/budget and an explicit no_reading. Insert-only,
-- PK tick_id (deterministic per tick). sql-writer's create_all makes the same table at boot; this file
-- is the jsonb-typed operator contract. Additive only.

CREATE TABLE IF NOT EXISTS physiology_state (
    tick_id       text PRIMARY KEY,
    generated_at  timestamptz NOT NULL,
    constraints   jsonb NOT NULL,
    warnings      jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_physiology_state_generated ON physiology_state (generated_at);
CREATE INDEX IF NOT EXISTS idx_physiology_state_created ON physiology_state (created_at);

-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Body economy Patch C1 (docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, Organ 3
-- Appetite + Organ 4 Choice), SHADOW.
--
-- want_ledger: one row per bid decision. Writer: orion-sql-writer off bus orion:physiology:want:decision
-- (WantDecisionV1; producer: orion-durable-runs POST /physiology/bid). Funded or deferred, the lane, the
-- prices seen, the net, the binding constraint, what it displaced; `decision` holds the full decision
-- with the want. shadow = true on every Patch C row: nobody obeyed it. Insert-only, PK decision_id.
-- sql-writer's create_all makes the same table at boot; this file is the jsonb-typed operator contract.
-- Additive only.

CREATE TABLE IF NOT EXISTS want_ledger (
    decision_id         text PRIMARY KEY,
    want_id             text NOT NULL,
    family              text NOT NULL,
    producer            text NOT NULL,
    verb                text NOT NULL,
    tier                integer NOT NULL,
    decided_at          timestamptz NOT NULL,
    outcome             text NOT NULL,
    reason              text NOT NULL,
    lane                text,
    net                 double precision,
    binding_constraint  text,
    shadow              boolean NOT NULL,
    correlation_id      text,
    decision            jsonb NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_want_ledger_decided ON want_ledger (decided_at);
CREATE INDEX IF NOT EXISTS idx_want_ledger_family_decided ON want_ledger (family, decided_at);
CREATE INDEX IF NOT EXISTS idx_want_ledger_want ON want_ledger (want_id);

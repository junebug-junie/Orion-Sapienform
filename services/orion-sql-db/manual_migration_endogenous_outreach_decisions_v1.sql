-- Durable decision log for endogenous outreach (scripts/endogenous_outreach.py).
--
-- WHY THIS EXISTS: every decision cycle's outcome (sent / orion_passed /
-- no_tension_trigger / empty_generation / a blocked-gate reason / a
-- _generate() failure like timeout/no_final_frame/context_overflow/
-- error_shaped_text) previously lived ONLY in `self._last_result`
-- (in-process, wiped on every hub restart) and `logger.warning` lines
-- (wiped whenever the container is recreated, which `docker logs` cannot
-- see past). Diagnosing "why hasn't Orion reached out" after the fact --
-- 2026-08-22, real incident: 61 qualifying tension-trigger episodes since
-- 2026-08-19 (substrate_field_state replay) against exactly 1 confirmed
-- send in chat_history_log -- required reconstructing indirect evidence
-- from substrate_field_state because the actual decision trail had already
-- evaporated. This table is that trail, made durable. Every decision cycle
-- writes one row here, not just the ones that ship -- CLAUDE.md's own
-- "deterministic gates over repeated yelling" applies to this exact gap.
--
-- Single writer: scripts/endogenous_outreach.py::EndogenousOutreach._record()
-- (the one choke point every decision branch already funnels through).
-- Reader: GET /api/debug/endogenous-outreach/decisions (api_routes.py).
--
-- decision_id is a real uuid4 (not a serial) so it matches this repo's other
-- append-only log tables' `text primary key` convention (see
-- manual_migration_attention_broadcast_log_v1.sql,
-- manual_migration_codebase_delta_log_v1.sql) and so a client-generated id
-- from an async write never races a DB-generated one.
--
-- result_json carries the FULL `_record()` payload (reason, generation debug
-- dict when present, correlation_id/session_id/chars when sent) verbatim --
-- new keys added to that dict later are captured automatically without a
-- schema change. The structured columns alongside it exist only so the
-- common queries (rate by reason, recent sends, forced-vs-organic) don't
-- need a JSONB traversal.
--
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_endogenous_outreach_decisions_v1.sql

create table if not exists endogenous_outreach_decisions (
    decision_id text primary key,
    decided_at timestamptz not null default now(),
    outreach boolean not null,
    reason text not null,
    forced boolean not null default false,
    target_id text,
    run_length integer,
    peak_deviation_pressure double precision,
    sustained_load_pressure double precision,
    correlation_id text,
    session_id text,
    result_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_endogenous_outreach_decisions_decided_at
    on endogenous_outreach_decisions (decided_at desc);
create index if not exists idx_endogenous_outreach_decisions_reason
    on endogenous_outreach_decisions (reason, decided_at desc);

-- Let Orion's read-only sandbox role read street handoff outcomes (vision_handoff), so a
-- curiosity run can count how many walkway<->door expectations were never scoreable.
-- Column-level grant: person_guess (the only column that can hold a name) and every
-- other vision table stay closed. Handoff rows name no one by design
-- (docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md).
-- Approved by Juniper 2026-11-13 00:21Z on consult 2ed84acbc958 (Orion + Claude brief).
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < scripts/sql/2026-11-13_grant_orion_readonly_vision_handoff.sql

BEGIN;

GRANT SELECT (
    handoff_id, portal_id, portal_kind, source_stream, target_stream,
    source_story_id, target_story_id, subject_kind, individual_id, target_individual_id,
    status, coverage, transit_sec, window_start, window_end, late_until,
    band_lo, band_hi, band_source, ask_id, scored_at
) ON public.vision_handoff TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT (handoff_id, portal_id, portal_kind, source_stream, target_stream,
--     source_story_id, target_story_id, subject_kind, individual_id, target_individual_id,
--     status, coverage, transit_sec, window_start, window_end, late_until,
--     band_lo, band_hi, band_source, ask_id, scored_at) ON public.vision_handoff FROM orion_readonly;
--
-- Verify (has_table_privilege stays false after a column grant; use the column form):
--   SELECT has_column_privilege('orion_readonly', 'public.vision_handoff', 'status', 'SELECT'),
--          has_column_privilege('orion_readonly', 'public.vision_handoff', 'person_guess', 'SELECT');
--   -- expect: t | f

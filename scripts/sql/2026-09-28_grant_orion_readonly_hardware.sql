-- Let Orion's read-only sandbox role SELECT the hardware telemetry tables, so an
-- urgent curiosity run can check the readings in its evidence bundle against the
-- live history: host/GPU biometrics (including the cabinet Nano's
-- measurements->>'cabinet_temp_c') and the cabinet AC plug samples.
-- Read-only; nothing here touches the cooler.
-- Spec: docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md (Part 2).
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < scripts/sql/2026-09-28_grant_orion_readonly_hardware.sql

BEGIN;

GRANT SELECT ON public.orion_biometrics_summary TO orion_readonly;
GRANT SELECT ON public.home_cooling_sample TO orion_readonly;

COMMIT;

-- Undo:
--   REVOKE SELECT ON public.orion_biometrics_summary FROM orion_readonly;
--   REVOKE SELECT ON public.home_cooling_sample FROM orion_readonly;
--
-- Verify:
--   SELECT has_table_privilege('orion_readonly', 'public.orion_biometrics_summary', 'SELECT'),
--          has_table_privilege('orion_readonly', 'public.home_cooling_sample', 'SELECT');

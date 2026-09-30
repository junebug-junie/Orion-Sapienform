-- GPU pool stage 5.6: drop the four dead legacy GPU-admission tables.
-- Spec: docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md (Decision 5).
--
-- Nothing writes or reads these any more:
--   durable_resource_demands, durable_resource_leases, durable_elastic_slot: frozen since stage 4.5 (2026-09-26);
--   durable_gateway_permits: the /capacity permit broker, no callers since stage 5.4 (2026-09-29 22:48 UTC),
--     code deleted in 5.6.
-- durable_admission_runs and durable_resource_events are the live run registry + outbox and STAY.
-- No kept table has a foreign key into a dropped one (permits -> leases -> demands -> runs only).
--
-- PRODUCTION: do not run this file by hand. Run scripts/gpu_pool_stage5_snapshot_and_drop.sh, which
-- pg_dumps these tables to /tmp/gpu-pool-stage5-drop/, verifies the dump row counts, refuses if any row
-- was written after its cutoff, and only then applies this file.
-- Tests: orion/durable_runs/registry_store.py setup() applies it after the v1 migration, so a test
-- schema matches production.
--
-- Idempotent. Order follows the foreign keys (dependents first).
DROP TABLE IF EXISTS durable_gateway_permits;
DROP TABLE IF EXISTS durable_resource_leases;
DROP TABLE IF EXISTS durable_resource_demands;
DROP TABLE IF EXISTS durable_elastic_slot;
-- Only durable_resource_leases.generation used this sequence (not OWNED BY, so the drop above leaves it).
DROP SEQUENCE IF EXISTS durable_resource_fencing_generation;

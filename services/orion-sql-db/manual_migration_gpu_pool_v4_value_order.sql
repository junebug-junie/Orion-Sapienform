-- ORION-MIGRATION-REQUIRED-BY: none the pool adds these columns itself at boot (app/store.py BOOT_ADDITIVE_COLUMNS)
-- orion-gpu-pool, attention patch 3b: the background queue served by value.
-- Spec: docs/superpowers/specs/2026-11-07-attention-patch3-market-decides-design.md (section 1).
--
-- The body-economy decision a lease serves (want id, net, family), carried on the lease request by
-- durable runs. The pool reads want_net/want_family to order its background class
-- (orion/gpu_pool/value_order.py); a lease with none keeps its arrival place.
--
-- Additive only, safe on the live table (all three nullable: no rewrite). The pool also adds them
-- itself at boot (app/store.py BOOT_ADDITIVE_COLUMNS, kept identical by tests/test_schema_drift_gate.py),
-- so running this first is recommended, not required. lock_timeout: re-run the file if it times out.
SET lock_timeout = '5s';

ALTER TABLE gpu_pool_leases ADD COLUMN IF NOT EXISTS want_id text;
ALTER TABLE gpu_pool_leases ADD COLUMN IF NOT EXISTS want_net double precision;
ALTER TABLE gpu_pool_leases ADD COLUMN IF NOT EXISTS want_family text;

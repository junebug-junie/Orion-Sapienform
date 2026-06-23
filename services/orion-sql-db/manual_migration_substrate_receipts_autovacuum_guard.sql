-- Guard: prevent autovacuum/TOAST on substrate_reduction_receipts from pinning
-- the Postgres cgroup and freezing the host (Athena incident Jun 2026).
-- Pruning is handled by orion-substrate-runtime receipt_pruner, not autovacuum.
ALTER TABLE substrate_reduction_receipts SET (
  autovacuum_enabled = false,
  toast.autovacuum_enabled = false
);

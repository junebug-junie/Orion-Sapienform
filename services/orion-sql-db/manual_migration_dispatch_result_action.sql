-- Name the action on its own result row (2026-08-22).
--
-- substrate_dispatch_results carries latency_ms for 100% of dispatched
-- actions, and did not record WHICH action it was -- so the allocator sourced
-- its cost estimates from substrate_action_outcomes instead, which has
-- dispatch_kind and target_id but only contains actions that DECLARED A
-- SIGNAL: 32.3% of dispatch volume.
--
-- Live consequence, caught on the allocator's first real verdict: 7 of 12
-- pending (kind, target) pairs had no cost estimate and were refused
-- `no_cost_estimate`, while their cost sat in dispatch_results the whole
-- time. The allocator was declining to spend on two thirds of Orion's
-- behaviour for want of data that already existed.
--
-- The alternative was joining results to the frame's candidate array on every
-- lookup. The worker knows both values at save time; storing them is one
-- column each and turns the cost query into a GROUP BY on one table.

ALTER TABLE substrate_dispatch_results
    ADD COLUMN IF NOT EXISTS dispatch_kind TEXT;

ALTER TABLE substrate_dispatch_results
    ADD COLUMN IF NOT EXISTS target_id TEXT;

-- The allocator's access pattern: median cost per action over a window.
CREATE INDEX IF NOT EXISTS substrate_dispatch_results_action_cost_idx
    ON substrate_dispatch_results (dispatch_kind, target_id, created_at DESC)
    WHERE latency_ms IS NOT NULL;

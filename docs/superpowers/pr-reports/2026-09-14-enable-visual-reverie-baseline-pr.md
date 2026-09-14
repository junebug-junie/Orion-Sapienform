# Enable recurring visual reverie baseline

Visual reverie has produced no new images because the merged baseline policy
remained disabled. The deployed activity API is healthy and reports the last
real artifact at `2026-09-08T08:31:33.836206Z`; motor allowance is available,
but proposal-runtime correctly emits no baseline candidate while the policy is
off. Production also lacks the required `reverie_visual_attempt` table.

This patch enables `config/proposals/visual_baseline.v1.yaml` and extends the
existing additive migration to install the scheduler checkpoint. It removes
runtime `CREATE TABLE` from proposal ticks so production schema changes happen
only through the explicit migration.

## Review and validation

- Focused policy and integrated replay tests: 14 passed.
- `git diff --check`: passed.
- Subagent review: no material code defect. Activation remains gated on the
  controlled receipt smoke below.
- Read-only live checks: baseline disabled in thought and proposal images;
  activity history `ok`; no active attempt; no production receipt rows;
  `reverie_visual_attempt` and `visual_baseline_checkpoint` absent;
  `action_outcomes.visual_outcome` present; allocator daily allowance 129,600s.

## Production rollout

Production writes and service recreation require explicit Juniper approval.
Run in this order:

1. Apply `services/orion-sql-db/manual_migration_reverie_visual_attempt.sql`.
2. Build/recreate thought only from this branch. Policy and proposal-runtime
   remain on the already deployed disabled image.
3. Submit one voluntary `POST /visual-chain/run-once`, then verify the selected
   source identity, thermal verdict, stored path/positive bytes/SHA, production
   receipt, activity advancement, and exact dispatch replay. Stop if any join
   is incomplete. A thermal deferral is acceptable evidence for refusal but is
   not production; retry after the configured ten-minute gap.
4. Build/recreate policy-runtime and execution-dispatch-runtime, then
   proposal-runtime last. This last step activates recurring due proposals.
5. Capture one calm baseline trace and one deferred/resumed trace through
   proposal → policy → allocation → dispatch → thought → activity.

Rollback: restore `enabled: false` and rebuild thought, policy, dispatch, and
proposal-runtime. Keep the additive tables and receipts for reconciliation.
Do not restart the retired visual cron or restore the resource-pressure reward.

Production remains `UNVERIFIED` until step 3 produces a joined receipt and
step 5 demonstrates recurrence.

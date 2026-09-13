# Visual reverie baseline implementation

A due visual baseline can now enter the existing proposal → policy → dispatch rail
without an action warrant or information-gain claim. It reserves one ordinary
capacity slot, pays estimated/actual motor cost, and retains policy, risk, scope,
thermal and resource gates. Voluntary image production also satisfies the pending
need. Baseline scheduling is disabled until consumer rollout and production verification.

Implements `6f7a341e9`'s revised design. Branch: `feat/visual-reverie-baseline`.
No graph files, env templates, production data, or running services were changed.

## Contracts and behavior

- Thought owns selected-source provenance, acknowledged production, read-only activity,
  and durable per-attempt claims. Repeated bytes count as a new production receipt,
  not a novel artifact row. Failures and refusals do not reset activity.
- Proposal-runtime owns a persisted singleton checkpoint: one need, bounded retry,
  restart recovery, no accumulated backlog. External candidates cannot mint eligibility.
- Optional eligibility travels through proposal, policy and dispatch, including blocked
  decisions. The real action warrant remains unchanged. Ordinary express confidence,
  operator approval and scope controls still apply; missing field confidence can require
  operator review. The baseline is not an approval bypass.
- `render_scene` no longer claims or learns resource-pressure reduction. Baseline uses
  `expected_effect=None`; ordinary extras can remain unmeasurable without a justified
  signal. Historical posterior rows remain as history.
- Explicit produced/deferred_thermal/deferred_busy/already_satisfied/failed/unknown
  outcomes survive verb, dispatch result, action-outcome SQL and feedback normalization.
  Deferrals and ambiguous results do not become effect observations.
- Unknown blocking diffusion work is held until positive receipt reconciliation or
  operator investigation. No claim is made that an asyncio timeout stops GPU work.

## Scheduling-state quality gate

1. Provenance: `app/store.py::acknowledge_visual_production` verifies file bytes/hash,
   canonical artifact and current chain receipt in one transaction. `load_visual_activity`
   projects those receipts; legacy rows use the matching artifact/chain join.
2. Independence: elapsed age and any count derive from that same production record;
   neither is an independent cognition signal. No pressure/debt/quality score is added.
3. Theory: periodic scheduling with one outstanding need and bounded retries. No
   biological, consciousness or information-theoretic measurement claim.
4. Live data: read-only PostgreSQL query found 1,433 max_steps and 9 generation_failed
   rows. Latest matching artifact: chain `1a7cbbcc-8511-42fe-b919-450994e798f2`, produced
   `2026-09-08T08:31:33.836206Z`, sha
   `f7db253957ac4c99d0cd126ac8468b90eb63f2630eaaef4f6d1f8792a9ed400f`.
   Reading its file from the existing thought container confirmed 868,728 bytes and
   the identical SHA-256. This verifies one historical artifact, not new execution.
   Receipt/reset, duplicate-byte, failure and replica behavior are verified in isolated
   PostgreSQL tests. **New production receipt and end-to-end live gate remain UNVERIFIED**;
   baseline remains disabled. It is not wired into a cognition model.
5. Existing mechanism: reuse proposal/policy/dispatch/allocator/thermal and existing
   thought tables; newest-row age is not used for baseline admission. No scheduler service.
6. Reversibility: disable baseline policy; retain readers and claims for reconciliation.
   Do not restore the retired resource-pressure reward or cron.

## Review findings fixed

- Finding: a positively completed ambiguous attempt could keep activity permanently active.
  - Fix: exclude/reconcile only positively acknowledged production before active gating.
  - Evidence: isolated PostgreSQL replay/activity regression.
- Finding: freshness validation and same-dispatch lookup could hide completed replay receipts.
  - Fix: immutable request replay/reconciliation before new authorization checks;
    preserve the typed execution receipt through the verb.
  - Evidence: delayed replay and changed-request regression.
- Finding: visual context evidence bound was smaller than the producing thought schema.
  - Fix: share the producer's 50-reference bound.
  - Evidence: maximum-bound context compatibility test.

## Validation

Tests: 148 admission/allocator/policy/dispatch/feedback tests passed; 142 thought
storage/source/thermal/deadline/HTTP tests pass across the final suite and corrected
fixture rerun (12 use isolated PostgreSQL). SQL-shape/idempotency: 7 passed; feedback
worker: 11 passed. The final thought fixture correction supplies a required nullable
coalition field; the full store suite rerun passed 57 tests.

Evals: integrated scheduler→proposal→policy→allocator→thermal→feedback replay passed;
image honest-degradation matrix passed separately. These use synthetic readings and
stored fixtures, not live diffusion.

Docker builds passed for thought, proposal-runtime, policy-runtime, dispatch-runtime,
all four cortex-exec lanes, feedback-runtime and SQL-writer. An isolated thought
container returned health ok and activity history_status=ok with confirmed empty
history against a disposable PostgreSQL database. Metric lineage and definition-drift
gates passed with zero changed metric definitions.

The initial subagent review completed and material findings were fixed and covered
by regression tests. The follow-up reviewer hit the account usage limit; independent
subagent re-review of those fixes is outstanding. Final status is DONE_WITH_CONCERNS
until that review and production activation evidence are supplied. Docker builds use `scripts/safe_docker_build.sh` from the
implementation worktree with project `orion-visual-baseline-test`; these do not
restart production containers.

## Rollout and rollback

Production changes require Juniper's explicit approval (AGENTS.md §13). None were run.
After approval, from the implementation worktree:

1. Apply the additive migrations `manual_migration_action_outcomes_visual_outcome.sql`
   and `manual_migration_reverie_visual_attempt.sql` using the operator Postgres DSN.
2. Keep baseline disabled. Build/recreate SQL-writer and every closed-schema reader
   of proposal/policy/dispatch/action outcome records, including policy-runtime,
   dispatch-runtime, feedback-runtime and any persisted-frame readers. Deploy cortex-exec
   and thought before proposal-runtime begins emitting eligibility.
3. Verify `GET /visual-chain/activity`; exercise one voluntary run and inspect selected
   source, thermal verdict, bytes/hash, committed receipt and activity reset. Check an
   actual deferred/resumed diffusion attempt; do not infer physical cancellation from
   a timeout. Confirm motor allowance accommodates the configured cadence.
4. Set policy `enabled: true` only after those gates pass, rebuild the policy-owning
   thought/proposal/policy/dispatch services, and collect calm baseline plus deferred/resumed
   traces named in the design. Keep the old visual worker disabled.

Use `scripts/safe_docker_build.sh <service> up -d --build` for each approved deployment.
For rollback set policy `enabled: false` and rebuild those four services. Preserve
attempt records and readers so ambiguous work can still be investigated.

Feedback-runtime has no dedicated periodic eval harness; follow-up: add a live
feedback replay harness. This patch's scheduler and image-degradation evals do not
claim production posterior coverage; deterministic integration tests verify its
non-observation contract.

# Reading Durable Admission

## Summary

Reading was outside the durable graph: Hub directly started unified turns, so
individual stance/harness requests competed for short HTTP admission budgets.
Both reading stages now submit an admitted `reading.turn` workflow. This is a
transport/admission change, not a timeout increase or GPU capacity-policy change.

## Current Architecture

- Services: Hub owns seed selection, wallets, parsing, evidence and landing;
  durable-runs owns checkpointed GPU admission and a single model turn.
- Entry points: Hub Stage 1/2 ticks; durable HTTP POST/GET /runs; held Hub bus RPC.
- Config: Hub settings and .env_example, HUB_READING_DURABLE_URL.
- Bus: orion:reading:turn:request and orion:reading:turn:reply:*.
- Registry: ReadingTurnRequestV1 and ReadingTurnResultV1.
- Compose: existing Hub host networking and durable port 8124; no new service.
- Tests: reading worker suites, disposable PostgreSQL, Lua-backed fake Redis,
  actual LangGraph interrupts, hold validation and RPC replay.
- Eval: existing offline reading handoff behavior evaluation.

## Contracts

- Persist the first prompt/run binding before HTTP submission; lost acknowledgments
  and Hub restarts reuse it. Binding generations are independent of failed-attempt
  counters because admission refusals do not increment those counters.
- Queue waiting does not debit wallets or increment failed attempts. Atomic Redis
  settlement markers prevent double charges on result replay.
- A held read-only turn preserves source-fetch receipts and its actual fenced
  correlation. Empty/error results cannot become completed reading.
- Actual model failures remain bounded by the existing queue policy. Holds are
  released on completion/failure; lost holds requeue.
- Post-processing retries retain their result binding; journal repair remains
  artifact replay. Operator cancellation stops the seed without auto-retry.
- New pending sources cannot jump ahead of an active stage binding; stale-source
  cleanup cannot discard a bound run.

## Scope and Rollback

No change to memory authority, private recall, identity, autonomy policy, source
trust, graph writes, thermal guards, model selection or physical GPU capacity.
Prompts/results now also persist in the existing durable checkpoint database;
they stay inside the trusted internal service boundary. The model still gets
reading-only tools with no_write. No new telemetry estimator is introduced.

Wallet windows gate submission, not eventual admission. A queued run can start
after its submission window closes. Settlement markers intentionally have no TTL.
These are explicit operational semantics, not claims of exactly-once inference
under arbitrary distributed crashes.

Pause both reading worker flags before rollout/rollback. Apply
`services/orion-sql-db/manual_migration_reading_durable_turn.sql`, deploy updated
durable-runs with admission enabled, deploy Hub, then re-enable reading. If
rolling back, cancel/drain active reading runs first; do not revive unheld turns
while their durable counterparts are still running. No production migration,
restart, requeue, or paper retry was performed in this patch.

Update field-digester after the migration as well: its existing head-of-line-age
query must follow the same active-binding-first order as the worker. This is
ordering coherence for the existing instrument, not a new signal, estimator or
cognition input.

Local .env was synced from the template. Existing window and curiosity overrides
were reported and preserved. ORION_BUS_URL remains the Tailscale bus address;
the sync script intentionally excludes host-specific bus and grammar values.

## Review Findings Fixed

- Hold loss incorrectly became a failed reading attempt.
  Fix: requeue the shared HoldLost condition; graph regression verifies no call.
- Failed graph events lost the actual fenced turn correlation.
  Fix: use existing harness_turn_meta; failure regression verifies correlation.
- Hub hold-validation failures silently dropped the RPC.
  Fix: explicit pre-work refusal reply with no slot/attempt charge.
- Queued operator cancellation was charged as a failed read.
  Fix: distinct cancellation outcome and no-charge/no-attempt queue transition.
- CI found the separate live schema lookup missing the new RPC registrations.
  Fix: register both maps; direct resolve plus parity regression passes.
- CI found head-of-line monitoring still using the old claim order.
  Fix: align the existing query with the worker and retain the equality gate.
- The definition-drift gate requires fingerprints for bus RPC contracts too.
  Fix: regenerate the lock; exactly the two reading request/reply channels are
  added, with no estimator or cognition-metric definition changes.

## Verification

- 217 reading/worker/real-PostgreSQL tests passed.
- Full reading CI selection: 658 passed, 1 unrelated opt-in skip.
- Durable-run suite: 89 passed, 54 opt-in integration tests skipped.
- Offline reading handoff eval: 11 passed.
- Hub, durable-runs and field-digester Docker images built under isolated check
  project names. Hub and durable reading imports also passed with networking off.
- Reply-channel catalog check: 13 prefixes resolved, zero uncovered.
- Schema lookup/parity regression: 11 passed. Queue contention SQL tests: 18 passed.
- Live admission-to-reading-to-journal path: UNVERIFIED until deployment and smoke.
- PR: #2370, conflict-free at submission. Latest CI status is on the PR.
- Independent final review: all four findings resolved; no material blocker.

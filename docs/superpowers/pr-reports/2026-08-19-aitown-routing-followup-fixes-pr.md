## Summary

Follow-up to the AI-Town table-split cutover (PR #1743). Code review on an unrelated branch (the `orion_substrate` concept-node purge, PR #1748) examined the live state of already-merged `services/orion-sql-writer/app/worker.py` and found 4 real, unfixed routing-awareness/locking gaps left over from #1743's cutover. This branch closes all 4, plus 3 smaller findings a second review round surfaced on this branch's own diff -- and, critically, **catches and reverts a real regression this branch itself introduced** in its first attempt at one of those fixes.

- **`_apply_spark_meta_patch`** now acquires `_lock_chat_history_row` before its cross-table lookup, matching the "callers MUST hold this lock" contract `_resolve_chat_history_model_cls`'s own docstring already stated. Without it, a patch racing a concurrent turn-path write for the same row could find the row missing from *both* tables and silently drop the patch (`spark_meta_patch_missing_row`).
- **`SparkTelemetrySQL`'s "Bi-Directional Metadata Sync" back-populate branch** -- extracted into a standalone, directly-testable function `_back_populate_chat_spark_meta_from_telemetry` -- now checks `aitown_chat_history_log` when the primary lookup misses. Previously it only ever queried `ChatHistoryLogSQL`, so telemetry for an AI-Town-routed correlation_id silently back-populated nothing.
- **`_chat_history_thought_for_merge`** gets the same mirror-table fallback, plus its own lock acquisition (its caller doesn't acquire one until ~160 lines later in the same function).
- **`_fetch_chat_turn_for_memory_emit`** -- a 4th primary-table-only blind spot, found by the *second* review round, not the first -- now falls back to the mirror table too. Before this, a turn already routed to `aitown_chat_history_log` silently dropped out of memory consolidation (`services/orion-memory-consolidation`'s real downstream consumer of `orion:memory:turn:persisted`) with zero trace.
- **A real regression this branch introduced and then caught itself**: the first attempt at the above fixes gated all three new lock calls behind `sql_writer_aitown_routing_enabled`, reasoning "no cross-table race to protect when routing is off." A second review round proved that reasoning wrong -- the pre-existing writer-side locks (`_write_row`, `_ensure_chat_history_from_message`) stay unconditional regardless of the flag, so gating only the reader side reopened the exact race this whole patch exists to close. Reverted; a third, narrowly-scoped review round mutation-tested the revert and confirmed it's both complete and correctly tested.

## Outcome moved

Four previously-silent data-loss paths in the AI-Town-routed half of `chat_history_log` traffic are closed: spark_meta patches, telemetry back-population, thought_process preservation, and memory-consolidation turn emission all now see AI-Town-routed rows. The routing-decision serialization contract (`_lock_chat_history_row`) is now held consistently by every reader that depends on it, not just the two writer call sites #1743 originally added it to.

## Current architecture

`services/orion-sql-writer/app/worker.py` routes each `chat_history_log` row to exactly one of two physical tables (`ChatHistoryLogSQL` / `AitownChatHistoryLogSQL`) based on `client_meta`, via `_resolve_chat_history_model_cls` (PR #1743). Multiple other functions in the same file read or patch rows by `correlation_id`/`id` after the fact (patch application, telemetry back-population, thought-merge, memory-turn emission) -- each of those needed its own mirror-table awareness, since routing only decides where a row is *written*, not where subsequent code should *look* for it.

## Architecture touched

`services/orion-sql-writer` only. No schema, contract, or config changes.

## Files changed

- `services/orion-sql-writer/app/worker.py`:
  - `_apply_spark_meta_patch`: added unconditional `_lock_chat_history_row` call.
  - `_chat_history_thought_for_merge`: added unconditional `_lock_chat_history_row` call and mirror-table fallback.
  - New `_back_populate_chat_spark_meta_from_telemetry`: extracted from an inline block in `_write_row`'s `SparkTelemetrySQL` branch; added mirror-table fallback, `.with_for_update()` on both lookups (matching `_apply_spark_meta_patch`'s existing lost-update protection for the same read-modify-write-of-`spark_meta` shape), and a debug-level log on a full miss.
  - `_fetch_chat_turn_for_memory_emit`: added mirror-table fallback (read-only, no lock -- called only after the row's own write has already committed).
- `services/orion-sql-writer/tests/test_aitown_chat_history_dual_write.py`, `test_spark_meta_patch.py`, `test_thought_candidate.py`: fake-session `execute()` signatures updated for the `params` kwarg (the advisory-lock call shape); new lock-acquisition tests.
- `services/orion-sql-writer/tests/test_spark_telemetry_chat_meta_back_populate.py` (new): 6 tests for the extracted function.
- `services/orion-sql-writer/tests/test_fetch_chat_turn_for_memory_emit.py` (new): 5 tests for the memory-emit mirror fallback.

## Schema / bus / API changes

None.

## Env/config changes

None -- `SQL_WRITER_AITOWN_ROUTING_ENABLED` already exists (#1743), unchanged here.

## Tests run

```text
.venv/bin/python3 -m pytest services/orion-sql-writer/tests/ -q
  -> 353 passed, 11 failed, 3 skipped, 34 warnings

The 11 failures are pre-existing and unrelated to this branch:
- Confirmed identical (same 11 test names) running the same suite against
  origin/main in the primary checkout, before any commit on this branch.
- None are in files this branch touches (they're in
  test_biometrics_summary_sql_shape.py, test_chat_history_response_identity_merge.py,
  test_journal_entry_payload_boundary.py, test_notify_attention_ack.py,
  test_notify_attention_escalate.py).
- The final review round found all 11 pass when run in isolation
  (`pytest tests/test_notify_attention_ack.py tests/test_notify_attention_escalate.py -q`
  -> 5 passed) -- a pre-existing test-order/pollution issue in the full-suite
  run on main, not a regression from this diff. Not fixed here (unrelated
  service concern, different owning area).
- 4 of the 11 (test_chat_history_response_identity_merge.py) are a different
  known gap: PR #1742 added _lock_chat_history_row-incompatible fake sessions
  before #1743 added that call -- also pre-existing, also out of scope.
```

## Evals run

No eval harness exists for `orion-sql-writer` beyond its `tests/` directory. The underlying `pg_advisory_xact_lock` mechanism this branch extends to 3 more call sites was already live-proved against real concurrent Postgres sessions in PR #1743 (two genuinely concurrent threads, message-path measurably blocked ~0.93s waiting for turn-path's commit). This branch applies that same, already-proven mechanism to additional call sites rather than introducing a new one -- a fresh real-Postgres concurrency eval for each individual call site was judged not proportionate to the risk for a locking primitive already proven correct at the mechanism level, and is noted below as a deferred gap rather than silently skipped.

## Docker/build/smoke checks

No runtime/config/Docker-boot-path changes -- pure application logic. Not run.

## Review findings fixed

Three review rounds ran on this branch. Findings from rounds 1 and 2, and their resolution:

- Finding (round 1, CONFIRMED): `_apply_spark_meta_patch` never acquired `_lock_chat_history_row` despite `_resolve_chat_history_model_cls`'s own docstring requiring it.
  - Fix: added.
  - Evidence: `TestApplySparkMetaPatchRouting` passes; mutation-tested in round 3 (see below).
- Finding (round 1, CONFIRMED): `SparkTelemetrySQL`'s back-populate branch and `_chat_history_thought_for_merge` both only ever queried the primary table.
  - Fix: mirror-table fallback added to both (extracted to `_back_populate_chat_spark_meta_from_telemetry` for the first).
  - Evidence: `test_spark_telemetry_chat_meta_back_populate.py`, `test_thought_candidate.py`.
- Finding (round 2, CONFIRMED): `_fetch_chat_turn_for_memory_emit` -- a 4th primary-table-only blind spot the first round missed entirely -- silently dropped AI-Town-routed turns from memory consolidation.
  - Fix: mirror-table fallback added.
  - Evidence: `test_fetch_chat_turn_for_memory_emit.py` (new, 5 tests).
- Finding (round 2, CONFIRMED): `_back_populate_chat_spark_meta_from_telemetry`'s miss case gave zero log signal, unlike `_apply_spark_meta_patch`'s analogous ERROR-level log.
  - Fix: added a debug-level log (`spark_telemetry_chat_backfill_missing_row`) -- debug, not error, since a telemetry tick with no matching chat row is an expected/common case, not an operator-actionable failure the way a named patch's missing row is.
- Finding (round 2, CONFIRMED): `_back_populate_chat_spark_meta_from_telemetry`'s two lookups lacked `.with_for_update()` despite doing the identical read-modify-write-of-`spark_meta` shape `_apply_spark_meta_patch` was already fixed for.
  - Fix: added to both lookups.
- **Finding (round 2, CONFIRMED, self-introduced regression)**: gating the three new `_lock_chat_history_row` calls behind `sql_writer_aitown_routing_enabled` -- an "optimization" this branch's own first commit added -- reopened the exact cross-session race the lock exists to close, because the pre-existing writer-side lock sites stay unconditional regardless of the flag.
  - Fix: reverted; all three locks are unconditional again, matching the writer-side sites.
  - Evidence: round 3's mutation test -- reintroduced the gating in a scratch copy, confirmed all 3 rewritten `..._acquires_lock_even_when_routing_disabled` tests fail against it, confirming they're not vacuous.
- Findings NOT fixed, documented as accepted/deferred (confirmed real by round 2, explicitly not re-litigated in round 3 per its own scope):
  - Lock-key inconsistency: some call sites lock on `correlation_id`, others on `id`, relying on the existing (already-documented-elsewhere, not this branch's invention) "every current producer sets `id == correlation_id`" convention rather than an enforced invariant. A real fix would need a canonical row-identity resolver or a DB-level constraint; judged out of scope for a thin bug-fix patch.
  - The primary-then-mirror-fallback pattern is now hand-duplicated across 5 call sites with no shared helper. A real refactor opportunity; deferred to avoid scope creep/behavior risk in a review-driven bug-fix patch -- round 2 itself is direct evidence a 5th ad hoc instance (`_fetch_chat_turn_for_memory_emit`) was missed by round 1 for exactly this reason.
  - `_chat_history_thought_for_merge` now acquires the same advisory lock twice within one `_write_row` call (once here, once again at `_write_row`'s own pre-existing call site further down) -- confirmed harmless (`pg_advisory_xact_lock` is reentrant per session, both target the same row id), left as a documented, accepted minor inefficiency rather than restructuring call order for a marginal perf gain.
  - No real-Postgres concurrency eval directly exercises these specific new lock sites (as opposed to the mechanism in general, already proven in #1743). Noted as a real gap; not built here given the mechanism itself is unchanged and already proven.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-sql-writer/.env \
  -f services/orion-sql-writer/docker-compose.yml \
  up -d --build
```

## Risks / concerns

- Severity: low
- Concern: `_back_populate_chat_spark_meta_from_telemetry` now costs up to 3 DB round trips (1 advisory lock + 2 `with_for_update()` SELECTs) on its common miss case (a telemetry tick with no matching chat row at all), versus 1 unlocked SELECT before this patch, on a genuinely hot path.
- Mitigation: accepted as the cost of correctness -- see the round-2 finding above and the in-code comment explaining why gating this away is unsafe. AI Town's own backend is confirmed dead (no live AI-Town traffic currently), so the mirror-table half of this cost is not presently being paid in practice; if it becomes a measured problem, the right fix is a cheaper existence check, not re-gating the lock.
- Severity: low
- Concern: the correlation_id-vs-id lock-key inconsistency (documented above) could theoretically let a future producer that violates the `id == correlation_id` convention defeat this branch's own locking.
- Mitigation: documented in-code at every lock call site; no current producer violates the convention.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1750

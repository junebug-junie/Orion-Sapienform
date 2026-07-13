# PR report — chat history compactor (feat/chat-history-compactor)

## Summary

- Ships the `chat_history_compactor_pass` cognition workflow: compacts bounded Hub `chat_history_log` windows into indexed (`compactor_index`-upserted) `high_recall` memory cards plus an idempotent journal append, exposed in the Hub Skill Runner and bootstrapped as a daily 06:00 `America/Denver` schedule.
- Recovers the branch Cursor stranded on 2026-07-10 (all 9 plan tasks were complete; push/review/PR were not), rebased onto main.
- Review round 1 fixed: conversational-alias hijack, quiet-day stub writes, bootstrap resurrecting cancelled/duplicating edited schedules, uncapped lookback, silent window-mode coercion, day-end 999µs gap, missing unique index, fragile lifespan import.
- Review round 2 (this round) fixed every remaining concern: deduped the chat/github digest runners onto shared helpers, card-persist failures now degrade instead of discarding the digest, the indexed upsert dropped from 5 to 3 DB round trips, the dead `find_active_card_by_compactor_index` export was removed, and the pre-existing `WorkflowScheduleStore.upsert_from_dispatch` request_id bug was fixed.
- Adds a deterministic eval lane (`services/orion-cortex-orch/evals/`) for digest input/output budgets and quiet-window honesty.
- Repairs three pre-existing broken tests on main (2 concept-induction host-state/module-collision failures, 1 world-pulse lifespan mock drift) so the touched suites are fully green.

## Outcome moved

Orion gains durable chat memory: bounded chat windows become one live, indexed, high-recall card per window key plus a journal trail — schedulable, quiet-safe (no empty-shell cognition), and resilient to transient DB failures. `services/orion-actions` schedule dispatch no longer duplicates schedules on re-dispatch.

## Current architecture

Before this branch there was no chat compaction path: `chat_history_log` grew unbounded, nothing digested it, and the only compactor (github) hardcoded its own runner plumbing. `WorkflowScheduleStore.upsert_from_dispatch` looked up `self._schedules.get(request.request_id)` in a dict keyed by `schedule_id`, so re-dispatch always created duplicates.

## Architecture touched

- `orion/cognition/chat_history_compactor/` — window resolution (day/rolling, capped, fail-loud), input trim, digest parse/budget, quiet builder.
- `orion/cognition/compactor/` — shared budget/parse/index helpers now used by **both** compactors (github digest module migrated onto them).
- `services/orion-cortex-orch` — workflow lane: `_execute_chat_history_compactor_pass`, shared `_build_compactor_digest_request` / `_compactor_digest_from_payload`, brain-lane verb `chat_history_compactor_digest_v1` with chat→quick route retry, degrade-on-persist-failure.
- `services/orion-actions` — daily schedule bootstrap (operator-respecting) + schedule store request_id upsert fix.
- `services/orion-hub` — Skill Runner workflow exposure.
- `orion/core/storage` — `upsert_indexed_compactor_card` (3 round trips, one transaction) + partial unique index `idx_mc_active_compactor_index` (idempotent DDL, auto-applies via `apply_memory_cards_schema`).

## Files changed (round 2 highlights)

- `services/orion-cortex-orch/app/workflow_runtime.py`: shared digest request/extract helpers; both runners rebuilt on them (~70 duplicated lines removed); card-persist failures degrade with `card_persist_skipped_reason` instead of discarding the digest.
- `orion/cognition/compactor/digest.py` (+README): shared `parse_compactor_digest_json`; chat and github digest modules delegate parse and budget checks.
- `orion/core/storage/memory_cards.py`: upsert lookup carries the before-snapshot, `UPDATE ... RETURNING to_jsonb(...)` carries the after — 5 → 3 round trips; removed dead `find_active_card_by_compactor_index` export (deviation from spec/plan docs, which listed it — nothing consumed it).
- `services/orion-actions/app/workflow_schedule_store.py`: `upsert_from_dispatch` matches records by `request_id` field (pre-existing bug: dict is keyed by `schedule_id`, so the lookup never hit).
- `services/orion-cortex-orch/evals/test_chat_history_compactor_digest_eval.py`: deterministic budget/honesty evals.
- `services/orion-cortex-orch/tests/test_workflow_lane.py`: new final_text-parse-path and persist-degrade tests; concept-induction tests made hermetic (sql-writer `app` package swap loader; placeholder-store path pinned to tmp — the real `/tmp/concept-induction-state.json` leaked from a live run flipped the old test).
- `services/orion-actions/tests/`: schedule-store request_id regression test; world-pulse lifespan test mocks updated to main.py's current async bus surface.
- `services/orion-actions/README.md`, `services/orion-cortex-orch/README.md`: full behavior contracts for the workflow (bootstrap semantics, window bounds, quiet gating, persist degrade, budgets, evals).

## Schema / bus / API changes

- Added: brain-lane verb `chat_history_compactor_digest_v1`; workflow id `chat_history_compactor_pass`; partial unique index `idx_mc_active_compactor_index` on active `compactor_index` cards.
- Removed: workflow alias “what have we been talking about” (substring matching hijacked ordinary chat into durable writes — **deliberate deviation from the design/plan docs**, which list the phrase); `find_active_card_by_compactor_index` export (dead).
- Behavior changed: `upsert_from_dispatch` with a repeated `request_id` now updates the existing schedule (revision bump) instead of duplicating — this is the documented upsert contract, previously broken.
- Compatibility notes: DDL additions are idempotent (`CREATE ... IF NOT EXISTS`); no bus channel or schema-registry changes.

## Env/config changes

- None. No `.env_example`, `orion/bus/channels.yaml`, or `orion/schemas/registry.py` changes; no env sync required.

## Tests run

```text
.venv/bin/python -m pytest services/orion-cortex-orch/tests/test_workflow_lane.py -q      → 61 passed (0 failures; 2 were failing on main)
.venv/bin/python -m pytest services/orion-actions/tests -q                                → 91 passed (1 was failing on main)
.venv/bin/python -m pytest orion/cognition/chat_history_compactor/tests \
    orion/cognition/compactor/tests tests/test_indexed_compactor_memory_cards.py \
    services/orion-hub/tests/test_workflow_request_builder.py -q                          → passed (101 total with orch lane+evals)
git diff --check                                                                          → clean
```

Note: orion-actions tests must run in a separate pytest invocation from cortex-orch tests (cross-service `app` package name collision at collection; pre-existing repo-wide constraint).

## Evals run

```text
.venv/bin/python -m pytest services/orion-cortex-orch/evals -q → 5 passed
```

Deterministic lane: adversarial 500×10k-char windows trim to ≤30 turns/≤45k serialized chars with newest-suffix retention; quiet digests stay in budget and claim nothing; digest JSON round-trips and rejects non-objects; over-budget digests fail loud. **Gap**: digest *quality* (summary faithfulness to transcript) still needs an LLM-in-the-loop eval — follow-up issue material.

## Review findings fixed

Round 1 (8 fixed): alias hijack; quiet-day journal/card stubs; bootstrap resurrect/duplicate; lookback uncapped; silent window-mode coercion; day-end 999µs gap; missing unique index; bootstrap import outside lifespan try.

Round 2 (all remaining concerns fixed):

- Finding: ~90-line digest-runner duplication between chat and github compactors.
  - Fix: shared `_build_compactor_digest_request` + `_compactor_digest_from_payload` in workflow_runtime; shared parse/budget in `orion/cognition/compactor/`.
  - Evidence: 25 compactor tests pass across both workflows; github error tokens unchanged.
- Finding: only `recall_pg_dsn_unavailable` tolerated on card persist; transient DB errors discarded the digest.
  - Fix: any persist exception degrades — journal still written, `card_persist_skipped_reason` in metadata, warning logged with traceback.
  - Evidence: `test_chat_history_compactor_pass_card_persist_error_degrades`.
- Finding: production final_text digest-parse path untested.
  - Fix/Evidence: `test_chat_history_compactor_pass_digest_from_final_text_only`.
- Finding: upsert did 5 sequential DB round trips; dead `find_active_card_by_compactor_index` export.
  - Fix: 3 round trips in one transaction; export removed.
  - Evidence: `tests/test_indexed_compactor_memory_cards.py` (2 passed).
- Finding (pre-existing on main): `upsert_from_dispatch` request_id lookup never matched.
  - Fix: match on record field; regression test `test_redispatch_same_request_id_updates_in_place`.
- Finding (pre-existing on main): 2 concept-induction + 1 world-pulse test failures (host-state dependence, `app` module collision, stale mocks).
  - Fix: hermetic loader/paths/mocks in the test files.
  - Evidence: full suites green (61 + 91 passed).

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-cortex-exec/.env -f services/orion-cortex-exec/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-cortex-orch/.env -f services/orion-cortex-orch/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-actions/.env -f services/orion-actions/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low. Concern: broadened persist tolerance could mask a persistent DB misconfiguration. Mitigation: reason is surfaced in workflow metadata and warning logs; journal trail still lands; a stuck reason string is grep-able (`chat_history_compactor_card_persist_skipped`).
- Severity: low. Concern: digest quality has no LLM-in-the-loop eval yet. Mitigation: deterministic budget/honesty evals in place; follow-up issue proposed.
- Severity: low. Concern: `test_handle_envelope_world_pulse_journal` lifespan test reads/writes the real default `/tmp/orion-actions/workflow_schedules.json`. Mitigation: noted for a follow-up test-hygiene pass; not compactor scope.
- UNVERIFIED: live Docker smoke (schedule fire → card row → journal row) was not run in this environment; deterministic gates and evals cover the seams.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/chat-history-compactor (filled in at PR creation)

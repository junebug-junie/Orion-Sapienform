## Summary

- New act signal `output_used`: "another part of me used what this act made". It is the act-level form of the integration metric.
- A resolver grades it from tables that already exist:
  - **1:** at the first use by another organ, read from `experience_use` (through the act's `correlation_id`) or `act_downstream_use.used`.
  - **0:** at the horizon, when the output was observable but nobody used it.
  - **Void, never graded:** when no use could ever have been seen.
- A producer's "I made it" close no longer closes an `output_used` act. It stays open for the grader. A producer's "nothing made" close grades a real 0 (`no_output`).
- Void closes are no grade in the shared `act_outcome`. This also fixes a pre-existing gap: censored promise voids read as "refuted" in the body economy's reliability weights.
- No producer opens `output_used` yet. That is Patch 1b. This patch is safe to deploy first: until a family switches, the resolver's cheap gate query finds nothing, and the new joins never run.

## Outcome moved

None live yet, by design. This is the grader Patch 1b switches families onto. The first measurable change comes with Patch 1b: act surprise per family stops sitting at 0.0000-0.004 nats.

## Current architecture

85% of acts bet `artifact_produced`, which is true 80-100% of the time, so act surprise is near zero. Downstream use is measured in `act_downstream_use` and `experience_use`, but no act bets on it. Spec: `docs/superpowers/specs/2026-11-25-act-consequence-bets-design.md` (PR #3953, merged). Juniper took all three defaults.

## Architecture touched

orion-sql-writer's act resolver (`orion/grammar/act_ledger.py`), the grammar schema, the self-monitor act grade (also read by `orion/physiology/weights.py`), and one index migration.

## Files changed

- `orion/schemas/grammar.py`: `ActSignal` += `output_used`.
- `orion/grammar/output_used.py`: the grade rule (pure), the producer-close rule, and the due query.
- `orion/grammar/act_ledger.py`: `close_act` defers a made close and grades a nothing-made close; `resolve_due_acts` runs the `output_used` pass; the main due list excludes `output_used`; `ResolveSummary` gets three new counts.
- `orion/self_monitor/adapters.py`: `act_outcome` returns None for a void close.
- `services/orion-sql-db/manual_migration_temporal_self_event_correlation_idx_v1.sql`: partial index on `temporal_self_event.correlation_id`. Additive, auto-applied post-pull. REQUIRED-BY orion-sql-writer.
- `orion/grammar/tests/test_output_used.py`, `services/orion-sql-writer/tests/test_output_used_acts_integration.py`: new tests.
- `.github/workflows/orion-sql-writer-tests.yml`: the new test runs in the must-not-skip act ledger step.
- `services/orion-sql-writer/tests/conftest.py`: the new integration test skips in the no-database unit job.
- `services/orion-sql-writer/Dockerfile`: copies `config/integration/organs.yaml`.
- `services/orion-sql-writer/app/act_downstream_use.py`: an `output_used` act's use window starts at its open.
- `orion/physiology/weights.py`: `output_used` acts are not reliability evidence (source (d)).
- `services/orion-sql-writer/tests/test_act_downstream_use_pg.py`: fixture gets `signal_id`; window test.
- `docs/superpowers/specs/2026-11-25-act-consequence-bets-design.md`: records Juniper's defaults.

## Schema / bus / API changes

- Added: `ActSignal` value `output_used`. Consumer-first: the readers (sql-writer, substrate-runtime grammar consumer) get it in this patch. No producer sends it yet.
- Removed / renamed: none.
- Behavior changed: `act_outcome` returns None for a `void:` note. Before, it returned "refuted" for an expired void.
- Compatibility notes: old rows and old signals parse unchanged.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Tests run

```text
tests/test_act_ledger_contract.py, orion/grammar, orion/physiology, orion/self_monitor: 562 passed
orion-sql-writer unit tests (not integration): 1387 passed, 188 skipped
Postgres integration (throwaway postgres:16-alpine), the whole CI act ledger step incl. new files: 112 passed, 0 skipped
promise_acts_mutations.py, camera_ask_acts_mutations.py: every mutant caught/killed
check_integration_seam: OK; check_definition_drift: 0 changed; env template parity: PASS; git diff --check: clean
substrate-runtime test_grammar_truth_reducer_health.py: 2 failed, SAME on clean main (Settings field missing in local env; pre-existing)
substrate-runtime test_grammar_consumer_integration.py: needs CI's postgres on :5432; runs in CI
```

## Evals run

```text
No new eval: Patch 1a grades nothing live. services/orion-sql-writer/evals/test_act_ledger_calibration_eval.py is unchanged.
The spec's live acceptance checks 1-3 (surprise spread, no age drift, AUC/Brier gate) apply after Patch 1b.
```

## Docker/build/smoke checks

```text
Not run: this changes library code the sql-writer image copies in. The live check after deploy is the
resolver log's summary carrying closed_output_used / closed_output_unused / closed_output_unobservable = 0,
and the index existing:
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select indexname from pg_indexes where indexname='idx_temporal_self_event_correlation'"
```

## Review findings fixed

Code review ran in a subagent (orion-repo-agent) on `git diff origin/main...HEAD`. All 8 findings are fixed.

- Finding (important): one turn's correlation_id also carries other organs' rows (attention_row, gpu_wait, chat_turn). A cite of those credited every act on the turn, and their presence made an act "observable".
  - Fix: an act now matches only experiences whose kind belongs to its own organ. The organ registry (`config/integration/organs.yaml`) gives both maps: `families` -> actors, `temporal_sources` -> kinds. A cite counts only if a different organ made it. `action_outcome` never counts. The sql-writer image now copies `organs.yaml`. If the registry can't be read, nothing is graded (`output_used_registry_unread`).
  - Evidence: `test_turn_rows_of_other_organs_neither_credit_nor_make_observable`, `test_a_cite_by_the_acts_own_organ_is_not_a_use`, `test_act_output_kinds_maps_actor_to_own_organ_kinds_never_action_outcome`, `test_live_registry_loads_and_names_every_switchable_family`.
- Finding (important): prefixing "no_output:" hid execution failures ("failed: ..."). They read as refuted instead of failed_to_execute.
  - Fix: the producer's note is kept as is. `no_output` is used only when the producer gave no note.
  - Evidence: `test_producer_nothing_made_note_keeps_execution_failure_readable`, `test_failed_producer_close_keeps_its_failure_note`.
- Finding (important): a deferred close dropped the producer's cost for good.
  - Fix: the deferral fills the close-time cost (wall time to the producer's close, GPU, energy) into `cost_json`. The resolver's close keeps that cost instead of re-filling GPU out to the horizon.
  - Evidence: `test_deferred_close_keeps_cost_and_correlation_id`.
- Finding: a close whose open never arrives stayed open forever.
  - Fix: the due time falls back to `created_at + horizon_sec`.
  - Evidence: `test_close_before_an_open_that_never_arrives_still_closes`.
- Finding: hits close by the resolver, which reliability source (d) skips, while misses counted. So (d) would see only failures. Also `settled_by` said "juniper".
  - Fix: `output_used` acts are left out of (d), because downstream use already reaches the weights as source (a). `settled_by` is "world" for them.
  - Evidence: `test_output_used_acts_are_not_reliability_evidence`, `test_output_used_resolver_close_is_settled_by_the_world_not_juniper`.
- Finding: `_persisted_close` recovery skipped `output_used`.
  - Fix: a persisted nothing-made close is applied before any expiry.
  - Evidence: `test_persisted_nothing_made_close_is_recovered_not_expired`.
- Finding: a deferred close dropped a correlation_id the open lacked.
  - Fix: the deferral writes `COALESCE(correlation_id, close's)`.
  - Evidence: same cost test.
- Finding: `act_downstream_use` dated an `output_used` act's output from its graded close, up to a day late.
  - Fix: for `output_used` it dates from the open, the same window the grade reads.
  - Evidence: `test_output_used_act_window_starts_at_open_not_at_its_graded_close`. It runs in CI in its own step (`act_downstream_use_test` database).
- Also: the promise mutation check targets the exact cost line, so that line keeps its shape. Every mutant is still caught.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

The post-pull rebuild applies the index migration first, then rebuilds the sql-writer.

## Risks / concerns

- Severity: low. Concern: act_outcome's void change shifts reliability weights for families with promise voids. Those voids were miscounted as failures. Mitigation: it is the correct grade; the reliability input is off in live weights (#3323).
- Severity: low. Concern: an `output_used` act's use window in `act_downstream_use` now starts at its open, so a use decided between the open and the producer's close counts. Mitigation: that gap is the act's run time (seconds to minutes), against a 72 h window.
- Issues opened: none. Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3994

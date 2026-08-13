# doc_semantic_drift: stop dropping every event

## Summary

- `orion:substrate:doc_semantic_drift` had `consumer_services: []`, and `OrionBusAsync.publish()` is Redis **pub/sub**, not a stream — so with no subscriber, every event was discarded the instant it was published.
- The only record was the producer container's stdout, which every redeploy wiped. Roughly two days of real doc-drift scores exist nowhere, including the first batch through the new scoring path.
- Adds `DocSemanticDriftSQL` + the routing to reach it. Smallest possible slice: no Hub panel, no threshold, no consumer behavior. Just stop throwing it away.
- Records `base_sha` so the scored *range* is visible, not only its endpoint.

## Outcome moved

"Come back in two weeks and look at the data" was not a plan — there would have been no data. Rows now accumulate durably, so the open threshold question becomes answerable whenever anyone gets to it, with no one needing to remember anything in the meantime.

## Current architecture

The producer polls `*.md` changes every 300s, scores each via max chunk-pair drift over chunked embeddings, and publishes `DocSemanticDriftV1`. Nothing consumed it. `orion-sql-writer` already had a generic envelope→model write path (`MODEL_MAP` + `route_map` + subscribe list) used by ~70 other channels; `dev_economics` was wired through it on 2026-08-12.

## Architecture touched

`orion-sql-writer` (new model + routing), `orion/schemas/doc_semantic_drift.py` (`event_id`, `base_sha`), `orion/structural_mass/doc_semantic_drift.py` (`base_sha` on `DocHunkChange`), `orion/bus/channels.yaml`.

## Files changed

- `services/orion-sql-writer/app/models/doc_semantic_drift.py`: new `DocSemanticDriftSQL`, table `doc_semantic_drift_log`.
- `services/orion-sql-writer/app/models/__init__.py`, `app/worker.py` (model + schema import, `MODEL_MAP`), `app/settings.py` (`DEFAULT_ROUTE_MAP`, subscribe list), `.env_example`.
- `orion/schemas/doc_semantic_drift.py`: `+event_id` (derived), `+base_sha`.
- `orion/structural_mass/doc_semantic_drift.py`: `+base_sha` on `DocHunkChange`.
- `orion/bus/channels.yaml`: `consumer_services: ["orion-sql-writer"]`, plus two now-false comments corrected.
- `services/orion-sql-writer/tests/test_doc_semantic_drift_sql_shape.py`: 16 tests.

## Why `event_id` is derived, not a uuid

`dev_economics` uses `uuid4`. Here `(sha, path)` fully determines the event, and the producer re-scores a whole range after a failed publish — with a random id every retry writes a duplicate row. Since the entire purpose of this table is the *distribution* of real scores, duplicates would corrupt exactly what is being collected.

**Honest bound on that claim** (review finding): it holds when HEAD has not moved. If HEAD moved between a partial publish failure and the retry, the same edit is re-scored over a wider range under a different key — two real rows for one edit. `base_sha` is what makes that overlap detectable; it is not prevented.

## Why `base_sha` matters

The score covers `git diff last_sha head_sha` — a range. A tick that spans several commits (missed poll, retry) collapses them into one hunk. Without `base_sha`, a six-commit row is indistinguishable from a one-commit row, and `chunk_count_*` does not substitute: it measures text length, not commit span. Any threshold derived from this table must be able to condition on it.

## Schema / bus / API changes

- Added: `DocSemanticDriftV1.event_id` (derived from `(sha, path)` by an after-validator), `DocSemanticDriftV1.base_sha` (nullable).
- Channel: `consumer_services: [] -> ["orion-sql-writer"]`.
- Compatibility: additive; `base_sha` is nullable so pre-2026-08-13 events validate unchanged. `orion/schemas/registry.py` already registered `DocSemanticDriftV1`, so no registry change.

## Env/config changes

- Added keys: none. `SQL_WRITER_SUBSCRIBE_CHANNELS` and `SQL_WRITER_ROUTE_MAP_JSON` gained entries.
- `.env_example` updated: yes.
- local `.env` synced: yes, **by hand**. `effective_subscribe_channels` REPLACES from env rather than merging (unlike `route_map`), so `.env_example` alone would have left the writer permanently unsubscribed with no error — confirmed live with `dev_economics` on 2026-08-12.

## Tests run

```text
pytest services/orion-sql-writer/tests/ -q --ignore=tests/test_dream_model_constraints.py
  18 failed, 211 passed, 3 errors
  baseline on clean main: 18 failed, 195 passed, 3 errors
  -> +16 passed, identical failure set, zero new breakage

pytest services/orion-cocreation-signals/tests/ orion/structural_mass/tests/ -q
  143 passed
```

The 18 failures / 3 errors and the `test_dream_model_constraints.py` collection error are **pre-existing on clean `main`** — verified before and after. That suite is not healthy; fixing it was out of scope.

Note: `orion-sql-writer` and `orion-cocreation-signals` both have an `app` package and collide if run in one pytest invocation.

## Evals run

None. This is a persistence slice with no scoring behavior; the eval that matters (a threshold) is what the accumulated rows are *for*.

## Docker/build/smoke checks

```text
safe_docker_build.sh orion-sql-writer build / up -d          -> Started
safe_docker_build.sh orion-cocreation-signals build / up -d  -> Started

redis-cli PUBSUB CHANNELS orion:substrate:doc_semantic_drift
  -> orion:substrate:doc_semantic_drift      (a live subscriber; none before)

doc_semantic_drift_log columns present, base_sha nullable, 0 rows
cocreation_doc_semantic_drift_resumed_from_durable_state last_sha=8299de4f0...
sql-writer log: no errors, no tracebacks
```

**No real row exists yet** — `main` has not moved since deploy, so the producer has had nothing to score. Wiring is verified end-to-end (live subscriber, live table, tested routing); the first actual row lands when this merges with a `.md` alongside it.

## Review findings fixed

- Finding: no record of the diff range start. A multi-commit row was indistinguishable from a single-commit one, and the idempotency claim was broader than the key supports.
  - Fix: `base_sha` through `DocHunkChange` → schema → SQL column (indexed, nullable); docstring corrected to the narrower claim.
  - Evidence: `test_base_sha_is_persisted`, `test_overlapping_rescore_is_detectable_rather_than_silent`.
- Finding: the subscribe-list test asserted on `Settings()`, which reads process env plus a cwd-relative `.env`. The live `.env` already contains the channel, so it passed **even with the code default deleted** — precisely the regression it was named for.
  - Fix: assert on `Settings.model_fields[...].default`, matching `test_drive_audit_sql_shape.py:85`.
  - Evidence: mutation-tested — deleting the default now fails the test; it did not before.
- Finding: missing the `.env_example` assertion both sibling shape tests carry.
  - Fix: `test_env_example_lists_the_channel`.
- Finding: two now-false statements in `orion/bus/channels.yaml` — the block header still said "Pure shadow write, no consumer yet", and the `dev_economics` entry still listed `doc_semantic_drift` as `consumer_services: []`. That file is the operator-facing bus contract; someone auditing which channels still drop events would have read it and concluded wrongly.
  - Fix: both corrected. `juniper_affective_state` is now named as the last channel still dropping events.
- Finding: `channels.yaml` cited this PR report before it existed.
  - Fix: it exists.
- Verified clean by review, by execution rather than reading: wiring completeness against every surface `dev_economics` touches; the generic `_write_row` path maps every field with no special case needed; the `event_id` validator cannot recurse (`validate_assignment` is unset) and preserves an explicit wire value; and the writer uses `sess.merge()` (not a plain INSERT), so a redelivered event upserts rather than raising a duplicate-key error.

## Restart required

Already deployed. One manual DDL was applied, since `create_all()` creates missing tables but **not** missing columns and the table was created by the previous commit:

```sql
ALTER TABLE doc_semantic_drift_log ADD COLUMN IF NOT EXISTS base_sha VARCHAR;
CREATE INDEX IF NOT EXISTS ix_doc_semantic_drift_log_base_sha ON doc_semantic_drift_log (base_sha);
```

Additive and reversible, against a table with 0 rows created 15 minutes earlier.

## Risks / concerns

- Severity: medium
- Concern: this repo has no migration system. `create_all()` handles new tables only, so **any future column added to an existing table silently does not appear** — the ORM then inserts against a column the DB lacks. This bit this patch within one commit of creating the table.
- Mitigation: none applied; the DDL above was run by hand. Worth a real answer before the next model change.

---

- Severity: low
- Concern: no retention window. Deliberate — measured real rate is well under 1 row/day (~2 `*.md` changes/day, most skipped as unscoreable), and a window would delete the dataset this table exists to accumulate.

---

- Severity: low
- Concern: still no valid threshold, and nothing consumes these rows yet. The table is a prerequisite, not a result.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/doc-drift-persistence

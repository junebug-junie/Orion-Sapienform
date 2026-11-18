## Summary

Orion's nightly memory step retires an old memory when a newer one changes or corrects it. This PR makes that step careful, measured, and reversible, all on local models.

- **Classify, don't pair.** A local model names how the new memory relates to the old one: update, correction, refine, compatible, or unrelated. It answers in JSON three times at temperature 0.7. The step acts only when all three votes agree on update or correction, each says "same attribute", and each is at least 0.8 confident. Anything else abstains, which leaves both memories visible.
- **When a fact stopped being true.** A retired row now records `invalid_at`, `invalidated_by`, and `invalid_reason`. An update was true until the new memory's time, and a past-tense read can still reach it (`EPISODE_VALID_AT_SQL`). A correction was never true, and nothing serves it again. Undo clears all three.
- **Measured before trusted.** `scripts/memory_supersede_eval.py` scores the classifier on made-up fixtures in the repo and on real pairs kept host-side. Write time applies only while the latest recorded eval for the current check version passed. Without that, it proposes and logs.
- **Juniper's words win.** Suppose a new memory is in Juniper's voice and is backed by a verified quote from her own message. It acts when every vote says update or correction, at any confidence, and exactly one old row qualifies.
- **Audit trail (#3660).** Every run writes one append-only `memory_supersede_run` row: ids, which candidate legs fired, votes, verdicts, and what was applied. It holds no statement text. The pass also keeps a run-history file next to its manifest, and a per-watched-row trace.
- **Tighter candidates.** A cosine alone no longer makes a candidate pair. Every pair also needs a rare shared word or referent.

## Outcome moved

EVAL_PLACEHOLDER

## Current architecture

PR #3617 added write-time supersede: the distill's check node asked a two-question yes/no check per candidate pair. A cosine of 0.80 alone could make a candidate. A "yes" set `status = 'superseded'`, which is binary, with no notion of when the fact stopped being true. Precision was never measured. The pass manifest was overwritten on every run. PR #3669 (merged) already stopped degraded applies and closed the eval gate.

## Architecture touched

- `orion/memory/episode/supersede.py`: the v3 classifier (`classify_pair`, `Vote`, `Classification`), the Juniper clear-target path in `check_pairs`, bi-temporal `apply_link`/`undo_link`, `record_run`, and candidate legs.
- `orion/memory/episode/supersede_eval.py`, `scripts/memory_supersede_eval.py`, `orion/memory/episode/fixtures/supersede_eval_synthetic.jsonl`: the eval.
- `orion/memory/lived_memory.py`: `EPISODE_VALID_AT_SQL`, `fetch_valid_at` (past-tense read).
- `orion/memory/episode/store.py`: persist writes the run audit row in a SAVEPOINT.
- `services/orion-durable-runs/app/episode_distill_graph.py`: votes carry their temperature; persist passes the plan.
- `scripts/memory_supersede_pass.py`: v3 classifier, a vote-aware cache, run history and watched trace, and a version check on apply.
- Migration `services/orion-sql-db/manual_migration_episode_memory_temporal_v1.sql` (additive, `REQUIRED-BY: orion-durable-runs`), with a rollback and a grant file.

## Files changed

- `orion/memory/episode/supersede.py`: the classifier, validity marks, audit writer, and candidate legs.
- `orion/memory/episode/supersede_eval.py`: labelled items and scoring (`passed` = zero false acts, zero errors, enough real items).
- `orion/memory/episode/fixtures/supersede_eval_synthetic.jsonl`: 26 made-up pairs that cover every relation.
- `scripts/memory_supersede_eval.py`: `build`, which makes the real set (read-only DB) into `~/.orion/evals/` (mode 600), and `run`, which scores and with `--record-db` records the verdict.
- `scripts/memory_supersede_pass.py`: v3, history, trace, and the check-version refusal.
- `orion/memory/lived_memory.py`: the as-of reader.
- `orion/memory/episode/store.py`: the audit row.
- `services/orion-durable-runs/app/episode_distill_graph.py`, `settings.py`, `.env_example`, `docker-compose.yml`: temperature per vote; `MEMORY_SUPERSEDE_MAX_CHECKS` now counts model calls (default 36).
- `services/orion-sql-db/manual_migration_episode_memory_temporal_v1{,_rollback}.sql`, `scripts/sql/2026-11-18_grant_orion_readonly_supersede_runs.sql`: schema.
- `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`: "Corrections rev 2".
- Tests: `test_supersede.py`, `test_supersede_eval.py`, `test_supersede_pass_script.py`, `test_supersede_pg.py`.
- `.github/workflows/orion-memory-episode-tests.yml`: the path filter now includes the eval script.

## Schema / bus / API changes

- Added: `episode_memory.invalid_at`, `invalidated_by`, `invalid_reason`. Tables `memory_supersede_eval_run` and `memory_supersede_run`. Plan key `decisions`. Link fields `relation`, `path`, `votes`.
- Removed: the v2 yes/no check (`CHECK_PROMPT`, `BOTH_TRUE_PROMPT`, `run_check`) and `DENSE_ALONE`.
- Renamed: `CHECK_VERSION` `supersede_check.v2` -> `supersede_classify.v3`. The eval gate is per version, so v3 needs its own passing eval.
- Behavior changed: an update keeps the old row's `confirmation_state`; only a correction marks it `corrected`.
- Compatibility notes: a pre-v3 checkpoint's links read as `correction` / path `model`, but #3669 already refuses to apply them. A pass pairs file from v2 is refused by `apply`. No bus changes.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- Changed meaning: `MEMORY_SUPERSEDE_MAX_CHECKS` now caps model calls (1-3 per pair). The default went from 12 to 36 (`le` 90).
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: the script reports value drift but does not change existing values, so the local `services/orion-durable-runs/.env` was set to 36 by hand.
- skipped keys requiring operator action: none.

## Tests run

TESTS_PLACEHOLDER

## Evals run

EVALS_PLACEHOLDER

## Docker/build/smoke checks

```text
Not run: no Dockerfile or dependency change. The migration was applied to a throwaway postgres:16 by the Postgres-backed tests.
scripts/check_migration_required_by.py: OK. orion.sql_migration_drift.auto_apply_refusals(migration, existing_tables={'episode_memory'}) == [] (fits the post-pull auto-apply allowlist).
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

The post-pull rebuild applies the migration first (additive, REQUIRED-BY header). Then, by hand:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "GRANT SELECT ON public.memory_supersede_run, public.memory_supersede_eval_run TO orion_readonly;"
.venv/bin/python scripts/memory_supersede_eval.py build && .venv/bin/python scripts/memory_supersede_eval.py run --record-db
```

Write time stays proposal-only until that eval run is recorded and passed.

## Risks / concerns

RISKS_PLACEHOLDER

## PR link

(set after open)

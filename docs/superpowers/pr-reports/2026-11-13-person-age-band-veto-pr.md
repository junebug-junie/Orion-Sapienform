## Summary

- Orion now stops guessing a person whose age band is wrong for the cluster. When a camera cluster clearly reads as a child (or as an adult), and Juniper said a person is in the other band, that person is struck from the cluster's candidates.
- The child/adult read comes from the SigLIP vector Orion already stores for each crop. It is compared with fixed text vectors committed in `orion/vision/age_band.py` and `age_band_text_vectors.json`. No model is loaded and nothing new is computed on a GPU.
- The veto uses only ages Juniper stated (`person_fact` `age`, newer than 2 years, aged since they were given). It does nothing on cameras with a prior (the office), and it reads only vectors from the pinned SigLIP v1 model.
- The veto only removes candidates. It never adds a name, and it never promotes a two-person accept to an accept.
- Trace: `vision_person_attribution.age_veto = {"struck": [names], "rule": "age_band_v1"}`. No age read is stored. The trace is cleared whenever a row is revoked.
- The eval scores veto off and veto on in one pass. The gate needs 0 wrong accepts in both arms, and the on arm must not name the right person less often than the off arm.
- Gender is not read anywhere (design PR #3278: NO-GO).

## Outcome moved

Wrong top guesses are struck before they become "Is this X?" asks. The eval numbers are below.

## Current architecture

`attribute()` scored every album person against each unnamed cluster. The only filter was Juniper's "not X" answers (`blocked`). Nothing used age.

## Architecture touched

- orion-sql-writer person model: the guess cycle, the attribution write, and the confirm-answer revokes.
- `orion/vision/age_band.py`: a new pure module.
- One additive migration.
- The person recognition eval.

## Files changed

- `orion/vision/age_band.py`, `orion/vision/age_band_text_vectors.json`: the pure read, the committed text side, and the age-answer parser.
- `scripts/generate_age_band_text_vectors.py`: a one-off regenerator. It runs in the vision-host image.
- `services/orion-sql-writer/app/vision_person_model.py`: the changes are:
  - `Crop.body_model`;
  - the veto in `attribute()`;
  - `cluster_age_bands`, `load_person_age_bands` and `veto_exempt_streams`;
  - the `age_veto` write and clearing;
  - the fail-open wiring and stats.
- `services/orion-sql-writer/app/vision_individuals.py`: confirm revokes clear `age_veto` in the same statement.
- `services/orion-sql-writer/app/settings.py`, `vision_individuals_loop.py`, `.env_example`, `README.md`: the `VISION_PERSON_AGE_VETO_ENABLED` flag (default true).
- `services/orion-sql-db/manual_migration_vision_person_age_veto_v1.sql` (+ `_rollback.sql`): adds the `age_veto jsonb` column.
- `services/orion-sql-writer/evals/run_person_recognition_eval.py`: scores both arms.
- Tests: `orion/vision/tests/test_age_band.py` and `services/orion-sql-writer/tests/test_vision_person_model.py`. The test vectors are synthetic. No real crop vectors are committed.
- `.github/workflows/orion-sql-writer-tests.yml`: the PG lane applies the new migration.

## Schema / bus / API changes

- Added: the column `vision_person_attribution.age_veto jsonb`. The migration has REQUIRED-BY: orion-sql-writer and is additive, so it is on the post-pull auto-apply allowlist (checked with `drift.auto_apply_refusals`, which reports no refusals).
- Removed / renamed: none.
- Behavior changed: an age-band-mismatched person is no longer a guess candidate.
- Compatibility notes: the writer checks that the column exists before it writes, so a deploy before the migration still writes guesses. No bus or registry change.

## Env/config changes

- Added keys: `VISION_PERSON_AGE_VETO_ENABLED=true` (orion-sql-writer).
- `.env_example` updated: yes.
- local `.env` synced: the worktree has no `.env`. `python scripts/sync_local_env_from_example.py` was run. The key was added to the primary checkout's `services/orion-sql-writer/.env` (gitignored).
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/vision/tests/test_age_band.py services/orion-sql-writer/tests/test_vision_person_model.py (throwaway Postgres 16, migrations applied) -> 79 passed
pytest services/orion-sql-writer/tests/test_vision_confirm_asks.py test_vision_office_sessions_pg.py (throwaway Postgres) -> 53 passed
pytest services/orion-sql-writer/tests services/orion-sql-writer/evals -> 1294 passed, 226 skipped
check_migration_required_by --base origin/main -> OK; env template parity PASS; metric lineage PASS; definition drift PASS; no personal terms
```

## Evals run

Full both-arm eval (`run_person_recognition_eval.py --live`, read-only, no `--write`) is running on the final code; numbers will be added here when it finishes (~100 min).

## Docker/build/smoke checks

The text vectors were generated inside the `orion-vision-host` image (`docker run --rm`, weights mounted read-only).

Read-only dry run of one live guess cycle (2026-11-13 ~06:20 UTC, final code, `SET TRANSACTION READ ONLY`, no writes):

```text
street clusters 1,781; names with a stated band 16; clusters with a confident band 387 (176 child, 211 adult)
clusters where the veto struck at least one name: 374 (walkway 292, doorbell 82, office 0 -- exempt)
top guess changed by the veto: 41; clusters left with no guess: 0
```

So the first person-model cycle after deploy (every `VISION_PERSON_MODEL_INTERVAL_SEC` = 600 s) should write non-null `age_veto` rows. Post-deploy check:
`docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "SELECT count(*) FROM vision_person_attribution WHERE revoked_at IS NULL AND age_veto IS NOT NULL"` (expect > 0 within ~10 min), and the sql-writer log summary keys `person_age_vetoes`, `person_age_veto_all_struck`. UNVERIFIED until deployed.

## Review findings fixed

The code review ran in a subagent. Each finding and its fix:
- Finding: "18 months", "born 2015" and "not an adult" were read as adult.
  - Fix: the parser handles months and weeks, refuses years and negation, and maps child words.
  - Evidence: new cases in `test_band_from_age_answer`.
- Finding: the veto could promote a double accept to accept.
  - Fix: the accept count before the veto decides the band.
  - Evidence: `test_age_veto_never_promotes_a_double_accept_to_accept`.
- Finding: the confirm revokes cleared `age_veto` only on the next cycle.
  - Fix: `_age_veto_clear()` is applied in the same UPDATE.
  - Evidence: a PG test.
- Finding: a broken priors file lifted the office exemption.
  - Fix: the veto is off for that cycle, and `person_age_veto_error` is reported.
  - Evidence: a PG test.
- Finding: stale answers near 13, and conflicting answers.
  - Fix: the newest answer wins, aged by the years since it was given.
- Finding: `not_a_person` names could carry a band.
  - Fix: both branches of the name query are filtered.
- Finding: the eval duplicated the exemption rule.
  - Fix: the shared `veto_exempt_streams()`. The gate also gets `unsure_right_not_lower`.
- Finding: the test used a wrong voice value.
  - Fix: changed to `ask_answer`.

## Restart required

Merge first, then from the clean primary main checkout:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

The post-pull rebuild auto-applies the migration first. If it does not, apply it by hand with `docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_person_age_veto_v1.sql`.

## Risks / concerns

- Severity: medium.
  - Concern: the read is right 83% of the time per crop. A child who reads as an adult on 2+ crops will strike that child's own name.
  - Mitigation: the veto needs a confident cluster mean. It only removes names, so the worst case is a missed "Is this X?" ask, never a wrong name. It is off with `VISION_PERSON_AGE_VETO_ENABLED=false`.
- Severity: low.
  - Concern: on the office camera the read is unmeasured (infrared).
  - Mitigation: the office camera is exempt.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

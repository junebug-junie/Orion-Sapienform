# PR: on the office camera, the person model starts from "it is almost always Juniper"

## Summary

- Juniper's first full office visit (2026-11-05, 18:23-18:31 MDT) was guessed as other people 15 times out of 16. Body vectors across cameras mostly match clothes and light, and her album holds only outdoor pictures.
- This PR adds a **camera prior** inside the existing person model. On the office camera, when no album match is confident, the guess is the person usually there (Juniper) instead of the highest raw score. A confident match (`accept`) still wins.
- A "not this person" bound learned on other cameras no longer rules anyone out on the office camera, until that person's album holds an office picture.
- Everything else is the existing walkway loop, unchanged: the guess feeds the existing "Is this Juniper?" asks (cap 6/day), her answers name the records, named records join her album, and the album's own bars arm over time.
- Orion's office line now says "probably Juniper (my guess from who is usually here, not confirmed)" instead of "unfamiliar people". A second person in view is still reported as unfamiliar.
- Household names stay out of git: `config/person_model/camera_priors.example.yaml` (committed) holds only Juniper; `camera_priors.local.yaml` (gitignored, read first) holds the household.

## Outcome moved

Live read-only replay (`evals/run_office_prior_replay.py`, 2026-11-05):

```text
cam0: 16 unnamed clusters, 16 guessed
  before: {'[household member]': 7, '[walkway person A]': 4, 'skeleton': 1, 'Juniper': 1, '[walkway person B]': 3}
  after:  {'Juniper': 16} (prior basis: 16)
  accepted guesses changed: 0
guesses changed on cameras without a prior: 0
VERDICT: PASS
```

Live effect after deploy (UNVERIFIED until it runs): the next office visit opens "Is this Juniper?" asks, not "Is this [household member]?", and the brief names her as a guess.

## Current architecture

- `vision_person_model.attribute` guessed the top raw score per unnamed cluster, on every camera alike. Bars (accept / reject) are conformal, fitted on street clusters; no accept bar is armed for anyone today.
- `_open_person_confirm_asks` opens "Is this X?" for `unsure` guesses with at least 2 sightings, 6 per day across all cameras.
- `perception_reader._who_line` shows only names Juniper gave; everyone else is a range of "unfamiliar people".

## Architecture touched

orion-sql-writer (person model, Dockerfile), `orion/person_model`, `orion/situational` (Hub and cortex-exec read it), one additive migration. No bus or schema-registry changes.

## Files changed

- `orion/person_model/camera_priors.py` (new): loads priors, local file first.
- `config/person_model/camera_priors.example.yaml` (new): redacted prior (Juniper only).
- `services/orion-sql-writer/app/vision_person_model.py`: `Crop.stream`, `cluster_stream`, `album_has_stream`, prior rule in `attribute`, `basis` in guesses and in `write_attributions` (guarded), `_camera_priors`, `person_prior_guesses` in the cycle summary.
- `services/orion-sql-writer/Dockerfile`: copies `people.yaml` and `camera_priors.*.yaml` (the priors and their local file).
- `services/orion-sql-writer/app/vision_individuals.py`: prior confirm asks rank last, one open per name.
- `services/orion-sql-db/manual_migration_vision_person_basis_v1.sql` (+ rollback): `vision_person_attribution.basis`.
- `orion/situational/perception_reader.py`: `_read_prior_guesses`; `_who_line` shows prior guesses and keeps extra people visible.
- `services/orion-sql-writer/evals/run_office_prior_replay.py` (new): live replay.
- Tests: `services/orion-sql-writer/tests/test_vision_person_model.py`, `orion/situational/tests/test_street_summary.py`.
- `.github/workflows/orion-sql-writer-tests.yml`: applies the new migration in the Postgres job.
- `services/orion-sql-writer/README.md`: camera priors.

## Schema / bus / API changes

- Added: `vision_person_attribution.basis text` (prior | evidence; NULL on old rows).
- Behavior changed: office-camera guesses (`unsure` only) follow the camera prior.
- Compatibility notes: the writer writes `basis` only when the column exists; the reader's guess statement fails open without it.

## Env/config changes

- Added keys: none. Optional override `VISION_CAMERA_PRIORS_PATH` (settings-only, not in any `.env`).
- New local file: `config/person_model/camera_priors.local.yaml` written in the primary checkout (gitignored).
- `.env_example` updated: no. Local `.env` sync: not needed.

## Tests run

```text
orion-sql-writer  pytest tests -q                                   1212 passed, 185 skipped
orion-sql-writer  test_vision_person_model.py with throwaway Postgres 58 passed (confirm-ask tests also pass with the basis column dropped)
orion/situational/tests (+ throwaway Postgres for the guess SQL)     298 passed
```

## Evals run

```text
evals/run_office_prior_replay.py (live, read-only)   PASS (16/16 office guesses -> Juniper; 0 accepted changed; 0 other cameras changed)
```

## Docker/build/smoke checks

```text
docker build -f services/orion-sql-writer/Dockerfile .   built; /app/config/person_model/camera_priors.example.yaml loads in the image
gates: check_personal_terms, check_definition_drift --gate, check_metric_lineage --gate,
       check_env_key_single_source, check_migration_required_by --base origin/main, git diff --check   ok
```

## Review findings fixed

Review ran in a subagent.

- Finding (HIGH): after Juniper's "no", the prior named the next 1% person ("probably <household member>") and walked the confirm asks down the small names.
  - Fix: the prior names a guess only for a dominant person (`PRIOR_DOMINANCE = 0.5`) she has not ruled out; otherwise the scores choose (`basis = evidence`).
  - Evidence: `test_after_her_no_a_small_prior_never_names_the_guess`.
- Finding (HIGH): out-of-camera rejects were relaxed for every person on the office camera.
  - Fix: only for the dominant prior person. A misspelled prior changes nothing.
  - Evidence: `test_a_small_prior_relaxes_no_reject_and_a_misspelled_prior_changes_nothing`.
- Finding (HIGH): a named "Juniper" plus a guessed "Juniper" counted as two names and hid a second person in view.
  - Fix: a guess counts as the same name. Evidence: extended `test_a_name_she_gave_wins_over_the_same_guess`.
- Finding (MEDIUM): with unread window counts, several guessed records collapsed into one; a stationary prop could produce "up to 0 unfamiliar people".
  - Fix: extra guessed records read "up to N"; props subtract from the window peak; no phrase when the count is 0. Evidence: two new tests.
- Finding (MEDIUM): prior guesses competed for the shared 6/day confirm cap.
  - Fix: prior guesses rank after evidence guesses; at most one prior ask open per name. Works before the migration (column checked first). Evidence: `test_office_prior_asks_rank_last_and_one_is_open_per_name` (real Postgres), confirm tests pass with the column dropped.
- Findings (LOW): Dockerfile copied every personal local file; column check ignored schema; duplicate prior names merged silently.
  - Fix: copy only `people.yaml` and `camera_priors.*.yaml`; `table_schema = current_schema()`; duplicate names rejected.

## Restart required

```text
No manual restart. The post-pull rebuild applies the additive migration first, then rebuilds orion-sql-writer, orion-hub and orion-cortex-exec.
```

## Risks / concerns

- Severity: medium. Concern: a visitor alone in the office (a child, a house tour) reads as "probably Juniper (my guess…)". Mitigation: it is labeled a guess, a second person in view is still reported, and a confident match for someone else overrides it. The office design (#3127, step B) adds the unknown-person alarm.
- Severity: low. Concern: editing `camera_priors.local.yaml` takes effect only after a rebuild of orion-sql-writer (the file is copied into the image).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3129

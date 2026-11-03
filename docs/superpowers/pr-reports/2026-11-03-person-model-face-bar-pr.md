# PR: person model -- split alpha over real people only, face bars on a 90-day street

## Summary

- Orion can say "probably X" only when a person clears an accept bar. A bar must be stricter for each extra name Orion could guess. That strictness was counting 16 names, but 3 of them are not one real person: a skeleton decoration, "UPS delivery", and the pair "Addy and Esme". It now counts only real people (13 live). The check reuses the existing person record (`person.status = 'not_a_person'`, set in the Hub People panel). It does not use a hand-kept list.
- Non-person names keep their albums, their reject bound, and their "Is this X?" asks. Their clusters still count as strangers for everyone else. They never get an accept bar, because "probably Addy and Esme" would name known people outside the bound. (A review finding; see below.)
- Face bars are fitted against unnamed clusters from the last 90 days (`VISION_PERSON_FACE_CANDIDATE_DAYS`, shipped ON). Faces are rare, and a bar needs about 259 stranger clusters with faces. Body bars and every guess stay on the 30-day street.
- `MODEL_VERSION` is now `median-topk-street-v2-people-split-face90`.

## Outcome moved

The count of stranger face clusters that a face bar needs is now reachable. Today: **nothing visible changed** (eval identical to v1). Every face crop is under 19 days old (the first is 2026-10-15), so the 90-day window has no extra clusters to add yet. At the measured rate (~2.8 street face clusters a day, plus ~2.7 when the doorbell is back via #3085), the face street reaches the ~245 clusters needed in about 35 days (~2026-12-08). With the old 30-day window it would level off near 165 and never arm. With the old 16-name split it would need ~305, about 11 more days.

This is not yet "probably X appears". Two more limits still bind after the count is met. Only 2 people (Juniper, Esme) have the 3 face clusters a bar needs. Face cross-day AUC is 0.614 (0.5 is chance).

## Current architecture

`services/orion-sql-writer/app/vision_person_model.py`. `fit_bar` set the accept bound at `alpha / len(album)`, over every album label, fitted on one street dict loaded with `candidate_days=30`. That one dict was used both to fit every channel and to make the guesses.

## Architecture touched

Only orion-sql-writer. It reads the `person` and `person_link` tables, which already exist (PR #2852), and writes nothing new. No bus, schema, or API change.

## Files changed

- `services/orion-sql-writer/app/vision_person_model.py`: adds `people_count`, `street_pools`, and `not_a_person_names`. `fit_bar`/`fit_all` take `face_street` and `not_people`. Non-person labels get no accept bar. Two new summary fields: `person_split_people` and `person_face_street_clusters`. `MODEL_VERSION` bumped.
- `services/orion-sql-writer/evals/run_person_recognition_eval.py`: uses the same street and name rules as the live cycle. Reports `split_people`, `not_people`, `face_street_clusters`. Adds `--face-candidate-days` and `--split-all-labels` (old rule) for A/B runs.
- `services/orion-sql-writer/app/settings.py`, `.env_example`, `app/vision_individuals_loop.py`: two new keys, wired into `PersonModelConfig`.
- `services/orion-sql-writer/tests/test_vision_person_model.py`: 9 new tests (2 Postgres-backed).
- `services/orion-sql-writer/README.md`, `docs/superpowers/specs/2026-10-20-walkway-person-learning-design.md`: rule and status.

## Schema / bus / API changes

- Added: none (the reads use existing tables).
- Removed: none.
- Renamed: none.
- Behavior changed: the accept bound uses alpha / real-people count. Non-person labels never accept. The face bar calibrates on 90 days.
- Compatibility notes: because `MODEL_VERSION` changed, every live `vision_person_attribution` row is revoked and rewritten once, on the first cycle after deploy. Guesses churn once. When the person tables are missing, every label counts as a person (the split only gets stricter).

## Env/config changes

- Added keys: `VISION_PERSON_FACE_CANDIDATE_DAYS=90`, `VISION_PERSON_ALPHA_SPLIT_PEOPLE_ONLY=true`.
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced: yes. `python3 scripts/sync_local_env_from_example.py --all-keys orion-sql-writer` added both keys to the primary checkout's `services/orion-sql-writer/.env` (lines 307-308). The default run does not sync the `VISION_PERSON_` prefix, so `--all-keys` was needed.
- Skipped keys requiring operator action: none.
- Off switch: `VISION_PERSON_FACE_CANDIDATE_DAYS=30` with `VISION_PERSON_ALPHA_SPLIT_PEOPLE_ONLY=false` gives the v1 rules (tested).

## Proposal-mode fields (person model is a memory/social-continuity change)

- **Capability change:** a face accept bar ("probably X" from a face) can arm once enough stranger face clusters exist. Non-person names can no longer produce "probably <name>".
- **Data touched:** reads `vision_individual`, `vision_crop_observation` (embeddings), `person`, `person_link`. Writes the same tables as before (`vision_person_bar`, `vision_person_attribution`). It writes no label. A guess never writes `vision_individual.label`. Retention: person crops are kept forever (`VISION_CROP_RETENTION_KEEP_LABELS=person`). Unnamed clusters are pruned after `VISION_SIGHTING_RETENTION_DAYS` (90), so a face window longer than 90 days finds nothing more (a test pins face window <= retention).
- **Privacy boundary:** none crossed. Same local Postgres, same camera data. No new consumer reads guesses.
- **Trace that proves it worked:** `vision_person_bar` rows with `channel='face' AND accept_at IS NOT NULL`. The cycle summary fields `person_split_people` (13 expected) and `person_face_street_clusters` (should grow past `person_street_clusters` after ~2026-11-14).
- **Dangerous failure:** a wrong "probably X" that names a stranger as a known person. Guards: the conformal bound per bar; the split counts every name that can accept; a bar needs >= 3 held-out own clusters with half above it; a cluster accepted for two people is downgraded to unsure; non-person names cannot accept; a name that an active person also carries is never treated as a thing. Residual risk: face bars calibrate on older strangers than the ones they are applied to. If lighting or season makes recent faces score higher, the real false-accept rate exceeds alpha / people (see Risks).
- **Disable / roll back:** set `VISION_PERSON_FACE_CANDIDATE_DAYS=30` and `VISION_PERSON_ALPHA_SPLIT_PEOPLE_ONLY=false`, then restart sql-writer. This restores the v1 rules (the version label stays v2, so attributions churn once more). A full revert of the PR restores v1 exactly. `VISION_PERSON_MODEL_ENABLED=false` stops all guesses.

## Tests run

```text
pytest services/orion-sql-writer/tests/test_vision_person_model.py (ORION_TEST_PG_URI = throwaway postgres:16 with
  walkway + person_album + person_record migrations)            -> 49 passed
pytest services/orion-sql-writer/tests (branch)                  -> 1182 passed, 183 skipped, 0 failed
pytest services/orion-sql-writer/tests (main, same venv)         -> 1 failed (test_the_ratchet_does_not_hide_a_channel_that_got_fixed, pre-existing), 1176 passed
Mutation check: removing "no accept bar for not_people" and the link-join half of not_a_person_names
  -> test_a_not_a_person_label_never_gets_an_accept_bar and test_not_a_person_names_come_from_the_person_record FAIL
python3 scripts/check_env_template_parity.py                     -> PASS
git diff --check                                                 -> clean
```

## Evals run

Read-only, `evals/run_person_recognition_eval.py --live postgresql://...@localhost:55432/conjourney`, without `--write`. Both runs are on the same data on 2026-11-03. Stored previous row: `eval-58d03b6ab5994a04b20f5cf39ddee4ee` (v1).

| | v1 rules today (`--split-all-labels --face-candidate-days 30`) | v2 (this PR) |
|---|---|---|
| labelled clusters / scored | 115 / 103 | 115 / 103 |
| **wrong accepts** | 0 | **0** |
| correct accepts | 0 | 0 |
| unsure (right person top guess) | 103 (50) | 103 (50) |
| street accepts | 0 | 0 |
| split across | 16 | 13 (not_people: addy and esme, skeleton, ups delivery) |
| street / face street clusters | 1352 / 1352 | 1352 / 1352 |
| face AUC (genuine / impostor clusters) | 0.6135 (11 / 764) | 0.6135 (11 / 764) |
| body AUC | 0.8503 | 0.8503 |
| osnet AUC | 0.7635 | 0.7635 |
| people_with_accept_bar face / body / osnet | 0 / 0 / 0 | 0 / 0 / 0 |
| McNemar vs stored v1 | 113 common, 0 / 0, p = 1.0 | 113 common, 0 / 0, p = 1.0, better = false |

**Gate: PASSED** (0 wrong accepts, no street accept flood). This version is not "better" on McNemar. On today's data it is the same model, by construction (see Outcome moved).

## Docker/build/smoke checks

```text
No docker build: no dependency or Dockerfile change. Behaviour reads env at boot through settings.py (tested).
No deploy and no live DB write (read-only eval only).
```

## Review findings fixed

- Finding (blocker): a pair label such as "A and B" kept its accept bar, so "probably A and B" could name known people outside the Bonferroni split.
  - Fix: a `not_a_person` label never gets an accept bar (`fit_bar`). It is still scored and asked about.
  - Evidence: `test_a_not_a_person_label_never_gets_an_accept_bar` (fails when the line is removed).
- Finding (should): the Postgres test for `not_a_person_names` passed even with the link-join half deleted.
  - Fix: the person's name now differs from the cluster label. The test adds a revoked link and an active-person link.
  - Evidence: the mutation check above fails without the join.
- Finding (should): a lagging link or a merge could mark a real person's name as a thing.
  - Fix: any name an active person row carries is subtracted from the set.
  - Evidence: the "Teddy" case in the same Postgres test.
- Finding (nit): there was no test for missing person tables.
  - Fix: `test_missing_person_tables_mean_everyone_counts_as_a_person`.
- Finding (nit): there was no test that flag-off ignores `not_people`.
  - Fix: assertion in `test_split_people_only_off_is_the_old_split_over_every_label`.
- Not fixed, follow-up: (a) a time-split drift check (fit face bars on 30-90-day strangers only, then measure the share of recent strangers above the bar against alpha / people). Today the 30-90-day slice is empty, so the check would read nothing; it becomes meaningful after ~2026-11-14. (b) A test that pins eval/live parity. Today both call the same `street_pools` / `not_a_person_names` / `fit_all`. (c) `street_pools` makes one extra `last_seen_at` query (nit).

## Restart required

Juniper runs this from the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d
```

Post-deploy check (after the first person-model cycle, <= 10 min). `person_split_people` should be 13:

```bash
docker logs --since 15m orion-athena-sql-writer 2>&1 | grep -o "'person_split_people': [0-9]*" | tail -1
```

Then, once face bars can arm (expected ~2026-12-08 or later), run:

```sql
SELECT person_label, channel, accept_at, n_impostor FROM vision_person_bar WHERE accept_at IS NOT NULL;
SELECT individual_id, person_label, channel, score FROM vision_person_attribution
 WHERE band = 'accept' AND revoked_at IS NULL ORDER BY decided_at DESC;
```

Juniper must review every `band='accept'` row by eye before any consumer reads "probably X".

## Risks / concerns

- Severity: medium. Concern: face bars calibrate on older strangers than the ones they are applied to. Seasonal light or camera moves could push the real false-accept rate above alpha / people. Mitigation: the drift check follow-up above; Juniper reviews accept rows by hand; the off switch.
- Severity: low. Concern: "Mail carrier" is still `active`, so it still counts in the split and can get an accept bar. Mitigation: if it is a role and not one person, mark it "not a person" in the Hub People panel. The model picks that up on the next cycle.
- Severity: low. Concern: the version bump revokes and rewrites every attribution once. Mitigation: the rewrite happens in the same cycle; open confirm asks are keyed by individual and name, not by version.
- Severity: info. Concern: the eval is identical to v1 today, and `vs_previous.better` is false. The nightly `make eval-person-recognition --write` will record a v2 row with p = 1.0. Mitigation: expected; the change only bites once face data is older than 30 days.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3087

🤖 Generated with [Claude Code](https://claude.com/claude-code)

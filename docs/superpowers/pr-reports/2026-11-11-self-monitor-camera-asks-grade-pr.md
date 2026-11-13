# PR report: self-monitor grades the camera guesses Juniper answers (orion_ask)

## Summary

- Orion's self-monitor now reads the camera asks Juniper answers in `orion_ask`. It grades the two
  ask kinds where Orion states a guess: "Is this <name>?" and "Are these the same person?".
- Three faculties appear in the organ's state with a hit rate: `vision.person_confirm`,
  `vision.person_confirm.camera_prior` (the office camera's "it is the usual person" guess) and
  `vision.same_person`. A fourth, `vision.same_person.handoff` (the street loop's "did this person
  walk from one camera to the next?"), is split out too; it had no answer in the last 14 days.
- None of them gets a status (reliable, unreliable and so on). Orion records no confidence before
  these asks, and the organ needs one to judge. They show as `ungradeable:no_stated_confidence`,
  with the hit rate visible. The catalog names the fix: a P(yes) bet in orion-prediction when the
  ask opens.
- `vision.same_person` is no longer marked `source_bug`. #3046 is live: since 11-03, 15 of 15 pair
  verdicts carry the ask's face cosine. The grade reads the ask itself, not the verdict table.
- The other camera asks ("Who is this?", walkway story names, person facts) are not graded. They
  carry no guess of Orion's, so there is nothing to be right or wrong about. Of the ~495 answered
  asks in 14 days, only ~117 hold a guess.

## Outcome moved

Before: the self-monitor knew nothing about the camera-ask loop. After: every hourly evaluation
carries the hit rate of each camera guess, with its Wilson interval, in `orion:self_monitor:latest`,
`GET /self_monitor/state` and the live report.

Live grade (dry run of `python -P -m orion.self_monitor.report` against live Postgres, 2026-11-11
04:10Z; decay-weighted hit rate, 95% Wilson interval):

| Faculty | Window | Answers | Graded | Hit rate | Interval |
|---|---|---|---|---|---|
| `vision.person_confirm` (album scores chose the name) | 7 d | 38 | 36 | 45% | 29-62% |
| `vision.person_confirm.camera_prior` (office prior chose it) | 7 d | 9 | 9 | 100% | 69-100% |
| `vision.same_person` | 7 d | 15 | 15 | 53% | 30-76% |
| `vision.person_confirm` | 14 d | 80 | 77 | 41% | 28-55% |
| `vision.same_person` | 14 d | 24 | 24 | 51% | 29-72% |

Raw 7-day answers: person confirm 20 yes / 18 no (2 of the yeses were mixed per-picture answers,
so not graded); office prior 9 yes / 0 no; same person 8 yes / 7 no. Status for all three:
`unknown`, `ungradeable:no_stated_confidence`. No source errors.

## Current architecture

The self-monitor (`orion/self_monitor/`, spec PR #2827) grades faculties from settled bets. One SQL
plus one pure adapter per source (`collect.SOURCES`), shared by the live report and the hourly host
node in orion-durable-runs. The reducer (`record.py`) gives a status only to faculties with at least
10 graded bets and a stated confidence. `catalog.py` lists faculties it knows but cannot grade.
`vision.same_person` sat in the catalog as `source_bug`. `orion_ask` was not read at all.

## Architecture touched

- `orion/self_monitor` only: a new `orion_ask` source, its adapter, two catalog entries and three
  plain names. No schema, bus, env or service change. The host picks the source up from `SOURCES`.

## Files changed

- `orion/self_monitor/adapters.py`: `camera_ask` and `camera_ask_outcome`; docstring records the live
  ask vocabulary.
- `orion/self_monitor/collect.py`: the `orion_ask` source (answered camera guesses, windowed on
  `answered_at`).
- `orion/self_monitor/catalog.py`: `vision.same_person` moves from `source_bug` to
  `no_stated_confidence`; `vision.person_confirm` and `vision.person_confirm.camera_prior` added. Shown only in a window with no answer.
- `orion/self_monitor/names.py`: plain names for the four faculties.
- `services/orion-durable-runs/app/self_monitor_store.py`: comment only; the source-count timeout
  budget is now 7 x 15 s = 105 s, under the node's 120 s.
- `orion/self_monitor/tests/test_self_monitor_camera_asks.py`: adapter and reducer tests, including
  the regression below.
- `orion/self_monitor/tests/test_self_monitor_host.py`: the "unread source" set now includes
  `orion_ask`.

## Metric quality gate (CLAUDE.md section 0A)

The metric is the hit rate of a camera guess: the share of answered guesses that Juniper said
were right.

1. **Provenance.** The guess: `services/orion-sql-writer/app/vision_individuals.py`
   `_open_person_confirm_asks` (name from `vision_person_attribution`, band `unsure`;
   `rule:camera_prior` marks the office-prior lane) and `_open_face_asks` (pair with face cosine >=
   `same_ask_min_face_cosine` 0.40). The outcome: Juniper's `answer` and `answer_detail.pictures`,
   read by the same rule the applier uses (`_apply_person_confirm`, `parse_yes_no`).
2. **Independence.** Every other self-monitor source is a different producer (acts, turn bets,
   action bets, forecasts, percept expectations). The two camera faculties share an upstream: the
   individual clustering and the face/body embeddings. An impure cluster can hurt both. So they are
   related, not redundant: one asks "which name", the other "same or not". The office-prior lane is
   a different mechanism (a config prior, not album scores), so it is its own faculty.
3. **Theory anchor.** A guess judged by an outside party is a Bernoulli trial; the hit rate is its
   success share, and the organ already puts a Wilson interval on it. Juniper is the ground truth
   for who people are.
   **What fails the gate:** the confidence. `face_cosine` is a similarity, not a probability. On 37
   answered same-person asks it barely ranks Juniper's "same" above "different" (AUC 0.59; mean
   0.58 vs 0.54). The attribution `score` is a median top-k cosine that the person model rewrites
   every cycle, so it is not even a pre-ask record. Using either as `stated` would invent a
   calibration claim. So `stated` is None and no status is given.
4. **Live data.** Real rows, not degenerate: the hit rate reads low and high (person confirm 41-45%,
   same person 51-53%, office prior 100%), and every answer is yes or no from Juniper. A low rate
   here is the asking rule working: Orion only asks when unsure. That is exactly why no status is
   given without a stated confidence: "unreliable" at 41% would be a false alarm.
5. **Existing mechanism.** No grade of these asks existed. `vision_individual_pair_verdict` holds
   the same answers for the pair ask; the grade reads the ask instead (one row per ask, while the
   verdict table also holds propagated pairs). #3046 is verified live: since 11-03, 15/15 verdict
   rows for the ask's own pair carry the ask's cosine.
6. **Reversibility.** Cheap. One `SOURCES` row, one adapter, nothing persisted beyond the hourly
   state. A faculty with status `unknown` writes no `faculty_status_changed` event and moves no gate.

## Schema / bus / API changes

- Added: none (new faculty ids inside the existing `FacultyRecordV1` state).
- Removed: none.
- Renamed: none.
- Behavior changed: the self-monitor state carries three more faculties.
- Compatibility notes: the cue's "I can't grade N other things" count grows by up to three.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest -q orion/self_monitor/tests                                   137 passed
pytest -q orion/temporal_self/tests/test_faculty_arcs.py tests/test_metric_lineage.py   47 passed
services/orion-thought  (cue tests + self_track_record truth eval)   17 passed
services/orion-durable-runs  tests/test_self_monitor_node.py          14 passed, 1 skipped
services/orion-hub  tests/test_self_monitor_questions.py              4 passed
```

Regression test: `test_a_low_hit_rate_without_a_stated_confidence_never_reads_unreliable`. Thirty
unsure guesses right a third of the time stay `unknown` / `no_stated_confidence`, never
`unreliable`.

## Evals run

```text
services/orion-thought/evals/test_self_track_record_truth_eval.py   passed (in the 17 above)
Live replay: python -P -m orion.self_monitor.report [--days 14] against live Postgres; numbers above.
```

## Docker/build/smoke checks

```text
No image change. The source runs inside orion-durable-runs' existing self_monitor node.
```

## Review findings fixed

Code review ran in a subagent on the branch diff. No blocker. It confirmed on live data that these
faculties cannot reach a notable status (no `failed_to_execute`, `event` claims, no `stated`), so the
act gate, E2 self-questions and E3 messages cannot fire; the SQL runs under psycopg.

- Finding (should): street-handoff "same person?" asks (`rule:handoff`, a story-crop cosine) were
  pooled into `vision.same_person`, a face-centroid guess.
  - Fix: split into `vision.same_person.handoff`, with a plain name and a test.
  - Evidence: `test_a_street_handoff_pair_is_its_own_faculty`.
- Finding (nit): pictures could grade a card whose answer was not yes/no, and pictures with no
  `image_ref` were counted; the applier does neither.
  - Fix: parse the answer first (no yes/no -> no bet), then read only pictures with an image.
  - Evidence: `test_unsure_only_pictures_and_pictures_without_an_image_are_read_like_the_applier`.
- Finding (nit): no catalog entry for `vision.person_confirm.camera_prior`.
  - Fix: added.
- Finding (nit): stale "5 sources x 15 s" timeout comment in orion-durable-runs.
  - Fix: comment updated (7 x 15 s = 105 s < 120 s).
- Finding (nit): test gaps (auto-closed card with `answer` NULL, unsure-only pictures, handoff).
  - Fix: tests added.
- Finding (nit): the yes/no word sets copy the sql-writer's.
  - Fix: comment names the twin. Not moved: the applier lives in a service app.
- Not fixed (noted): in a window where every bet is `no_change`, the reducer says
  `never_settled`, whose unlock text reads oddly for answered asks. Pre-existing reducer wording,
  not reachable today (2 of 38 answers are mixed).

## Restart required

Deploy only from the clean primary `main` checkout, and only when no chat turn is running. After
merge, the post-pull rebuild rebuilds orion-durable-runs (it imports `orion/`). Check that
orion-durable-runs is in that run's rebuild list; if not:

```bash
cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Then check: `docker exec orion-orion-athena-bus-core redis-cli GET orion:self_monitor:latest` holds
`vision.person_confirm` within an hour.

## Risks / concerns

- Severity: low
- Concern: the grade has no status, so the organ cannot act on it yet.
- Mitigation: next patch, a P(yes) bet in orion-prediction when a camera ask opens (the same seam
  the memory-confirmation idea needs). With that, the organ judges calibration and Brier skill.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3189

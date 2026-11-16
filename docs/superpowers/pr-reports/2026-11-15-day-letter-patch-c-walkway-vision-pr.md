# Patch C: walkway vision (day-letter fix plan, PR #3439)

Fixes #3387. Fixes #3389. Refs #3388.

## Summary

- The walkway wrote about 60 to 87 "a vehicle is observed" rows an hour at night against about 6 by day. Two causes, both fixed. The larger one was not in the ticket.
- The 0.8 confidence on every vision row was the number in the prompt example. It is gone. A row's confidence is now the detector's own score for what the row names, or empty when the detector saw none of it.
- Each row a model wrote now names the model that really answered.
- A car that sat parked in view during its expected arrival window is graded "cannot tell" (`unscorable`), not "did not come" (`missed`).
- The hourly walkway line now says how many parked sightings it left out: "nobody passed (54 parked sightings not counted)".
- Handoff rows (door to walkway and back) now say what they were about: `["person"]`, not `[]`.

## Outcome moved

Replay of 96 hours of real walkway data through the real code (48 hours scored, the first 48 warm the fixture tracker). The number is how many times a label change would make the council call the model:

| Case | Calls in 48 h |
|---|---|
| Production as recorded | 3,949 |
| Belief fix alone | 1,174 |
| Belief fix and fixture rule | 300 |

Night hours 05-13Z on 11-14 went from 166-326 an hour to 3-24 an hour. The remaining busy hours (01-03Z on 11-14, 30 to 68) are early evening with real traffic. Script: `services/orion-vision-council/evals/replay_walkway_gate.py`.

The replay treats each model call as instant, so it is an upper bound. Live row counts are lower than replay counts in every column.

## Current architecture

Before this patch:

- `orion-vision-window` keeps a "believed" label set per camera: a label enters after 3 votes and leaves after 3 windows without it. An empty window inherited the last labels seen, as a guard against one dropped frame.
- `orion-vision-council` asks the model to describe the scene whenever the believed label set changes, or every 600 s.
- The council prompt showed an example row with `"confidence": 0.8`. The model copied it. 9,165 of 9,242 walkway rows in 7 days carried 0.8.
- `orion-sql-writer` rhythm loop: an expected arrival counts only if a sighting in the window is not stationary. A parked sighting in the window gave `missed`.
- The walkway rollup counted excluded parked sightings and never printed the count.
- The handoff writer always stored `entities = []`.

## Architecture touched

- `orion-vision-window`: belief tracker rule; two new keys in `summary.evidence` (`label_scores`, `fixture_only_labels`).
- `orion-vision-council`: transition gate, prompt, confidence, served model on rows.
- `orion/schemas/vision.py`: `confidence` may be empty on vision event rows.
- `orion-sql-writer`: rhythm grading, handoff event writer.
- `orion/vision/walkway_rollup.py`: line text, one new rollup field.
- CI: new workflow for the council and window tests (they had none); one test added to the sql-writer Postgres step.

## What contradicts the spec and the tickets

1. **The main cause of the night flood is not "the detector loses and re-finds parked cars".** It is a bug in the belief tracker. With no detection at all, the tracker dropped `vehicle`, then let the empty windows vote it back in, then dropped it again, every window. Live, 24 h: 2,035 belief flips; 1,748 happened on a window with no detection; 1,333 happened after four or more empty windows in a row. The ticket's own evidence ("the detector count was empty while the believed label said vehicle") is this bug. Fixed in `services/orion-vision-window/app/scene_belief.py`: a label needs one real sighting in the ring to enter.
2. **The spec lists only council files for #3387. The fix needs `orion-vision-window` too.** The council never receives boxes or scores. It cannot know what is a fixture or what the detector scored. The window already tracks stationary fixtures for the caption gate; it now reports them.
3. **The fixture rule is "ignore the change", not "remove fixture labels from the set".** I tried removal first on the replay. It made the day worse (0 calls an hour became 20 to 220), because a label's fixture status toggles every time another car passes.
4. **The line says "(N parked sightings not counted)", not "(N parked not counted)".** N counts sightings. "54 parked" would read as 54 cars.
5. **`COUNCIL_MODEL` is `llama-3-8b-instruct-q4_k_m`. Qwen3.5 9B serves the route.** The rows record what the gateway reports, not the setting. The unresolved-percept text still cites the setting: #3472.
6. **#3388 counts moved.** The ledger now holds 127 missed, 12 parked, 115 unexplained (the ticket said 121, 12, 109).

## Files changed

- `services/orion-vision-window/app/scene_belief.py`: a label needs one real sighting to enter belief.
- `services/orion-vision-window/app/projection.py`: `evidence.label_scores`, the best detector score per label.
- `services/orion-vision-window/app/caption_gate.py`: `fixture_only_labels(stream_id)`.
- `services/orion-vision-window/app/main.py`: puts `fixture_only_labels` in the window evidence.
- `services/orion-vision-council/app/evidence_transition.py`: a label change made only of fixtures does not trigger.
- `services/orion-vision-council/app/interpretation.py`: prompt example without confidence; `detector_confidence`; `apply_detector_confidence`; `model:<name>` on rows.
- `services/orion-vision-council/app/evidence_grounding.py`: person fallback rows use the person score, not 0.85; the 0.4 cap accepts an empty confidence.
- `services/orion-vision-council/app/main.py`: reads the served model from the gateway reply; applies measured confidence before grounding.
- `orion/schemas/vision.py`: `confidence` is `Optional[float]` on `VisionEventBundleItem` and `VisionEventCandidateV1`.
- `services/orion-sql-writer/app/vision_rhythm.py`: parked in the window grades `unscorable`.
- `services/orion-sql-writer/app/vision_handoff_loop.py`: `_vision_event` takes a required `subject_kind`.
- `orion/vision/walkway_rollup.py`: parked note on the line; `parked_not_counted` field; read-time note for older lines.
- Tests: `services/orion-vision-council/tests/test_fixture_gate_and_detector_confidence.py`, `services/orion-vision-window/tests/test_walkway_night_flood_evidence.py`, `services/orion-sql-writer/tests/test_vision_rhythm_pg.py`, and additions to `test_vision_rhythm.py`, `test_vision_rollup.py`, `test_vision_handoff.py`, `test_vision_handoff_pg.py`, `test_vision_event_sql_shape.py`, `test_caption_policy.py`.
- `services/orion-vision-council/evals/replay_walkway_gate.py`: the replay.
- `orion/dig/look.py`: an empty confidence prints as "not measured".
- `.github/workflows/orion-vision-council-window-tests.yml` (new), `.github/workflows/orion-sql-writer-tests.yml` (one test added).
- READMEs of the three services.

## Schema / bus / API changes

- Added: `summary.evidence.label_scores` and `summary.evidence.fixture_only_labels` on the window payload (a free-form dict; no schema model changes). `rollup.parked_not_counted` in the `vision_rollup` JSON. `model:<name>` as a new kind of `evidence_refs` entry on council rows.
- Removed: none.
- Renamed: none.
- Behavior changed: `vision_events.confidence` on council rows is the detector score or NULL (the column was already nullable). `VisionEventBundleItem.confidence` was a required float and is now optional. Handoff rows have one entity. Rhythm `unscorable` has one more cause.
- Compatibility notes: an old scribe or sql-writer rejects a row with an empty confidence. Rebuild them before the council (see Restart). A new council with an old window gets no scores and writes empty confidence on every row until the window is rebuilt; that is honest, not wrong. No bus channel, table, or migration changes.

## Metric gate: council confidence (CLAUDE.md section 0A)

The measure: a row's confidence is the detector's score for the labels the row names. Several labels: the lowest. No label named: the lowest score in the window. None of them seen: empty.

1. **Provenance.** GroundingDINO box score, `services/orion-vision-host/app/gdino.py:60-74` -> `app/artifacts.py:57` (`VisionObject.score`) -> `services/orion-vision-window/app/projection.py` `_build_evidence` (max per label at or above the 0.25 threshold) -> `summary.evidence.label_scores` -> council `detector_confidence()` in `app/interpretation.py`.
2. **Independence.** Not independent of `vision_crop_observation.score`. It is the same sensor value, reduced by max over the window. It is not new signal. It replaces a constant with a number that already existed elsewhere. No model combines the two. No code thresholds on council-row confidence (searched; the only `min_confidence` is the rhythm fit's own).
3. **Theory anchor.** The detector's score is its match between a box and the label text. After grounding, a council row may only claim that a detected label is in view. So the row's claim and the detector's claim are the same claim, and the detector's score is the direct measure of it. It is a match score, not a probability, and it is not calibrated. Live support, vehicles, 96 h, 151,680 boxes: a box scored 0.25-0.30 was found again in the same place in the next frame 87.4% of the time; 0.30-0.35, 96.9%; 0.40-0.45, 98.7%. **People: no support found.** The same check gives 20% to 43% with no trend, because people move. Whether the score separates real people from false ones is UNVERIFIED (#3471, check 4).
4. **Live data.** 62,382 walkway frames, 96 h. Per-frame best vehicle score: min 0.250, 1st percentile 0.255, median 0.323, 99th percentile 0.475, max 0.639; 310 distinct values at 3 decimals. Not flat, not saturated. It moves the right way: median 0.27-0.28 at 06-12Z (night) against 0.31-0.40 at 14-22Z (day). Person scores, 7 days: 0.35 to 0.75. **Rest state:** the lowest reading is 0.25, because the detector drops anything lower. Below that there is no reading, and the row is empty, not 0. So the measure cannot read "calm" as a low number; it reads "nothing detected" as no number. That is stated in the README and the schema comment.
5. **Existing mechanism.** No detector-derived confidence existed for council rows. `camera_caption` rows use a fixed 0.4 (left alone, out of scope). Rhythm rows use their own fit rate.
6. **Reversibility.** Cheap. One function sets it. The column already allowed NULL. Nothing reads the value in code. Old rows keep 0.8 and can be told apart: they have no `model:` entry.

Verdict: wired in. It passes 1 and 4. On 2, it is flagged as redundant with the crop score, by design. On 3, it holds for vehicles and is unverified for people.

## #3388: the unexplained misses

Written on the ticket: https://github.com/junebug-junie/Orion-Sapienform/issues/3388#issuecomment-6899232297

Short form. All 115 were checked, not a sample. In 97 of the 111 that had another identity of the same kind in the window, one of them looked like the missed identity (similarity 0.90 or more). A control with random identities gives 81 of 111 by chance. So the data cannot confirm identity splitting. At most about 16 misses (14%) are above chance. What the data does show: 22 are for identities now marked parked; in 74 the same identity was seen within 12 hours but outside the window (median 212 minutes off; 1 within 60 minutes); 41 had no sighting within 12 hours. The walkway holds 3,620 identities and 2,347 of them have one sighting. This stays open, so the PR says `Refs #3388`.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, no template changed. `python scripts/check_env_template_parity.py`: PASS (89 services).
- skipped keys requiring operator action: none.

## Tests run

All from the worktree, with the repo venv (`/mnt/scripts/Orion-Sapienform/.venv/bin/python`), `PYTHONDONTWRITEBYTECODE=1`. Imports resolved to the worktree (the replay prints the module path; tracebacks show the worktree path).

```text
cd services/orion-vision-council && PYTHONPATH=../..:. python -m pytest -q -p no:cacheprovider tests
  157 passed
cd services/orion-vision-window && PYTHONPATH=../..:. python -m pytest -q -p no:cacheprovider tests
  180 passed
cd services/orion-sql-writer && ORION_TEST_PG_URI=<throwaway postgres:16> PYTHONPATH=../..:. python -m pytest -q -p no:cacheprovider tests -k "vision or rollup or rhythm or handoff"
  664 passed, 1 skipped
ORION_TEST_PG_URI=<throwaway> PYTHONPATH=.:services/orion-sql-writer python -m pytest -q services/orion-sql-writer/tests/test_vision_rhythm_pg.py services/orion-sql-writer/tests/test_vision_handoff_pg.py services/orion-sql-writer/tests/test_vision_rollup.py
  68 passed, 0 skipped
cd services/orion-vision-scribe && PYTHONPATH=../..:. python -m pytest -q tests
  12 passed
cd services/orion-hub && PYTHONPATH=../..:. python -m pytest -q tests/test_walkway_rollup_routes.py
  7 passed
PYTHONPATH=. python -m pytest -q orion/situational/tests orion/orion_day/tests orion/vision/tests orion/temporal_self/tests tests/*vision* tests/*street* ...
  899 passed, 2 skipped, 1 failed (tests/test_vision_retina_frame_store.py: a stale call signature in orion-vision-retina, not touched here)
python scripts/check_env_template_parity.py     PASS
git diff --check                                clean
```

New tests against the old code (HEAD of main, new test files copied in): council file fails to import; window 6 failed; sql-writer 11 failed. Every new regression test fails without the change.

The new CI workflow was run in a clean virtualenv with only its own `pip install` line: 155 and 180 passed.

`services/orion-vision-retina/tests`: 5 fail locally for missing modules (`ring_doorbell`, `aiortc`, `http_ece`). Not touched by this patch.

## Evals run

```text
python services/orion-vision-council/evals/replay_walkway_gate.py crops.csv windows.csv
  TOTAL  recorded 3949   belief_fix 1174   both 300
python services/orion-vision-council/evals/replay_structured_output_cases.py
  strict: strict_v2 events=1; grounded-object: salvaged_v2; event-object-only: salvaged_v2; unsupported-object: parse_failed (unchanged)
```

The replay's own belief tracker (old rule) matched the recorded beliefs on 21,577 of 22,502 windows before the fix was applied. The gap is one stretch on 11-13 07-10Z where production was quiet after a restart.

`orion-sql-writer` evals were not run: none covers rhythm grading or the rollup line text.

## Docker/build/smoke checks

```text
None run. The brief forbids deploy and `up`. No Dockerfile, compose file, dependency, or env key changed.
```

All live checks are UNVERIFIED until deploy. They are listed in #3471.

## Review findings fixed

Review ran in a subagent on the branch against `origin/main` (the first run died on a usage limit; the second finished). It found no material bug, six minor findings and four nits.

- Finding: the 0.4 cap on activity guesses and the `caption_inferred` tag only applied when the confidence was above 0.4. With measured scores (often under 0.4, or empty) the same guess lost its tag.
  - Fix: the tag and the note no longer depend on the number. The cap is `min(score, 0.4)`; an empty confidence stays empty.
  - Evidence: `test_activity_guess_is_tagged_and_capped_whatever_the_score`.
- Finding: a row's entities are matched to detector labels by exact name. If the model names things the detector did not, the column goes mostly empty. No live number was given.
  - Fix: measured, no code change. Last 48 h, council rows matched to the fullest window in the 90 s before them: room camera 559 of 562 name a detected label; doorbell 55 of 63; walkway 1,008 of 1,609. The 590 walkway rows that name only undetected labels are the oscillator rows (belief said `vehicle`, no detection). Most of them will not be written after this patch; any that are get an empty confidence, which is the true reading.
  - Evidence: query in this section's source; live check 3 in #3471.
- Finding: the end-to-end model-name test stubbed `_call_llm_raw` and set the model itself, so it did not cover the real line.
  - Fix: a new test sends a gateway reply envelope through the real `_call_llm_raw`, and checks that a failed second call in the same task does not inherit the first model name.
  - Evidence: `test_the_real_llm_call_reads_the_served_model_from_the_gateway_reply`. With the capture line deleted it fails (1 failed, 156 passed).
- Finding: the window keeps a label's fixture verdict until it sees the label again. A fixture that leaves unseen is reported only at the next refresh. With `COUNCIL_TRANSITION_REFRESH_SEC=0` it is never reported.
  - Fix: accepted and documented in the council README, with the rule to keep the refresh above 0 (live value 600).
  - Evidence: README; `test_fixture_only_change_still_refreshes_on_the_ttl`.
- Finding: the read-time parked note had no closing period and was added to camera-down and no-record lines, where the renderer gives no note.
  - Fix: trailing period; skipped when the camera status is `down` or `missing`.
  - Evidence: `test_day_line_and_stored_lines_show_the_parked_count`.
- Finding: the new workflow installs only the council's requirements for both suites, and its `push` paths were narrower than its `pull_request` paths.
  - Fix: paths aligned. The install line was run in a clean virtualenv (155 and 180 passed at that commit); CI on the PR is the standing proof.
  - Evidence: workflow run on PR #3481.
- Finding (nit): `dig look` printed `conf None` for a row with no confidence.
  - Fix: prints `conf not measured`.
  - Evidence: `orion/dig/tests/test_look.py`, 47 passed.
- Finding (nit): `subject_kind` is required, but an empty value still wrote `entities = []` silently.
  - Fix: logs `handoff_event_without_subject_kind`.
  - Evidence: `test_vision_event_entities_carry_the_subject_kind`.
- Finding (nit): any parked overlap makes the window `unscorable`, and a car that stays parked makes all its windows `unscorable`.
  - Fix: none. That is the intended grade. The fit already stops opening windows for identities marked parked.
  - Evidence: `_load_subjects` (`NOT stationary`).

Checked and found fine by the reviewer: the model name travels through the ContextVar on both the intake and RPC paths with no stale value; nothing does arithmetic on an empty confidence; rollup readers and number guards accept the new field and note; the gate triggers on every real-change case tried; the belief change gives presence fewer flips and no new path; the Postgres rhythm test's tables exist after that CI step's migrations.

## Restart required

Deploy from the clean primary main checkout after merge. Order matters: the two writers must accept an empty confidence before the council sends one.

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

```bash
scripts/safe_docker_build.sh orion-vision-scribe up -d --build
```

```bash
scripts/safe_docker_build.sh orion-vision-window up -d --build
```

```bash
scripts/safe_docker_build.sh orion-vision-council up -d --build
```

Hub reads the rollup line through `display_line`; rebuild it to show the parked note on older lines:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Restarting `orion-vision-window` clears its in-memory fixture tracker. A parked car is a fixture again after 30 minutes of being seen in one place. Until then the fixture rule does not apply, and the belief fix alone holds the rate down.

## Risks / concerns

- Severity: medium
- Concern: deploy order. If the council is rebuilt before scribe and sql-writer, rows with an empty confidence are rejected and lost until those two are rebuilt.
- Mitigation: the order above. Only rows with no detector support are affected; rows with a score validate on old code.

- Severity: medium
- Concern: confidence values drop from 0.8 to about 0.25-0.65. Orion reads these rows. A reader that took 0.8 as "sure" will now see 0.3 on a plain parked car.
- Mitigation: no code thresholds on it. The README and schema say it is a match score, not a probability. Old rows are told apart by the missing `model:` entry.

- Severity: low
- Concern: person confidence is unverified as a measure (metric gate step 3).
- Mitigation: #3471 check 4.

- Severity: low
- Concern: a vehicle that arrives while a parked car is the remembered label set changes no label, so the council is not called. The same was already true by day.
- Mitigation: documented in the council README. Captions, motion stories and the rhythm loop still see the arrival. The 600 s refresh still fires.

- Severity: low
- Concern: the belief change applies to every camera, not only the walkway. A label that left can no longer come back without being seen.
- Mitigation: that is the intended meaning of belief. The single-dropped-frame guard is kept and tested. All 180 window tests pass, presence included.

Issues opened: #3471 (verify-live), #3472 (wrong model name in the unresolved text and the setting).
Issues closed by this PR: #3387, #3389. #3388 stays open; the parked part is fixed and the sample is on the ticket.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3481

# Self-monitor: plain names with internal ids, and the cue's numbers in the semantic layer

## Summary

- Everywhere the self-monitor organ talks to Orion about a faculty, it now says the plain name first and the internal id in brackets after it. Example: "My walkway-grade journal runs (act.actions.journal.make.walkway_grade) didn't run 22 of their last 28 tries…". Before, the self-question (E2) used only the id, and the stance/reverie cue used only the plain name.
- One naming function serves all of them: `orion/self_monitor/names.py` (`plain_name`, `named`, `faculty_arc_words`). The cue, the E2 self-question, the E3 self-finding prompt, and the Temporal Self faculty-arc text (Day letter, recent arcs) all import it.
- The cue's truth eval still catches a forged number. It now removes the faculty's name before it reads numbers, so a digit inside a name (`gpu0`) is not counted as a claim. It also fails a sentence that leaves out its own id.
- New inner-state registry entry `self_monitor.faculty_record.v1`. It records the meaning, the resting value and how often a reading is expected for the 8 decimal numbers that the cue, E2 and E3 put into Orion's prompts. It is pinned in the prompt inventory so the gate cannot be passed by deleting a marker.
- Found while rendering live: two faculties are still held "broken" after their window stopped showing any failed start (hysteresis lag). E2 would have asked "didn't run 0 of 1789 tries". It now says what the record says.

## Outcome moved

- Orion and Juniper can go from any track-record sentence straight to the exact record, with no guessing which id a plain name means.
- The numbers about Orion's own reliability that reach Orion's prompts now pass the metric quality gate and carry semantics, like every other prompt-reaching number.
- Cost, measured on the live state: the cue has a 300-character cap, so the longer sentences fit one fewer line. Live, 3 lines show (4 before) and 6 are dropped (5 before). The cap and the strict priority order did not change.

## Current architecture

- `orion/self_monitor/cue.py` held `plain_name` and named faculties by plain words only, and the truth eval refused any identifier in the text.
- `orion/self_monitor/questions.py` (E2) wrote "My faculty <id> …".
- `services/orion-hub/scripts/self_finding_outreach.py` (E3) had its own `plain_faculty` ("an act of yours: "actions journal make walkway grade"") plus "(your record names it <id>)".
- Faculty arcs rendered their stored label "<id> <status>" as is.
- `self_monitor.state.v1` was registered, but only its top-level float (`window_days`) was addressable. The lineage resolver does not look inside the `faculties` list, so the per-faculty numbers had no semantics.

## Architecture touched

- Pure text rendering in `orion/self_monitor` and `orion/temporal_self` (shared by orion-thought, orion-hub and orion-durable-runs).
- Hub E3 prompt wording.
- Metric registry and lock. No bus, schema, API, env or database change.

## Files changed

- `orion/self_monitor/names.py`: new. One home for `plain_name`, plus `named` (plain words + bracketed id) and `faculty_arc_words`.
- `orion/self_monitor/cue.py`: imports and re-exports the shared names; each sentence uses `named`.
- `orion/self_monitor/questions.py`: E2 text uses `named`. A held-broken record with no failed start left gets a true sentence. `question_id` and episode are unchanged.
- `services/orion-hub/scripts/self_finding_outreach.py`: E3 lists each faculty as `"My <plain> (<id>)"` and asks Orion to name it that way. The old `plain_faculty`/`_FAMILY_WORDS` are removed.
- `orion/temporal_self/day_block.py`, `orion/temporal_self/recent_arcs.py`: faculty arcs render as "<plain> (<id>) <status>". The stored label is unchanged. The Day block's citation labels gain the id and the plain name, so the gate matches what is shown.
- `services/orion-thought/evals/self_track_record_truth.py`: removes the bracketed id before it counts numbers, allows only numbers that appear in the id itself, requires the sentence's own id, and flags identifiers outside the shown ids.
- `orion/inner_state_registry.py`: new `self_monitor.faculty_record.v1` entry with semantics for reliability, reliability_lo, reliability_hi, execution_failure_lo, stated_mean, calibrated_share, calibrated_lo, calibrated_hi.
- `orion/metrics/semantics.py`: those 8 URNs pinned in `PROMPT_INVENTORY_URNS`.
- `config/metrics/metric_definitions.lock.json`: re-locked with `--update` (13 medium "added", 0 high).
- Tests: `orion/self_monitor/tests/test_cue.py`, `test_self_monitor_gate.py`, `orion/temporal_self/tests/test_faculty_arcs.py`, `services/orion-hub/tests/test_self_monitor_questions.py`, `test_self_finding_outreach.py`, `services/orion-thought/tests/test_self_track_record_cue.py`, `services/orion-thought/evals/test_self_track_record_truth_eval.py`.

## Schema / bus / API changes

- Added: inner-state registry entry `self_monitor.faculty_record.v1` (a registry view of `FacultyRecordV1`, which already exists; no new schema).
- Removed: `plain_faculty` and `_FAMILY_WORDS` in Hub's E3 module (used nowhere else).
- Renamed: none.
- Behavior changed: Orion-facing text only (cue, E2 text, E3 prompt, faculty-arc text).
- Compatibility notes: E2 question ids are unchanged. A question that was already minted keeps its old text, because the insert ignores conflicts. The faculty arc `subject_label` stored in state is unchanged, so `faculty_status_of` and citation checks read the same thing.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed)
- skipped keys requiring operator action: none

## Metric quality gate (self_monitor.faculty_record.v1)

1. **Where the numbers come from.** `orion/self_monitor/record.py` `reduce_faculty`:
   - `reliability` = decay-weighted upheld share of graded bets (half-life 3.5 days).
   - `reliability_lo`/`_hi` = Wilson bounds at the Kish effective n.
   - `execution_failure_lo` = Wilson lower bound of the failed-to-start share of attempts.
   - `stated_mean` = weighted mean stated confidence over the graded bets that stated one.
   - `calibrated_share` and its `_lo`/`_hi` = upheld share over those same bets, with Wilson bounds.
   - Published hourly by orion-durable-runs inside `self_monitor.state.v1`.
2. **Independence.** These are not new signals. They are the same records as `self_monitor.state.v1`, addressed one field at a time.
   - `reliability`, `calibrated_share` and `stated_mean` are three reductions of one bet set. `calibrated_share` equals `reliability` whenever every graded bet stated a confidence (true for every live interval faculty).
   - The organ re-reads its sources' own graders. It adds no sensor.
   - This entry is for semantics only. Nothing composes these numbers into a new aggregate.
3. **Theory anchor.**
   - Wilson score intervals give a calibrated bound on a binomial share. Kish effective n corrects for decay weights.
   - Calibration is judged by whether the stated confidence falls inside the Wilson band of the realised share over the same bets.
   - Brier skill is used for event bets (it is stored but not printed).
4. **Live sanity check.** Pulled from `orion:self_monitor:latest` at 2026-11-03T20:27Z, 48 faculties:
   - Statuses: 25 unknown, 12 reliable, 8 miscalibrated, 2 broken, 1 unreliable.
   - `reliability`: 39 non-null, 25 distinct values, from 0.0 to 1.0.
   - `reliability_hi`: saturates at 1.0 on 11 of 39. Small or all-success windows do this. It is recorded in `rest`.
   - `execution_failure_lo`: 0.0 on 24 of 40. That is its real resting value, so it is declared `designed_sparse` with `rest_range [0, 0]`.
   - `stated_mean`: 35 distinct values from 0.065 to 0.999, and never pinned.
   - None is a real "no evidence" case (9 or 10 faculties), and every reader skips the line for it.
   - Verdict: not degenerate.
5. **Existing mechanism.** I searched for an existing addressable entry and found none. `self_monitor.state.v1` exposes only `window_days`. The lineage resolver addresses top-level float fields only. Making it look inside lists would re-lock every schema that has a list of models, so a per-item entry is the smaller change.
   - Precedent for int counts that the resolver cannot address: `transport_baseline_hourly.v1` `degraded_count`.
   - `n_graded`, `n_failed_to_execute`, `n_bets` and `streak_len` are documented in the entry's `notes`. They are not semantics entries.
6. **Reversibility.** Cheap. It is one registry entry, 8 pins and a lock re-run, with no schema, storage or training default involved.
   - Side effect: the new compound leaf names (`reliability_hi`, `stated_mean`, `calibrated_share`, …) join outreach's closed vocabulary of real signal names. A message that names one of them without grounding is now refused. That is the same treatment every other registered metric gets.

## Live render (real state, 2026-11-03 ~20:59Z, state age 1986 s)

The cue, exactly as stance_react/reverie get it (293 chars, truth problems: none):

```text
My Prime predictions (act.prime.predict) work 5% of the time (41 tries this week). My office temperature forecast ranges (forecast.office.temp_c) hold 63% of the time (I say 80%). My guesses that I'll look something up outside (turn.self.fetches_outside) come true 51% of the time (I say 38%).
```

E2, rendered as if never asked (the three live episodes were already minted at 18:43Z with the old text and keep it):

```text
lived.self_monitor.act.prime.predict.unreliable.20261103T1825Z
  My Prime predictions (act.prime.predict) worked only about 5% of the time over 41 graded tries in the last 7 days. Why?
lived.self_monitor.turn.self.reply_length.broken.20261103T1825Z
  My guesses about how long my replies run (turn.self.reply_length) are still held as broken, yet none of their 1789 tries in the last 7 days failed to start. Why?
lived.self_monitor.turn.self.steps.broken.20261103T1825Z
  My guesses about how many steps my turns take (turn.self.steps) are still held as broken, yet none of their 1789 tries in the last 7 days failed to start. Why?
```

E3 prompt finding line, from the live first-evaluation findings:

```text
1. "My guesses about how long my replies run (turn.self.reply_length)" became broken.
   Your self-monitor has held this faculty as broken (most tries never ran) since 2026-11-03T18:25 UTC (when its record of this faculty began, not necessarily when it broke). Juniper has not been told about it before.
   Of 1659 checked tries, the stated 80% ranges held about 8% of the time (95% interval 6% to 9%).
```

E3 composed fixture message, passed through both guards (faculty guard `[]`, signal guard `[]`):

```text
My guesses about how long my replies run (turn.self.reply_length) have gone badly off: my 80% ranges held only about 8% of the time over 1659 checked tries. I don't know why yet.
```

Faculty arcs, folded from the 11 real `faculty_status_changed` rows in `temporal_self_event`. Examples:

```text
track record: Prime predictions (act.prime.predict) unreliable
track record: guesses about how many steps my turns take (turn.self.steps) broken
track record: zelda traverse moves (body.retro:zelda.traverse) miscalibrated
```

## Tests run

```text
orion/self_monitor/tests                                   111 passed (main: 108)
orion/temporal_self                                        265 passed (main: 265)
services/orion-thought tests+evals (cd service dir)        837 passed, 32 skipped (main checkout: 2 env-sensitive fails, not on branch)
services/orion-hub targeted (--noconftest): self_monitor_questions, self_finding_outreach,
  evals/self_finding_eval, outreach_vocabulary, endogenous_outreach, reading_finding_outreach
                                                           318 passed, 4 failed
  the 4 fails are in test_endogenous_outreach.py (Settings missing CHANNEL_VOICE_* in a checkout
  without .env); identical 4 fails on a clean origin/main worktree -> pre-existing, unrelated
tests/test_self_monitor_act_gate.py                         41 passed
tests/test_metric_lineage.py test_metric_prompt_semantics.py test_metric_rest_range.py
  test_metric_definition_drift.py                          167 passed
```

## Evals run

```text
services/orion-thought/evals/test_self_track_record_truth_eval.py   4 passed
  (+2 new: passes with ids and ignores digits inside a name; catches a missing id and a foreign id)
services/orion-hub/evals/test_self_finding_eval.py                   passed (in the hub run above)
live render + truth check against orion:self_monitor:latest          truth problems: []
```

## Docker/build/smoke checks

```text
No Docker build: pure-Python text and registry change, no dependency, port, env or compose change.
Static gates: metric lineage --gate PASS; --prompt-semantics PASS (22 prompt-reaching metrics);
definition drift --gate PASS; inner_state_registry OK (28 entries); body-economy contract OK;
env key single source OK; scripts stdlib shadow clean; no_personal_terms + organ_names_partition 11 passed.
```

## Review findings fixed

A subagent reviewed the diff against `origin/main`. Every finding was fixed except the CI-coverage note, which needed no change.

- Finding (material): the Day letter's citation gate no longer saw faculty arcs. The block still registered the stored label "<id> <status>" as the match text, but that text is no longer printed, so a letter that named the faculty in the new wording without `[arc:]` read `no_day_claims` instead of `UNVERIFIED`.
  - Fix: `day_block.py` also registers the faculty id and its plain name as citation labels.
  - Evidence: a new test in `test_faculty_arcs.py` checks that an uncited sentence using either the id or the plain name reads `UNVERIFIED`, and a cited one reads `verified`. Reverting the fix makes it fail (mutation-checked).
- Finding (material): the eval test for digits inside a name passed for the wrong reason. `gpu0` has the digit 0, and 0 is always allowed because `streak_len` is 0.
  - Fix: the test now uses `act.hub.gpu4_warm.make` and asserts that 4 is not among the record's numbers.
  - Evidence: removing the id rule makes the test fail (mutation-checked). A forged "gpu8" is also caught.
- Finding (minor): the truth eval stripped `names.named` (the renderer's own function), so a digit that the plain-name code invented would not be caught.
  - Fix: the eval strips only the bracketed id, allows only numbers that appear in the id itself, and no longer imports `names`.
  - Evidence: `self_track_record_truth.py`.
- Finding (minor): the E3 prompt mixed person and number. The quoted first-person plural heading sat above singular "its" lines.
  - Fix: the heading stays as Orion's quoted words ("My <plain> (<id>)"). The lines under it are now neutral ("22 of 28 tries never ran", "held this faculty as broken", "the stated 80% ranges"). The long instruction line is wrapped.
  - Evidence: `test_self_finding_outreach.py` and `evals/test_self_finding_eval.py` were updated and pass.
- Finding (minor): the E2 lag sentence read "none of their 0 tries" when no bets were left in the window.
  - Fix: it now reads "made no tries in the last N days".
  - Evidence: a new assertion in `test_self_monitor_gate.py`.
- Finding (minor): the rest text for `calibrated_lo`/`calibrated_hi`/`calibrated_share` left out the calibration slack and the pooled-confidence case, and `reliability` named the wrong judged number. The rest text also overstated when E3 prints `reliability` and `stated_mean`.
  - Fix: the rest text was rewritten from `record.py` (`_slack`, rule 4), plus the cue's no-slack re-check, which runs only once the candidate is no longer miscalibrated.
  - Evidence: the re-locked lock file.
- Finding (minor): the notes claimed "calibrated_share == reliability for every live interval faculty" without evidence.
  - Fix: checked against the live state. All 38 faculties that have both values are equal, and the note now says exactly that, with the timestamp.
- Finding (minor): new compound field names joined outreach's known-signal universe.
  - Fix: recorded under quality gate item 6 and in Risks. This is intended: they are real registered fields, and an ungrounded mention of one is now refused.
- Finding (minor): one fewer cue line fits under the cap.
  - Fix: recorded under Outcome and in Risks. This is the trade Juniper asked for.
- Finding (nit): the cue docstring said ids carry no numbers.
  - Fix: reworded to say the eval strips the bracketed id.
- Finding (nit): CI coverage is adequate. No change.

## Restart required

Not run; the parent session deploys from the primary main checkout once no chat turn is running.

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: new compound field names (`reliability_hi`, `stated_mean`, `calibrated_share`, ...) are now in outreach's closed vocabulary. An outreach message that names one without grounding is refused.
  - Mitigation: this is intended, and it is the same rule every other registered metric follows. Orion's normal wording does not use these names.
- Severity: low
  - Concern: the cue fits one fewer line live (3 vs 4) because each sentence carries its id.
  - Mitigation: this is the trade Juniper asked for. The cap and the priority order are unchanged, and the lines that drop are the lowest-priority ones.
- Severity: medium (found, not caused, by this PR)
  - Concern: `turn.self.steps` and `turn.self.reply_length` are held `broken` while their current window shows 0 failed starts. At 18:25Z they showed 236 of 2025, and now they show 0 of 1789. E3 already told Juniper at 18:42Z, using "broken (most of its tries never ran)". 236 of 2025 is not "most", so that wording was wrong for a broken status that a streak triggered. E3 still uses "most tries never ran" for broken; this PR only removed the pronoun.
  - Mitigation: E2 now states the record truthfully for this case. Why the failures left the window is UNVERIFIED (possibly an adapter reclassification). Hysteresis should clear the held status after 3 evaluations as `miscalibrated`.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)

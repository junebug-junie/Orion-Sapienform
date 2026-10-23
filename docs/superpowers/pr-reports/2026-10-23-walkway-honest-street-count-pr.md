## Summary

Spec PR #2763, Item 2: patch B, plus decisions 5 and 6. This builds on PR #2781 (rollup people ranges).

- The street line that reaches every chat turn no longer says "13 unfamiliar people". It now gives unnamed people as a range: "at least 2, at most 14 people I have no name for (...; the camera splits one person into several)". The low end is the most people in one camera window in the same 15 minutes. The high end is the person clusters.
- The patio head count now reads "At least 3 people are on the patio." It is one frame's count, so it is a floor.
- Decision 6 (Juniper, 2026-10-23) is now a rule. A one-frame glimpse story counts as a person only when it is clearly a person (`orion/vision/walkway_count.py`, `clear_glimpse`). The thresholds come from the stories Juniper checked in the Hub.
- A new read-only eval scores the count estimators against hand-counted hours. No hand counts exist yet, so it says INSUFFICIENT LABELS and lists the 10 hours most worth counting.
- Gates: an empty hour reads 0, and a person standing 2 minutes stays 1. The eval runs them on synthetic data, and on live data where live data exists.

## Outcome moved

- Street line, read-only replay of 673 moments (every 15 min, last 7 days):
  - Before (origin/main): 55 who-is-around or patio lines stated a bare people count. The largest was "13 unfamiliar people" (2026-10-22 00:09 UTC).
  - After: 0. That moment now reads "at least 2, at most 14 people I have no name for (some seen here on other days; the camera splits one person into several)". The high is 14, not 13, because people seen on other days are now in the same phrase.
- Count estimators, 168 live hours: clusters 427, most at once 107, stories 104, stories held in range 148, multi-frame stories only 69, sum of run peaks 275, runs 206. With no hand counts, this cannot say yet which estimator is right.

## Current architecture

- `perception_reader._who_line` counted unnamed person individuals from `vision_individual_sighting` and printed "N unfamiliar people". It printed people seen on 2+ days as a separate phrase, "N people I have each seen on D days". Both numbers are clusters, and the matcher splits one person into several.
- Rollup lines already used a range (#2781). The street line did not read rollups.
- Motion stories gave glimpses `drop_reason=glimpse` (fewer than 3 detections). Glimpses counted toward nothing (decision 6 was open).

## Architecture touched

- `orion/situational/perception_reader.py` (runs in cortex-exec and hub turn assembly) has two new reads, each with its own failure:
  - `vision_scene_inventory.counts` over the 15-minute window. This is the floor. `walkway_rollup.window_peak` maps compound labels the same way the rollup does.
  - The count of stationary person individuals in that window, taken off the floor like the rollup's props.
  - If the window read fails, the line can only say "up to N".
  - Live cost: 1.0 ms and 3.1 ms (`EXPLAIN ANALYZE`; index `vision_scene_inventory_stream_observed_idx`).
- The people phrase:
  - low = `count_range(peak, clusters incl. named, props).low` minus the named people listed. high = unnamed clusters, raised to low.
  - "an unfamiliar person" when low = high = 1. "at least N" when low = high. "possibly an unfamiliar person" or "up to N" when low is 0 or unread.
  - People seen on other days become "some seen here on other days" (no count). One person who is surely one keeps "a person I have seen on 2 days".
  - A cut list that hides the people range ends ", and others", never a number.
- `orion/vision/walkway_count.py` is new and pure. It holds the estimators, `clear_glimpse`, `story_count`, `presence_runs` and `within_range`. It has no settings, no schema column, and no change to the rollup JSON.
- **Where decision 6 changes counts.** Only in the `stories` estimator and in this eval. The Orion-facing range does not change:
  - The low stays the window count, which is a frame fact. A glimpse cannot be more certain than the frame it came from.
  - The high stays the clusters. Each glimpse detection already made a sighting, so a glimpse is already in the clusters. Adding glimpses would count them twice.
  - A story-based count may reach Orion-facing text only through `within_range`, held inside [most at once, clusters]. The live data shows why: in 10-20 10:00 MDT the raw story count (2) was below the most-at-once (3).

## Metric gate (CLAUDE.md 0A): "clearly a person" glimpse rule

1. **Provenance.** `services/orion-sql-writer/app/vision_motion.py` `build_stories` sets `drop_reason="glimpse"` when a passage has fewer than `min_detections` (3) boxes. Each `evidence[]` item has the detector `score`, the `box`, `clipped` (`crop_clipped`) and `zone` (`zone_for_box`). These come from the crop rows. The rule reads only these fields (`walkway_count.clear_detection`).
2. **Independence.** It is not independent. Clusters, most at once, stories and presence runs are four readings of the same detector boxes. This rule is a filter on that one sensor, not a new signal.
3. **Theory anchor.** A passage is one continuous track. A glimpse that looks like a detection from a hand-confirmed passage, with no counted track within 11 s, is a passer-by the tracker saw once. Within 11 s it is flicker beside a track. 11 s is the gap the motion reducer closes a story on.
4. **Live data** (walkway person stories, `vision_motion_story` and `vision_motion_story_verdict`, read 2026-10-23):
   - Juniper's Hub checks: 40 stories marked "right", 2 marked "wrong_other".
   - Best detection of a "right" story: p10 0.407, p25 0.437, p50 0.489. Rule: score >= 0.44.
   - Detections in "right" stories: box height p5 121 px, p10 149 px of a 1920 px frame. Rule: height >= 6.25 % of the frame (120 px).
   - The 2 "wrong" stories: scores 0.36-0.40, boxes 55-77 px (p5-p50). Both thresholds exclude them.
   - Glimpse population: 206 walkway person glimpses (10-17..10-23), 1.17 detections each, score p10 0.359 and p50 0.405.
   - In the 7-day eval window: 35 clear and counted, 15 clear but covered by a track, 156 unclear.
   - Rest state: 100 live camera-up hours with no person in any window read 0 for `stories` and `stories_in_range`.
   - A live person standing 2+ minutes: none in the last 7 days. That gate is synthetic only (UNVERIFIED live).
   - Weak spot: the anchor is 40 right and 2 wrong stories. The 2 wrong stories are too few to bound the false-positive rate.
5. **Existing mechanism.** The rule reuses `drop_reason`, the story evidence, `count_range`, `peak_at_once` and the rollup label mapping (`window_peak` is a public wrapper of `_peaks`). Nothing is rebuilt.
6. **Reversibility.** The rule is constants in one pure module. There is no env key, no column and no rollup JSON field. Removing it changes only the eval's `stories` column.

## Files changed

- `orion/situational/perception_reader.py`: the people range in the who line, two new reads, and the patio "at least".
- `orion/vision/walkway_count.py` (new): the estimators and the decision-6 rule.
- `orion/vision/walkway_rollup.py`: public `window_peak`.
- `services/orion-sql-writer/evals/run_walkway_count_estimator_eval.py` (new): the read-only eval.
- `services/orion-sql-writer/evals/fixtures/walkway_count_labels.csv` (new): the labels format, with no rows. None were invented.
- `services/orion-sql-writer/evals/test_walkway_count_estimator_eval.py` (new): offline eval tests. Added to the sql-writer CI list.
- `services/orion-cortex-exec/evals/run_street_who_line_eval.py`: now checks for bare people counts. It treats "at least L, at most H" as one phrase.
- `services/orion-sql-writer/README.md`: the eval and the glimpse rule.
- `.github/workflows/session-scope-tests.yml`: runs `orion/situational/tests` when `orion/vision/walkway_rollup.py` changes, because the street line now imports it.
- Tests: `orion/situational/tests/test_street_people_range.py` (new), `orion/vision/tests/test_walkway_count.py` (new), and updated expectations in `test_street_who_line.py`, `test_street_summary.py` and `test_street_material.py`.

## Schema / bus / API changes

- Added: none. No bus, schema or registry change.
- Removed: none.
- Renamed: none.
- Behavior changed: the text of `PerceptionContextV1.street_summary` (people as a range; the patio count says "At least N").
- Compatibility notes: the field is still a plain string. Consumers (`chat_stance`, hub turn orchestrator) read it as text.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/situational/tests orion/vision/tests services/orion-sql-writer/evals/test_walkway_count_estimator_eval.py   303 passed (after review fixes)
pytest orion/situational/tests                                        250 passed (before review fixes)
pytest services/orion-sql-writer/tests/test_vision_rollup.py orion/vision/tests \
  services/orion-sql-writer/evals/test_walkway_reducers_eval.py \
  services/orion-sql-writer/evals/test_walkway_count_estimator_eval.py \
  services/orion-sql-writer/tests/test_vision_handoff.py              148 passed, 1 skipped
(cortex-exec) pytest tests/test_chat_stance_street_context.py tests/test_situation_perception_context.py   59 passed
pytest tests/test_no_personal_terms.py tests/test_no_private_eval_artifacts.py tests/test_gitignore_no_bare_catchall.py   13 passed
python scripts/check_metric_lineage.py --gate                         PASS
git diff --check                                                      clean
```

## Evals run

```text
PYTHONPATH=. python services/orion-sql-writer/evals/run_walkway_count_estimator_eval.py --days 7
  synthetic: empty hour reads 0 (all estimators); standing 2 min stays 1 (all but clusters=5)
  live: 168 rollup hours; 100 empty hours read 0; 0 standing stories of 2+ min
  glimpses: clear+counted=35 clear-but-covered=15 unclear=156
  VERDICT: INSUFFICIENT LABELS (0 of 10) -- lists 10 hours to count, e.g.
    Tue 10-20 10:00 MDT: clusters=14 most_at_once=3 stories=2 stories_in_range=3 run_peaks=7
    Mon 10-19 11:00 MDT: clusters=13 most_at_once=2 stories=7 run_peaks=15
  PASS (gates)
python services/orion-cortex-exec/evals/run_street_who_line_eval.py          PASS
python services/orion-cortex-exec/evals/run_street_who_line_eval.py --live   PASS
Read-only street replay, 673 moments / 7 days: bare people counts in who/patio lines 55 -> 0
```

## Docker/build/smoke checks

```text
No image build: no dependency or Dockerfile change. Every affected image already COPYs the whole orion/ package.
scripts/rebuild_affected_services.py for the changed paths: orion-actions, orion-cortex-exec,
orion-durable-runs, orion-hub, orion-sql-writer, orion-substrate-runtime, orion-thought.
```

## Review findings fixed

The code-review subagent ran on `origin/main...HEAD`. It found no must-fix items. It found 3 should-fix items and 5 nits.

- Finding: if the stationary-people read failed, the line treated that as "0 props". The patio skeleton then stayed in the floor, and "at least L" could overstate.
  - Fix: when the props read fails, the window floor counts as unread, and the line falls back to "up to H".
  - Evidence: `test_fetch_with_the_props_read_failing_says_up_to`.
- Finding: `prop_ids` was turned into a set again at each level. A generator argument would be used up after the first check, and the prop guard would then be skipped silently.
  - Fix: a `frozenset` is built once in `clear_detection`, `clear_glimpse` and `story_count`.
  - Evidence: `test_prop_ids_may_be_a_generator`.
- Finding: a prop seen as a person in every window would make the live empty-hour gate check 0 hours, with no warning.
  - Fix: the eval prints a WARNING when no empty hour exists although camera-up hours do. Live today: 100 empty hours checked.
- Nits fixed:
  - The hand-check list prints the real zone name (`%Z`), not a hardcoded "MDT". MST starts 11-01.
  - The verdict says the best estimator was picked on the same hours the sign test uses.
  - New dedupe boundary test: a glimpse at end + 11 s is covered, at end + 12 s it is counted.
  - A comment says the named-person subtraction is conservative.
- Nit left as is: the eval's default DSN has local credentials inline. This is the same pattern as `run_vision_rollup_replay_eval.py`, and `POSTGRES_URI` or `--pg` overrides it.

## Restart required

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

The post-pull rebuild covers the other importers. Only cortex-exec and hub build the street line.

## Risks / concerns

- Severity: medium
  - Concern: camera captions in the same street line can still state a people count from one frame ("Two persons are observed walking"). The replay found 3 such caption lines in 7 days. They come from `street_material`'s caption line, which another agent is editing now.
  - Mitigation: follow-up. Either the caption line prefixes counts with "at least", or the guard covers captions.
- Severity: low
  - Concern: other kinds are still cluster counts ("6 unfamiliar dogs"). They over-count the same way.
  - Mitigation: follow-up, if Juniper wants decision 5 for every living kind.
- Severity: low
  - Concern: the clear-glimpse thresholds rest on 40 right and 2 wrong Hub checks. Hand counts are needed before any estimator is trusted.
  - Mitigation: the eval lists the 10 hours to count. The rule does not reach Orion-facing text.
- Severity: low
  - Concern: the floor subtracts named people. If a named person was not in the busiest window, the floor is lower than it could be.
  - Mitigation: none needed. A low floor is still true.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)

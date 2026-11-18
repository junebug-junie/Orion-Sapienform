## Summary

- The street now reaches Orion's mind, always labelled as the street: the latest walkway camera captions go into the chat situation brief ("On the walkway just now: ..."), into reveries as a separate "OUTSIDE, ON THE WALKWAY (... NOT your room)" section, and into curiosity's study material ("WHAT THE STREET CAMERA SAID LATELY").
- Graded street predictions (did the expected dog/van/person come, plus that subject's 14-day hit rate) now reach reveries and dreams. In dreams they are replay material only, never sleep pressure.
- Endogenous curiosity now reads the walkway camera's own surprise node (`node:substrate.perception_walkway`). Attention still does not: the one shared fence became two per-consumer fences.
- The room readers are unchanged and still room-only; nothing street-side is merged into the room percept or seen by the room-only reverie expectation judge.
- Glossary entry for the walkway node (it now has a consumer), metric-definition lock re-locked for that one addition.
- Follow-up for sibling PR #2683 (patio fingerprinting on, stationary fixtures): the street summary leaves stationary sightings (the patio skeleton) out of "who is around", finds the patio head-count row by `head_count_zone` (falling back to `no_embed_zone`), and its privacy comments now say patio people can be individuals.

## Outcome moved

Before: the walkway camera wrote captions and graded predictions that only the nightly walkway forecast journal read. Reverie and dreams were blind to the street, chat saw only who was recognised (no scene), curiosity saw only things the camera could not name, and the walkway surprise node had no consumer at all.

After: four consumers read the street, each under a street label, each fail-open (a failing street read costs only the street line). Live read-only smoke against the real database, 2026-10-15 ~05:35 UTC:

```text
BRIEF   StreetSummary(stream_id='walkway', lines=['On the walkway just now: A person is present in the frame; A person is visible in the frame.', 'Someone is on the patio.'], read_ok=True)
REVERIE captions [{'narrative': 'A person is present in the frame.', 'age_sec': 81.6}, {'narrative': 'A person is visible in the frame.', 'age_sec': 185.4}]
REVERIE graded   []      <- expected: no expectation has 5 days of support yet
ROOM    [{'narrative': 'Two doors, two tables, one chair, and one desk are present in the scene.', ...}]   <- room reader unchanged, no street rows
```

Graded predictions will stay empty for about a week. The rhythm reducer only emits an expectation after 5 or more days of support; on 2026-10-15 the walkway has 4 sightings on 1 day and `vision_percept_expectation` has 0 met/missed rows. All graded paths are tested with fixtures only. Live: UNVERIFIED until ~2026-10-20.

## Current architecture

- `vision_events` holds every camera's narratives. Room readers (`orion/situational/perception_reader.py` `_room_percept_stmt`, `services/orion-thought/app/vision_reader.py` `read_recent_vision_events`) read room streams only.
- The situation brief had `fetch_street_summary` (who is around, expectation outcomes in the last hour, could-not-name items, patio count) but no captions.
- `vision_percept_expectation` met/missed rows reached only `services/orion-actions/app/walkway_forecast.py` (nightly grade/forecast journal).
- Curiosity study material carried up to 3 `vision_unresolved` items.
- Reverie (`services/orion-thought`) and dreams (`services/orion-dream`, replay sources in `cycle_store.SOURCE_QUERIES` + arcs) read nothing from the walkway.
- `node:substrate.perception_walkway` was fenced out of curiosity, the attention broadcast and the attention runtime by one set, `NO_COGNITION_CONSUMER_NODE_IDS`.

## Architecture touched

- New shared, stdlib-only module `orion/situational/street_material.py`: what counts as a caption (reducer rows `arrived_as_expected`/`expected_absent`/`attention_worthy` are grades, not captions), freshness + case/punctuation dedupe, graded-prediction wording, and the sqlalchemy SQL used by thought and dream.
- Situation brief: a fifth, independently fail-open street read (captions, last 15 min, up to 2) in `fetch_street_summary`; its line comes first. Chat stance street clip 240 -> 420 characters so the caption line does not push the rhythm/could-not-name lines out.
- orion-thought: `read_recent_street_events` / `read_recent_street_graded` in `vision_reader.py`, fetched concurrently in the reverie tick, passed as context keys `street_percepts` / `street_predictions` (absent when empty), rendered by `reverie_narrate.j2` in both branches (concern cards and coalition).
- orion-dream: new replay source kind `walkway` (one item per subject, weight 0.5 if it missed in the window else 0.3, at most a sixth of the replay), read after the pressure read like arcs.
- orion-hub curiosity: `_read_street_captions` (own read, own failure), `StudyMaterial.street_captions`, kickoff prompt section.
- Fence: `NO_COGNITION_CONSUMER_NODE_IDS` split into `NO_ATTENTION_CONSUMER_NODE_IDS` (walkway node; used by `attention_broadcast.py` x2 and attention-runtime `worker.py`) and `NO_CURIOSITY_CONSUMER_NODE_IDS` (empty; used by `endogenous_curiosity.py`).

## Sibling PR #2683 follow-up (patio fingerprinting, stationary fixtures)

PR #2683 (not merged yet) turns patio fingerprinting on, adds `vision_individual_sighting.stationary` / `vision_individual.stationary`, and adds a `head_count_zone` flag on the patio presence row (still writing `no_embed_zone` too). In this branch:

- `fetch_street_summary` sightings read adds `AND NOT s.stationary`. If that statement fails (column not there until `manual_migration_walkway_privacy_v2.sql` is applied), it falls back to the old statement, so the "who is around" line keeps working either way. Live today: column absent, fallback path ran, summary returned normally.
- The patio presence read matches `head_count_zone = true OR no_embed_zone = true` (SQL and Python check). Identity fields on that row are still dropped: it is a count row; people on the patio can now appear by name/kind through their own sightings.
- Stale comments fixed in `perception_reader.py` (the "patio is never named / counts only" block) and `PerceptionContextV1.street_summary`.
- Other readers this branch adds: captions (`vision_events` narratives) carry no stationary flag, so a caption can still mention the skeleton; graded predictions come from `vision_percept_expectation`, which #2683's rhythm reducer already builds from non-stationary individuals and sightings only. No other stationary filter applies.
- Tests: `test_stationary_sightings_are_excluded_with_fallback_before_the_migration`, `test_head_count_row_found_by_head_count_zone_flag_and_kept_a_count`; the missing-tables test now expects 3 rollbacks (stationary read + fallback + expectations).

## Files changed

- `orion/situational/street_material.py`: new shared rules/wording/SQL.
- `orion/situational/perception_reader.py`: caption read + first line in the street summary; stationary exclusion with pre-migration fallback; `head_count_zone` patio row; privacy comments updated for #2683.
- `orion/schemas/situation.py`: `street_summary` field comment only.
- `orion/situational/tests/test_street_material.py`, `test_street_summary.py`: new tests; the all-reads-failed test now also fails the caption read.
- `services/orion-hub/README.md`: one paragraph on the street in study material.
- `services/orion-cortex-exec/app/chat_stance.py`: street clip 240 -> 420.
- `services/orion-thought/app/{vision_reader,reverie,settings}.py`, `.env_example`, `README.md`, `tests/test_reverie_street_context.py`: reverie street input.
- `orion/cognition/prompts/reverie_narrate.j2`: "OUTSIDE, ON THE WALKWAY" section in both branches; stated expectations stay about the room.
- `orion/schemas/dream_cycle.py`: `ReplaySourceKind` gains `walkway`.
- `services/orion-dream/app/{replay,cycle,cycle_store,main,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`, `tests/test_replay_walkway_candidates.py`: dream walkway replay.
- `orion/curiosity/study_material.py`, `kickoff_prompt.py`, `README.md`; `services/orion-hub/scripts/{curiosity_investigation,main}.py`, `app/settings.py`, `.env_example`, `tests/test_curiosity_street_captions_read.py`: curiosity street captions.
- `orion/schemas/attention_candidate.py`, `orion/substrate/{endogenous_curiosity,attention_broadcast}.py`, `services/orion-attention-runtime/app/worker.py`, `orion/substrate/tests/test_endogenous_curiosity_walkway_fence.py`: per-consumer fence.
- `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json`, `tests/test_metric_rest_range.py` (count 10 -> 11), `config/field/orion_field_topology.v1.yaml` and `services/orion-substrate-runtime/README.md` (comments): the walkway node's glossary entry and docs.

## Schema / bus / API changes

- Added: `ReplaySourceKind` value `walkway` (`orion/schemas/dream_cycle.py`). Only orion-dream builds and validates `ReplayItemV1`; `dream_replay.source_kind` is plain text, no CHECK constraint (checked `manual_migration_dream_cycle_v2.sql`). Producer and consumer ship together.
- Added: `StudyMaterial.street_captions` (dataclass in `orion/curiosity`, in-process only).
- Added context keys `street_percepts` / `street_predictions` on the reverie plan request context (cortex-exec passes context through to the template; no model validates them).
- Removed/renamed: `NO_COGNITION_CONSUMER_NODE_IDS` -> `NO_ATTENTION_CONSUMER_NODE_IDS` + `NO_CURIOSITY_CONSUMER_NODE_IDS` (repo-wide references updated; none outside this repo).
- Behaviour changed: `PerceptionContextV1.street_summary` (unchanged type) can now start with a caption line.
- No bus channel changes.

## Env/config changes

- Added keys:
  - orion-thought: `ORION_REVERIE_STREET_ENABLED=true`, `ORION_REVERIE_STREET_STREAM_IDS=walkway`, `ORION_REVERIE_STREET_MAX_AGE_SEC=900`, `ORION_REVERIE_STREET_MAX_EVENTS=2`, `ORION_REVERIE_STREET_GRADED_LOOKBACK_HOURS=24`, `ORION_REVERIE_STREET_GRADED_MAX=4`, `ORION_REVERIE_STREET_TIMEZONE=America/Denver` (compose uses `env_file`).
  - orion-dream: `DREAM_REPLAY_WALKWAY_ENABLED=true`, `DREAM_WALKWAY_STREAM_IDS=walkway`, `DREAM_WALKWAY_TIMEZONE=America/Denver` (also in compose `environment:`).
  - orion-hub: `HUB_CURIOSITY_STREET_STREAM_IDS=walkway` (compose uses `env_file`).
- Removed / renamed keys: none.
- `.env_example` updated: yes, all three services.
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys <svc>`: yes; it added exactly the 7 thought keys, 3 dream keys and 1 hub key above (written to the primary checkout's `.env`).
- skipped keys requiring operator action: none. (Hub reported 5 pre-existing diverged keys, unrelated, left alone.)

## Metric quality gate: `node:substrate.perception_walkway` -> endogenous curiosity

1. **Provenance.** `orion/substrate/prediction_error.py::perception_prediction_error` (1 - cosine of the embedding vs the stream's own EWMA, then z-scored against its own surprise EWMA, negatives clipped, `min(1, z/3)`), called from `services/orion-substrate-runtime/app/worker.py::_handle_perception_prediction_error_message`; streams in `SUBSTRATE_PERCEPTION_WALKWAY_STREAM_IDS` go to the walkway cache, and `_publish_perception_group` writes the node every ~30 s tick (0.0 while warming or stale).
2. **Independence.** Same formula as `node:substrate.perception` (the room cameras), different input: its own camera, its own baseline row keyed by stream (`walkway`), its own cache, so the street never writes the room node. Shared only through the algorithm and through anything that affects both cameras at once (e.g. dusk). Not a transform of any other curiosity input.
3. **Theory anchor.** Prediction error as surprise against a learned per-stream expectation (the same predictive-processing anchor the room node already uses in this consumer): a scene unlike what this camera usually sees is a candidate for an evidence-gap scan.
4. **Live data (the step that matters).** The node has been live since 2026-10-15 04:57 UTC; data pulled ~05:10-05:35 UTC (night, 23:00 MDT). This is about 15 minutes of a dark street, so it is thin:
   - Per-embedding scores from the substrate-runtime log (172 real scores after a fresh baseline): 61.6% exactly 0.0, mean 0.056, p50 0.0, p90 0.146, p99 0.508, max 1.0. Above curiosity's threshold `min_error=0.55` (`EndogenousCuriosityConfig.min_prediction_error`): 2 of 172 (1.2%). One full saturation (raw 0.00753 vs a mean of 0.00321).
   - Tick writes in `substrate_node_prediction_error_history` (what curiosity actually reads): 18 rows, 9 exactly 0.0 (50%), p50 0.0006, p90 0.239, max 0.249. Crossings of 0.55: 0.
   - It returns to calm: long runs of exact 0.0 between non-zero ticks; zeros come from the z-score clip (at or below the street's usual surprise), not from staleness decay (written fresh every tick, `embedding_staleness` channel carries staleness separately).
   - Room node for comparison: same 15-minute window per-embedding (cam0, 191 scores) 0 crossings, p99 0.385; last 24 h history 3.9% of ticks >= 0.55 (61% exact 0.0, p95 0.473); last 7 days 2.7%.
   - Verdict: not degenerate (not flat, not stuck at 0, not saturated), can rest at 0, crossing rate (~1% per embedding, 0/18 ticks) is at or below the room node's. No calibration change proposed.
   - NOT measured: daytime street traffic. The EWMA baseline adapts to a busy street, but how often a busy day crosses 0.55 is UNVERIFIED. Re-check after the first full day: `SELECT avg((value>=0.55)::int) FROM substrate_node_prediction_error_history WHERE node_id='node:substrate.perception_walkway' AND observed_at > now()-interval '24 hours'`. If it is far above the room node's ~3-4%, the fix is a per-node threshold or a raw-magnitude floor, not a change to the shared `min_error`.
5. **Existing mechanism.** No new mechanism: curiosity's generic prediction-error scan already reads every `node:substrate.*` node; this patch only stops excluding one. Budget cap, staleness decay and the rung-6 governance chain bound what a crossing can do.
6. **Reversibility.** One line: add `PERCEPTION_WALKWAY_NODE_ID` to `NO_CURIOSITY_CONSUMER_NODE_IDS` in `orion/schemas/attention_candidate.py` (and drop the glossary `consumers:` line). Nothing persisted depends on it.

## Tests run

```text
Static gates (.github/workflows/orion-static-gates.yml, every python step): 32/32 ok
  incl. check_metric_lineage.py --gate PASS, --prompt-semantics PASS,
  check_definition_drift.py --gate PASS (after --update for the one added walkway metric),
  tests/test_metric_rest_range.py + test_metric_prompt_semantics.py 67 passed
New/updated focused tests:
  orion/situational/tests                         179 passed  (baseline 166)
  services/orion-thought tests/test_reverie_street_context.py   9 passed
  services/orion-dream tests                     270 passed, 7 skipped (baseline 261 passed)
  services/orion-hub tests/test_curiosity_street_captions_read.py + unresolved  6 passed
  orion/substrate/tests/test_endogenous_curiosity_walkway_fence.py + world_first  17 passed
  services/orion-attention-runtime tests/test_world_first_worker.py   7 passed
```

Baseline comparison, origin/main (detached worktree at b3a5a6555) vs this branch, same commands:

```text
suite                                   origin/main                    branch
orion/situational/tests                 166 passed                     177 passed
orion/curiosity                         130 passed                     130 passed
orion/substrate/tests                   3 failed, 1129 passed          3 failed, 1131 passed   (same 3 felt_state failures)
orion/schemas                           1 failed, 77 passed            1 failed, 77 passed     (same test_context_provenance failure)
tests/test_metric_rest_range.py         25 passed                      25 passed
services/orion-thought                  1 failed, 632 passed           1 failed, 641+ passed   (same test_mind_enrichment_defaults_off)
services/orion-dream                    261 passed                     270 passed
services/orion-hub (curiosity/situation files)  450 passed             458 passed
services/orion-attention-runtime        53 passed, 1 collection error  53 passed, 1 collection error (same)
services/orion-cortex-exec              133 failed, 1081 passed, 15 errors   identical lists
services/orion-substrate-runtime        14 failed, 432 passed, 11 errors     identical lists
```
The FAILED/ERROR lists are identical between the two trees for every suite (diffed); every failure is pre-existing on main.

## Evals run

```text
No new eval harness. The reverie/dream/curiosity prompt sections are pinned by golden-render tests
(reverie_narrate.j2 rendered through build_reverie_context; kickoff _material_section rendered).
Live read-only smoke of brief/reverie/room readers above. Graded-prediction paths: fixtures only
(no live rows exist until ~2026-10-20).
```

## Docker/build/smoke checks

```text
No docker build run. Live read-only DB smoke of the new readers and the Hub caption SQL (index-only scan on
vision_events_stream_created_idx) passed. Nothing deployed.
```

## Review findings fixed

Fresh review subagent on `origin/main...HEAD`: no must-fix; two should-fix, both fixed; nits mostly fixed.

- Finding: a reverie prompt with the street but no room block (room camera stale) still said "any expectation you state is about the room", inviting a claim that would later be judged against a room percept -- a street-to-room leak.
  - Fix: the sentence renders only when a room block is present; `run_reverie_once` drops a stated expectation when the street was offered and the room was not (no street and no room is unchanged from main).
  - Evidence: `test_expectation_is_dropped_when_only_the_street_was_offered`, `test_expectation_kept_when_the_room_was_offered`; existing `test_reverie_expectation_scoring.py` still passes (62 passed together).
- Finding: reducer rows (`attention_worthy`, ...) were filtered in Python after the SQL `LIMIT`, so a burst could hide real captions.
  - Fix: all three caption statements exclude them in SQL from one shared clause (`street_material.REDUCER_EVENT_TYPES_SQL`); the Python check stays as a backstop.
  - Evidence: `test_reducer_rows_are_excluded_before_the_row_limit` (25 newer reducer rows, caption still found), `test_reducer_types_are_excluded_in_every_caption_sql`.
- Nit: two 160-char captions first in the brief could push the "did not come" rhythm line past chat stance's clip.
  - Fix: brief captions clipped to 100 chars. Evidence: `test_brief_captions_are_clipped_so_the_rhythm_line_survives_the_stance_clip`.
- Nit: Hub README did not mention `HUB_CURIOSITY_STREET_STREAM_IDS`. Fixed.
- Not changed: textual dedupe misses near-synonyms (listed under risks); bad-timezone fallback in the kickoff section prints UTC without a marker (pre-existing pattern from the unresolved section); `metric_definitions.lock.json` may conflict if main re-locks first (re-run `--update` after rebase).

## Restart required

Not deployed. After merge, from the primary checkout on main:

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-dream up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-attention-runtime up -d --build
```

(cortex-exec: chat stance clip + the shared `perception_reader`; substrate-runtime hosts endogenous curiosity's scan; attention-runtime only for the renamed fence constant. Hub also builds the situation brief for unified turns.)

## Risks / concerns

- Severity: medium. Concern: daytime walkway crossing rate for curiosity is unmeasured (night sample only). Mitigation: re-check query above after one day; one-line revert.
- Severity: low. Concern: dedupe is textual, so near-synonyms survive ("A person is present in the frame" / "A person is visible in the frame", seen live). Mitigation: capped at 2-3 captions per consumer; not patched with word lists.
- Severity: low. Concern: captions are the council's free text and can describe a person on the patio or a stationary prop (the skeleton); no stationary flag exists on `vision_events`. Mitigation: narrative-only (no entities, no names); Juniper lifted the patio limits in #2683; capped at 2-3 captions.
- Severity: low. Concern: merge order with #2683. This branch works before and after its migration (fallback read), and reads both patio flags. If #2683 later stops writing `no_embed_zone`, this reader already uses `head_count_zone`.
- Severity: low. Concern: graded predictions are fixture-tested only. Mitigation: fail-open, empty renders nothing; verify after ~2026-10-20.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2682

🤖 Generated with [Claude Code](https://claude.com/claude-code)

# Orion's now: fair ranking for unscored kinds, and what Prime holds comes first (#3783)

Fixes #3783. Refs #3425.

## Summary

- Orion's now no longer puts an experience last just because its kind has no salience score. Every organ now gets
  one seat before any organ gets a second, so a reading, a chat turn or a painting run reaches the reader.
- Salience still orders rows, but only as a place in each kind's own 30-day history. An unscored row stands at 0.5
  (the expected place of a row in its own history). It is never treated as 0. The item still shows "salience unknown".
- A reader can now say what it holds (`GET /temporal-self/now?held=<ref>&held=<ref>`). A row whose own ids equal one
  of those refs exactly ranks first (at most 3) and carries `held_refs`. No text matching, no word list.
- Prime sends each held concern's event_ref, `arc:<arc_id>` and `prime_concern:<id>`, and its frame marks a matched
  item "ABOUT YOUR CONCERN <id>".
- Readers that send no `held` (reverie, curiosity, the thought service, dream day) get the old body shape.
- New read-only tool `scripts/replay_prime_now.py` replays stored Prime now reads with the new ranking.

## Outcome moved

Replay of Prime's last 40 real now reads (2026-11-20 .. 11-24, read-only, `scripts/replay_prime_now.py --last 40`):

| | before (shown live) | after (replay) |
|---|---|---|
| reads that had a reading in their window and showed one | 3 of 14 | 14 of 14 |
| items at salience 0.00 (the smallest of their kind) | 96 | 31 |
| unscored items shown (pinned chat turns not counted) | 89 | 240 |
| organs shown per read (mean) | 6.3 | 7.45 |

With Prime's real MemOnce concern held (concern `520a5a9f8d084495`, opened 2026-11-23 22:37 on
`world_pulse_read_seed:reading:50753c3f-...`, held until 11-24 07:55), all 3 of its now reads in that time put the
MemOnce GitHub reading (`reading_turn:act_48c17d7e041191a6ead65d30`) first, marked. Live, none of the 3 showed it.

Before/after for read `8151e652...` (2026-11-24 01:14Z, since 11-23 22:35Z, concern held):

```text
BEFORE (what Prime was shown)                AFTER (this ranking, held = concern's event_ref + id)
chat_turn            None (hunger pin)       reading_turn  None  act_48c17d7e... HELD world_pulse_read_seed:reading:50753c3f-...
field_dominance_run  0.97                    field_dominance_run 0.97
field_dominance_run  0.97                    gpu_wait            0.87
gpu_wait             0.87                    reverie_chain       0.75
reverie_chain        0.75                    chat_turn           None
reverie_chain        0.75                    visual_run          None
curiosity_run        0.43                    action_outcome      None
gpu_wait             0.36                    vision_percept      None
prediction_surprise  0.00                    curiosity_run       0.43
prediction_surprise  0.00                    attention_loop_raised None
curiosity_run        0.00                    reading_turn        None  (a second, unrelated reading)
vision_percept       None                    expectation_verdict None
```

The replay does not rebuild hunger pins; a live read keeps them first, as before.

Without any held concern (read `6cd45133...`, 2026-11-23 19:39Z, the read named in the issue), the newest reading
now gets the reading organ's seat; before, 4 prediction/dream/curiosity rows at 0.00 filled the slots instead.

## Current architecture

`GET /temporal-self/now` (orion-durable-runs) read at most 2 rows per kind (salience first, NULLs last) and
`orion/temporal_self/now.py build_now` sorted all of them by "has salience, then salience, then newest", capped 2 per
organ and 12 in all. Kinds without a magnitude rule in `config/integration/organs.yaml` (reading, chat, memory,
painting, reverie, most of camera/body/metacognition) always sorted after every scored row, even a 0.00.

## Architecture touched

- `orion/temporal_self/now.py`: `_fair_order` (the new order), `held_ref_keys` / `held_lookup` / `held_refs_of`
  (exact id matching), `build_now(held=, arc_subjects=)`.
- `orion/schemas/temporal_self.py`: `TemporalNowItemV1.held_refs` (omitted from the body when empty),
  `NOW_HELD_REFS_CAP`.
- `services/orion-durable-runs`: `held` query param; `NOW_HELD_SQL` (rows of the last 24 h whose ids equal a held key),
  `NOW_ARC_SUBJECTS_SQL`; candidate rows now carry `source_ref`, `correlation_id`, `subject_ref`, `kind_total`.
- `orion/prime/thought.py`, `services/orion-mind/app/prime_loop.py`: Prime sends `held=`, marks matched items, the
  section header and experience prompt say the new order.
- `orion/curiosity/now_section.py`, `services/orion-thought/app/temporal_self_cue.py`: order wording only.

## Files changed

- `orion/temporal_self/now.py`: fair order and held matching.
- `orion/temporal_self/tests/test_now.py`: new tests (below); the old "highest salience first" and "unknown last"
  tests are replaced, since that order is the bug.
- `orion/schemas/temporal_self.py`: `held_refs` field.
- `services/orion-durable-runs/app/main.py`, `app/temporal_self_store.py`: endpoint and SQL.
- `services/orion-durable-runs/tests/test_experience_now.py`: endpoint test with `held=`.
- `orion/prime/thought.py`, `services/orion-mind/app/prime_loop.py`: Prime side.
- `services/orion-mind/tests/test_prime_now.py`: Prime sends held refs, frame marks the item.
- `services/orion-mind/tests/fixtures/prime_replay_golden.json`, `tests/test_prime_replay.py`: re-recorded (5 parts
  changed, 0 after mapping the `&held=` URL keys and the order words back).
- `orion/integration/evals/fixtures/now_ablation_cases.jsonl`: regenerated (header wording).
- `orion/curiosity/now_section.py`, `services/orion-thought/app/temporal_self_cue.py`: wording.
- `scripts/replay_prime_now.py`: read-only replay.

## Schema / bus / API changes

- Added: `TemporalNowItemV1.held_refs: list[str]` (max 4), omitted from JSON when empty. `GET /temporal-self/now`
  accepts repeatable `held=`.
- Removed / renamed: none.
- Behavior changed: the now's order (see Summary).
- Compatibility notes: a read without `held` returns exactly the old field set, so images built before this field
  (extra="forbid") still validate it. Prime reads the now as a dict, so deploy order between orion-mind and
  orion-durable-runs does not matter (an old durable-runs ignores `held=`; an old mind sends none).

## Metric gate (two new ranking signals)

**Held-ref match.**
1. Provenance: `held_refs_of` (now.py) compares a row's stored `event_id`, `source_ref`, `correlation_id`,
   `subject_ref`, `related_refs` (written by `orion/temporal_self/sources.py`) with keys of the reader's refs. The
   MemOnce link is `reading_turn(...)` setting `correlation_id` = the seed uuid, and the event want's event_ref
   `world_pulse_read_seed:reading:<uuid>`.
2. Independence: not derived from salience or any other ranking input; it is identity, not magnitude.
3. Theory: an experience that is literally the concern's own event (or on its arc) is evidence about the concern.
   Issue evidence: the only Prime cites were of items directly on the thought's subject.
4. Live sanity: on the current held concerns (promise `turn:fc22c1a0-...`, dream `dream_day:2026-11-23#new1`) the
   last 24 h match 4 rows: the dream day itself, the 2 memory episodes that name it, and the cortex stance row of the
   promise turn. No unrelated row matched. A 10-character date tail is not a key (`HELD_TAIL_MIN = 12`).
5. Existing mechanism: hunger pins (`include=`) pin a need's row by event id; nothing matched concern refs.
6. Reversibility: a query param and an omitted-when-empty field; removing it is a revert.

**Kind window count (tie-break) and 0.5 standing.**
1. Provenance: `kind_total` = `count(*) OVER (PARTITION BY source_kind)` in `NOW_CANDIDATES_SQL` (already computed).
   Standing = the stored `salience` column (`source_salience`), 0.5 when NULL.
2. Independence: the count is a base rate, used only to break ties at equal step and standing. It is not salience.
3. Theory: a percentile of a row drawn from its own history is uniform on 0..1, so 0.5 is the no-information value.
   Fewer-rows-first is the relative-rate rule: a busy kind is not news because it is busy.
4. Live sanity: over Prime's last 40 reads, one kind had 1 to 361 rows in a window (median 5): not flat, never
   null. Organs present per read: 1 to 12 (12 is every organ Prime can read). When pins or held rows take seats in
   a 12-organ read, the organs left out of round one are the ones with the lowest standing.
5. Existing mechanism: world-first attention ranks each source against its own history the same way.
6. Reversibility: pure ordering code; no stored state.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no. Local `.env` sync: not needed. Skipped keys: none.

## Tests run

```text
pytest orion/temporal_self orion/integration orion/curiosity orion/prime          1047 passed, 29 skipped
services/orion-mind: pytest tests                                                 454 passed
services/orion-durable-runs: PYTHONPATH=<repo> pytest tests                       1102 passed, 157 skipped
services/orion-thought: pytest tests                                              573 passed, 31 skipped
services/orion-hub: pytest tests/test_curiosity_now.py tests/test_curiosity_now_arm_blind.py   18 passed
tests/scripts/test_schema_skew_discovery.py tests/test_gitignore_no_bare_catchall.py           21 passed
```

New tests: `test_an_unscored_kind_reaches_the_now_past_many_scored_organs`,
`test_a_row_about_a_held_concern_ranks_first_and_is_marked`, `test_held_matching_is_exact_ids_only`,
`test_no_held_is_the_same_body_shape_as_before`,
`test_build_now_caps_two_per_organ_and_twelve_in_all_every_organ_seated_before_any_second`,
`test_unscored_rows_of_one_kind_are_newest_first_and_a_quiet_kind_goes_before_a_busy_one`,
`test_held_refs_bring_a_concerns_reading_from_before_the_window_first_and_mark_it` (durable-runs),
`test_prime_sends_what_it_holds_and_the_frame_marks_an_item_about_a_held_concern` (orion-mind).
`test_the_now_never_shows_a_blinded_experiment_arm` passes unchanged.

## Evals run

```text
python -m orion.integration.evals.now_ablation_fixtures --write   (fixtures regenerated; test_now_ablation_eval passes)
scripts/replay_prime_now.py --last 40                             (read-only replay; numbers in "Outcome moved")
```

## Docker/build/smoke checks

```text
Not run: no Dockerfile, requirements or compose change. The live path is UNVERIFIED until deploy (see below).
```

## Review findings fixed

Review ran in a subagent (no must-fix findings). Fixed:

- Finding: rows that only share a held arc's broad subject (e.g. `orion:body:energy`, 40 rows/day) could take all 3
  held seats ahead of the concern's own event.
  - Fix: matches carry a tier; a direct id match (ref, ref without #fragment, tail id) always takes a held seat
    before an arc-subject match (`held_tier`, `TIER_DIRECT`, `TIER_ARC_SUBJECT`).
  - Evidence: `test_a_direct_match_takes_a_held_seat_before_rows_that_only_share_a_held_arcs_subject`.
- Finding: a held seat could be spent on a row the organ cap then dropped.
  - Fix: `_fair_order` takes `max_per_organ` and only spends a held seat on a row its organ cap lets through.
  - Evidence: `test_a_held_seat_is_never_spent_on_a_row_its_organ_cap_drops`.
- Finding: held rows from before the window were not counted in `candidates`, so shown + dropped could exceed it.
  - Fix: `NOW_HELD_SQL` returns `before_window`; the endpoint adds those rows to the total.
  - Evidence: durable-runs endpoint test asserts `candidates == 13` and shown + dropped <= candidates.
- Finding: the back-compat test validated with the new model, so it proved nothing.
  - Fix: the test builds the pre-#3783 forbid model without `held_refs`; a no-held body validates against it, a
    held body is refused (why only Prime, which reads a dict, sends `held`).
  - Evidence: `test_no_held_is_the_same_body_shape_as_before`.
- Nits fixed: `held_refs` now in the reader's order (`test_held_refs_are_named_in_the_readers_order`); a stored
  `"None"`/`"null"` id never matches; held-seat counter instead of O(n^2) sum; replay help text says pins are not
  rebuilt.
- Not changed (kept as designed, stated in Risks): a held row can lead every Prime read while its concern is held
  (at most 24 h, at most 3 seats).

## Restart required

Deploy from the clean primary main checkout after merge (the post-pull rebuild covers both):

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Live check after deploy

1. Next Prime thought: `docker logs orion-athena-orion-mind 2>&1 | grep prime_now_read | tail -1` and confirm the
   GET URL in durable-runs logs carries `held=` for each held concern.
2. `python3 scripts/replay_prime_now.py --last 5 --held <a held concern's event_ref>` against the new rows, and
   check `experience_shown.event_ids` of new Prime reads contain a `reading_turn` when one landed in the window.
3. After ~3 days: count Prime thoughts that cited an experience (was 1 of 70, 11-16..11-24), split by whether the
   cited item carried `held_refs`.

## Risks / concerns

- Severity: low. Concern: the frame now carries more unscored telemetry rows (vision percepts, action outcomes) in
  seats that used to go to a second row of a scored organ. Mitigation: seats are one per organ first; held items and
  pins still lead; the cite count after deploy measures whether this helps.
- Severity: low. Concern: a held item from the last 24 h repeats in every Prime read while the concern is held.
  Mitigation: at most 3 held rows; a concern lives 24 h unkept.
- Issues opened: none. Issues closed: #3783 (on merge; `verify-live` evidence still due after deploy).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3841

🤖 Generated with [Claude Code](https://claude.com/claude-code)

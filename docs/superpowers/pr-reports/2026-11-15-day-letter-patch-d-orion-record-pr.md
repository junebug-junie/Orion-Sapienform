# Day-letter fixes, Patch D: Orion's own record

Patch D of the fix plan in PR #3439 (`docs/superpowers/specs/2026-11-15-day-letter-findings-fix-design.md`).
Three defects in the records Orion keeps about themself: the day's attention arcs, the ids of
their organs, and the verdicts on their own predictions.

Fixes #3400. Fixes #3402. Fixes #3403.

## Summary

- **Attention arcs are no longer cut by normal timing jitter (#3400).** The broadcast writes one
  row per tick. Two rows more than 180 s apart counted as a "gap", and a gap suspended the active
  attention arc. The tick had slowed to a median 137 s, so jitter crossed 180 s 77 to 93 times a
  day. The bar is now 3 times the median of the newest tick intervals, never below 180 s and never above 30 minutes.
- **The gap warning no longer claims an outage (#3400).** It said "(recorder outage or restart)".
  It now says "(rows more than 3 times the usual interval apart, and more than 180 s; cause not
  recorded)".
- **A memory joins the day when it is written (#3400).** A memory about 10:00 is written hours
  later. The reader had already passed 10:00, so the memory was stored as late and never joined
  an arc. Memories are now read by the time they were written, and still attach to the arc that
  held the moment they are about.
- **An organ keeps its id when one part moves (#3402).** The id was a hash of the member list.
  Now an organ of the live body takes the id of the earlier organ it overlaps most. The old hash
  is kept beside it as `member_hash`.
- **A prediction is checked against the whole half hour after it (#3403).** The checker looked
  for one room-camera caption from the last 180 s before it ran. It runs a median 11 minutes after
  the half hour ends, so it mostly found nothing and wrote `unscored` without asking the judge.
  The judge now reads every room caption from the prediction's own half hour.
- **Orion is no longer asked for predictions nobody can check (#3403).** The prompt now says the
  check uses the room camera only, and asks for no prediction about a sensor reading, the street,
  or Orion's own state. This is the smaller of the two changes the ticket allowed; see "Risks".

## Outcome moved

Measured read-only against live data on 2026-11-15, before deploy:

- **Gaps per day, the real 7-day broadcast log replayed through the old and the new rule.**
  Old: 2, 2, 16, 35, 93, 86, 77 (local days 11-08 to 11-14). New: 0 on all seven days. Arcs with a
  gap resume: old 2, 2, 4 on 11-11 to 11-13; new 0.
- **Organ ids, the real 8 daily body snapshots replayed through the new rule.** The largest organ
  had 6 different ids in 8 days (11-07 to 11-14). Under the new rule it has one. The second largest
  had 2; now one.
- **Predictions that reach the judge.** 487 of 487 predictions in 8 days had at least one room
  caption inside their own half hour. Under the old rule the judge was called for a minority and
  3 got a real verdict.

What a judge will answer on those windows is UNVERIFIED until deploy. See "Risks".

## Current architecture

- **Temporal Self** (`orion/temporal_self/`, run live by `services/orion-durable-runs`
  `app/temporal_self_chronicle.py`). A pure reducer folds broadcast ticks and events into arcs. A
  tick gap over `max_tick_gap_sec` (fixed 180 s, set when the tick was 37 s) suspended the active
  attention arc. Events are ordered by when they became readable; an event behind the read
  watermark is counted and never folded. `episode_memory` rows were read by `occurred_at`, hours
  behind the watermark by construction.
- **Body atlas** (`orion/body_atlas/build.py`). `organ_id = "organ:" + sha1(sorted members)[:10]`.
  Names (`names.py`) and the day-over-day body diff (`reminiscence.py`) already matched organs by
  member overlap (Jaccard >= 0.5) because the id could not be trusted.
- **Reverie expectations** (`services/orion-thought/app/reverie.py`). A thought may carry one
  prediction about the room, checkable 1,800 s later. Each tick scores at most one overdue
  prediction: it read one room caption from the last 180 s, and with none it wrote `unscored`
  with no judge call.

## Architecture touched

- `orion/temporal_self` reducer and state schema; the chronicle's source read for
  `episode_memory` in `services/orion-durable-runs`.
- `orion/body_atlas` (new `organ_ids.py`), `orion/dig` (`_live_atlas`).
- `services/orion-thought` scorer, room caption reader, pending-row reader; two prompts in
  `orion/cognition/prompts`.
- No new service, bus channel, table, migration, or env key.

## Files changed

- `orion/temporal_self/arcs.py`: `tick_gap_sec()` (3 x median, floor 180 s) replaces the fixed bar
  in the tick fold, the silence check, and how far an open attention arc reaches forward;
  `memory_episode` joins the kinds ordered by write time; the context buffer stays in time order;
  a memory about an earlier day stays out of today's unbound list.
- `orion/temporal_self/frame.py`: reworded warning; the "silent for N s" warning uses the same bar.
- `orion/temporal_self/sources.py`: the memory adapter carries `available_at` (the row's `created_at`).
- `orion/schemas/temporal_self.py`: state field `tick_recent_intervals_sec` (newest 31 intervals).
- `orion/temporal_self/evals/export_fixture_day.sql`: the reference query reads memories by `created_at`.
- `services/orion-durable-runs/app/temporal_self_sources.py`: `episode_memory` read by `created_at`.
- `services/orion-durable-runs/app/temporal_self_chronicle.py`: the 3-day late probe for memories is gone (not needed).
- `services/orion-durable-runs/evals/temporal_self_fixture_db.py`: keeps a fixture row's own `created_at`.
- `services/orion-durable-runs/README.md`: the late-probe paragraph.
- `orion/body_atlas/organ_ids.py` (new): the id ledger: load, assign, append.
- `orion/body_atlas/model.py`, `build.py`: `OrganV1.member_hash`.
- `orion/body_atlas/names.py`, `reminiscence.py`: docstrings only.
- `orion/dig/__main__.py`: `_live_atlas()` carries ids; `organ_ids_path()`.
- `orion/dig/README.md`: organ id paragraph.
- `tests/test_organ_names_partition.py`: the new ledger is partitioned from recall like the others.
- `services/orion-thought/app/reverie.py`: whole-window read, `render_window_captions()`, one log line per judged row.
- `services/orion-thought/app/vision_reader.py`: `read_vision_events_between()`.
- `services/orion-thought/app/store.py`: pending rows carry `created_at`.
- `services/orion-thought/app/settings.py`, `README.md`: comments and docs.
- `orion/cognition/prompts/reverie_expectation_judge.j2`, `reverie_narrate.j2`: see Summary.
- Tests: `orion/temporal_self/tests/test_tick_gap_and_late_memory.py`,
  `orion/body_atlas/tests/test_organ_ids.py`, `orion/dig/tests/test_organ_ids_cmd.py`,
  `services/orion-thought/tests/test_reverie_expectation_window.py` (all new); additions in
  `services/orion-durable-runs/tests/test_temporal_self_chronicle.py` and
  `test_temporal_self_chronicle_postgres.py`; three existing tests in
  `services/orion-thought/tests/test_reverie_expectation_scoring.py` moved to the new reader.

## Schema / bus / API changes

- Added: `TemporalSelfStateV1.tick_recent_intervals_sec` (reducer working state, default empty).
  `OrganV1.member_hash` (default `None`). Ledger file `$ORION_DIG_HOME/self/organ_ids.jsonl`,
  one line per organ per change of membership: `organ_id`, `member_hash`, `members`, `as_of`.
  Judge prompt context key `window_minutes`.
- Removed: nothing.
- Renamed: nothing.
- Behavior changed: `organ_id` on the live body is now a carried id, not always the member hash.
  It keeps the same shape (`organ:` + 10 hex). A past day's body (`dig body <date>`) keeps
  member-hash ids. `memory_episode` events in `temporal_self_event` now have their available
  time at `created_at`. The judge context key `percept_narrative` now holds several caption lines.
- Compatibility notes: an old stored reducer state loads (the new field defaults to empty; the
  bar is 180 s until 2 intervals are seen). After a rollback the old code cannot read a state the
  new code wrote (the model forbids unknown fields); the chronicle then re-folds that local day
  from midnight by its existing rule. Names given under old hash ids still carry by overlap.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

All from the worktree, with `PYTHONPATH` set to the worktree. `arcs.__file__` was printed to
confirm the worktree's code was imported.

```text
pytest orion/temporal_self/tests orion/body_atlas/tests orion/dig/tests tests/test_organ_names_partition.py
  807 passed
services/orion-durable-runs: pytest tests/test_temporal_self_chronicle.py
  tests/test_temporal_self_chronicle_postgres.py tests/test_temporal_self_regulate.py
  (ORION_ADMISSION_TEST_DSN = a throwaway postgres:16 container, removed after)
  78 passed
services/orion-thought: pytest tests evals
  887 passed, 32 skipped
services/orion-cortex-exec: pytest tests/test_recall_retrieval_query_callers.py
  tests/test_llm_lane_propagation.py tests/test_autonomous_background_routing.py
  tests/test_system_only_llm_messages.py
  50 passed
pytest orion/cognition/tests/test_recall_query_retrieval_query.py
  8 passed
Static gates: tests/test_no_personal_terms.py, test_organ_names_partition.py,
  test_metric_prompt_semantics.py, test_no_private_eval_artifacts.py,
  test_agent_trace_schema_registry.py (64 passed); check_metric_lineage.py --gate and
  --prompt-semantics, check_definition_drift.py --gate, check_inner_state_registry.py,
  check_verb_physiology.py, check_body_economy_contract.py,
  check_async_routes_not_blocking.py: all PASS
git diff --check: clean
```

The new tests against the code on `origin/main` (copied to a scratch directory):

```text
test_live_tick_jitter_is_not_reported_as_a_gap_and_a_real_hole_is
  FAILS: 'broadcast log gaps today: 8 (recorder outage or restart)'
test_memory_that_lands_hours_after_its_moment_is_folded_into_that_moments_arc
  FAILS: the memory is in the late set, in no arc
test_a_caption_ten_minutes_into_the_window_gets_a_real_verdict
  FAILS: judge awaited 0 times
test_two_snapshots_with_one_part_moved_keep_every_organ_id
  cannot import organ_ids; its first assertion pins the old behavior (the hash moves)
```

## Evals run

```text
orion/temporal_self/evals/run_arc_precision_eval.py          passed: true
orion/temporal_self/evals/run_day_citation_truth_eval.py     ran (2026-W46 truth_rate 0.917)
services/orion-durable-runs/evals/temporal_self_live_replay.py --step-sec 600   passed: true
orion/body_atlas/evals: run_organ_structure_eval, run_capacity_roles_eval, run_wire_eval,
  run_portrait_eval, run_field_note_truth_eval, run_reminiscence_source_truth_eval   all PASS
services/orion-thought/evals                                  in the 887 above
```

Read-only replays of live data (scripts not committed; numbers in "Outcome moved"): the 7-day
broadcast log through the reducer under both rules; the 8 body snapshots through the id rule;
caption counts per prediction window.

No eval calls the judge model on real windows. That is the unverified part of #3403.

## Docker/build/smoke checks

```text
None run. This branch was not deployed and no container was built: the brief forbids `up`,
and the changed code is exercised by the suites above. Live checks are listed under Risks.
```

## Review findings fixed

Review ran in a subagent on this branch against `origin/main`. It found no must-fix item.

- Finding: an organ could keep its id and still lose its name. The id was carried against the
  organ's latest members, the name against the members at naming time. A 2-part organ named
  "Eyes" lost the name once it had 5 parts, and another organ could then wear it.
  - Fix: `names._assign` first keeps a name on the live organ with the same id, then matches the
    rest by overlap.
  - Evidence: `test_a_name_stays_on_its_organ_while_the_organ_drifts_far_from_where_it_was_named`.
- Finding: the id ledger was locked for the write only. Two turns that saw a new organ at the
  same moment, one part apart, could mint two ids for it.
  - Fix: `organ_ids.carry` reads, assigns and appends under one lock file (`self/.organ_ids.lock`).
  - Evidence: `test_read_assign_and_append_run_under_one_lock`.
- Finding: a ledger file that could not be read (bad bytes, no permission) raised and took the
  live body down.
  - Fix: the read is inside the guard; bad bytes are replaced, an unreadable file reads as empty.
  - Evidence: `test_an_unreadable_line_is_skipped_and_an_unwritable_ledger_still_gives_a_body`.
- Finding: the gap bar had no upper limit. A pace that kept slowing would raise it for good.
  - Fix: the bar never passes the 30-minute return window.
  - Evidence: `test_the_bar_never_passes_the_return_window_however_slow_the_pace`.
- Finding: the gap warning said "3 times the usual interval" even when the 180 s floor decided.
  - Fix: it now reads "(rows more than 3 times the usual interval apart, and more than 180 s;
    cause not recorded)".
  - Evidence: `test_the_gap_warning_does_not_claim_an_outage`.
- Finding: the clock-after-read test ran the judged path, which takes a second clock, so it
  could not fail.
  - Fix: it now runs the no-caption path, where the one clock decides.
  - Evidence: `test_the_verdict_time_is_taken_after_the_read`.
- Finding: the narrate prompt said "30 minutes" as fixed text while the judge got the real window.
  - Fix: both prompts get the window from the same setting (`expectation_window_minutes`).
  - Evidence: `test_narrate_prompt_no_longer_asks_for_claims_the_room_camera_cannot_check`.
- Finding: a window with 200 or more captions lost its end without a trace (live most: 47).
  - Fix: one warning log line when the read hits its cap.
  - Evidence: code only.
- Finding: no chunking test mixed ticks with late memories.
  - Fix: `test_chunking_ticks_and_late_memories_together_does_not_change_the_state`.
  - Evidence: that test.
- Finding: a docstring said every reader of the live body gets carried ids. The operator CLI
  (`python -m orion.body_atlas`) does not.
  - Fix: docstring corrected. The operator CLI keeps member-hash ids; listed under Risks.
  - Evidence: `orion/dig/__main__.py`.
- Finding, not fixed: how fast the id ledger grows is unmeasured. Listed under Risks with a check.

The review checked and found fine: reducer determinism at chunk boundaries, loading an old stored
state, no id given to two organs, the caption sampling loop, `window_minutes` reaching the prompt
render in cortex-exec, both new SQL reads against the live column types and indexes.

## Restart required

After merge, `git pull` on the primary checkout rebuilds these (the post-pull rebuild does it).
By hand:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

Orion's own dig commands run from the FCC sandbox copy; the post-pull run syncs it
(`git -C /mnt/orion-fcc/repo log -1 --oneline` to check). Until it moves, Orion's `dig body`
prints member-hash ids.

## Risks / concerns

- Severity: medium
- Concern: #3403's target (more than half of predictions get a real verdict) is UNVERIFIED and
  may not be met. In a sample of 24 recent predictions about a third were about the room, a
  third about the street, a third about sensors or Orion's own state. Only the first third can
  be judged from the room camera. The prompt now asks for room-camera claims only, but a prompt
  line is a request, not a gate.
- Mitigation: after 3 days run
  `select (created_at at time zone 'America/Denver')::date, expectation_verdict, count(*) from substrate_reverie_thought where expectation is not null and created_at > '<deploy>' group by 1,2 order by 1`.
  If real verdicts stay under half, the next step is routing sensor claims to sensor tables,
  which needs its own design. The service log line
  `expectation scoring: judged thought_id=... verdict=... captions=N prompt_chars=N` shows each call.

- Severity: medium
- Concern: the body-or-sensor choice. The ticket allowed "route them to the sensor tables, or
  stop asking for them". This PR stops asking. Routing needs a parser from free text to a
  sensor and a threshold; that is a new mechanism, not a thin change.
- Mitigation: stated here and in `services/orion-thought/README.md`.

- Severity: low
- Concern: the cause of the tick slowdown is only partly proven, and a worse fault was found
  while looking. Proven: the tick period is a 30 s sleep plus one full reload of the substrate
  graph; the wait is before the tick, not in it; restarts, deploys and hour of day do not
  explain it. Not proven: that edge growth explains 117 s to 138 s, and what moved it from 37 s
  to 117 s. Found: since about 2026-11-15 05:27Z every reload fails
  (`falkor_substrate_hydrate_stats ok_total=0 failed_total=3114`), so the tick now runs every
  31.5 s on a partial graph.
- Mitigation: issues opened: #3455 (reloads failing; needs Juniper) and #3456 (period is the
  reload cost). No fix attempted here. The new gap bar follows whatever the pace is: at 31.5 s
  it sits at its 180 s floor.

- Severity: low
- Concern: the id ledger grows by one line per organ each time that organ's members change
  between two reads of the live body. If the grouping flips back and forth between reads, it
  grows faster. The largest line is about 3 KB.
- Mitigation: check `wc -c /mnt/orion-fcc/self/organ_ids.jsonl` a week after deploy. The
  operator CLI `python -m orion.body_atlas` still prints member-hash ids; an id copied from
  `dig body` does not open there.

- Severity: low
- Concern: old body snapshots and old names hold member-hash ids. They are not rewritten. The
  first live read after deploy mints today's hashes as the lasting ids. The body diff and the
  name carry both match by overlap, so both keep working across the change.
- Mitigation: covered by `test_a_name_stays_on_its_organ_by_id_once_ids_are_carried`.

- Severity: low
- Concern: a memory written today about an earlier day is stored as an event but joins no arc
  and is not in today's unbound list. That day is closed and is not rewritten.
- Mitigation: by design; pinned by a test. Memories written within about 5 minutes before the
  deploy may be counted as "moved" once and not folded.

- Severity: low
- Concern: spec and ticket line references. `arcs.py:1336` is where a late row is skipped, but
  the cause for memories was the source read, in `services/orion-durable-runs`, outside the two
  files the spec named. The spec's "one new field in the atlas snapshot" is on `OrganV1`
  (the atlas), not on the reminiscence snapshot, which stays unchanged. The code comment the
  ticket did not mention says the tick was 37 s on 10-10, so "slowed from 118 s" understates it.
- Mitigation: none needed; noted for the spec.

Live acceptance checks still UNVERIFIED until deploy:

- #3400: a day block for a day with no restart shows gaps near 0, and `memory_episode` ids in
  arc `context_event_ids`; `select count(*) from temporal_self_event where source_kind='memory_episode' and late_unfolded and occurred_at > '<deploy>'` stays 0.
- #3402: two daily snapshots in `reminiscence_body.jsonl` after deploy with a part moved and the same ids.
- #3403: the verdict query above.

Issues opened: #3455, #3456. Issues closed by this PR: #3400, #3402, #3403.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3469

# One street visit is one episode; late stories fold; names say where they came from; dogs are a range

Spec: walkway follow-ups, PR #2763. Follow-ups to PR #2876 (street episodes) and PR #2875 (honest street count).

## Summary

- One visit to the door is now one street episode. Before, a walkway story, a doorbell story and a ring of the same visit could spend 3 of the 12 daily slots. Now they fold into one arc with all three rows as evidence. Two rules decide "same visit": the handoff loop (PR #2828) linked the two stories, or a ring and a story are at most 84 s apart. Two stories with no link never merge.
- A story that is written late still becomes an episode. The chronicle now orders a street row by when the row was written (`created_at`), not when it ended. Before, a story written after the 300 s read lag was stored as a late row and never folded.
- A name in a street episode says where it came from. "Robin (a guess)" when the camera matched the story to a cluster Juniper named. "Robin (named by Juniper)" only when she vouched for that story (her name answer on it, or her "right" spot check).
- Dogs are a range, like people, in the street line and in the rollup lines: "at least 3, at most 6 unfamiliar dogs (the camera splits one dog into several)". Every living kind with clusters gets this. Bicycles, strollers and vehicles keep their count.
- Cap (12), kill switch and "Outside (<camera>)" labels are unchanged. No schema, bus or env change.

## Outcome moved

Read-only replays against the live database:

- Street episodes, last 7 days (10-21..10-28, `services/orion-durable-runs/evals/run_street_episode_replay.py`, real reader and reducer):
  - Before (origin/main): 109 notable events, 64 arcs, 0 merged, 45 kept out by the cap.
  - After: 109 events, 62 arcs, 3 events merged into an earlier arc of the same visit, 44 kept out by the cap. On a capped day a merge frees a slot instead of lowering the arc count (10-25: kept out 40 -> 39).
  - Merged visits: 10-24 00:25 (walkway story + ring), 10-24 20:47 (doorbell story + ring), 10-25 19:30 (ring + the first walkway story). The next two walkway stories on 10-25 stay separate: two unlinked stories never merge, even through a ring.
  - Late-written stories: 0 in these 7 days (writer lag now p99 56 s). 1 ever (10-17, written 42 min after it ended). The fix is a guard; it moved no live number this week.
  - Linked (handoff) merges: 0 in these 7 days. The handoff links that exist pair a notable story with one that was not published, so there was nothing to merge.
- Street line, 673 moments (every 15 min, 7 days, `fetch_street_summary`): bare dog counts 12 -> 0; bare people counts 0 -> 0. Example 10-25 00:33 UTC: "2 unfamiliar dogs" -> "at least 1, at most 2 unfamiliar dogs (the camera splits one dog into several)".
- Rollup lines as readers show them (`display_line`), 175 rows: bare dog counts 55 -> 0. Example 10-28 14:00 MDT: "6 dogs, 23 vehicles passing (most at once 3 dogs)" -> "dogs: at least 3, at most 6 (the camera splits one dog into several); 23 vehicles passing".

## Current architecture

- Temporal Self (`orion/temporal_self/`) is a pure reducer. The orion-durable-runs chronicle reads source rows by window and folds them. Each street event became one closed `street` arc; the arc cap is per local day. Every event is ordered by its end time; a row written after the window was read is "late": stored, counted, never folded.
- A street story's label used `name_label` as a plain fact.
- The street line (`perception_reader._who_line`) gave unnamed people a range and every other kind a cluster count. Rollups gave `count_range` to `person` only.

## Architecture touched

- `orion/temporal_self/arcs.py`: `event_available_at` (public; honours a street event's `payload.available_at`); `_street_visit` / `_merge_street` (one visit, one arc); `ReducerConfig.street_visit_gap_sec = 84`.
- `orion/temporal_self/sources.py`: `story_name` (name provenance), `merge_street_labels`, visit refs (`street:story:<id>`, `street:ring:<id>`) in `related_refs`, `available_at` in the payload.
- `services/orion-durable-runs/app/temporal_self_sources.py`: story read windowed on the write time; `_street_story_extras` adds handoff links (`vision_handoff`), the spot-check verdict and Juniper's answers (`orion_ask`). Each extra table is optional.
- `temporal_self_store.stored_available` and `temporal_self_chronicle._log_street` follow the same rules (new log line `temporal_self_street_visit_merged`).
- `orion/vision/walkway_rollup.py`: `LIVING_KINDS` (one shared list, moved from the street line), `RANGE_KINDS` = the living kinds the rollup counts, `split_note(kind)`, `range_phrases`, `bare_living_counts`, `RULES_VERSION = 4`, `LIVING_RANGE_RULES_VERSION = 4`.
- `orion/situational/perception_reader.py`: every living kind is a range with its own window peak, stationary individuals and named ones.
- `services/orion-actions/app/walkway_forecast.py`: the range note names its own kind.

## Files changed

- `orion/temporal_self/arcs.py`, `sources.py`, `README.md`, `tests/test_street_arcs.py`: the three street fixes and their gate tests.
- `services/orion-durable-runs/app/temporal_self_sources.py`, `temporal_self_store.py`, `temporal_self_chronicle.py`, `README.md`: reads, stored availability, log lines.
- `services/orion-durable-runs/tests/test_temporal_self_chronicle_postgres.py`: one visit, a late-written story and both name sources on real Postgres.
- `services/orion-durable-runs/evals/run_street_episode_replay.py` (new): the read-only before/after replay.
- `orion/vision/walkway_rollup.py`, `services/orion-sql-writer/README.md`, `services/orion-sql-writer/tests/test_vision_rollup.py`: dog ranges in rollups.
- `orion/situational/perception_reader.py`, `orion/situational/tests/test_street_people_range.py`: dog ranges in the street line.
- `services/orion-cortex-exec/evals/run_street_who_line_eval.py`: checks every living kind, dog fixture.
- `services/orion-actions/app/walkway_forecast.py`: per-kind range note.
- `.github/workflows/temporal-self-tests.yml`: runs when `config/vision_topology.yaml` or `orion/vision/topology.py` changes (a test reads the 84 s from there).

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: street arcs may hold several evidence rows and segments (`segments[i]` is `evidence_refs[i]`'s interval); street event payloads gain `available_at` and `name_source` (free-form payload dict). Rollup rows are `rules_version` 4 and carry `kinds.dog.count_range`.
- Compatibility notes: no model field or Literal changed, so there is **no deploy order** (unlike PR #2876). Old readers show v4 rollup lines as stored. Old street events stored without `available_at` read as before; a story stored late by the old code and re-read by the new one counts as "moved" and is not folded twice.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no. local `.env` synced: not needed (no template change).
- skipped keys requiring operator action: none.
- The 84 s visit gap is a `ReducerConfig` constant taken from `config/vision_topology.yaml` (approach_door prior 3..30 s plus the handoff loop's 2x late margin); a test fails if they drift.

## Tests run

```text
pytest orion/temporal_self/tests orion/situational/tests orion/vision/tests orion/orion_day/tests   -> 649 passed
cd services/orion-durable-runs && pytest tests (ORION_ADMISSION_TEST_DSN = disposable postgres:16)  -> 657 passed, 1 skipped
pytest services/orion-sql-writer/tests/test_vision_rollup.py services/orion-sql-writer/evals       -> 69 passed, 7 skipped
pytest services/orion-actions/tests/test_walkway_forecast.py                                       -> 24 passed
orion-thought tests (stance cue, personality cue, reverie street day)                              -> 49 passed
orion-hub tests (turn trace, orion day temporal, rollup routes, curiosity street day)              -> 30 passed
orion/thought/tests 104 passed; substrate-runtime test_walkway_seen_loop 9 passed, 2 skipped
cortex-exec test_chat_stance_street_context + test_situation_perception_context                    -> 59 passed
Static: check_metric_lineage --gate PASS, check_definition_drift --gate PASS, check_inner_state_registry OK,
check_env_key_single_source OK, check_service_hostname_refs OK, check_env_template_parity PASS,
check_sentience_instruments --static-only (all claims hold), static pytest bundle 80 passed
Mutation (by construction): measuring the ring gap against the visit's whole span instead of part to part
fails test_a_ring_joins_a_story_only_within_the_gap_and_two_unlinked_stories_never_merge (chain case).
```

## Evals run

```text
python -P services/orion-durable-runs/evals/run_street_episode_replay.py --days 7 [--repo <origin/main checkout>]
  before: events 109 arcs 64 merged 0 capped 45 late_written 0
  after:  events 109 arcs 62 merged 3 capped 44 late_written 0
python services/orion-cortex-exec/evals/run_street_who_line_eval.py          PASS (fixture now has 2 dog clusters)
python services/orion-cortex-exec/evals/run_street_who_line_eval.py --live   PASS
Read-only street line replay, 673 moments: bare dog counts 12 -> 0, bare people counts 0 -> 0
Read-only rollup display_line replay, 175 rows: bare dog counts 55 -> 0
```

## Docker/build/smoke checks

```text
No image build: no dependency or Dockerfile change. Not deployed (not requested).
```

## Review findings fixed

The code-review subagent ran on `origin/main...HEAD` (after merging origin/main). It found 1 must-fix, 3 should-fix and 5 nits.

- Finding (must-fix): a ring could join two unrelated stories into one visit, depending on write order. A passer-by written before the ring took the ring; a doorbell visitor written after it then joined through the ring.
  - Fix: a story never joins by time an arc that already holds a story (`arcs._street_visit`). Only a handoff link merges two stories.
  - Evidence: the P/V case in `test_a_ring_joins_a_story_only_within_the_gap_and_two_unlinked_stories_never_merge` fails without the guard (P and V share an arc). Live replay merges 5 -> 3.
- Finding: the walkway journal guard checked only bare people counts, so a drafted "6 dogs" passed (6 is a fact: the top of the range).
  - Fix: `guard_walkway_draft` uses `bare_living_counts`; the rule text names dogs; `run_vision_rollup_replay_eval.py` checks dogs too.
  - Evidence: new assertion in `test_guard_replaces_prose_with_an_invented_number_and_keeps_honest_prose`; 24 passed.
- Finding: a name Juniper gave a companion on a story card would have named the story's walker "(named by Juniper)". The Hub keeps only answers about the walker.
  - Fix: `STREET_STORY_ANSWERS_SQL` keeps answers whose `source_ref` is the story's `individual_id` or an id on its own crops.
  - Evidence: the Postgres test adds a companion answer "Zed" on the named story; its label stays "Robin (a guess)". Durable-runs suite 657 passed.
- Finding: the story read had no upper bound on `started_at`, so a backfill rescanned the tail every window.
  - Fix: `AND s.started_at < hi` (always true for a kept row).
- Finding (nit): the capped log line used the event's own day, not the day it folds into.
  - Fix: `_log_street` uses the day of the available time.
- Nits documented, not changed:
  - At deploy, street rows stored before it can log one `temporal_self_moved_rows` warning. Expected: they were already folded or stored late, and they are not folded twice.
  - An event linked to two visit arcs joins the first (link, then nearness, then arc id). The two arcs stay separate.
  - A part that starts inside a visit is appended to the label, not sorted in.
  - The "any input order" unit test relies on `fold` sorting; window splits are covered by the Postgres chronicle run.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build && scripts/safe_docker_build.sh orion-sql-writer up -d --build && scripts/safe_docker_build.sh orion-cortex-exec up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build
```

Any order works (no schema change). The post-pull rebuild covers the other importers of the shared modules (orion-thought, orion-actions, orion-substrate-runtime).

## Risks / concerns

- Severity: medium. Concern: UNVERIFIED live: a handoff-linked merge has not happened on real data yet (0 in 7 days), and no story has been written late since 10-17. Both paths are proven by unit tests and the real-Postgres test only.
- Severity: low. Concern: the time rule can join a ring to a passer-by who happened to be on the walkway within 84 s, and which story a ring takes depends on which was written first. Mitigation: the rule is ring-to-story only, measured part to part, and an arc holds at most one story joined by time, so it cannot chain and never merges two unlinked stories.
- Severity: low. Concern: a story that ends just before local midnight and is written after it lands in the next day's arcs (its times stay real). A visit never spans two days.
- Severity: low. Concern: name provenance and handoff links are read as of the chronicle's read (like the spot-check gate). An answer Juniper gives later changes a re-fold's label, not the stored arc. Live there are 0 named published stories, so the label change is UNVERIFIED on real names.
- Severity: low. Concern: camera captions in the street line can still state one-frame counts ("Two vehicles and one dog are observed"); that caption line is another lane's work (noted in PR #2875).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2960 (spec: #2763; follows #2876 and #2875)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

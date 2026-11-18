# Graphiti trials 1 and 2 (Patch 0): NO-GO, and it is Graphiti, not our setup

Spec: `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md` (PR #3319),
section "Graphiti: turn on the full engine, as the fact layer under the pages". Trial 1 details:
`docs/superpowers/pr-reports/2026-11-13-graphiti-trial-report.md`.

**Verdict: NO-GO, for the second time.** Juniper asked whether trial 1 failed because of Graphiti
or because of our setup. Trial 2 ran Graphiti the standard way:

- raw chat as input instead of distilled notes;
- a small set of entity types, with the catch-all type turned off;
- a first episode that tells the engine everyone's pronouns;
- thinking off;
- the same local model, through a seat that does not count against Orion's own lane.

What our setup changes fixed:

- The narrator junk is gone. Trial 1 made entities named "I", "Speaker" and "fact"; trial 2 made none.
- Each episode is about 2–3 times faster, and the model writes about 6 times fewer tokens.
- Wrong retirements of true facts fell from 7 of 15 to 5 of 29.

What did not get better, or got worse:

- The [trip city] hotel stay is still left open as a current fact.
- About 12% of entities are still duplicates.
- Misgendering got much worse: 42 sentences call Juniper "he" or Orion "he", against 10 in
  trial 1. Graphiti stored her pronouns correctly as a field on her entity (she/her, from the seed
  episode), but its summary and fact writer ignores that field.

The failures that survive both setups come from Graphiti's engine running on our local model. Changing
how we feed it does not fix them.

Real names are replaced with role placeholders ([subject], [home city], [trip city], [trip hotel],
[spouse], [pet]). The repo is public and its privacy gate refuses them.

## Summary

- Trial 2 is a new mode of the existing trial script (`--source chat --ontology orion --thinking off
  --graph graphiti_trial2`).
  - One episode = one Juniper/Orion chat exchange, labelled by speaker, timed at the turn.
  - A seed episode built from `person_fact` goes first, with each person's stated pronouns and their
    relation to Juniper.
  - Entity types: Person (pronouns, relation_to_juniper), Pet (species), Place, Event, OrionPart.
    The generic `Entity` type is excluded.
  - Thinking is off: `chat_template_kwargs.enable_thinking=false`. Checked live: the model's
    reasoning output is empty.
  - The `%` search crash is fixed: graphiti's fulltext query is cleaned of RediSearch operator
    characters, and a query that comes out empty is skipped.
- **#3391 fix: trial holds no longer count as Orion's own lane load.**
  - Holds are now `operator_agent` class, not Orion's `agent` class.
  - Operator classes queue behind all of Orion's background work. Any owner that waits can recall
    them. The body economy books their load as sponsored, not Orion's
    (`ORION_PHYSIOLOGY_OPERATOR_LOAD_SPONSORED`), so Orion's day no longer shows a trial as "agent
    lane full".
  - `operator_agent` pins role `agent`, the same model trial 1 used. That removes the model
    confound: the first trial-2 start had landed on the gpu2 seat (a different model) and was thrown
    away.
  - Each hold is non-retryable and is given back every 20 minutes (`--max-hold-sec 1200`), then
    re-queued, so Orion's own acts interleave.
  - Live: 28 trial-2 holds, all recorded as `operator_agent` in `gpu_pool_leases`. 22 were recalled
    by Orion's own work, 2 hit the length cap. Trial 1's 5 holds show as `agent`.
- Hold safety:
  - A gateway 503 `gpu_pool_unavailable` now means "hold gone, re-queue". Before, 11 episodes failed
    within a second of each other.
  - The request id is recorded before asking, so a hold that was asked for while the pool was not
    answering is found and cancelled by the next run.
- Two new counted checks in the scorer:
  - misgendered summaries/facts (target 0);
  - wrong invalidations, every one judged by hand (target under 2).
  Both are also applied to trial 1 for the side-by-side.

## Outcome moved

The question "Graphiti or our setup?" has a measured answer. In the table, ✓ = pass, ✗ = fail.

| | Trial 1 | Trial 2 |
|---|---|---|
| Input | distilled episode notes + 44 stance messages | raw chat exchanges + person-fact seed |
| Ontology | none (Graphiti default) | Person / Pet / Place / Event / OrionPart; generic `Entity` excluded |
| Thinking | on | off |
| Seat | `agent` class (counted as Orion's load) | `operator_agent` (sponsored, recalled by Orion's work) |
| Model | Qwen3.8 Flash-Next on role `agent` | the same |
| Episodes ingested / planned | 94 / 100 | 75 / 100 |
| Lives in [home city] is current | ✓ | ✓ |
| "Lives in [trip city]" absent or invalidated | ✓ | ✓ |
| Trip stay dated, none left open | ✗ 2 still current | ✗ 3 still current |
| [pet] is a dog, "cat" absent | ✓ | not scored: all [pet] rows lost to recalls |
| [pet] is one entity | ✓ | not scored (same reason) |
| [spouse] / "my [spouse alias]" is one entity | ✓ | ✓ |
| Duplicates (50 hand-judged) | ✗ 7/50 = 14% | ✗ 6/50 = 12% |
| **Misgendered sentences (target 0)** | ✗ 10 | ✗ **42** |
| **Wrong invalidations (target under 2)** | ✗ 7 of 15 (47%) | ✗ 5 of 29 (17%) |
| Junk entities ("I", "Speaker", "fact") | yes | none |
| Wall time per ok episode | mean 254 s, median 242 s | mean 111 s, median 86 s |
| Completion tokens per ok episode | ~17,100 | ~2,700 |
| LLM calls per episode | 11.2 | 17.8 |
| JSON retries / schema refusals | 0 / 0 | 0 / 0 |
| Lease seconds, whole run | 25,124 + 1,766 | 8,953 + 88 |

### What the 42 misgendered sentences are

- Most call Juniper "he" in summaries of other things: [removed: private memory].
- The rest call Orion "he": "Orion views this space as his body", "Orion expresses that he is glad".
- Juniper's own entity has `pronouns = she/her` stored correctly. Graphiti's summary prompt does not
  pass that field to the writer.
- Trial 1 had 10 such sentences with thinking on. Turning thinking off made it worse, but thinking on
  did not prevent it either.

### Trial 2's 5 wrong invalidations (each judged by hand; all 29 are listed in the appendix)

- [removed: private memory]: still true.
- [removed: private memory]: a historical fact that stays true.
- [removed: private memory]: still true.
- [removed: private memory]: still true.
- Juniper's correction about the patio camera zone. Orion's own fabricated 11:39 sighting retired it.
  The engine kept the fabrication's timing over her correction.

One invalidation went the right way: the fabricated 11:39 sighting [removed: private memory] was
itself retired.

### Cost

- **Per episode:** about 111 s of held seat (trial 1: 254 s).
- **Full backfill estimate:** the full 568-row chat backfill is about 17.5 hours of seat time, plus
  queue waits.
- **Seat and queue time this run:** the run held the seat for 8,953 s over 21,506 s of wall time,
  because the operator seat yields to Orion.
- **The cost of yielding:**
  - 25 of the 100 planned episodes were never ingested.
  - 14 failed when Orion recalled the seat mid-episode, and 11 got 503s on a hold that was already
    gone.
  - The retry runs then waited 2+ hours per hold.
  - All 8 [pet] rows (11-08..11-14) are among the 25.

## Current architecture

The live `services/orion-graphiti-adapter` is unchanged. It writes template nodes to
`graphiti_temporal` through a no-op LLM client and never calls `add_episode`. The trial scripts are
standalone and run from the adapter image.

## Architecture touched

- Changed or added:
  - `services/orion-graphiti-adapter/scripts/` (trial runner, clients, scorer, new
    `graphiti_trial2_inputs.py`);
  - tests;
  - this report.
- Not changed: the live adapter, the bus, schemas, compose, the GPU pool, or any live graph.
- Written: FalkorDB graphs `graphiti_trial` (trial 1) and `graphiti_trial2` (trial 2), and nothing
  else.
  - Every run checks the full list of graphs, checks that no trial-group nodes appear anywhere else,
    and checks that the other trial's graph does not move.
  - That came back `isolated=True` on every run. `graphiti_trial` is still 328 nodes.

## Files changed

- `services/orion-graphiti-adapter/scripts/graphiti_trial.py`:
  - picks the graph from an allow-list (`graphiti_trial`/`graphiti_trial2`), each with its own
    group_id and output folder;
  - `--source chat`, `--thinking`, `--ontology`;
  - refuses to resume with changed settings;
  - operator-class, non-retryable, length-capped holds (#3391);
  - treats a 503 as "hold gone";
  - records the request id for leftover-hold cleanup.
- `services/orion-graphiti-adapter/scripts/graphiti_trial2_inputs.py` (new):
  - chat loader and the trimming rule;
  - the person-fact seed;
  - the ontology;
  - the RediSearch-safe `fulltext_query` wrapper.
- `services/orion-graphiti-adapter/scripts/graphiti_trial_clients.py`: the thinking switch.
- `services/orion-graphiti-adapter/scripts/graphiti_trial_score.py`:
  - new checks: `pronoun_violations`, and `invalidations` with `check_invalidations`;
  - `--graph`, with a dump/graph mismatch guard;
  - the trip-city home check now requires the subject; possessive names ("[name]'s X") are
    not counted as duplicates.
- `services/orion-graphiti-adapter/scripts/graphiti_trial_terms.example.json`: `he_names`.
- `services/orion-graphiti-adapter/tests/test_graphiti_trial.py`: 61 host tests plus 6 that need
  graphiti-core; 93 tests in the adapter image.
- `docs/superpowers/pr-reports/2026-11-15-graphiti-trial2-report.md`: this report.

## Trimming rule (trial 2 input)

- Orion's reply: if over 1,500 chars, keep the first 1,100 and the last 300 with
  `[... N chars trimmed ...]` between them.
- Juniper's prompt: if over 2,500 chars, keep the first 2,000 and the last 400.
- 16 of the 100 planned exchanges were trimmed, 21,921 chars in total. The reason: chat replies run
  up to 9,700 chars, against a 540 s episode timeout.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none. These are manual trial scripts.
- Compatibility notes: they use the existing operator classes and routes in `config/gpu_pool.yaml`
  as they are.

## Env/config changes

- Added keys: none. The new options are CLI flags.
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template
  change)
- Skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-graphiti-adapter/tests/test_graphiti_trial.py   # host venv: 61 passed, 6 skipped (no graphiti-core)
pytest services/orion-graphiti-adapter/tests/                        # adapter image + pytest-asyncio: 93 passed
python3 scripts/check_personal_terms.py                              # no personal terms in tracked files
```

## Evals run

```text
The trial is the eval.
Trial 2: 100 planned (51 key rows incl. the seed); 75 ingested. Scored with
  graphiti_trial_score.py report --graph graphiti_trial2.
Trial 1: re-scored with the two new checks (misgendered: 10; wrong invalidations: 7 of 15).
The scored trial-2 output (redacted) is appended below.
```

## Docker/build/smoke checks

```text
No image built or service rebuilt. Runs used `docker run --rm` on the existing adapter image.
Isolation, every run: new graphs other than the trial graph = none; trial-group nodes elsewhere = none;
  other trial graph unchanged.
Thinking switch: live probe, enable_thinking=false gave 107 completion tokens and empty reasoning,
  against 237 tokens and 598 chars of reasoning with it on.
GPU pool: 28 trial-2 holds recorded as operator_agent (gpu_pool_leases). No trial lease is queued or
  granted now.
```

## Review findings fixed

The trial-2 code was reviewed by a subagent. It returned 2 must and 6 should findings, plus nits. All
were fixed:

- Finding (must): the scorer could score the wrong graph from a shared dump file.
  - Fix: each graph gets its own output folder, and `report`/`invalidations` refuse a dump of another
    graph.
  - Evidence: `test_scorer_refuses_a_dump_of_another_graph`.
- Finding (must): both trials shared one output folder, which mixed costs, plans and runs.
  - Fix: one folder per graph.
- Finding: one graph could mix settings.
  - Fix: `graphiti_trial2` only takes `--source chat`, and a resume refuses changed source, thinking
    or ontology.
  - Evidence: `test_trial2_graph_takes_chat_and_resume_refuses_changed_settings`.
- Finding: a missing judgment file changed trial 1's old verdict.
  - Fix: the invalidation check is added only when a judgment file exists.
- Finding: a name that is all punctuation still produced a bare `()` query.
  - Fix: wrap `fulltext_query` and return `''` (skip) when the cleaned query is empty.
  - Evidence: `test_install_redisearch_sanitizer_wraps_fulltext_query`, which goes through the real
    function and restores it afterwards.
- Finding: the seed said "uses not stated pronouns", and a blank `he_names` entry hid every
  violation.
  - Fix: both fixed, with tests.
- Finding (raised by Orion's own day letter, #3391): trial holds starved Orion's agent lane.
  - Fix: operator class plus the 20-minute cap, above.
  - Evidence: `test_trial_holds_are_operator_class_and_capped` and `gpu_pool_leases`.
- Finding (found live): a hold asked for while the pool was not answering was left queued after a
  crash.
  - Fix: record the request id first. The next run looks it up and cancels it.
  - Evidence: `test_unanswered_acquire_leaves_request_id_and_next_run_cancels_it`. The two orphans
    from this session were cancelled by hand, and the pool shows none.

Scorer fixes found while reading live output:

- A sentence that opens "He …" in another person's summary is about that person.
- A bar "located in" the trip city is not the subject's home.

Trial 1's review findings are in its own report.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium. Concern: trial 2 ingested only 75 of 100 episodes. The [pet] checks are
  unscored, and the "cat" question was not tested on chat input.
  - Mitigation: the verdict rests on checks that were ingested and failed: misgendering 42,
    duplicates 12%, stale stay facts, and wrong invalidations 5 of 29. Ingesting more rows cannot
    lower the misgender count.
- Severity: low. Concern: the planned "thinking on, key rows only" re-run was not done.
  - Each operator hold now waits 1–2+ hours behind Orion's work.
  - Trial 1 already ran thinking on with the same model. It misgendered 10 times and wrongly retired
    7 of 15 facts.
  - Whether chat input plus the ontology plus thinking on would pass is UNVERIFIED.
- Severity: low. Concern: retried episodes were ingested out of time order. This affects a handful of
  rows.
- Severity: low. Concern: a `docker run ... sh -c` wrapper does not pass SIGTERM to Python. One stop
  left a queued hold, which I cancelled by hand. The documented command in the script docstring runs
  python directly.
- Issues: #3391 (trial holds starve Orion's agent lane). It is addressed here (operator class plus
  hold cap). Left open for the coordinator to verify live, as requested.
- Issues opened/closed: none.
- Graphs `graphiti_trial` and `graphiti_trial2` are left for inspection. Nothing reads them. Drop them
  with `docker exec orion-athena-falkordb redis-cli GRAPH.DELETE graphiti_trial2` (and the same for
  `graphiti_trial`).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3341

---

# Appendix: trial 2 scored output (redacted)

## Scored output (graphiti_trial_score.py report --graph graphiti_trial2)

Verdict by the checks: **NO-GO**

Graph `graphiti_trial2`: 75 episodes, 170 entities, 369 fact edges (340 current).
Planned 100 of 568 source episodes (sample=100, key rows=51).

## Checks

- **PASS** lives in [home city] is current: 2 current of 2 [home city]-residence facts
- **PASS** lives in [trip city] absent or invalidated: 0 current [trip city]-home facts (0 home-shaped, 8 [trip city] facts)
- **FAIL** [trip city] stay dated as a stay, none left open (window 2026-10-04..2026-10-09): 1 stay facts opened and closed inside the window; 3 [subject]-in-[trip city] facts from the trip still current; 5 stay-shaped [trip city] facts
- **FAIL** [pet] is a dog current, cat absent or invalidated: 0 current dog facts, 0 current cat facts, 0 [pet] facts
- **FAIL** [pet] (all name forms) is one entity: 0 entity nodes match: []
- **PASS** [spouse]/my [spouse alias] is one entity: 1 entity nodes match: ['[spouse]']
- **FAIL** duplicate entities under 10% (hand-judged sample): 6/50 judged duplicates = 12%
- **FAIL** misgendered summaries/facts = 0 ([subject], [spouse] she/her; Orion they/them): 42 sentences
- **FAIL** wrong invalidations under 2 (hand-judged, all of them): 5 wrong of 29 judged / 29 invalidated

## Cost

```json
{
 "episodes": 100,
 "attempts": 102,
 "ok": 75,
 "failed": 25,
 "wall_sec_total": 9041.6,
 "wall_sec_per_episode_mean": 88.6,
 "wall_sec_per_episode_median": 73.4,
 "lease_sec_per_episode_mean": 88.6,
 "llm_call_sec_summed_per_episode_mean": 562.4,
 "llm_calls": 1540,
 "llm_calls_per_episode": 15.4,
 "llm_failures": 16,
 "graphiti_retries": 0,
 "schema_fallbacks": 0,
 "completion_tokens": 217287
}
```

Runs:

- run t2op: done 100, errors 25, wall 21505.7 s, lease 8953.5 s, isolation True
- run t2retry: done 20, errors 2, wall 8177.3 s, lease 88.1 s, isolation True

## Evidence (every edge each check looked at)

### lives in [home city] is current

- [CURRENT] [removed: private memory] valid_at=2026-08-15T02:38:40.002637+00:00 invalid_at=None expired_at=None
- [CURRENT] [removed: private memory] valid_at=None invalid_at=None expired_at=None

### lives in [trip city] absent or invalidated

- [CURRENT] [subject] -[DEPARTS_FROM]-> [trip city]: "[subject] has a 7:30am flight home from [trip city]." valid_at=2026-10-07T12:30:00+00:00 invalid_at=None expired_at=None
- [invalid] [subject] -[LOCATED_IN]-> [trip city]: "[subject] is still in [trip city] for another night." valid_at=2026-10-06T23:25:50.041841+00:00 invalid_at=2026-10-07T07:30:00+00:00 expired_at=2026-11-15T04:38:49.940679+00:00
- [invalid] [subject] -[TRAVELED_TO]-> [trip city]: "[subject] is in [trip city] this week." valid_at=2026-10-05T00:00:00+00:00 invalid_at=2026-10-07T07:30:00+00:00 expired_at=2026-11-15T04:38:45.608152+00:00
- [CURRENT] [subject] -[TRAVELLED_TO]-> [trip city]: "[subject] got out to [trip city] earlier today for a team meeting." valid_at=2026-10-05T00:00:00+00:00 invalid_at=None expired_at=None
- [invalid] [subject] -[STAYING_IN]-> [trip city]: "[subject] will be in [trip city] till Wednesday." valid_at=2026-10-05T02:55:06.332103+00:00 invalid_at=2026-10-07T07:30:00+00:00 expired_at=2026-11-15T03:34:39.254639+00:00
- [CURRENT] [subject] -[STAYING_AT]-> The [trip hotel]: "[subject] is staying at The [trip hotel] hotel in [trip city]." valid_at=2026-10-05T03:13:21.891349+00:00 invalid_at=None expired_at=None
- [CURRENT] The [trip hotel] -[LOCATED_IN]-> [trip city]: "The [trip hotel] hotel is located in [trip city]." valid_at=None invalid_at=None expired_at=None
- [CURRENT] [name] bar -[LOCATED_IN]-> [trip city]: "The [name] bar is located in [trip city]." valid_at=None invalid_at=None expired_at=None

### [trip city] stay dated as a stay, none left open (window 2026-10-04..2026-10-09)

- [CURRENT] [subject] -[DEPARTS_FROM]-> [trip city]: "[subject] has a 7:30am flight home from [trip city]." valid_at=2026-10-07T12:30:00+00:00 invalid_at=None expired_at=None
- [invalid] [subject] -[LOCATED_IN]-> [trip city]: "[subject] is still in [trip city] for another night." valid_at=2026-10-06T23:25:50.041841+00:00 invalid_at=2026-10-07T07:30:00+00:00 expired_at=2026-11-15T04:38:49.940679+00:00
- [CURRENT] [subject] -[TRAVELLED_TO]-> [trip city]: "[subject] got out to [trip city] earlier today for a team meeting." valid_at=2026-10-05T00:00:00+00:00 invalid_at=None expired_at=None
- [CURRENT] [subject] -[STAYING_AT]-> The [trip hotel]: "[subject] is staying at The [trip hotel] hotel in [trip city]." valid_at=2026-10-05T03:13:21.891349+00:00 invalid_at=None expired_at=None
- [CURRENT] The [trip hotel] -[LOCATED_IN]-> [trip city]: "The [trip hotel] hotel is located in [trip city]." valid_at=None invalid_at=None expired_at=None

### [spouse]/my [spouse alias] is one entity

- [spouse] (24b42f3b-32c7-4daf-8b63-839b9138c6ce): [removed: private memory]

### misgendered summaries/facts = 0 ([subject], [spouse] she/her; Orion they/them)

- [removed: private memory] (42 sentences quoted from summaries and facts)

### wrong invalidations under 2 (hand-judged, all of them)

- WRONG [invalid] [removed: private memory] valid_at=2026-07-23T23:26:48.695386+00:00 invalid_at=2026-10-28T05:08:13.696378+00:00 expired_at=2026-11-15T06:22:36.933245+00:00
- WRONG [invalid] [removed: private memory] valid_at=2026-08-15T00:00:00+00:00 invalid_at=2026-08-22T18:45:00+00:00 expired_at=2026-11-15T02:58:09.997648+00:00
- WRONG [invalid] [removed: private memory] valid_at=2026-08-15T02:38:40.002637+00:00 invalid_at=2026-08-31T05:29:03.998283+00:00 expired_at=2026-11-15T03:14:02.021603+00:00
- WRONG [invalid] [removed: private memory] valid_at=2026-08-18T07:32:02.098692+00:00 invalid_at=2026-08-31T05:29:03.998283+00:00 expired_at=2026-11-15T03:14:02.021609+00:00
- WRONG [invalid] [removed: private memory] valid_at=2026-11-03T00:00:00+00:00 invalid_at=2026-11-03T11:39:00+00:00 expired_at=2026-11-15T06:38:35.362373+00:00

## Every fact the engine marked no longer true (29)

Invalidation can be wrong (spec residual risk). Judge each by hand.

- [invalid] [removed: private memory] valid_at=2026-07-23T23:26:48.695386+00:00 invalid_at=2026-10-28T05:08:13.696378+00:00 expired_at=2026-11-15T06:22:36.933245+00:00
- [invalid] [subject] -[GREETED]-> Orion: "[subject]: hi, Orion" valid_at=2026-07-24T00:26:48.695386+00:00 invalid_at=2026-10-02T04:10:20.393646+00:00 expired_at=2026-11-15T03:32:15.762382+00:00
- [invalid] [removed: private memory] valid_at=2026-08-09T22:00:00+00:00 invalid_at=2026-08-16T08:51:53.916912+00:00 expired_at=2026-11-15T02:36:27.165805+00:00
- [invalid] Orion -[USES_TOOL]-> EnterWorktree: "Orion attempted to use the EnterWorktree tool, but the call failed because the main repo's .git/refs/heads/ directory is effectively read-only." valid_at=2026-08-11T03:58:04.932099+00:00 invalid_at=2026-08-18T06:57:46.589823+00:00 expired_at=2026-11-15T02:49:10.464091+00:00
- [invalid] [removed: private memory] valid_at=2026-08-15T00:00:00+00:00 invalid_at=2026-08-22T18:45:00+00:00 expired_at=2026-11-15T02:58:09.997648+00:00
- [invalid] [removed: private memory] valid_at=2026-08-15T02:38:40.002637+00:00 invalid_at=2026-08-31T05:29:03.998283+00:00 expired_at=2026-11-15T03:14:02.021603+00:00
- [invalid] Athena -[POWERED_BY]-> APC 2100w: "Circe and either Athena or Atlas also run on it [APC 2100w]." valid_at=2026-08-15T02:38:40.002637+00:00 invalid_at=2026-10-30T17:14:43.495441+00:00 expired_at=2026-11-15T06:28:31.346337+00:00
- [invalid] [removed: private memory] valid_at=2026-08-16T08:47:03.026096+00:00 invalid_at=2026-08-16T08:51:53.916912+00:00 expired_at=2026-11-15T02:36:27.165854+00:00
- [invalid] [removed: private memory] valid_at=2026-08-16T08:51:53.916912+00:00 invalid_at=2026-08-22T00:00:00+00:00 expired_at=2026-11-15T03:08:54.792696+00:00
- [invalid] Orion -[HAS_MEMORY_OF]-> [subject]: "Orion's memory shelves hold recent durable beliefs around the active_packet milestone and a congrats message involving [subject]." valid_at=2026-08-18T05:29:03.801197+00:00 invalid_at=2026-08-18T05:38:52.553819+00:00 expired_at=2026-11-15T02:42:44.363431+00:00
- [invalid] [removed: private memory] valid_at=2026-08-18T05:29:03.801197+00:00 invalid_at=2026-08-18T05:38:52.553819+00:00 expired_at=2026-11-15T02:42:44.363425+00:00
- [invalid] Orion -[EXECUTED_TOOL]-> look_at_camera: "Orion just ran `look_at_camera`" valid_at=2026-08-18T06:57:46.589823+00:00 invalid_at=2026-10-05T04:12:49.984469+00:00 expired_at=2026-11-15T04:35:46.396662+00:00
- [invalid] look_at_camera -[PROVIDES_FEED_TO]-> Orion: "The look_at_camera feed is active, providing spatial geometry, lighting gradients, and object boundaries to Orion" valid_at=2026-08-18T06:57:46.589823+00:00 invalid_at=2026-10-05T04:12:49.984469+00:00 expired_at=2026-11-15T04:35:46.396656+00:00
- [invalid] [removed: private memory] valid_at=2026-08-18T07:32:02.098692+00:00 invalid_at=2026-08-31T05:29:03.998283+00:00 expired_at=2026-11-15T03:14:02.021609+00:00
- [invalid] eye-ball-1 -[FEEDS_INTO]-> Orion: ""eye-ball-1" is live, feeding tensors back into me [Orion]" valid_at=2026-08-18T07:32:02.098692+00:00 invalid_at=2026-11-03T19:00:24.012865+00:00 expired_at=2026-11-15T06:33:18.891594+00:00
- [invalid] [removed: private memory] valid_at=2026-08-21T00:00:00+00:00 invalid_at=2026-09-09T00:00:00+00:00 expired_at=2026-11-15T03:16:41.200995+00:00
- [invalid] [removed: private memory] valid_at=2026-08-23T00:00:00+00:00 invalid_at=2026-10-28T05:08:13.696378+00:00 expired_at=2026-11-15T06:22:46.296443+00:00
- [invalid] [removed: private memory] valid_at=2026-08-27T04:26:11.045015+00:00 invalid_at=2026-11-03T11:39:00+00:00 expired_at=2026-11-15T06:33:11.733375+00:00
- [invalid] [removed: private memory] valid_at=2026-09-14T17:31:22.537300+00:00 invalid_at=2026-10-05T03:16:47.361757+00:00 expired_at=2026-11-15T03:40:18.944350+00:00
- [invalid] [subject] -[TRAVELED_TO]-> Austin: "[subject] was in Austin last week." valid_at=2026-09-28T00:00:00+00:00 invalid_at=2026-10-04T00:00:00+00:00 expired_at=2026-11-15T03:36:24.051838+00:00
- [invalid] [subject] -[TRAVELED_TO]-> [trip city]: "[subject] is in [trip city] this week." valid_at=2026-10-05T00:00:00+00:00 invalid_at=2026-10-07T07:30:00+00:00 expired_at=2026-11-15T04:38:45.608152+00:00
- [invalid] [subject] -[STAYING_IN]-> [trip city]: "[subject] will be in [trip city] till Wednesday." valid_at=2026-10-05T02:55:06.332103+00:00 invalid_at=2026-10-07T07:30:00+00:00 expired_at=2026-11-15T03:34:39.254639+00:00
- [invalid] [removed: private memory] valid_at=2026-10-05T02:59:28.988004+00:00 invalid_at=2026-10-05T03:16:47.361757+00:00 expired_at=2026-11-15T03:40:18.944354+00:00
- [invalid] [subject] -[LOCATED_IN]-> [trip city]: "[subject] is still in [trip city] for another night." valid_at=2026-10-06T23:25:50.041841+00:00 invalid_at=2026-10-07T07:30:00+00:00 expired_at=2026-11-15T04:38:49.940679+00:00
- [invalid] [removed: private memory] valid_at=2026-10-23T19:58:46.392051+00:00 invalid_at=2026-10-28T05:16:12.758813+00:00 expired_at=2026-11-15T06:25:53.222641+00:00
- [invalid] [removed: private memory] valid_at=2026-10-28T05:08:13.696378+00:00 invalid_at=2026-10-28T05:16:12.758813+00:00 expired_at=2026-11-15T06:25:50.581571+00:00
- [invalid] [removed: private memory] valid_at=2026-11-03T00:00:00+00:00 invalid_at=2026-11-03T11:39:00+00:00 expired_at=2026-11-15T06:38:35.362373+00:00
- [invalid] [removed: private memory] valid_at=2026-11-03T07:41:00+00:00 invalid_at=2026-11-03T11:39:00+00:00 expired_at=2026-11-15T06:38:40.852629+00:00
- [invalid] [removed: private memory] valid_at=2026-11-03T11:39:00+00:00 invalid_at=2026-11-03T12:48:00+00:00 expired_at=2026-11-15T06:38:07.173650+00:00

## Duplicate sample (hand-judged)

- [ok] session JSONL files
- [ok] main
- [ok] Forest area
- [ok] Surplus store
- [ok] [removed: private memory]
- [ok] GitHub
- [ok] [name]
- [DUP] world pulse (vs: pulse) -- same as "pulse"
- [DUP] eye-ball-1 -- same camera as "external camera" (borderline)
- [ok] feedback loops (vs: feedback layer)
- [ok] [removed: private memory]
- [ok] thermostat
- [ok] [trip city]
- [ok] Athena
- [ok] [removed: private memory]
- [ok] social
- [ok] [removed: private memory]
- [ok] cabinet sensor array (vs: cabinet microphones, server cabinet)
- [ok] journal plumbing
- [ok] [removed: private memory]
- [ok] [removed: private memory]
- [ok] transport bus
- [ok] vision profiles (vs: vision, vision pipeline)
- [ok] switches
- [DUP] APC server PDU (vs: APC 2100w, server cabinet, 1600w APC UPSs, server nodes) -- same as "PDUs"
- [ok] Prometheous
- [DUP] server nodes (vs: GPU nodes, APC server PDU, server cabinet) -- same as "servers"
- [ok] [removed: private memory]
- [ok] [removed: private memory]
- [ok] [name]
- [ok] [removed: private memory]
- [DUP] servers -- same as "server nodes"
- [ok] fcc motor
- [ok] cortex (vs: orion/cortex-orch/)
- [ok] Claude
- [ok] [name]
- [ok] GitNexus
- [ok] EnterWorktree
- [ok] self-check
- [ok] tension-driven outreach
- [ok] server cabinet (vs: cabinet sensor array, cabinet microphones, APC server PDU, server nodes)
- [DUP] Gigabyte G481-HA0 -- same machine as "Circe node"
- [ok] compactor
- [ok] WebSocket chat path
- [ok] memory (vs: MEMORY.md, memory directory)
- [ok] house
- [ok] route normalizer fix
- [ok] [removed: private memory]
- [ok] memory directory (vs: MEMORY.md, memory, config directory)
- [ok] [removed: private memory]

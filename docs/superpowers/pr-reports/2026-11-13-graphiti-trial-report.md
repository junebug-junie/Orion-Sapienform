# Graphiti full-engine trial (Patch 0): NO-GO

Spec: `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md` (PR #3319),
section "Graphiti: turn on the full engine, as the fact layer under the pages".

**Verdict: NO-GO.** Graphiti's real engine runs on Orion's local model. Every structured-output call
worked. But the engine gets too many facts wrong to be the fact layer under the pages:

- It left the hotel stay in the trip city open, so "[subject] is in [trip city]" is still a current fact.
- 14% of a 50-entity sample are duplicates. The bar was under 10%.
- 7 of the 15 facts it marked "no longer true" were still true. One of them is "[subject] refers to
  [spouse] as [her] wife".

It is also slow: about 4 minutes of a held agent seat per short memory sentence.

Per the spec's fail branch: pages fold straight from `episode_memory`, and the "corrections outrank"
rule carries the load alone.

Real names are replaced with role placeholders ([subject], [home city], [trip city], [trip hotel],
[spouse], [pet]). The repo is public and its privacy gate refuses them.

## Summary

- Added a runnable trial: `services/orion-graphiti-adapter/scripts/graphiti_trial.py`.
  - It feeds Orion's episode memories and the 44 high-salience stance messages through graphiti-core
    0.19's real `add_episode`.
  - It writes only to the FalkorDB graph `graphiti_trial`.
  - It runs on one GPU pool hold on the agent lane.
- Added local-model clients for Graphiti (`graphiti_trial_clients.py`):
  - an LLM client that sends a real `json_schema` grammar through llm-gateway's OpenAI passthrough,
    with the hold header;
  - a strict embedder over orion-vector-host;
  - a no-op reranker.
- Added a scorer, `graphiti_trial_score.py`, with the spec's checks, cost totals, a hand-judged
  duplicate sample, a list of every invalidated fact, and a redacted copy of the report that is safe
  to commit.
- Ran a 100-episode representative sample: all 45 key rows plus 55 seeded random rows. 94 were
  ingested and scored. The full run would take about 35 hours (see Cost).
- Did not change the live adapter. It still writes template nodes through a no-op LLM client. Nothing
  reads `graphiti_trial`.

## Outcome moved

The go/no-go question has an answer with numbers. Before this PR, the json_schema compatibility was
UNVERIFIED, and so was the engine's quality on Orion's data.

### Checks (94 episodes ingested, 234 entities, 424 fact edges, 409 current)

| Check | Result | What happened |
|---|---|---|
| lives in [home city] is current | PASS | 4 current residence facts, none invalidated |
| lives in [trip city] absent or invalidated | PASS | No episode ever said "lives in [trip city]", so the engine never made that fact. The correction was stored as a current `NOT_LIVES_IN` fact. |
| [trip city] stay dated as a stay, none left open | **FAIL** | 3 trip facts were closed correctly. "until Wednesday" closed on 10-07, "another night" closed on 10-07 07:30, and "team meeting" closed late, on 10-18. Two facts are still current: "[subject] is staying at a hotel in [trip city], so [subject] is in [trip city]" and "[subject] is staying at the [trip hotel]". |
| [pet] is a dog current, "cat" invalidated or absent | PASS | "el [pet] is a dog" is current, plus a current "el [pet] is not a cat". No episode ever said "cat", so there was nothing to invalidate. |
| el [pet] / [pet] is one entity | PASS | 1 entity |
| [spouse] / "my [spouse alias]" is one entity | PASS | 1 entity. "my wife" in Juniper's own messages resolved to [spouse]. |
| Duplicate entities under 10% (50 hand-judged) | **FAIL** | 7/50 = 14%. It is still 5/50 = 10% if the two borderline calls (dBA, gpu3) are dropped. |

### Invalidation quality (hand-judged, all 15 invalidated facts)

8 were right:

- the trip-city stays;
- "on her way home from the hospital";
- "36 PRs in the last 24 hours";
- the dated surgery-tomorrow messages;
- the 12:48 mailbox sighting.

7 were wrong:

- "[subject] refers to [spouse] as [her] wife". A later unrelated message marked it no longer true.
- "needs to spend time with [her] wife".
- "spends all [her] time with Orion".
- "Catherine is [spouse]'s sister and Alice's mom".
- "[spouse] might have multiple surgeries through December".
- Both "the surgery is intended to improve vascular capabilities" facts.

A wrongly retired true fact is the dangerous failure the spec named. Here it happens to about half of
all invalidations.

### Other quality findings

- 11 entity summaries call [subject] "he". She is she/her. Graphiti's summary prompt assumed a gender.
- The [spouse] summary says she is "Alice's mother". The source says Catherine is Alice's mother.
- There are junk entities: "I", "Speaker", "fact", "people", "system". The episodes are written in
  Orion's first person ("Juniper told me…"), and Graphiti turned the narrator into entities.
- There is a deterministic graphiti 0.19 + FalkorDB bug: an entity name containing `%` ("38% outreach
  figure") breaks the fulltext query with `RediSearch: Syntax error`, so that episode always fails.

### Cost (hold-seconds = seconds the GPU pool seat was held for this work)

- Wall time per episode: mean 254 s, median 242 s, p90 390 s. Hold-seconds per episode is the same,
  254 s.
- 11.2 LLM calls per episode. 1,119 calls in total.
- About 17,000 completion tokens per episode. Most of them are the model's thinking.
- Structured output:
  - 0 Graphiti JSON retries and 0 `json_schema` refusals in 1,119 calls. llama.cpp `json_schema`
    works with Graphiti 0.19's schemas.
  - The UNVERIFIED in the spec is now verified.
- Failures: 6 of 100 episodes.
  - 3 were connection errors, because llm-gateway was restarted mid-run at 23:58 UTC.
  - 2 hit the 540 s episode timeout. Both were long messages, with 28 and 9 calls.
  - 1 was the `%` RediSearch bug.
- Hold totals: 25,124 hold-seconds for the 100-episode run, plus 1,766 for the discarded retry.
- Queue waits before each hold was granted were 22 to 45 minutes. Orion's own durable runs fill the
  agent lane.
- **A full 494-episode backfill would take about 35 hours** (494 × 254 s) of one agent seat, plus
  queue waits.
- Live, it would cost about 49 minutes of agent seat per day (about 11.6 episodes a day × 254 s).
- Episodes got a little slower as the graph grew. The first 20 episode-memory rows averaged 224 s.
  The last 20 averaged 232 s.

## Current architecture

`services/orion-graphiti-adapter` (`app/backends/graphiti_core.py`):

- It writes fixed-template EntityNode and EntityEdge payloads to FalkorDB `graphiti_temporal`.
- It passes a no-op LLM client and never calls `add_episode`.
- `/v1/search` embeds through orion-vector-host.

## Architecture touched

- New, standalone trial scripts under `services/orion-graphiti-adapter/scripts/`. They run in the
  adapter image, never in the live container.
- No change to `app/`, the bus, schemas, compose, or the live graph.
- Seams used, read or call only:
  - GPU pool hold verbs over the bus (`orion/gpu_pool/client.py`);
  - llm-gateway OpenAI passthrough `POST /v1/chat/completions` with `model=agent` and
    `X-Orion-Gpu-Lease`;
  - orion-vector-host `/embedding`;
  - Postgres `episode_memory` and `memory_crystallizations`, read-only.

## Files changed

- `services/orion-graphiti-adapter/scripts/graphiti_trial.py`: the runner.
  - It holds the agent seat, with heartbeat, recall and leak-safe cancel/release. `hold.json` records
    the hold so a killed run's hold gets cancelled.
  - It logs progress, writes per-episode cost to `results.jsonl`, and can resume.
  - `--plan-file` re-runs exactly the same sample.
  - An isolation guard checks that no other graph appears and that no trial nodes appear elsewhere.
- `services/orion-graphiti-adapter/scripts/graphiti_trial_clients.py`: the Graphiti LLM client
  (json_schema, no retry on truncation or a dead hold, sticky schema fallback), the strict embedder
  and the no-op reranker.
- `services/orion-graphiti-adapter/scripts/graphiti_trial_score.py`: the checks, cost totals, the
  duplicate sample and judgments, the invalidated-fact list, and the redacted report.
- `services/orion-graphiti-adapter/scripts/graphiti_trial_terms.example.json`: placeholder names. The
  real ones live in a gitignored `.local.json`.
- `services/orion-graphiti-adapter/tests/test_graphiti_trial.py`: 39 tests.
- `.github/workflows/graphiti-trial-tests.yml`: a CI job for those tests.
- `services/orion-graphiti-adapter/README.md`: how to run and score the trial.
- `.gitignore`: `services/orion-graphiti-adapter/scripts/*.local.json`.
- `docs/superpowers/pr-reports/2026-11-13-graphiti-trial-report.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none. The trial is a manual script.
- Compatibility notes: the trial uses the existing GPU pool hold verbs and the gateway passthrough as
  they are.

## Env/config changes

- Added keys: none. The trial takes CLI flags. It reads `ORION_BUS_URL`, `POSTGRES_URI` and
  `CRYSTALLIZER_EMBED_HOST_URL` if they are set, with the tailscale bus address as the default.
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template
  change)
- Skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-graphiti-adapter/tests/test_graphiti_trial.py   # host venv: 34 passed, 5 skipped (graphiti-core absent)
pytest services/orion-graphiti-adapter/tests/                        # adapter image + pytest-asyncio: 59 passed (incl. graphiti client subclass tests)
python3 scripts/check_personal_terms.py                              # no personal terms in tracked files
CI: graphiti-trial (pass), Static repo gates (pass after the privacy fix)
```

## Evals run

```text
The trial is the eval. 100-episode sample: 94 ingested. Scored with graphiti_trial_score.py report.
The scored report (redacted) is appended below.
Smoke first: 3 episodes, 0 errors, json_schema accepted. A direct probe of 4 Graphiti schemas through
the agent passthrough: all returned valid JSON.
```

## Docker/build/smoke checks

```text
No image built or service rebuilt. The trial ran in `docker run --rm` containers from the existing
orion-graphiti-adapter image, with the worktree's scripts and orion/ mounted read-only.
Isolation guard, every run: new_graphs_other_than_trial=[], trial_group_nodes_elsewhere={}, isolated=True.
Hold lifecycle seen live:
- 3 holds were granted and released.
- 2 queued holds were cancelled by docker stop (SIGTERM). The pool showed no graphiti-trial lease
  afterward.
```

## Review findings fixed

The first review was run in a subagent on the full diff. It returned 2 must, 14 should and several nit
findings. Everything material was fixed:

- Finding: the home-city scorer dropped the real correction fact ("lives in [home], not [trip city]")
  as negated. The trip-city scorer had the mirror hole.
  - Fix: the negation must govern the place (`negated_before`).
  - Evidence: tests cover both phrasings.
- Finding: a timed-out or killed episode left a half-saved episode that resume then skipped.
  - Fix: an episode is done only when it is in the graph AND has an ok row. Partial episodes are
    removed with `remove_episode`.
  - Evidence: `test_resume_plan_*`.
- Finding: the hold leaked on a failed first RPC, on the exit path, and when heartbeats failed.
  - Fix: re-ask idempotently to learn the lease id and cancel it. Shield the release. Use
    `hold.json` for leftovers. Mark the hold lost after 3 missed beats. Add
    `--stop-timeout 60`.
  - Evidence: 6 hold-lifecycle tests and live cancels.
- Finding: the protected-graph node-count guard was noise. Live graphs change on their own.
  - Fix: check the graph list plus a count of trial-group nodes in every other graph.
  - Evidence: `isolated=True` in every run.
- Finding: the cost totals dropped retried attempts.
  - Fix: cost is summed over all attempts.
  - Evidence: `test_cost_summary_counts_every_attempt_but_last_outcome`.
- Finding: truncation was retried at the same max_tokens. Any 400 counted as a schema refusal. There
  was no transport retry.
  - Fix: RefusalError for truncation and for a dead hold. The fallback runs only for schema errors and
    is sticky. `max_retries=2`.
  - Evidence: client subclass tests.
- Finding: the episode timeout of 1,800 s was longer than the pool's 600 s recall grace.
  - Fix: the timeout is now 540 s.
- Finding: re-running `sample` overwrote the hand judgments.
  - Fix: it refuses without `--force`.
- Finding: the [pet]/cat and "home from" checks gave false failures.
  - Fix: tighter predicates, with tests.

Issues found by the live run, not by review:

- The privacy CI gate failed on real names.
  - Fix: the terms file and redaction.
- A retry with `--sample` re-drew the sample, because new rows had landed in Postgres. It ingested 4
  out-of-sample rows.
  - Fix: `--plan-file`. The 4 were removed with `remove_episode`.
  - Check: none of the 4 had invalidated any fact. No `expired_at` was set after the retry started.

A second review covered the post-review commits. Its findings are listed in the PR body.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium. Concern: sample, not full run.
  - 94 of 494 episodes, chosen so every row the checks are about was included.
  - Mitigation: the failure modes found (stale open stay, wrong invalidations, duplicates) come from
    the engine's design and the model, not from sample size. The full run is about 35 hours and was
    not run.
- Severity: low. Concern: six episodes were not ingested.
  - 3 are [spouse]-related rows from 11-03, lost to connection errors during the gateway restart.
  - The retry was cancelled after a 45-minute queue wait. It would not change the verdict: two checks
    fail on rows that were ingested.
- Severity: low. Concern: thinking was left on (the model's default).
  - Turning it off could cut cost several-fold. It was not tested. Whether quality holds without
    thinking is UNVERIFIED.
- Severity: low. Concern: the graph `graphiti_trial` is left in place for inspection. Nothing reads it.
  - Drop it with `docker exec orion-athena-falkordb redis-cli GRAPH.DELETE graphiti_trial`.
- Issues opened: none. The NO-GO means the Graphiti bugs found here (`%` fulltext escape, gendered
  summaries) have no live consumer to fix them for.
- Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3341

---

# Appendix: scored report (redacted)

## Scored output (graphiti_trial_score.py report)

Verdict by the checks: **NO-GO**

Graph `graphiti_trial`: 94 episodes, 234 entities, 424 fact edges (409 current).
Planned 100 of 494 source episodes (sample=100, key rows=45).

## Checks

- **PASS** lives in [home city] is current: 4 current of 4 [home city]-residence facts
- **PASS** lives in [trip city] absent or invalidated: 0 current [trip city]-home facts (0 home-shaped, 10 [trip city] facts)
- **FAIL** [trip city] stay dated as a stay, none left open (window 2026-10-04..2026-10-09): 1 stay facts opened and closed inside the window; 2 [subject]-in-[trip city] facts from the trip still current; 7 stay-shaped [trip city] facts
- **PASS** [pet] is a dog current, cat absent or invalidated: 3 current dog facts, 0 current cat facts, 10 [pet] facts
- **PASS** [pet] (all name forms) is one entity: 1 entity nodes match: ['el [pet]']
- **PASS** [spouse]/my [spouse alias] is one entity: 1 entity nodes match: ['[spouse]']
- **FAIL** duplicate entities under 10% (hand-judged sample): 7/50 judged duplicates = 14%

## Cost

```json
{
 "episodes": 100,
 "attempts": 101,
 "ok": 94,
 "failed": 6,
 "wall_sec_total": 25664.0,
 "wall_sec_per_episode_mean": 254.1,
 "wall_sec_per_episode_median": 241.8,
 "lease_sec_per_episode_mean": 254.1,
 "llm_call_sec_summed_per_episode_mean": 1187.1,
 "llm_calls": 1119,
 "llm_calls_per_episode": 11.2,
 "llm_failures": 9,
 "graphiti_retries": 0,
 "schema_fallbacks": 0,
 "completion_tokens": 1705570
}
```

Runs:

- run sample100-retry: done 26, errors 1, wall 2069.5 s, lease 1766.5 s, isolation True
- run sample100-retry2: done 0, errors 0, wall 2260.3 s, lease 0.0 s, isolation True
- run sample100: done 100, errors 6, wall 27390.1 s, lease 25124.0 s, isolation True

## Evidence (every edge each check looked at)

### lives in [home city] is current

- [CURRENT] [subject] -[LIVES_IN]-> [home city], UT: "We live in [home city], UT." valid_at=2026-08-15T02:38:40.723711+00:00 invalid_at=None expired_at=None
- [CURRENT] [subject] -[CORRECTED]-> Circe node: "[subject] corrected me that she lives in [home city], Utah, not [trip city]." valid_at=2026-10-06T03:50:17.429363+00:00 invalid_at=None expired_at=None
- [CURRENT] Orion -[LIVES_IN]-> [home city], UT: "[subject] and Orion live in [home city], Utah." valid_at=2026-10-05T03:51:48.036489+00:00 invalid_at=None expired_at=None
- [CURRENT] Orion -[LIVES_IN]-> [home city], UT: "We live in [home city], UT." valid_at=2026-08-15T02:38:40.723711+00:00 invalid_at=None expired_at=None

### lives in [trip city] absent or invalidated

- [CURRENT] [subject] -[CORRECTED]-> Circe node: "[subject] corrected me that she lives in [home city], Utah, not [trip city]." valid_at=2026-10-06T03:50:17.429363+00:00 invalid_at=None expired_at=None
- [invalid] [subject] -[STAYING_IN]-> [trip city]: "[subject] is staying in [trip city] for another night." valid_at=2026-10-07T00:25:00+00:00 invalid_at=2026-10-07T07:30:00+00:00 expired_at=2026-11-13T21:16:18.133347+00:00
- [CURRENT] [subject] -[NOT_LIVES_IN]-> [trip city]: "[subject] does not live in [trip city]." valid_at=2026-10-06T03:50:17.429363+00:00 invalid_at=None expired_at=None
- [CURRENT] [subject] -[LOCATED_IN]-> [trip city]: "[subject] is staying at a hotel in [trip city], so [subject] is in [trip city]." valid_at=2026-10-05T03:07:00+00:00 invalid_at=None expired_at=None
- [invalid] [subject] -[LOCATED_IN]-> [trip city]: "[subject] is in [trip city] and will be there until Wednesday." valid_at=2026-10-05T02:55:00+00:00 invalid_at=2026-10-07T00:00:00+00:00 expired_at=2026-11-13T20:36:25.713701+00:00
- [invalid] [subject] -[ATTENDS]-> team meeting: "[subject] is in [trip city] for a team meeting." valid_at=2026-10-05T02:55:00+00:00 invalid_at=2026-10-18T08:40:00+00:00 expired_at=2026-11-13T21:58:40.397191+00:00
- [CURRENT] team meeting -[LOCATED_IN]-> [trip city]: "The team meeting is in [trip city]." valid_at=2026-10-05T02:55:00+00:00 invalid_at=None expired_at=None
- [CURRENT] The [trip hotel] -[LOCATED_IN]-> [trip city]: "The [trip hotel] is in [trip city]." valid_at=2026-10-05T03:07:00+00:00 invalid_at=None expired_at=None
- [CURRENT] The [trip hotel] -[LOCATED_ON]-> Lake Michigan: "The [trip hotel] is right on the lake in [trip city], i.e., Lake Michigan." valid_at=2026-10-05T03:07:00+00:00 invalid_at=None expired_at=None
- [CURRENT] Speaker -[NOTED_TIME_CHANGE]-> [trip city]: "Speaker noted that the 4:30am CDT wakeup is brutal due to the [trip city] time change." valid_at=2026-10-07T03:45:00+00:00 invalid_at=None expired_at=None

### [trip city] stay dated as a stay, none left open (window 2026-10-04..2026-10-09)

- [invalid] [subject] -[STAYING_IN]-> [trip city]: "[subject] is staying in [trip city] for another night." valid_at=2026-10-07T00:25:00+00:00 invalid_at=2026-10-07T07:30:00+00:00 expired_at=2026-11-13T21:16:18.133347+00:00
- [CURRENT] [subject] -[LOCATED_IN]-> [trip city]: "[subject] is staying at a hotel in [trip city], so [subject] is in [trip city]." valid_at=2026-10-05T03:07:00+00:00 invalid_at=None expired_at=None
- [invalid] [subject] -[ATTENDS]-> team meeting: "[subject] is in [trip city] for a team meeting." valid_at=2026-10-05T02:55:00+00:00 invalid_at=2026-10-18T08:40:00+00:00 expired_at=2026-11-13T21:58:40.397191+00:00
- [CURRENT] [subject] -[STAYING_AT]-> The [trip hotel]: "[subject] is staying at the [trip hotel]." valid_at=2026-10-05T03:07:00+00:00 invalid_at=None expired_at=None
- [CURRENT] team meeting -[LOCATED_IN]-> [trip city]: "The team meeting is in [trip city]." valid_at=2026-10-05T02:55:00+00:00 invalid_at=None expired_at=None
- [CURRENT] The [trip hotel] -[HAS_BRAND]-> Marriott: "The [trip hotel] is a Marriott hotel." valid_at=2026-10-05T03:07:00+00:00 invalid_at=None expired_at=None
- [CURRENT] Speaker -[NOTED_TIME_CHANGE]-> [trip city]: "Speaker noted that the 4:30am CDT wakeup is brutal due to the [trip city] time change." valid_at=2026-10-07T03:45:00+00:00 invalid_at=None expired_at=None

### [pet] is a dog current, cat absent or invalidated

- [CURRENT] [subject] -[REPORTED_INCIDENT_INVOLVING]-> [spouse]: "[subject] told me that [spouse] tripped over el [pet] and fell down the stairs, requiring a CT scan." valid_at=None invalid_at=None expired_at=None
- [CURRENT] [subject] -[TOLD]-> I: "[subject] told me that el [pet] had a difficult post-anesthesia recovery night." valid_at=None invalid_at=None expired_at=None
- [CURRENT] [subject] -[HAS_DOG]-> el [pet]: "[subject]'s dog is el [pet]." valid_at=2026-11-11T03:34:00+00:00 invalid_at=None expired_at=None
- [CURRENT] [spouse] -[TRIPPED_OVER]-> el [pet]: "[spouse] tripped over el [pet]" valid_at=None invalid_at=None expired_at=None
- [CURRENT] el [pet] -[REPORTED_BY]-> [subject]: "el [pet] had a difficult post-anesthesia recovery night, as reported by [subject]." valid_at=None invalid_at=None expired_at=None
- [CURRENT] el [pet] -[WAS_TAKEN_TO_BY_JUNIPER]-> hospital: "[subject] took her dog, el [pet], to the hospital." valid_at=None invalid_at=None expired_at=None
- [CURRENT] el [pet] -[IS_A]-> dog: "el [pet] is a dog" valid_at=2026-11-11T20:30:00+00:00 invalid_at=None expired_at=None
- [CURRENT] el [pet] -[IS_NOT_A]-> cat: "el [pet] is not a cat" valid_at=2026-11-11T20:30:00+00:00 invalid_at=None expired_at=None
- [CURRENT] el [pet] -[UNDERWENT_SURGICAL_PROCEDURE]-> surgery: "el [pet] underwent a surgical rectal prolapse repair." valid_at=None invalid_at=None expired_at=None
- [CURRENT] el [pet] -[HAD_CONDITION]-> rectal prolapse: "el [pet] had rectal prolapse." valid_at=None invalid_at=None expired_at=None

### [pet] (all name forms) is one entity

- el [pet] (ea25f901-99ae-4e86-9314-6296b040bf47): el [pet] is [subject]'s dog, not a cat; [subject] asked the speaker to correct this in records. [subject] took el [pet] to the hospital for a surgical rectal prolapse repair after he would not stop scooting. 

### [spouse]/my [spouse alias] is one entity

- [spouse] (16ed03d3-652b-43be-a1d9-a05084a083ba): [spouse] is [subject]'s [spouse alias], Catherine's sister, and Alice's mother. She has significant vascular health issues (DVT/PE), limited walking, and possible multiple surgeries through December; a surgery was 

## Every fact the engine marked no longer true (15)

Invalidation can be wrong (spec residual risk). Judge each by hand.

- [invalid] [subject] -[SPENDS_TIME_WITH]-> Orion: "[subject] says he spends all his time with Orion." valid_at=2026-08-15T03:09:45.192963+00:00 invalid_at=2026-10-18T01:55:00+00:00 expired_at=2026-11-13T21:53:05.668506+00:00
- [invalid] [subject] -[HAS_WIFE]-> [spouse]: "[subject] refers to [spouse] as his [spouse alias]." valid_at=2026-08-15T03:09:45.192963+00:00 invalid_at=2026-10-18T01:55:00+00:00 expired_at=2026-11-13T21:53:05.668496+00:00
- [invalid] [subject] -[NEEDS_TO_SPEND_TIME_WITH]-> [spouse]: "[subject] says he needs to spend time with his [spouse alias] [spouse]." valid_at=2026-08-15T03:09:45.192963+00:00 invalid_at=2026-10-18T01:55:00+00:00 expired_at=2026-11-13T21:53:05.668512+00:00
- [invalid] [subject] -[COMMUNICATES_WITH]-> Orion: "[subject] (to Orion): Yasss! I'm ready to dive into sleep. [spouse], my [spouse alias], has surgery tomorrow early so gotta be fresh to support. Best!" valid_at=2026-08-20T05:54:45.793561+00:00 invalid_at=2026-10-21T12:00:00+00:00 expired_at=2026-11-13T22:28:13.754089+00:00
- [invalid] [subject] -[SUPPORTS]-> [spouse]: "[spouse], my [spouse alias], has surgery tomorrow early so gotta be fresh to support." valid_at=2026-08-20T05:54:45.793561+00:00 invalid_at=2026-10-21T12:00:00+00:00 expired_at=2026-11-13T22:28:13.754095+00:00
- [invalid] [spouse] -[CURRENTLY_TRAVELING_FROM]-> hospital: "[spouse] is on her way home from the hospital." valid_at=2026-08-22T21:10:29.183196+00:00 invalid_at=2026-10-28T05:52:00+00:00 expired_at=2026-11-13T23:02:15.579546+00:00
- [invalid] [spouse] -[COMMUNICATED_WITH]-> [subject]: "[subject] reports that [spouse] said she might be having multiple surgeries between here and December for her ongoing issues." valid_at=2026-09-09T00:00:00+00:00 invalid_at=2026-11-03T21:01:00+00:00 expired_at=2026-11-14T00:11:34.958713+00:00
- [invalid] [subject] -[LOCATED_IN]-> [trip city]: "[subject] is in [trip city] and will be there until Wednesday." valid_at=2026-10-05T02:55:00+00:00 invalid_at=2026-10-07T00:00:00+00:00 expired_at=2026-11-13T20:36:25.713701+00:00
- [invalid] [subject] -[ATTENDS]-> team meeting: "[subject] is in [trip city] for a team meeting." valid_at=2026-10-05T02:55:00+00:00 invalid_at=2026-10-18T08:40:00+00:00 expired_at=2026-11-13T21:58:40.397191+00:00
- [invalid] [subject] -[STAYING_IN]-> [trip city]: "[subject] is staying in [trip city] for another night." valid_at=2026-10-07T00:25:00+00:00 invalid_at=2026-10-07T07:30:00+00:00 expired_at=2026-11-13T21:16:18.133347+00:00
- [invalid] [subject] -[TOLD]-> Speaker: "[subject] told me that Catherine is [spouse]'s sister and Alice's mom." valid_at=2026-10-20T07:05:00+00:00 invalid_at=2026-11-03T21:01:00+00:00 expired_at=2026-11-14T00:11:34.958718+00:00
- [invalid] [subject] -[HAS_MERGED_PULL_REQUESTS]-> Pull Requests: "[subject] had 36 merged Pull Requests in the last 24 hours." valid_at=2026-10-20T07:05:00+00:00 invalid_at=2026-10-21T07:05:00+00:00 expired_at=2026-11-13T22:17:36.946523+00:00
- [invalid] [subject] -[MENTIONED]-> surgery: "[subject] mentioned that the surgery is intended to improve [spouse]'s vascular capabilities." valid_at=2026-10-23T20:20:00+00:00 invalid_at=2026-10-24T03:53:00+00:00 expired_at=2026-11-13T22:39:29.863521+00:00
- [invalid] surgery -[INTENDED_TO_IMPROVE]-> vascular capabilities: "the surgery is intended to improve [spouse]'s vascular capabilities" valid_at=2026-10-23T20:20:00+00:00 invalid_at=2026-10-24T03:53:00+00:00 expired_at=2026-11-13T22:39:29.863515+00:00
- [invalid] [spouse] -[LOCATED_AT]-> community mailbox: "[spouse] was at the mailbox when the camera matched her at 12:48 today." valid_at=2026-11-03T12:48:00+00:00 invalid_at=2026-11-03T21:01:00+00:00 expired_at=2026-11-14T00:28:44.612127+00:00

## Duplicate sample (hand-judged)

- [ok] Pull Request #2512 (vs: Pull Request #2514, Pull Requests, Pull Request #2513)
- [DUP] dBA (vs: office_sound_dba) -- same metric as office_sound_dba (summary says so)
- [ok] white wires (vs: black wires)
- [ok] cat -- generic, not a duplicate
- [ok] self-inquiry organ (vs: cognitive organ)
- [DUP] [name] -- same person as "[subject]'s son"
- [ok] contractor
- [ok] v2 tool
- [ok] [spouse]
- [ok] The [trip hotel] (vs: The Legend of Zelda, A Link to the Past)
- [ok] GitHub Issues board (vs: GitHub CLI)
- [ok] corporate card
- [ok] dig access (vs: dig capacities, body dig) -- distinct from dig capacities (borderline)
- [ok] accelerators
- [ok] latent-action-space claim (vs: latent modes, Deep Space Network, National Aeronautics and Space Administration)
- [ok] nostalgia
- [ok] Tracking and Data Relay Satellite (vs: National Aeronautics and Space Administration, real data)
- [ok] Claude Haiku 4.5 (vs: Claude Sonnet 4.6, Claude)
- [DUP] recovering patient -- "recovering patient" is el [pet] (summary says so)
- [ok] stance parallelization
- [ok] Deep Space Network (vs: latent-action-space claim, National Aeronautics and Space Administration)
- [ok] biometrics prediction error (vs: prediction.surprise.v1, prediction layer)
- [DUP] Cursor (vs: Cursor seat) -- same as "Cursor seat"
- [ok] plumbing questions (vs: curiosity questions)
- [ok] porch
- [ok] Salt Lake City (vs: Lake Michigan)
- [ok] system (vs: system prompt, Nintendo Entertainment System, reactive system) -- generic junk entity, not a duplicate
- [ok] drift
- [ok] dog -- a dog [child] wants, not el [pet]
- [ok] Features
- [ok] crate
- [ok] prediction.surprise.v1 (vs: prediction layer, biometrics prediction error)
- [ok] walkway forecasts (vs: walkway cam, walkway)
- [ok] viral infection
- [ok] Claude Sonnet 4.6 (vs: Claude Haiku 4.5, Claude)
- [ok] field-note feature (vs: Field of Fear)
- [ok] people -- aggregate junk entity, not a duplicate
- [ok] Marriott
- [ok] expected presence
- [ok] Arturo Sol
- [ok] pending traces
- [DUP] Gigabyte G481-HA0 high-density GPU expansion node (vs: busiest GPU lanes, Circe node) -- same machine as "Circe node" (Circe node summary names this model)
- [ok] language model
- [ok] health status
- [ok] Curiosity Door-A (vs: Curiosity mechanism, autonomous curiosity, Curiosity Daily digest, front door, curiosity questions)
- [ok] National Aeronautics and Space Administration (vs: latent-action-space claim, Tracking and Data Relay Satellite, Deep Space Network)
- [ok] backup failure
- [ok] Gravity sound meter (vs: office_sound_dba)
- [DUP] gpu3 -- gpu3 is the metacog lane's card; "metacog lane" also exists (borderline)
- [DUP] server nodes (vs: server cabinet) -- same as "servers"

# Memory recall step 2: query rewrite, link-following, degraded fallback

Fixes #3534
Fixes #3573

## Summary

- Before searching, a small local model reads the chat message and the last 3 turns of the same conversation. It writes up to 3 short searches that name what the message points at ("okay, I beat it!" -> the game). Each search runs through the same meaning and exact-word search as the message. The reranker scores memories against the message followed by the searches.
- Memories that share a person, project or place with the top candidates join the rerank as extra candidates. This is two-hop personalized PageRank over `episode_memory_referent`, in the style of HippoRAG 2. A neighbour reaches the prompt only if the reranker scores it over the bar.
- When the reranker or embedder is down or too slow, recall no longer drops to nothing. It ranks by exact words and labels the block as degraded in the prompt. A degraded search that finds nothing still puts a one-line note in the prompt: "not proof there is no memory". Telemetry records `lived_memory_degraded`.
- The 5-word floor from step 1 stays. Replay tested letting the rewrite rescue shorter messages. Small-talk injection rose from 13% to 35%, and no referring message gained.
- The replay bank now holds 9 explicit and 13 implicit references. It is host-side, mode 600, ids only.

## Outcome moved

Same bank, same day, same 150 recent messages. A replayed message only sees memories stored before it.

| | Step 1 (before) | Step 2 (after) |
|---|---|---|
| Explicit references (message names the thing) | 9 of 9 | **9 of 9** |
| Implicit references (leans on the shared past without naming it) | 4 of 13 | **7 of 13** |
| Small-talk messages that inject anything (23 messages of 8 words or fewer) | 3 of 23 (13%) | **3 of 23 (13%)** |
| Lived-memory time per message, p50 / p90 | 1.41 s / 2.64 s | **2.85 s / 4.70 s** |
| Rewrite time p50 / p90 (part of the line above) | none | 0.93 s / 2.50 s (the cap) |
| Rewrite timed out (raw message searched alone) | none | 19 of 166 rewrites (11%) |

**Added latency.** Lived recall now takes 1.44 s more at p50 and 2.06 s more at p90. The causes are the rewrite (about 0.9 s at p50) and a larger rerank (up to 32 pairs instead of 24). Lived recall starts at intake and overlaps the other recall backends. Live, it already set the continuity wall time (`lived_memory_wait` about 2.4 s on 2026-11-17). So expect the chat turn's recall step to get about 1.5 s slower at p50. That live figure is **UNVERIFIED** until deploy (#3623).

**Ceiling.** With no time limit on the rewrite, implicit references reached 8-10 of 13. At the 2.5 s cap, the fast lane sometimes answers too late (it is shared, and its queue varies). Then the turn falls back to the message alone.

**Links.** On this bank, links moved nothing at 2 neighbours. At 6 neighbours they traded one explicit hit for one implicit hit. They ship on at 2 (neutral and cheap). #3624 re-checks them with the scoreboard (#3536).

**Degraded (reranker endpoint dead).** Every searched message (149 of 149) ran degraded and said so. Nothing was skipped. Results: explicit 4 of 9, implicit 9 of 13, small talk 3 of 23 (13%), p50 0.83 s / p90 1.76 s. Exact words alone miss explicit references the reranker would catch. With the rewritten searches, they still find most implicit ones. The prompt labels these lines as loose.

## Current architecture

- Step 1 (PR #3571, live since 2026-11-17 04:35Z) searched only the message's own words, by meaning (bge-large in Chroma `orion_lived_memory`) and by exact words (BM25). The local cross-encoder then reranked, and at most 4 lines went in.
- Implicit messages ("okay, I beat it!") found nothing.
- A down or slow reranker made the whole lived step return nothing, with no sign in the prompt that recall had failed.

## Architecture touched

- **orion-recall**
  - `app/lived_rewrite.py` (new): the prompt, the parser, the context read from `chat_history_log`, and the bus RPC to the gateway.
  - `app/lived_links.py` (new): the referent graph and the two-hop spread.
  - `app/lived_memory.py`:
    - the pipeline: rewrite, per-search candidates, step 1's pool kept whole, then links;
    - the rerank against the message and the searches;
    - degraded mode and its notes;
    - one deadline end to end;
    - `degraded_now`, the fallback when the worker's backstop fires.
  - `app/worker.py`:
    - passes `session_id`, the active-turn ids and a budget taken from the caller's deadline;
    - on backstop timeout or error, serves the exact-word fallback;
    - writes `lived_memory_degraded` and `lived_memory_{rewrite,rerank,total}` into the persisted telemetry.
  - `app/main.py`, `app/settings.py`: wiring.
- **Shared**: `orion/memory/lived_memory.py` holds the two referent reads (edges, and name-class aliases only).
- **Bus catalog**: orion-recall is added as a producer on `orion:exec:request:LLMGatewayService` and as a consumer on `orion:exec:result:LLMGatewayService:*`.

## Files changed

- `services/orion-recall/app/lived_rewrite.py`, `app/lived_links.py`: new.
- `services/orion-recall/app/lived_memory.py`, `app/worker.py`, `app/main.py`, `app/settings.py`: pipeline and wiring.
- `services/orion-recall/.env_example`, `docker-compose.yml`, `README.md` §17a/§17b: config and docs.
- `orion/memory/lived_memory.py`: the referent SQL.
- `orion/bus/channels.yaml`, `config/metrics/metric_definitions.lock.json`: catalog entries and the re-lock.
- `services/orion-recall/tests/test_lived_memory_step2.py` (new), `tests/test_lived_memory.py`: tests.
- `services/orion-recall/evals/run_lived_memory_replay_eval.py`: variants (step1, rewrite, links, step2, degraded), a split by kind, rewrite latency.
- `services/orion-recall/evals/test_lived_memory_replay_eval.py`, `evals/fixtures/lived_memory_replay_fixture.json`: implicit fixtures.
- `.github/workflows/lived-memory-recall-tests.yml`: the new files and tests.

## Schema / bus / API changes

- **Added**:
  - orion-recall sends one `llm.chat.request` (`ChatRequestPayload`) per turn on route `quick_turn` and reads the reply on a per-call `orion:exec:result:LLMGatewayService:<uuid>`. Both are existing channels and schemas.
  - Telemetry keys: `backend_counts.lived_memory_degraded`, `timings_ms.lived_memory_{rewrite,rerank,total}`, and `recall_debug.lived_memory.{backend, degraded, rewrite.queries, links}`.
- **Removed**: none.
- **Renamed**: none.
- **Behavior changed**:
  - the lived block can carry a degraded header or a degraded empty note;
  - `MemoryItemV1.tags` can carry `degraded` or `link`.
- **Compatibility**: no schema model changed. `LivedMemoryRecall.recall` gains keyword arguments only.

## Env/config changes

- **Added keys, orion-recall**:
  - `RECALL_LIVED_MEMORY_REWRITE_ENABLED=true`
  - `RECALL_LIVED_MEMORY_REWRITE_ROUTE=quick_turn`
  - `RECALL_LIVED_MEMORY_REWRITE_TIMEOUT_SEC=2.5`
  - `RECALL_LIVED_MEMORY_LINK_TOP_K=2`
  - `CHANNEL_LLM_INTAKE=orion:exec:request:LLMGatewayService`
- **Changed default**: `RECALL_LIVED_MEMORY_TIMEOUT_SEC` 4.0 -> 5.5. The rewrite plus a larger rerank need the room. The caller's own deadline (30-90 s) still caps it.
- **Removed or renamed**: none.
- `.env_example` updated: yes. All flags ship ON.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-recall`. The 4.0 -> 5.5 change and the earlier draft values (1.8, 6) were set to the final values by hand.
- Skipped keys: none.

## Metric gate (the new telemetry)

1. **Provenance.**
   - `lived_memory_degraded` is set in `worker.py` from `LivedRecall.backend`. That backend comes from `lived_memory.py`: rerank raised or timed out, all dense calls failed or timed out, or the worker backstop fired.
   - `lived_memory_rewrite` is the wall time from the rewrite task's start to its answer or its cap.
2. **Independence.**
   - `lived_memory_degraded` is a direct readout of a failure path, not a new signal. It is not an input to any model or ranking.
   - The rewrite time is part of `lived_memory_total`, by construction.
3. **Theory.** #3573 asks for it by name: an empty result must not read as "no memory" when the search could not run.
4. **Live data.**
   - Degraded fired on 149 of 149 searched messages with the reranker URL dead (eval).
   - Degraded fired on 0 of 149 with it up.
   - Rewrite time ranges 0.6-2.5 s (not flat).
   - Live: **UNVERIFIED**.
5. **Existing mechanism.**
   - Step 1 had `dense_error` in the debug only, and nothing persisted.
   - `backend_counts` already carries per-backend counts, so a key there is the existing place.
6. **Reversibility.** Two jsonb keys. No schema, no consumer.

## Integration gate

This is not a new producer of Orion experience. It is a reader on the existing chat recall path. The rewrite is an internal LLM call, not an organ output. `scripts/check_integration_seam.py`: OK.

## Tests run

```text
pytest services/orion-recall/tests/test_lived_memory.py tests/test_lived_memory_step2.py evals/test_lived_memory_replay_eval.py   51 passed
pytest services/orion-recall/tests   all lived tests pass; the same 2 failures as main (#3569)
mutation check: removing the "never serve a link neighbour without a rerank" filter fails its test
scripts/check_env_template_parity.py PASS; check_integration_seam.py OK; check_personal_terms.py clean;
check_bus_reply_channels.py OK; check_single_consumer_channels.py OK; check_chat_route_poachers.py PASS;
check_definition_drift.py: 0 changed after re-lock
CI on PR #3622: green
```

Degraded is tested with the endpoint down:
- `test_reranker_endpoint_down_over_real_http_degrades` makes the real HTTP call to a port nothing listens on;
- the replay eval's `degraded` variant points the reranker at `127.0.0.1:9`.

## Evals run

```text
services/orion-recall/evals/run_lived_memory_replay_eval.py --variants step2 | step1,degraded \
  --link-top-k 2 --rewrite-timeout 2.5 --min-fit 0.004   (host-side bank, real fast lane, real reranker)
```

Exploration runs on the same bank (rewrite cap 10 s, referring messages only), used to choose the design:

| variant | explicit | implicit | note |
|---|---|---|---|
| step 1 | 9/9 | 4/13 | baseline |
| links only (6) | 8/9 | 4/13 | links alone gain nothing |
| rewrite only | 9/9 | 8/13 | |
| rewrite + links 6, fused pool | 7/9 | 10/13 | fused pool pushed explicit targets out of the rerank |
| rewrite + links 6, step 1 pool kept | 8/9 | 9/13 | |
| rewrite + links 2, step 1 pool kept (shipped) | 9/9 | 8/13 | |
| reranking against message and joined, max | 7/9 | 9/13 | no better, double the rerank cost; dropped |
| rewrite rescue of short messages | (no short refs in bank) | | small talk 13% -> 35%; dropped |

## Docker/build/smoke checks

```text
Rewrite live smoke: BusRewriter -> gateway quick_turn from the host. 4 hand cases; small talk -> NONE in 0.6 s, an implicit game reference -> 3 searches.
Replay evals above used the live vector host (/embedding, /rerank), Chroma and the gateway.
No docker build: no Dockerfile or requirements change (new modules are under app/, already copied).
compose config: env parity gate PASS for orion-recall (the 2 missing compose keys predate this PR).
Not deployed (as asked).
```

## Review findings fixed

A code-review subagent reviewed the branch after the first two commits.

- **Finding (must-fix):** the embed and nearest calls had no deadline. A hung embedder ate the budget, the worker backstop fired, and the turn was skipped silently.
  - **Fix:** the meaning search waits at most until the rerank reserve, then cancels and marks the turn `embedder_timeout` (degraded). Exact-word candidates are still reranked.
  - **Evidence:** `test_a_hanging_embedder_degrades_inside_the_deadline`.
- **Finding (must-fix):** the worker backstop still skipped silently, and the reader's deadline ignored the caller's.
  - **Fix:** the reader takes a budget from the caller's deadline. On backstop timeout or error, the worker runs `degraded_now`: exact words from the in-memory snapshot, with a bounded re-check, labelled degraded.
  - **Evidence:** `test_worker_falls_back_to_exact_words_when_the_backstop_fires`.
- **Finding:** dense tasks could be orphaned on cancel.
  - **Fix:** every child task is cancelled and gathered in `finally`.
- **Finding:** the rewritten searches go into recall debug.
  - **Kept on purpose:** the spec's recall trace asks for them. They are Orion's own query text, not memory text. The docstring now says so.
- **Finding:** the degraded empty note was wrong when the reranker ran, and it fired on small talk.
  - **Fix:** two notes: exact words only, or meaning search down. No note when the rewrite said NONE and no word matched.
  - **Evidence:** `test_degraded_small_talk_gets_no_note`, `test_embedder_down_and_nothing_fits_says_meaning_search_was_down`.
- **Finding:** the link-neighbour test passed for the wrong reason.
  - **Fix:** rebuilt so the neighbour shares words but enters only as a link.
  - **Evidence:** it fails with the filter removed.
- **Finding:** the referent graph reloaded inside a turn.
  - **Fix:** it loads once, then refreshes in the background every 10 minutes.
- **Nits fixed:**
  - `link_top_k` is clamped.
  - The rewrite result is cached (no double timing).
  - Names in the rewritten searches also seed links.
  - The "never logged" claim is softened to what orion-recall itself does.
  - The docstring no longer names a person.

## Restart required

Run from the clean primary `main` checkout after merge. The post-pull rebuild covers it:

```bash
scripts/safe_docker_build.sh orion-recall up -d --build
```

## Risks / concerns

- **Severity: medium. Concern: chat turns get slower.** Lived recall is about 1.5 s slower at p50 and 2 s slower at p90 in replay. Live is UNVERIFIED.
  - **Mitigation:** `RECALL_LIVED_MEMORY_REWRITE_ENABLED=false` removes the rewrite cost. #3623 measures it live.
- **Severity: medium. Concern: the fast lane is shared.** One more short call per searched turn goes on `quick_turn` (interactive priority). About 11% of rewrites hit the 2.5 s cap in replay. A late rewrite is cancelled on our side, but the gateway may still finish it.
  - **Mitigation:** messages under 5 content words or over 1,500 characters never call it.
- **Severity: low. Concern: no measured gain from links.**
  - **Mitigation:** they ship at 2 (neutral). #3624.
- **Severity: low. Concern: merge order with #3603** (scoreboard), which also edits `lived-memory-recall-tests.yml` and the recall README. Whoever merges second resolves a small conflict.
- **Severity: low. Concern: bar met on small talk, partly on implicit.** Implicit references rose 4 -> 7 of 13. At the live latency cap, that is short of the 8-10 the rewrite reaches with time.
- **Issues:** opened #3623 (verify-live), #3624 (follow-up). This PR closes #3534 and #3573.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3622

🤖 Generated with [Claude Code](https://claude.com/claude-code)

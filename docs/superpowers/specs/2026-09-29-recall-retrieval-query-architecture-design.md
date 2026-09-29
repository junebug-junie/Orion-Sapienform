# Recall retrieval architecture: separate the search query from the prompt

Status: APPROVED by Juniper 2026-09-29 (all three missing questions answered with the recommendations below). Phase 1+2 in progress; Phase 3 follows.
Date: 2026-09-29
Evidence base: `recall_telemetry` (live since PR #2398, 2026-09-29 18:14 UTC), `grammar_events` RPC timeouts, two read-only code traces (upstream callers, downstream pipeline) against main @ 4ed5ee36e.

## Arsonist summary

Recall is slow because Orion hands it the wrong thing to search for. For self-initiated turns (curiosity self-inquiry, world-pulse reading), cortex-exec uses the whole instruction prompt as the search query. That can be 30,000 characters. Recall has no guard against that, and everything it does next makes it worse:

1. **It splits the query into one search per capitalized word, with no limit.** The 30,663-char prompt became 268 searches, including "No", "It", "The" and "You".
2. **It runs every backend once per search, one after another.** Most of those backends ignore the search text entirely (recent chat, bus anomalies, recent timeline), so 267 of the 268 runs are identical repeats.
3. **It fetches first and time-filters afterwards.** 1,608 chat turns were fetched, and all 1,608 were thrown away.
4. **It has no overall deadline and handles one request at a time.** If a caller gives up at 90s, recall keeps working anyway, and every other recall waits behind it.
5. **stance_react runs this twice per turn**: once before the language model runs, once after.

Across all live rows so far, query length and latency correlate at **0.98**. The 72-second recall is simply the longest query.

This is a textbook RAG failure. The prompt is being used as the retrieval query. Standard practice keeps the two separate: the caller decides what to search for, and the retriever turns a bounded query into bounded work.

## Evidence

| Fact | Source |
|---|---|
| 71 telemetry rows; `stance_react` max 30,663 chars / 71.7s; every other verb ≤ 92 chars and ≤ 2.1s | `recall_telemetry` grouped by verb, 2026-09-29 18:45 UTC |
| corr(length(query), latency_ms) = 0.98 | same |
| 30,663 chars → 268 signals (the fragment, the verb, plus 266 extracted "entities") | `_expand_query` run on the stored query text |
| bus_synaptic_anomaly = 1,340 = 268 × 5, and falkor_chat = 1,608 = 268 × 6 (per-signal cap), with 1,608 dropped out of window | `backend_counts` on the slow row |
| 44 recall RPC timeouts in 24h, 26 of them in 09-28 14:07–14:28 | `grammar_events` `rpc_transport_timeout` |
| reverie_narrate / reverie_expectation_judge send **empty** queries (34 calls) because the router enables recall by default (`recall_enabled=True recall_cfg={}`) | `recall_telemetry`; cortex-exec log `Exec plan start ... verb=reverie_narrate recall_enabled=True` |

## Current architecture

**Upstream (query formation).** Every cortex-exec recall uses `_last_user_message(ctx)` as `fragment` (`services/orion-cortex-exec/app/executor.py:2454`, `:1603`), with no cap.

For self-initiated turns there is no human message:
- `turn_orchestrator.py:1101-1113` sets `stance_user_message = mind_appraisal_text or user_message`.
- For self-inquiry and curiosity, the appraisal comes from an in-memory dict (`services/orion-hub/scripts/curiosity_investigation.py:3214`). When that lookup misses, recall gets the whole prompt (`orion/curiosity/self_inquiry_prompt.py:334`). Even when it hits, the appraisal text is nearly empty ("Investigation claim: not yet chosen").
- World-pulse reading never passes an appraisal at all (`services/orion-hub/scripts/reading_turn_listener.py:164-169`). Its stage-2 prompt embeds the whole stage-1 handoff JSON (`world_pulse_read_stage2.py:158`).

stance_react recalls twice per turn: PCR phase 0+1 at `router.py:1106-1128`, then phase 3 at `grounding_capsule.py:152`. Each gets a 90s timeout.

The one query-trimming helper, `_derive_chat_general_query` (`services/orion-recall/app/worker.py:374`), only runs for `chat_general`. The contract has no "search text" separate from `fragment` (`orion/core/contracts/recall.py:47-95`).

**Downstream (recall pipeline).** Line numbers are in `services/orion-recall/app/worker.py` unless noted.

| Stage | Location | Problem |
|---|---|---|
| Expansion | `_expand_query` :514, `_extract_entities` :212 | No cap and no stopword filter; every capitalized word becomes a signal |
| Backend loop | :1707-1727 | Sequential `await _query_backends(sig)` for every signal |
| bus_synaptic_anomaly | :1130 | Never receives the query, yet runs per signal |
| falkor_chat | :1143, `falkor_chat_adapter.py:68` | Query-independent (`ORDER BY ts DESC LIMIT n`) and runs per signal; time window applied after the fetch |
| sql_chat pairs/msgs | :1198 | Query-independent and runs per signal; new asyncpg connection each call |
| sql_timeline recent / related | :1253, `sql_timeline.py:415` | "recent" is query-independent. "related" re-extracts **all** entities from the full query on every iteration: ~265 `ILIKE '%x%'` patterns × 2 columns × 268 runs |
| Anchor rail | `_anchor_tokens` :264 | **Dead.** `r"\\b..."` matches a literal backslash. Verified: `_anchor_tokens("p4 v100 gpu1") == []`. The only test monkeypatches the function. |
| Memory-browse shortcut | :282 | **Dead**, same `\\b` bug |
| Entity boost | :618 | The "first 3 entities" come from set iteration order, so they are arbitrary |
| Fusion | `fusion.py:187-201` | Re-tokenizes the whole query for every candidate. No signal provenance, so extra signals add no ranking value. |
| Deadline | none | No overall budget. Handlers run serially (`RECALL_RABBIT_CONCURRENT_HANDLERS=false`). A caller timeout does not cancel recall. |
| Timing | :1815 | `latency_ms` stops before the boost and fusion run. `backend_fetch` is one lump. |

**Also observed, separate:** Falkor's newest `chat.history` ChatTurn is from 04:11 UTC. Postgres has 4 newer turns. Writer lag, UNVERIFIED cause.

## Target architecture (RAG best practice, mapped to Orion)

```
caller ──► RecallQueryV1{ fragment (turn text, provenance only),
                          retrieval_query (what to search for),
                          deadline_ms }
recall:
  1. intake guard     : pick query = retrieval_query or condense(fragment); hard char cap
  2. plan             : context feeds (query-independent) + retrievers (query-dependent)
                        over a bounded sub-query set (<= K, stopword-filtered, specificity-ranked)
  3. fetch            : all feeds once + retrievers × sub-queries, concurrently,
                        time window pushed into each backend query, under one deadline
  4. fuse             : tokenize once; candidates carry provenance (backend, sub-query)
  5. select / render  : existing caps and render budget
  6. telemetry        : per-stage timings, sub-query count, candidates fetched/kept, deadline_hit
```

The principles behind it:

1. **Query formulation belongs to the caller.** Only the caller knows what a self-initiated turn is *about*: the standing question, the source being read. Recall must not have to guess that from a 30k-character instruction prompt.
2. **The retriever defends itself anyway.** A deterministic intake guard (length cap, then condensation) means no caller can bring recall down again. This is a backstop, not the primary path.
3. **Context feeds are not retrievers.** "Recent chat", "recent timeline" and "bus anomalies" answer "what is going on", not "what matches X". They run once per recall.
4. **Bounded expansion.** At most K sub-queries (proposed K = 4). Drop stopwords and sentence-starters. Rank by specificity using the entity-degree data the boost already fetches, rather than set order.
5. **Filter at the source.** Push time windows into Cypher and SQL, so nothing is fetched only to be discarded.
6. **Anytime retrieval.** Run under a deadline set below the caller's timeout, and return what has arrived when it expires (`deadline_hit=true`). A partial answer beats a timeout that looks like "no memory".
7. **Provenance-aware fusion.** Tokenize the query once, and record which backend and sub-query found each candidate. That makes a future reranker or reciprocal-rank fusion possible without re-plumbing. The reranker itself is a non-goal here.

## Missing questions (answered 2026-09-29: Juniper accepted each recommendation)

1. **What should a self-initiated turn remember *about*?** Options:
   - (a) the standing question text;
   - (b) the question plus the last investigation note;
   - (c) the reading's source title and URL plus the Stage-1 claim.

   This changes what Orion recalls during reflection, so it is a cognition choice, not a plumbing one. Recommendation: (a) for self-inquiry and curiosity, (c) for reading.
2. **Should stance_react recall twice per turn?** Phase 0+1 and phase 3 now both run on the same text. Recommendation: keep both, but phase 3 should reuse phase 0+1's query and plan, not redo expansion.
3. **Should verbs with no user text recall at all?** reverie_narrate and reverie_expectation_judge currently send an empty query and get the generic "recent context" feeds. That may be exactly what reverie wants, just unlabelled. Recommendation: make it explicit, `recall.mode = context_only` (feeds only, no retrievers), rather than an empty query.

## Proposed schema / API changes

**`RecallQueryV1`** (`orion/core/contracts/recall.py`, `extra="forbid"`). The rollout must be consumer-first (see the "additive fields on forbid models" memory):
- `retrieval_query: Optional[str]` (max 1,000 chars): what to search for. When absent, recall condenses `fragment`.
- `deadline_ms: Optional[int]`: the caller's remaining budget. Recall aims to finish at 80% of it.
- `mode: Literal["retrieve", "context_only"] = "retrieve"`: `context_only` runs the feeds only.

**`RecallDecisionV1` / telemetry:** add `query_chars`, `retrieval_query_source` (`caller` | `condensed` | `fragment`), `sub_query_count`, `candidates_fetched`, `candidates_kept`, `deadline_hit`, and a per-stage `timings_ms`. Add matching nullable columns to `recall_telemetry`: `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, done in the writer's once-per-process DDL.

**Callers:**
- `StanceReactRequestV1` gets `retrieval_query: Optional[str]`.
- `execute_unified_turn(retrieval_query=...)` threads it through.
- cortex-exec `run_recall_step` and `build_recall_query_v1` prefer `ctx["retrieval_query"]`.

Registry and channel catalogs: no new channels. Schema registry entries update for the changed models.

## Files likely to touch

**Phase 1: recall-side backstop (no contract change, one service)**
- `services/orion-recall/app/worker.py`:
  - intake cap and condensation (reuse `_derive_chat_general_query`'s clause scoring for all verbs);
  - `_expand_query` cap K plus a stopword filter;
  - split `_query_backends` into `_fetch_context_feeds` (once) and `_fetch_retrievers(sub_query)`;
  - `asyncio.gather` under a semaphore with an overall `asyncio.timeout`;
  - hoist `_extract_entities` out of the loop;
  - fix the `\\b` regexes at :264 and :282;
  - specificity-ordered entities for the boost;
  - move `latency_ms` to the end.
- `services/orion-recall/app/storage/falkor_chat_adapter.py`: `WHERE t.ts >= $cutoff`.
- `services/orion-recall/app/fusion.py`: tokenize the query once.
- `services/orion-recall/app/sql_timeline.py`: cap the `related_by_entities` patterns to K.
- `services/orion-recall/settings.py`, `.env_example`: `RECALL_MAX_QUERY_CHARS`, `RECALL_MAX_SUB_QUERIES`, `RECALL_DEADLINE_MS_DEFAULT`, then run the env sync.
- `services/orion-recall/tests/`: signal-count, run-once-feeds, deadline, and regex tests with no monkeypatching.
- `services/orion-recall/evals/` (new): latency and recall-quality eval over the stored `recall_telemetry` queries.

**Phase 2: contract (consumer first)**
- `orion/core/contracts/recall.py`: new fields.
- Schema registry.
- orion-recall reads them.

**Phase 3: callers**
- `services/orion-cortex-exec/app/executor.py:2454`, `orion/cognition/recall_query.py:88`: prefer `retrieval_query`.
- `orion/hub/turn_orchestrator.py`, `orion/schemas/thought.py`, `services/orion-thought/app/bus_listener.py`: thread the field through.
- `services/orion-hub/scripts/curiosity_investigation.py`, `reading_turn_listener.py`: set it durably on the run request, not in an in-memory dict.
- reverie verbs: `mode=context_only`.
- `grounding_capsule.py`: phase 3 reuses phase 0+1's query.

## Non-goals

- No LLM-based query rewriting. Condensation stays deterministic, and the caller supplies intent.
- No reranker model. Provenance plumbing only, so one can be added later.
- No change to fusion weights, render budgets, or which backends exist.
- No fix for the Falkor ChatTurn writer lag here. File it separately after confirming the cause.
- No change to caller timeouts. Recall's deadline sits under them.

## Proposal-mode disclosure

- **Capability change:** recall returns results for self-initiated turns in seconds instead of timing out, and searches for what the turn is *about* rather than for words in its instructions. Recall content for those turns **will change** (Phase 3). Phases 1–2 alone mostly change latency, plus which top-3 entities feed the boost.
- **Data touched:** read paths only, plus `recall_telemetry` gaining columns. No memory is written or deleted.
- **Privacy boundary:** unchanged. Same stores, same session/node scoping. Telemetry already stores the query text; it will now store a capped query instead of 30k-character prompts.
- **Trace that proves it:** in `recall_telemetry`, `stance_react` p99 latency below 5s with `sub_query_count ≤ K+2`; no `RecallService` entries in `grammar_events` `rpc_transport_timeout` for 24h; `candidates_fetched ≈ candidates_kept` for falkor_chat.
- **Dangerous failure mode:** an over-aggressive cap drops the one entity that mattered, and recall quietly gets worse while getting faster. That is why the quality eval over stored real queries is a Phase 1 gate, not a follow-up.
- **Rollback:** Phase 1 knobs are env settings. `RECALL_MAX_SUB_QUERIES=0` means no cap, and restores today's fan-out. Contract fields are optional, so callers can stop sending them.

## Acceptance checks

1. Unit: a 30,663-char query yields ≤ K+2 sub-queries, and none of them is a stopword.
2. Unit: with N sub-queries, each context feed (bus_synaptic_anomaly, falkor_chat, sql_chat, sql_timeline recent) is called exactly once.
3. Unit: a backend hanging past the deadline returns partial results with `deadline_hit=true` within `deadline_ms`.
4. Unit: `_anchor_tokens("p4 v100 gpu1") == ["p4","v100","gpu1"]` with no monkeypatch; the memory-browse regex matches "show recent memories".
5. Eval: over every distinct query in `recall_telemetry`, the new pipeline keeps ≥ 90% of today's top-8 selected ids for queries ≤ 500 chars, and all long queries finish in < 5s.
6. Live: 24h after deploy, `recall_telemetry` `stance_react` max latency < 10s, and `grammar_events` shows 0 RecallService RPC timeouts outside restart windows.

## Recommended next patch

Phase 1 alone, in orion-recall. It is service-bounded and needs no contract change, and it removes the timeout risk for every caller, including ones added later. Then Phase 2 (contract, consumer-first) and Phase 3 (callers). Phase 3 uses the approved answers: the standing question for self-inquiry/curiosity, source title + claim for reading, phase 3 reusing the phase 0+1 query, and reverie on `mode=context_only`.

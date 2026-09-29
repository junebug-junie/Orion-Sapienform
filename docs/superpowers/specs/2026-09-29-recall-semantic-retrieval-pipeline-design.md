# Recall semantic retrieval pipeline: hybrid + graph + rerank, measured first

Status: PROPOSAL (design mode + proposal mode). Nothing here is implemented.
Date: 2026-09-29
Builds on: `docs/superpowers/specs/2026-09-29-recall-retrieval-query-architecture-design.md` (APPROVED, in progress on `docs/recall-retrieval-architecture`). That spec fixes *what* recall searches for (`retrieval_query`), bounds the fan-out, adds a deadline, and puts provenance on every candidate. This spec is about *how well* recall ranks once the query is sane. It does not redo any of that work and assumes it has landed.

Evidence base: read-only live queries against Postgres (`conjourney`), FalkorDB, Chroma (`localhost:8500`), container env/logs, and main @ 38d65a36e. All numbers below were pulled 2026-09-29 between 22:00 and 23:40 UTC unless marked otherwise.

## Arsonist summary

Recall today does not search by meaning. It picks from whatever is newest, and adjusts that list with a word-overlap tie-breaker. The live evidence is blunt:

- **The same few memories win almost every recall, whatever was asked.** Across all 401 `recall_telemetry` rows, recall selected 5,003 items but only **106 distinct ones**. One chat turn ("Unwinding after a long day of travel and team socializing. You?", 2026-09-29 04:11) was selected in **343 of 401** recalls. The next seven most-selected items appear in 291–362 recalls each. They are simply the most recent chat turns.
- **Bus-anomaly background noise is 30% of everything recall hands back.** 1,484 of 5,003 selected items are `bus_synaptic_publish:*` fragments. They do not depend on the query at all.
- **There is no dense (embedding) retrieval anywhere in recall, and it was switched off on purpose.** It was deleted in May 2026 (PR report `2026-05-29-orion-recall-vector-amputation-pr.md`). The one leftover semantic rail, in the active-packet collector, still embeds the query on every call (160 calls to vector-host in 24h) and then throws the vector away, because `chromadb` is not installed in the recall image. That is the "chromadb not installed" log line: 29 of them in the last 4h. The Chroma collection it would have searched does not exist either.
- **The "score" in fusion is mostly a per-backend constant.** Every adapter hard-codes a base score (falkor_chat 0.50, sql_chat 0.75, sql_timeline 0.7–0.95). Fusion multiplies that by 0.7, while query-dependent text overlap gets 0.15 and recency 0.1. The candidate *set* itself is chosen by `ORDER BY ts DESC`. So the ranking is: which backend it came from, then how new it is, with word overlap breaking ties.

The fix is not a smarter formula on top of a recency-selected set. The fix is standard hybrid retrieval: several independent retrievers that each *search* (by meaning, by keywords, by linked entities, by time), fused by rank, reranked by a cross-encoder, then diversified. Orion already has most of the ingredients: a running embedding model, a spaCy entity graph, Postgres full-text search, and GPUs. They are just not wired into recall. And we will measure before we build. Phase 0 is a labeled eval set plus today's baseline number, because today nobody can say how bad recall is, only that it is recency.

What I am **not** recommending: GraphRAG global/community summaries, graphiti, HyDE, LLM query rewriting, or a vector database migration. Reasons are below.

## Current architecture

### Semantic asset inventory (live-verified unless marked)

| Asset | What it is | State | Evidence (2026-09-29) |
|---|---|---|---|
| **Embedding model** `BAAI/bge-large-en-v1.5` (1024-dim) in `orion-athena-vector-host` | Turns text into vectors, on **CPU** (`VECTOR_HOST_EMBEDDING_DEVICE=cpu`). Bus RPC on `orion:embedding:generate`, HTTP `POST :8320/embedding` | **LIVE**, busy | Container up 8 days. Bus log: received→published 170–250 ms per ~450-char text. 86,677 HTTP `/embedding` calls in 24h from topic-foundry (172.18.0.51), 160 from recall (172.18.0.30) |
| Chat embeddings | vector-host embeds every chat message/turn (`orion:chat:history:log`/`:turn`) and feeds only OrionTissue | **Computed then discarded** | vector-host README "Dead pipeline killed (2026-08-14)": the vector upsert to `orion_chat`/`orion_chat_turns` was removed because recall had no reader |
| **Chroma** 0.4.24 `orion-athena-vector-db` (`:8500`) | Vector store | **LIVE but not a usable memory index** | `orion_main_store` 6,612 rows, all `original_channel=orion:embedding:generate`, 6,608 from `llm-gateway`: JSON blobs of LLM outputs, not memories. Oldest row 2026-09-29 04:44:17, which is 3 s after the container started (`StartedAt 04:44:14`), despite a bind mount (`/mnt/postgres/collapse-mirrors/chroma`). Why earlier rows are gone is **UNVERIFIED**. `orion_chat` 4 rows. `orion_reading_results` 29 rows. `doc_semantic_drift` 85 rows. **`orion_memory_crystallizations` does not exist** |
| Recall dense path | `source=vector` in recall | **DEAD by design** | Amputated in commit 710c7bfc3 (2026-05-30). `RECALL_ENABLE_VECTOR=false`. `backend_counts.vector` is 0 in all 401 rows. `test_recall_vector_amputation.py` guards the removal |
| Active-packet Chroma rail (`orion/memory/crystallization/retriever.py`) | Embeds the query, then queries `orion_memory_crystallizations` in Chroma | **DEAD, wastes a call** | Recall container: `ModuleNotFoundError: No module named 'chromadb'`. 29 "chromadb not installed" in 4h. Even if it returned hits, `extra_crystallization_ids` is computed but never merged into the packet (code read, `retriever.py:121-164`) |
| Memory cards (`memory_cards`) | Self-authored cards, ranked by Postgres full-text search (`ts_rank_cd` over a weighted, GIN-indexed `search_vector` tsvector) × confidence | **LIVE, lexical only** | 1,166 cards, 436 active, newest 2026-09-29 12:01. Cards appear in 16/401 recalls (71 candidates). The cosine path was tried in July and deleted: embedding cards *at query time* on CPU took 5.9 s for 40 cards (commit abd0d841f) |
| **pgvector** | Postgres vector type | **ABSENT** | `pg_extension`: plpgsql, pgcrypto, pg_stat_statements only. The only vector-ish column in the DB is `memory_cards.search_vector` (tsvector) |
| Falkor `orion_recall` | Chat/social turn graph | **LIVE, partly stale, mostly textless** | 12,157 `ChatTurn` (10,241 `social.turn.stored.v1` up to 09-17 23:45, 1,916 `chat.history` up to **09-29 04:11**, while Postgres has newer turns), 1,615 `ChatSession`, 318 `Entity`, 3 `CollapseEvent`. Edges: `HAS_TURN` 12,157, `MENTIONS_ENTITY` 3,975. **No entity–entity edges.** ChatTurn keys: `turn_id, correlation_id, sentiment, source_kind, ts` (no text). **Only 324 of 1,916 chat.history turn_ids resolve to a Postgres row with text** (`chat_history_log` id/correlation_id; `chat_message` matches 0). Cause **UNVERIFIED** |
| Entity extraction | spaCy NER (`en_core_web_trf`) in `orion-meta-tags` → `falkor_recall_writer.py` writes `MENTIONS_ENTITY` | **LIVE, entities only, no relations** | 720 of 1,916 chat.history turns mention ≥1 entity (279 distinct). Top: orion 307, tessa 135, nico 134, juniper 85, sofia 70. Several are AI Town agent names. No relation extraction exists for chat, journal or reading |
| Entity relatedness boost (`storage/falkor_entity_relatedness.py`) | Co-occurrence over `MENTIONS_ENTITY` with degree discounting, added to the composite score | LIVE | In fusion. The first 3 query entities come from set order (companion spec) |
| Falkor neighborhood (`falkor_neighborhood_adapter.py`) | Turns around query entities | LIVE, rare | Non-zero in 70/401 recalls, 349 candidates total |
| Falkor `orion_worldview` | Curiosity/investigation graph: Hop 548, TurnOutcome 175, Finding 171, Prior 120, Concept 13 … Edges SUPPORTS 165, ABOUT 56, CONTRADICTS 25 | LIVE, **not read by recall** | Label and edge counts above. Concepts are investigation titles, not an entity ontology |
| Falkor `graphiti_temporal` | graphiti temporal KG | **STALE, not a KG** | 25 `Entity` nodes whose `name` is a raw user message ("how are things going over at AI Town?"), 25 `RELATES_TO`, newest 2026-09-04 04:55. graphiti-adapter log shows a startup failure against Postgres recovery before its current run |
| Falkor `orion_substrate*`, `orion_kg` | Substrate self-model nodes; `orion_kg` empty | Not retrieval corpora | `SubstrateNode` 4,510 / 9,245; `orion_kg` returns no labels |
| **Fuseki / RDF** | SPARQL store the recall RDF adapter targets | **DOWN: no container** | `docker ps -a` shows no fuseki container, `localhost:3030` refuses, yet `RECALL_ENABLE_RDF=true` in the live env |
| Community detection / summaries | GraphRAG-style | **ABSENT** | Nothing in Falkor. `topic_foundry` clusters text segments (994,423 segments, newest 18:37 today, topic −1 = outliers dominate) but stores no embeddings and recall does not read it |
| Crystallizations (`memory_crystallizations`) | Distilled memories: stance 701, reflection 356, semantic 344, open_loop 6 | LIVE | 1,407 rows, newest 04:11 today. `memory_crystallization_sources` links 1,051 of them to 2,280 chat turns (544 resolve to `chat_history_log`). Many summaries are verbatim copies of the source turn |
| Corpus sizes (Postgres canonical text) | | | `chat_history_log` 531 (07-24 → 09-29 17:31). `journal_entries` 101,014, of which 98,428 are `metacog/digest`, ~2,600 other. `evidence_units` 113,592 (101,013 journal, 12,526 notify). `world_pulse_article` 929 |
| Existing recall eval | `app/recall_eval.py` + `recall_eval_corpus.json` | Exists, too weak to decide anything | 5 hand-written cases that score *term coverage* (expected words appear in snippets). No gold document ids. A lexical system scores well by construction. No `services/orion-recall/evals/` directory |

### What fusion actually rewards (`services/orion-recall/app/fusion.py:504-520`)

```
composite = backend_weight * (0.7*base_score + 0.15*text_similarity + 0.1*recency) + tag + turn_effect + entity_boost
```

(weights from `reflect.v1`/`chat.general.v1`; `chat.continuity.v1` uses recency 0.4)

- `base_score` is a constant chosen per adapter (`worker.py:732, 1010, 1217, 1237, 1287, 1613`; `falkor_chat_adapter.py:118`). It is a prior on the backend, not a relevance score.
- `text_similarity` is `max(token-overlap fraction, any rare token ≥6 chars appears)`. That is bag-of-words with no IDF, no stemming and no synonyms.
- The candidates were already fetched by `ORDER BY ts DESC` (falkor_chat, sql_chat, sql_timeline recent, cards always-inject). Anything older than the fetch window can never be ranked.

### Where recall spends its selections (all 401 telemetry rows, 18:14–22:14 UTC)

| backend | candidates fetched | recalls where non-zero |
|---|---|---|
| sql_timeline | 37,171 | 340 |
| falkor_chat | 16,238 | 401 |
| bus_synaptic_anomaly | 12,490 | 401 |
| sql_chat_pairs | 7,071 | 359 |
| falkor_neighborhood | 349 | 70 |
| concept_region | 274 | 6 |
| active_packet | 180 | 30 |
| cards | 71 | 16 |
| vector / graph_compression / sql_chat_msgs | 0 | 0 |

Latency (p50 / p95 ms): reverie_narrate 417 / 529, journal.compose 959 / 2,047, stance_react 896–1,432 / 19,880–25,103 (the query-length bug the companion spec fixes).

### Hardware that retrieval models can use

- **athena** (where recall runs): 96 cores, 566 GB RAM. Tesla P4 (8 GB, 4.9 GB used) and Tesla T10 (16 GB, 10.7 GB used), each hosting other processes today. Whether the T10's ~5.7 GB is reliably free is **UNVERIFIED**; it needs a week of `orion_biometrics` GPU samples.
- **circe**: seven GPUs (V100s, P100, PG500) managed by the GPU pool (`gpu_pool_cards` gpu0–gpu3 with `chat`/`agent`/`fast`/`metacog` roles). Pool leases load and unload models on demand. A cold load inside a recall request would blow the p95 budget, so **recall models must be resident and must not lease**.

## Target pipeline (best practice, mapped to Orion, with verdicts)

```
RecallQueryV1{retrieval_query, deadline_ms, mode}        ← companion spec
 │
 1. understand   deterministic: temporal scope, entity links, intent (existing intent.py)
 │
 2. retrieve     independent ranked lists, each with provenance, concurrently, under the deadline
 │   ├─ dense    bge-large query embedding (one call) × in-memory matrix of doc embeddings
 │   ├─ sparse   Postgres full-text search over chat / journal(non-metacog) / cards / crystallizations
 │   ├─ graph    entity-linked personalized PageRank over the MENTIONS_ENTITY graph (Phase 4, gated)
 │   ├─ recency  "what happened lately" list (today's feeds), now one list among several
 │   └─ feeds    bus anomalies etc. are context feeds, excluded from ranking (companion spec)
 │
 3. fuse         weighted reciprocal rank fusion (RRF, k=60) over the lists
 4. rerank       cross-encoder on the top ~40, hard latency budget, skip-on-timeout recorded
 5. diversify    MMR over embeddings (λ≈0.7) + existing transcript dedupe
 6. assemble     existing render budget; best first; each line keeps its source
 7. trace        per-list ranks, RRF score, rerank score, and which stage cut each item → recall_telemetry
```

### Stage by stage

**1. Query understanding. ADOPT, deterministic only.**
- *Temporal scope*: a small regex/date parser ("yesterday", "last week", "in August", "this morning", ISO dates) produces a `[since, until]` window. When present it is a **hard filter** pushed into every retriever's query. When absent there is no filter, and recency is only one voice in the fusion. This is the fix for recency-as-winner.
- *Entity linking*: match query tokens and bigrams against the Falkor `Entity` name set (318 names, cached in-process and refreshed every few minutes). Exact and case-folded matching, nothing fuzzy. Entities are ranked by specificity (inverse degree, data the boost already fetches).
- *Intent*: keep `intent.py`. Do not add a model.
- REJECT: LLM query rewriting, HyDE (hypothetical-document embeddings), multi-query expansion. Each costs an LLM call (seconds on this mesh) inside a request that must finish in a few seconds. The companion spec already moved query formation to the caller, where the intent is actually known.

**2a. Dense retrieval. ADOPT, and store embeddings at write time.**
- The July lesson (commit abd0d841f): embedding documents inside a recall request on CPU is too slow (0.15 s per card). So documents are embedded **when they are written**, and a query costs one embedding call (~200 ms on the current CPU host).
- Producer: vector-host already embeds every chat message and turn for OrionTissue and then drops the vector. Add a persist step there, plus a small backfill job for journals (non-metacog), cards, crystallizations and world_pulse claims.
- Store: a new Postgres table `recall_doc_embedding` (below), with the vector stored as `real[]`. **Not Chroma**, because its main collection lost everything older than today's container start despite a bind mount (cause UNVERIFIED), it is pinned at 0.4.24, and recall deliberately dropped the client dependency. **Not pgvector**, because the production DB runs stock `postgres:15` without the extension, and swapping the image of the primary database is a large risk for a corpus this small.
- Search: recall loads the matrix into memory at boot and refreshes it incrementally by `created_at` cursor. Brute-force cosine over about 10k × 1024 float32 vectors is about 40 MB and a few milliseconds of numpy. No approximate-nearest-neighbour index is needed below roughly 1M vectors. Revisit only if the eval shows metacog digests (98k) belong in the corpus. My prior is that they do not.
- Model: keep `bge-large-en-v1.5`. It is already running, and changing it invalidates every stored vector. Query embeddings need the bge query instruction prefix ("Represent this sentence for searching relevant passages: "), and vector-host must support it. **UNVERIFIED** whether it does today.
- Latency risk: topic-foundry sends ~1 embedding per second to the same CPU host. The query embed therefore needs its own timeout (500 ms). If it misses, dense is skipped for that request and `dense_skipped=timeout` is recorded, never silently.

**2b. Sparse / keyword retrieval. ADOPT via Postgres full-text search.**
- It is already proven in production for cards. Add expression GIN indexes (`CREATE INDEX CONCURRENTLY … USING gin (to_tsvector('english', …))`) on `chat_history_log(prompt||response)` and on `journal_entries(title||body) WHERE source_kind <> 'metacog'`. No table rewrite and no new columns.
- `ts_rank_cd` is not true BM25 (no term-frequency saturation, weak IDF). That matters little at this corpus size. Switch to an in-process BM25 (`bm25s`) only if the Phase 0 eval shows lexical queries losing. Dense retrieval reliably misses exact identifiers ("p4", "v100", "PR #2398"), so this list is not optional.

**2c. Graph retrieval**
- *GraphRAG local search over entity neighborhoods*: **ADOPT in its Orion-sized form.** This is `falkor_neighborhood` fed by the new entity linker, emitting its own ranked list rather than a score boost.
- *HippoRAG-style personalized PageRank (PPR)*: **ADOPT CONDITIONALLY (Phase 4).** Seed PPR at the linked entities, run it over the bipartite turn–entity graph (~3k turns, 318 entities, 3,975 edges; numpy power iteration in-process, <10 ms), and rank turns by stationary mass. This is the principled version of what the entity-relatedness boost approximates by hand. Gate: build it only if Phase 0 shows entity-anchored queries are ≥20% of the eval set and Phase 2 still misses them. If built, it **replaces** `_entity_relatedness_boost` (no parallel boost left ticking). Known limit: only 324 of 1,916 chat.history graph turns resolve to text, and AI Town names dominate the entity set. Fix the join or filter by `source_kind` before trusting it.
- *GraphRAG global search / community summaries*: **REJECT.** It needs an LLM-written summary per community, over a graph with no relations whose biggest entities are "orion" and AI Town agents. No consumer asks global sense-making questions of recall. Crystallizations and topic_foundry already play the "distilled themes" role. This is the keyword-cathedral pattern: a producer with no consumer and no eval.
- *graphiti_temporal*: **REJECT and retire from the recall path.** 25 nodes of raw message text, stale since 09-04. The active-packet graphiti rail should be killed together with the dead Chroma rail (Phase 1), per the "retire completely" rule.
- *orion_worldview*: **NOT NOW.** It is a curiosity-investigation graph. It could become a retriever for self-inquiry turns later, but only once a labeled self-inquiry slice exists to prove it helps.

**3. Temporal handling. ADOPT: a filter when asked, a prior otherwise, never the sort.**
- With a temporal scope, every retriever filters at the source. Without one, the recency list is one RRF input with a modest weight (start at 0.5 against 1.0 for dense and sparse), so a relevant month-old turn can beat an irrelevant one from an hour ago. `chat.continuity.v1` ("what were we just talking about") keeps a higher recency weight. That is its job, and the eval checks it with its own slice.

**4. Fusion. ADOPT weighted RRF.**
- `score(d) = Σ_lists w_list / (60 + rank_list(d))`. Rank-based fusion needs no score calibration across backends. That matters here because today's base scores are hand-set constants that cannot be compared with each other.
- The companion spec's provenance (`backend`, `sub_query`) supplies the list membership. Tag and turn-effect boosts either become lists or get dropped, decided by ablation in the eval. They are not kept by default.
- The old composite runs **in shadow** for one phase: both rankings are computed and both are logged to telemetry. Then it is deleted.

**5. Reranking. ADOPT, with a hard budget, after fusion shows a gap.**
- Model: `BAAI/bge-reranker-v2-m3` (568M params, multilingual) or `bge-reranker-base` (278M). Score the top 40 fused candidates, with each passage truncated to 256 tokens.
- Where it runs: resident on athena, as a new `/rerank` verb in **vector-host** (the existing semantic model host), not a new service. It is reached over a bus RPC channel `orion:rerank:request` with a reply prefix, registered in `orion/bus/channels.yaml`. Device: the T10 if a week of biometrics shows ≥3 GB free headroom. Otherwise CPU with `bge-reranker-base`.
- Budget: 400 ms p95 on GPU, 1,200 ms on CPU. Both are **UNVERIFIED** estimates that Phase 3 must measure. On timeout, recall keeps the RRF order and records `rerank_skipped=timeout`.
- Not through the GPU pool: pool leases load models on demand, and a cold load inside a recall is a multi-second stall.
- Why it is worth it: a cross-encoder reads query and passage together, which is the single biggest precision gain in standard RAG benchmarks. But Orion's gain is **UNVERIFIED** until the eval says so, which is why it comes after Phase 2.

**6. Diversity. ADOPT MMR, which is cheap.** Maximal marginal relevance over the doc embeddings already in memory (λ=0.7), on top of the existing transcript dedupe. It directly targets the "same seven turns every time" failure.

**7. Context assembly. KEEP, small changes.** Keep the existing per-profile `render_budget_tokens`, `max_per_source` and `max_total_items`. Order the output by final score, best first. Every rendered line keeps its source tag so the model and the debug surface can tell a memory from a feed. Context feeds (bus anomalies, recent timeline) render in their own labelled block **after** retrieved memories, with a separate small cap (e.g. ≤3), so they can never again take 30% of the slots.

## Evaluation plan (Phase 0 is built first)

### The labeled set: three slices, built cheaply and honestly

1. **Real back-reference turns (the primary slice, n≈60–100).** Take `chat_history_log` prompts that refer back to something ("remember", "last time", "you said", "I told you", "we talked", "last week"; 31 match today) plus a hand-picked sample of other real Juniper turns. Add the self-inquiry and reading `retrieval_query` strings once the companion spec's Phase 3 records them in `recall_telemetry`. Label by **pooling**: for each query, take the union of the top-20 from every candidate system (today's recall, sparse, dense, graph) and grade each pooled item 0/1/2. An LLM judge drafts the grades, and **Juniper audits a random 25%**. Report the judge–Juniper agreement (Cohen's κ). If κ < 0.6, the judge grades are not used.
2. **Known-item paraphrase slice (synthetic, labeled as such, n≈100).** Use `memory_crystallization_sources` rows whose source turn resolves in Postgres (544). The query is a paraphrase of the crystallization summary, and the gold answer is the linked source turn(s). Drop pairs where the summary is a near-verbatim copy of the source (many are). Paraphrases are written by one LLM and filtered by a second check. Caveat: LLM paraphrases lean away from the original wording, which favours dense retrieval. Report this slice separately and never average it into slice 1.
3. **Continuity slice (n≈30).** "What were we just talking about" turns, where the gold is the previous 1–3 turns of the same session. This protects the one case where recency *is* right, so the new pipeline cannot regress it silently.

Storage: `services/orion-recall/evals/retrieval_set/v1.jsonl` (query, slice, gold doc ids with grades, pool provenance, created_by, audited_by). The set is frozen and versioned. Private chat text stays out of git: the file stores **doc ids only**, and the harness resolves text from Postgres at run time.

### Metrics

- **Recall@k (k=8, 20)**: did the relevant memories make it into the candidates? Measured before the reranker to judge first-stage retrieval.
- **nDCG@8**: is the best material at the top? Measured on the final selected list, which is what reaches the model.
- **MRR** on the known-item slice.
- **Distinctness**: distinct selected ids ÷ total selections over a replay of real telemetry queries. Today it is **106 / 5,003 = 2.1%**. A healthy value is not a target by itself, but a number that stays near 2% after the change means the change did nothing.
- **Groundedness (periodic eval, not a gate)**: for 30 slice-1 queries, generate the final answer with the recalled context and have a judge label each factual claim about the past as supported / unsupported by the recalled items. Juniper audits 10. Reported, not gated, because it depends on the chat model too.
- **Latency**: p50/p95 per stage from the telemetry `timings_ms` the companion spec adds.

### Baseline comparison

The harness calls `process_recall` in-process against live read-only stores, with the pipeline selected by a profile flag, so **today's pipeline is the baseline run on the same set**. Each phase reports the baseline and the new numbers side by side, per slice, with a paired bootstrap 95% CI on nDCG@8. A phase ships only if slice 1 nDCG@8 improves with a CI excluding 0, and slices 1 and 3 do not regress beyond −0.02.

### Metric quality gate (CLAUDE.md), applied to every new retrieval signal

Each new ranked list (dense, sparse, graph-PPR, rerank score) must, before it gets a fusion weight:
1. **Provenance**: name the producing function and line (e.g. `dense_rank` comes from `retrievers/dense.py::search`, cosine against `recall_doc_embedding` rows written by vector-host `persist_embedding`).
2. **Independence**: dense and rerank both read text semantics. Reranking a dense list is a refinement, not an independent vote, so rerank never enters RRF as a list. PPR and the old entity boost read the same `MENTIONS_ENTITY` edges, so PPR replaces the boost. It is never added next to it.
3. **Theory anchor**: dense = semantic similarity via contrastive bi-encoder (bge). Sparse = lexical exact match (BM25 family). PPR = HippoRAG associativity from seed entities. Cross-encoder = joint query–passage relevance. RRF = Cormack et al. 2009. MMR = Carbonell & Goldstein 1998.
4. **Live-data sanity**: on the eval set and on a telemetry replay, check the list is not degenerate. It must not return the same top items for every query (the exact failure recency has today). Similarity must spread (report the cosine distribution, not just its mean). There must be a real "no good match" state: a nonsense query must produce low top-1 similarity, not a confident hit. Dense lists always return *something*, so record top-1 similarity and let context assembly drop items below a calibrated floor.
5. **Existing mechanism**: sparse reuses the cards FTS pattern. Dense reuses vector-host and bge. Graph reuses `falkor_neighborhood` and the entity names. Nothing new where something exists.
6. **Reversibility**: every list has a fusion weight in the profile YAML. Weight 0 removes it with no schema unwinding. The `recall_doc_embedding` table is additive and droppable.

Findings from this gate are recorded in each phase's PR report.

## Phased plan

| Phase | What | Acceptance checks | Rollback | Cost |
|---|---|---|---|---|
| **0: Eval set + baseline** | Build `evals/retrieval_set/v1.jsonl` (3 slices), the harness `evals/run_retrieval_eval.py`, and a telemetry replay (distinctness). Run today's pipeline. Commit the baseline report | Set exists with ≥60 slice-1 queries and judge κ ≥ 0.6 on the audited 25%. Baseline numbers for recall@8/20, nDCG@8, MRR, distinctness and latency committed in a report. Re-running is deterministic (same numbers twice) | Delete the eval dir; nothing in runtime changes | No GPU. LLM-judge calls for ~2k pooled pairs, one-off, on the agent lane. ~Half a day of Juniper audit |
| **1: Kill the dead rails, fix the feeds** | Remove the active-packet Chroma rail and the query embed that feeds it (and its graphiti rail). Set `RECALL_ENABLE_RDF=false` (Fuseki is gone). Render context feeds in their own capped block. Log `candidates_per_list` | Zero "chromadb not installed" logs over 24h. Zero recall `POST /embedding` calls until Phase 2. Bus-anomaly share of selected items drops from 30% to ≤3 per recall. Eval: no regression on any slice | Revert the PR. Env flags restore RDF | Saves ~160 embed calls/day and one wasted HTTP round trip per active_packet recall |
| **2: Hybrid first stage + RRF** | Postgres FTS retriever (expression GIN indexes). Write-time embeddings (`recall_doc_embedding`, vector-host persist, backfill). In-memory dense retriever. Entity linker. Weighted RRF in shadow for one week, then primary. MMR | Metric gate passed per list. Slice-1 nDCG@8 better than baseline with CI excluding 0. Slice 3 no worse than −0.02. Distinctness on replay ≥5× baseline. Recall p95 (non-stance) < 2.5 s. Embedding freshness: newest `chat_history_log` row embedded within 60 s (query) | Profile weight 0 for dense/sparse restores today's ranking. Shadow mode means the old path stays live during the trial | Query embed ~200 ms CPU. Backfill ~10k docs × ~0.2 s ≈ 35 min CPU, off-peak. ~40 MB RAM in recall. Index build `CONCURRENTLY` |
| **3: Cross-encoder rerank** | `/rerank` verb in vector-host, bus RPC, top-40, deadline, `rerank_skipped` telemetry | Slice-1 nDCG@8 improves over Phase 2 with CI excluding 0 (if not, **do not ship**). Rerank p95 ≤ 400 ms on GPU or ≤ 1.2 s on CPU, measured live. `rerank_skipped` < 5% of recalls | `RECALL_RERANK_ENABLED=false` | ~1–2 GB VRAM on the athena T10 (headroom UNVERIFIED), or 2–4 CPU cores |
| **4: Graph PPR (gated)** | Only if Phase 0 shows ≥20% entity-anchored queries and Phase 2/3 still miss them. PPR list replaces `_entity_relatedness_boost` | Entity slice recall@20 improves. The old boost code is deleted in the same PR. PPR latency < 50 ms | Fusion weight 0, then revert | CPU only, in-process |

## Missing questions

1. **Should Orion's own metacog digests (98,428 journal rows) be retrievable memories?** They are 97% of the journal corpus. Including them multiplies the embedding cost ~10× and probably floods results with self-telemetry. My recommendation: exclude them from dense/sparse indexing. Metacog has its own paths.
2. **Should AI Town social turns (10,241 ChatTurns) be in Orion's personal recall corpus?** They dominate the entity graph. Recommendation: exclude them from the default profile, and allow them only in the AI Town-aware profiles, consistent with the existing source-tagging work.
3. **Is the athena T10 available for a resident reranker, or should reranking be CPU-only?** This needs Juniper's call on GPU ownership, plus a week of biometrics.
4. **Who audits the eval labels?** The plan assumes Juniper audits ~25% of LLM-judged grades (roughly 2–3 hours). If that is too much, the fallback is a smaller slice 1 (n≈40) that is fully Juniper-graded.

## Proposed schema / API changes

- **New Postgres table** `recall_doc_embedding` (created by the writer's idempotent DDL, `CREATE TABLE IF NOT EXISTS`):
  `doc_kind text, doc_id text, model text, dim int, embedding real[], text_hash text, source_ts timestamptz, created_at timestamptz default now(), PRIMARY KEY (doc_kind, doc_id, model)`.
- **New indexes** (`CREATE INDEX CONCURRENTLY`): FTS expression GIN on `chat_history_log`, and on `journal_entries` filtered to `source_kind <> 'metacog'`.
- **Bus**: `orion:rerank:request` plus a reply prefix `orion:rerank:result:*` (with the catalog wildcard, per the dynamic-reply-channel memory). New schemas `RerankRequestV1{query, passages[{id,text}], top_n, deadline_ms}` and `RerankResultV1{scores[{id,score}], model, elapsed_ms}` in `orion/schemas/vector/`, registered in `orion/schemas/registry.py`. Embedding persistence rides the existing `orion:embedding:generate` / chat-history consumers inside vector-host, with no new channel.
- **`recall_telemetry`** (nullable columns, `ADD COLUMN IF NOT EXISTS`): `list_counts jsonb` (candidates per retriever), `fusion_mode text` (`composite`|`rrf`|`rrf_shadow`), `rerank_ms int`, `rerank_skipped text`, `dense_top1_sim real`, `selected_provenance jsonb` (per selected id: which lists had it and at what rank).
- **Recall profile YAML** gains `retrieval.lists: {dense: w, sparse: w, graph: w, recency: w}`, `retrieval.rrf_k`, `retrieval.mmr_lambda`, `rerank.enabled`, `rerank.top_n`, `feeds.max_items`.
- **Env** (orion-recall `.env_example`, then sync local `.env`): `RECALL_DENSE_ENABLED`, `RECALL_DENSE_QUERY_TIMEOUT_MS`, `RECALL_RERANK_ENABLED`, `RECALL_RERANK_TIMEOUT_MS`, `RECALL_FUSION_MODE`. Vector-host: `VECTOR_HOST_PERSIST_EMBEDDINGS`, `VECTOR_HOST_RERANK_MODEL`, `VECTOR_HOST_RERANK_DEVICE`.
- No change to `RecallQueryV1` beyond the companion spec.

## Files likely to touch

- Phase 0: `services/orion-recall/evals/` (new: `retrieval_set/v1.jsonl`, `run_retrieval_eval.py`, `build_pool.py`, `replay_distinctness.py`, README). Retire `app/recall_eval.py` + `recall_eval_corpus.json` or fold them in.
- Phase 1: `orion/memory/crystallization/retriever.py`, `services/orion-recall/app/collectors/active_packet.py`, `services/orion-recall/app/fusion.py` / `render.py` (feeds block), `services/orion-recall/.env_example`, `settings.py`.
- Phase 2: `services/orion-recall/app/retrievers/` (new: `dense.py`, `sparse.py`, `entity_link.py`, `temporal_scope.py`), `app/fusion.py` (RRF + MMR), `app/worker.py` (lists wiring), `orion/recall/profiles/*.yaml`, `services/orion-vector-host/app/` (persist), `scripts/backfill_recall_doc_embeddings.py` (backfill protocol, snapshot under `/tmp/`), `services/orion-sql-writer` or recall's DDL for the table, tests in both services.
- Phase 3: `services/orion-vector-host/app/rerank.py`, `orion/schemas/vector/schemas.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`, vector-host `requirements.txt` / `.env_example` / `docker-compose.yml` (GPU device reservation if the T10 is used), `services/orion-recall/app/rerank_client.py`.
- Phase 4: `services/orion-recall/app/retrievers/graph_ppr.py`. Delete `_entity_relatedness_boost` and `storage/falkor_entity_relatedness.py`'s boost-map path.

## Non-goals

- Query formation, fan-out caps and the deadline: the companion spec owns these.
- GraphRAG global/community summaries, graphiti, LLM query rewriting, HyDE, multi-vector (ColBERT) indexes, fine-tuning the embedder or reranker.
- Migrating Chroma, installing pgvector, or changing the embedding model.
- Fixing the Falkor ChatTurn writer lag, the 83% textless chat.history ChatTurns, or deciding what `orion_main_store` is for. Each gets a separate investigation ticket, because they affect Phase 4 inputs.
- Changing what gets written to memory (cards, crystallizations). This is read-path only, plus one additive embedding table.

## Proposal-mode disclosure

- **Capability change:** recall stops returning mostly the newest turns. It returns the memories that match what the turn is about, including older ones. This changes what Orion "remembers" in every recalling verb (chat, stance_react, journal.compose, reverie). That is a cognition change, so every phase is gated by the eval and ships behind profile weights.
- **Data touched:** reads chat_history_log, journal_entries (non-metacog), memory_cards, memory_crystallizations, world_pulse claims and the Falkor `orion_recall` graph. Writes only the new `recall_doc_embedding` table (derived, rebuildable) and new telemetry columns. Eval files store doc ids, not text.
- **Privacy boundary:** unchanged scoping. Existing lane/visibility filters (`visibility_allows_card`, session/node scoping, AI Town source tagging) are applied **inside each retriever before ranking**, not after, so a dense hit cannot pull a private item across a boundary that the old path respected. Embeddings are derived from the same private text and live in the same Postgres. Nothing leaves the host. The reranker runs locally.
- **Trace that proves it:** `recall_telemetry.selected_provenance` shows, for each selected memory, which retrievers found it and at what rank. Distinctness on live telemetry rises from 2.1%. The eval report shows the nDCG@8 lift with a CI. `rerank_ms` / `dense_top1_sim` are populated (not null, not constant).
- **Dangerous failure mode:** (a) semantic retrieval confidently surfaces a *related-sounding but wrong* memory, and Orion asserts it as fact. That is worse than recency, which is at least honestly "recent". Mitigated by the top-1 similarity floor, the groundedness eval and the per-line source tags. (b) Recency regressions: "what were we just talking about" loses the last turn. Guarded by the continuity slice. (c) Retrieval crosses a privacy lane. Guarded by filtering inside retrievers, with a test per retriever.
- **Rollback:** `RECALL_FUSION_MODE=composite` restores today's ranking in one env flip. Per-list weights go to 0. `RECALL_RERANK_ENABLED=false`. The embedding table can be dropped without affecting anything else.

## Acceptance checks

1. Phase 0 report committed with baseline recall@8/20, nDCG@8, MRR, distinctness (expected ≈2%) and latency, and a judge κ ≥ 0.6 on the audited sample.
2. Unit: each retriever applies the lane/visibility filter before ranking (a private-lane doc never appears for a public-lane query).
3. Unit: RRF of known lists matches a hand-computed fixture. MMR drops a near-duplicate.
4. Unit: a nonsense query yields `dense_top1_sim` below the floor and no dense items rendered.
5. Eval (per phase, per slice) meets the gates in the phased plan, with numbers in the PR report.
6. Live 24h after each deploy: zero "chromadb not installed" (Phase 1). `selected_provenance` populated on ≥95% of rows (Phase 2). `rerank_skipped` < 5% and rerank p95 within budget (Phase 3). Recall p95 for non-stance verbs < 2.5 s throughout.

## Recommended next patch

**Phase 0 only**, in `services/orion-recall/evals/`: the pooled labeled set (three slices), the harness, the telemetry-replay distinctness script, and a committed baseline report. It touches no runtime path, it makes every later phase decidable, and it will put a number on "most recent N wins" (today's replay suggests ~2% distinctness). Phase 1 (killing the dead Chroma/graphiti/RDF rails and boxing the context feeds) can proceed in parallel, since it is cleanup with its own acceptance checks and no ranking change.

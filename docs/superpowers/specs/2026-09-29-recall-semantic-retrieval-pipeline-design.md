# Recall by referent, not by resemblance

Status: PROPOSAL, revision 2 (design mode + proposal mode). Nothing here is implemented.
Date: 2026-09-29
Revision 2 replaces revision 1, a generic dense + BM25 + rerank pipeline, which Juniper rejected. What changed and why is at the bottom.
Builds on: `docs/superpowers/specs/2026-09-29-recall-retrieval-query-architecture-design.md` (APPROVED, in progress). That spec fixes *what* recall searches for (`retrieval_query`), bounds the fan-out, adds a deadline and puts provenance on candidates. This spec decides *what counts as a match*. It assumes the companion spec has landed and does not redo it.

Evidence base: read-only live queries against Postgres (`conjourney`), FalkorDB, Chroma (`localhost:8500`), `nvidia-smi` on athena, the published graphify bundle, and container env/logs. Main is at 38d65a36e. Numbers were pulled 2026-09-29 between 22:00 and 23:59 UTC. The analysis scripts ran on exported CSVs and are reproduced in Phase 0.

## Arsonist summary

Recall should bring back the things a turn is **about**: the same PR, the same service, the same person, the same question, the same article, the same claim. It should not bring back things that merely **sound like** the turn. Today it does neither. It brings back whatever is newest.

- **Today recall ignores the query.** 411 recalls selected 5,095 items, and only **107** were distinct. The most recent chat turn was selected in **369 of 411** recalls. When `journal.compose` asked about `transport:rpc_timeout:orion:exec:request:LLMGatewayService`, recall answered with "Unwinding after a long day of travel and team socializing. You?". Of the text-bearing items recall selected for queries that contain a distinctive term, **98.4% (1,222 of 1,242) share no distinctive term with the query**.
- **Embeddings would swap "unrelated" for "cousins".** An embedding says two texts are near. It does not say they are about the same thing. In Orion's own bge vectors, a NASA Starliner update's nearest neighbour is a NASA Armstrong anniversary (cosine 0.80). Orion's own design doc on reading internal documents lands nearest a sensor paper (0.76). A real same-referent match (two GGUF quantization docs) scores 0.87. No threshold separates those two cases, and some neighbour always comes back, even when nothing in the store is related.
- **Orion already has what referent-based recall needs; none of it is wired into recall.** It has:
  - its own vocabulary, which makes rarity measurable: "p4" appears in 3 of 4,046 documents, "stance_react" in 7, "memory" in 275, "feel" in 652;
  - typed links: curiosity findings are `ABOUT` priors, world-pulse claims point to their article, crystallizations point to their source turns, Falkor turns point to entities;
  - the graphify graph of the codebase and PR reports: 77,966 nodes, 530 PR-report files.

The proposal:
1. Pull the **referents** out of the query: explicit identifiers, plus terms that are rare in Orion's own corpus.
2. Look them up in a small **referent index** built at write time.
3. Follow **one hop of real, typed links**.
4. Rank by how many distinctive referents an item shares with the query, not by a similarity score.
5. Attach **why** each item was recalled and **whose words** it is.

When the query names nothing distinctive, recall says so and returns the boxed "what's going on" context, instead of eight confident-looking strangers.

Vectors are out of recall entirely. The evidence and the reasoning are in "Vectors: the verdict" below.

## Current architecture

### What "shitty cousins" means, precisely

A **cousin** is a retrieved item that resembles the query through genre, register, topic area or template, while the thing it is *about* (its referent) is different: same vibe, different referent.

- **Operational test** (used in the eval below): an item is a cousin if it shares **no referent key** with the query. Referent keys are explicit identifiers (PR number, file, service, channel, URL), linked entities, and terms that are rare in Orion's corpus.
- **Not a cousin**: an item that shares the referent in different words. That is a legitimate match, and this design must not lose it. The known limit of that goal is discussed under the vector verdict.

**Live examples.** These come from Orion's own `bge-large-en-v1.5` vectors in Chroma `orion_reading_results` (29 docs, read-only nearest-neighbour queries using the stored embeddings; cosine = 1 − d/2).

| Query document | Nearest neighbour | Cosine | Same referent? |
|---|---|---|---|
| NASA/Boeing Starliner development update | NASA Armstrong celebrates 80 years of flight | 0.80 | **No**: same publisher and genre (NASA press release) |
| WHO statement on US withdrawal (2026-01-24) | WHO tribute to David Nabarro | 0.73 | **No**: same publisher |
| Orion's own spec `2026-09-28-reading-internal-documents-design.md` | MDPI Sensors paper; Bonsai-2 27B model card | 0.76; 0.74 | **No**: "technical document" vibe |
| Anti-trans attack ads article | Utah governor's weekly schedule | 0.57 | **No**: nothing related exists, but a neighbour still comes back |
| HF blog "transformers + llama.cpp quants" | HF docs "GGUF quantization" | 0.87 | **Yes**: same referent (GGUF) |

Two properties matter:
- The true match (0.87) and the worst cousin (0.80) sit 0.07 apart, and there is no calm "nothing matched" state.
- This is a 29-document sample. How the gap behaves at scale is **UNVERIFIED**. The mechanism (similarity measures resemblance, not identity) is not.

### Semantic asset inventory (live-verified unless marked)

| Asset | State | Evidence |
|---|---|---|
| Recall ranking (`services/orion-recall/app/fusion.py:504-520`) | LIVE, effectively recency | `composite = backend_weight*(0.7*base_score + 0.15*overlap + 0.1*recency) + boosts`. `base_score` is a constant per adapter (`worker.py:732,1010,1217,1237,1287,1613`; `falkor_chat_adapter.py:118`). Candidates are fetched `ORDER BY ts DESC` |
| Recall dense path | DEAD by design | Removed in commit 710c7bfc3 (2026-05-30). `backend_counts.vector` is 0 in every row. Guarded by `tests/test_recall_vector_amputation.py` |
| Active-packet Chroma rail (`orion/memory/crystallization/retriever.py`) | DEAD, and it wastes a call | Embeds the query (160 recall→vector-host calls in 24h), then `chromadb` is missing (`ModuleNotFoundError` in the container; 29 "chromadb not installed" logs in 4h). Target collection `orion_memory_crystallizations` does not exist. Its hits would never be merged anyway (`extra_crystallization_ids` is unused, `retriever.py:121-164`) |
| Fuseki / RDF | DOWN | No container exists (`docker ps -a`), `:3030` refuses, yet `RECALL_ENABLE_RDF=true` in the live env |
| graphiti_temporal (Falkor) | STALE, not a knowledge graph | 25 `Entity` nodes whose names are raw user messages. Newest 2026-09-04 |
| Embedding model (vector-host, bge-large, CPU) | LIVE | ~200 ms/text. Used by topic-foundry (86,677 calls/24h) and OrionTissue. Recall does not need it under this design |
| Chroma 0.4.24 | LIVE, not a memory store | `orion_main_store` 6,612 rows of llm-gateway output JSON, oldest 3 s after today's 04:44 container start despite a bind mount (why is **UNVERIFIED**) |
| Memory cards | LIVE | Postgres full-text search over a weighted tsvector. 1,166 cards (436 active). Appear in 16/401 recalls |
| Falkor `orion_recall` | LIVE, partly stale | 12,157 ChatTurn (10,241 AI Town `social.turn.stored.v1`, 1,916 `chat.history`, newest 09-29 04:11), 318 Entity, `MENTIONS_ENTITY` 3,975 (spaCy NER via `orion-meta-tags`). No entity–entity edges. Only **324 of 1,916** chat.history turn ids resolve to Postgres text (cause **UNVERIFIED**) |
| Falkor `orion_worldview` (curiosity) | LIVE, not read by recall | Hop 548, Finding 171, Prior 120, Concept 13 … Typed edges: `SUPPORTS` 165, `ABOUT` 56 (Finding→Prior 19, HelpRequest→Prior 25, Finding→Concept 10), `CONTRADICTS` 25, `ANSWERS` 19 |
| graphify graph | LIVE on disk, **18 days stale**, not usable in the request path | Published bundle at `/mnt/storage-warm/orion-graphify/published/graphify-out/graph.json`: 108 MB, 77,966 nodes, 169,111 edges, built at aff23fac0 (2026-09-10). Node types: code 46,110, document 19,919 (8,107 from 530 PR-report files), rationale 10,752, concept 1,174. Edges: contains, calls, references, imports, rationale_for … `graphify query` took **10.1 s and 0.98 GB RSS** for one query, and there is no importable Python API. Precedent: `services/orion-cortex-exec/app/self_study.py:872` reads `graph.json` directly as JSON, mounted read-only at `/graphify` |
| Orion's own corpus for rarity | Available | Juniper–Orion chat 531 turns + non-metacog journals 2,586 + world-pulse claims 929 = 4,046 docs. Document frequencies: p4 3, t10 6, stance_react 7, v100 17, falkordb 18, circe 68, recall 33, memory 275, orion 339, think 432, feel 652, the 3,157. 5,555 terms appear in 2–5 docs (the raw material for the automatic eval) |
| Explicit identifiers in chat | Sparse but real | 15 chat turns cite PR numbers. 24 distinct `orion-*` service names. File names (finalize.py 11, test_finalize_reflect_llm_fallback.py 9, executor.py 3) |
| Existing eval | Too weak | `app/recall_eval.py` with 5 term-coverage cases and no gold ids |

### Where selections go today (411 telemetry rows)

- 5,095 selections: 1,524 bus-anomaly feed items (30%) and 1,899 for verbs that send an empty query (reverie). Of the rest, 1,222 items share no distinctive term with a query that has one, and **20 share one**.
- By verb, for short queries (≤500 chars) with distinctive terms: `journal.compose` 1,204 no-share vs 19 share.
- `harness_finalize_reflect` and `orion_response_repair` queries mostly contain **no** distinctive term (181 and 66 items). For those, "no referent" is the honest answer.

### Hardware (for the reranker question)

`nvidia-smi` on athena, measured now:

| GPU | Used / total | Used by |
|---|---|---|
| P4 | 4,927 / 7,680 MiB | vision-host (4,924) |
| T10 | 10,693 / 16,384 MiB | vision-host-qwen 6,522 · kev-0.8b 3,884 · vision-edge 284 |

Seven days of `orion_biometrics` for the T10 (21,955 samples): peak 10,693 MiB, mean 9,001, min 4,947, and utilization peaks at **100%**. Worst-case free VRAM is **5.69 GB**. That is enough memory for a small cross-encoder (bge-reranker-base needs roughly 1 GB in fp16, **UNVERIFIED** until loaded). But the card is shared with vision at full utilization, so rerank latency would be at vision's mercy.

Verdict: a reranker is **not** in this plan. It is another resemblance judge, and ranking by shared referents does not need one. Revisit only if Phase 2's eval shows ordering failures *among* referent-matched items.

## Sources and their epistemic status

Recall must say **whose words** an item is and **how settled** it is. "Careful on which we'd use" is concrete here: some sources are Orion's guesses about Juniper.

| Source | What it is | Epistemic status | Use in recall |
|---|---|---|---|
| Juniper–Orion chat (`chat_history_log`, 531 turns) | The conversation | prompt = **Juniper's words**; response = **Orion's words in conversation** (speculative) | Primary grounding for "what was said". Always label the speaker |
| Journals, non-metacog (`journal_entries`: embodiment 1,982, self_study 312, world_pulse 124, manual 83, scheduler 54, notify 25, self_reflection 6) | Orion's own writing, and sensor digests | **Orion's reflection**; embodiment = sensor-derived summary | Allowed, labelled "Orion's journal". **Metacog digests (98,428) excluded** (Juniper, 2026-09-29) |
| Curiosity self-questions (`curiosity_self_questions`, 13 open) | Questions Orion minted | **Open question**, not a claim | Allowed as a referent (a query can be "about" a question). Never grounding |
| Curiosity findings / hops (Falkor `Finding` 171, `Hop` 548) | Orion's investigation notes | **Orion's speculation**, with evidence attached | Allowed only via a direct referent match, labelled "Orion's investigation note" |
| Curiosity priors (Falkor `Prior` 120: open 57, supported 24, revised 24, refuted 13, active 1, confirmed 1) | Orion's hypotheses with confidence | **Orion's hypothesis**. Only `confirmed`/`supported` are conclusions | Label with status + confidence + times tested. **Refuted priors excluded.** Open priors average confidence 0.80 after 1.9 tests, so their confidence is not evidence. Example of why: a `supported` prior (0.95) asserts that Juniper personally approves stances by hand. That is Orion's guess about Juniper and must never be recalled as fact |
| Reading / world pulse (`world_pulse_article` 929, `world_pulse_claim` 929: 833 `candidate`, 96 `observed`, caveats like `requires_corroboration`; `reading_durable_turn` 98) | External articles, claims extracted from them, and Orion's reading turns | Article = **external source**. Claim = **external claim**, unverified unless `observed`. Reading turn = Orion's interpretation | Allowed, labelled with source URL/title and claim status |
| Crystallized beliefs (`memory_crystallizations` 1,407: stance 701, reflection 356, semantic 344, open_loop 6) | Orion's distilled conclusions | **Orion's conclusion**. Many `reflection` rows are machine lines ("Belief revision: same between crys_… and …") | Allowed only when the crystallization's source turn resolves to `chat_history_log` (544 of 2,280 source links). Some sources are AI Town role-play ("What's behind the door?"); that share is **UNVERIFIED**, which is why the filter is by source. Reflection machine lines excluded |
| graphify graph (published bundle) | Code, docs, PR reports as of aff23fac0 | **Code/document artifact.** Rationale nodes are LLM-extracted and so derived | Allowed for code, PR and spec referents, labelled with the build commit/date. Not a claim about current runtime |
| Memory cards | Self-authored cards | Existing path, existing labels | Unchanged |
| **AI Town** (social turns, AI Town-sourced crystallizations, town-resident entity names) | Role-play | n/a | **Excluded** from personal recall (Juniper, 2026-09-29) |
| Bus anomalies, recent timeline | "What's going on right now" | Context feed, not memory | Boxed separately, capped, never ranked against memories |

## Target design

```
retrieval_query (companion spec)
 │
 1. referent extraction (deterministic, in-process, <5 ms)
 │     explicit ids: PR #N, file paths/*.py, orion-* services, orion:* channels, env KEYS, snake_case symbols, URLs/domains
 │     rare terms and bigrams: IDF over Orion's own corpus, df ≤ 0.5% (≤ 20 of 4,046 today)
 │     alias dictionary hits: graphify artifacts, Falkor entities (non-AI-Town), curiosity question/prior ids, article titles
 │     ──► if none: ABSTAIN → return boxed context feeds + "no referent in query"
 │
 2. direct lookup in the referent index (Postgres, btree, one query)
 │     postings: referent_key → (doc_kind, doc_id, ts, speaker, epistemic_status)
 │
 3. one hop along typed links (bounded: ≤ 10 per referent)
 │     crystallization → source turn            (memory_crystallization_sources)
 │     claim → article                           (world_pulse_claim.article_id)
 │     finding/help-request → prior → concept    (orion_worldview ABOUT/SUPPORTS)
 │     PR # → PR report → files touched          (git merge log + graphify source_file)
 │     file/service → PR reports that touched it (graphify)
 │     entity → chat turns                       (MENTIONS_ENTITY, chat.history only)
 │
 4. rank: Σ idf(matched referents) × hop factor (direct 1.0, one hop 0.5);
 │        explicit time scope = hard filter; recency = tie-breaker only
 │
 5. diversity: ≤ 2 items per referent; ≤ 1 per chat session window; no item repeated from
 │        the previous recall for the same session unless it is the only match
 │
 6. render: memories block (best first), each line tagged
 │        "[recalled: mentions PR #2287 · chat 09-22 · Juniper's words]"
 │        then a separate capped context block (≤ 3 feed items)
 │
 7. telemetry: query referents, per-item {referents, via, hop, epistemic_status}, abstained, timings
```

### Stage notes

**Referent extraction.** Every piece is deterministic: regex for identifiers, a term→df table for rarity, and an in-memory alias dictionary loaded from the referent index, refreshed every 5 minutes.
- Rarity is computed over **Orion's own corpus**, not a general-English list. That is why "circe" (df 68) counts and "memory" (df 275) barely does.
- Pure numbers are ignored unless they appear as `#N` or `PR N`.
- The threshold (df ≤ 0.5%) is a starting knob, not a finding. Phase 0 reports the known-item hit rate at 0.25%, 0.5% and 1%.

**Referent index** (`recall_referent`, `recall_referent_posting`, below). It is built **at write time** by a cursor-based catch-up loop inside orion-recall that reads new rows from Postgres every 60 s. It does not depend on the bus, and it restarts from its cursor. That makes it robust, restartable and single-source-of-truth.
- Sources: chat, non-metacog journals, world-pulse claims/articles, reading turns, crystallizations (filtered by source).
- Curiosity nodes come from Falkor. graphify artifacts and PR numbers come from an **offline builder** that runs at graphify publish time and on a nightly git-log pass. It reads the published `graph.json` and never calls `graphify query` (10 s, 1 GB).
- Size: postings grow with the number of distinct rare terms per document. For about 4k documents that is tens of thousands of rows. Estimate **UNVERIFIED**; Phase 2 measures it.

**Typed links.** They are real edges that already exist, with the reason attached. No link is inferred from similarity. PR→files comes from the merge commit's diff (`git log --merges`, offline) and from graphify `source_file`, not from an LLM.

**Ranking.** Items sharing more, and rarer, referents with the query come first. That is the whole ranking. There is no fusion across independent retrievers any more, because there is one kind of evidence: shared referents, found directly or one hop away. Reciprocal rank fusion was right for revision 1's multiple similarity lists and is unneeded here.

**Abstention.** If the query has no referent (e.g. "how are you feeling tonight"), recall returns the boxed context feeds plus at most 3 most-recent Juniper turns, labelled "recent, not matched", and sets `abstained=true`. This is the honest version of today's behaviour: recency is presented as recency, not as relevance.

**Diversity.** The caps above directly target "one turn won 369 of 411 recalls". Per-referent and per-session caps keep one chatty turn from filling every slot.

### Vectors: the verdict

**Drop vectors from recall.** "Vectors propose, referents dispose" does not survive a closer look.

- If a vector hit must share a referent or rare term with the query to be admitted, the referent index already finds it directly: the same item, cheaper, with a reason attached.
- What vectors add is the items that share **no** referent. Those are exactly the cousins, or the paraphrase matches.
- The paraphrase case (Juniper says "the camera by the door" about something Orion knows as "walkway camera") is real. Vectors cannot tell it apart from a cousin: 0.80 for a cousin vs 0.87 for a true match in the sample above.
- The fix for paraphrase is an **alias** on the referent ("walkway camera" ↔ "front door cam"). Aliases can be added by Juniper, by Orion when it notices a correction, or from Falkor entity names. This spec does not build an alias proposer. It only leaves `aliases text[]` in the schema.

Production cost of keeping them: a second store, an embedding call on a CPU shared with topic-foundry, a model version pinned to every stored vector, and a failure mode ("some neighbour always returns") that no threshold fixes. The active-packet rail shows the typical result: it has quietly done nothing for weeks.

What would change this verdict: Phase 2's abstention data shows many queries that name a referent in words the index has never seen, and aliases cannot keep up. Then revisit with that evidence, not before.

## Evaluation without human labels (Phase 0)

No human labeling and no LLM-judged gate. Human-graded relevance and answer groundedness are **explicitly deferred**. Every check below is computed automatically from Orion's own data.

1. **Automatic known-item test.**
   - Draw terms and identifiers that appear in 2–5 documents (5,555 candidates today, excluding pure numbers; a stratified sample of 200 across chat, journal and claim). The query is the term, optionally wrapped in a fixed template ("what did we say about X"; the template adds no rare terms). The gold set is every document containing it, which is known exactly by construction.
   - Second variant: build the query from the 2–3 rarest terms of one document; the gold answer is that document.
   - Metrics: hit@8, recall@8, MRR.
   - *Honest caveat:* this test measures "does recall find the thing a named referent points to". That is the requirement. It is biased toward lexical/referent systems by construction and says nothing about paraphrase recall. The report must say so.
2. **Cousin rate.**
   - Over replayed `recall_telemetry` queries that contain ≥1 distinctive term: the share of selected, text-bearing, non-feed items that share no referent with the query.
   - Today: **98.4%** (1,222/1,242). That figure is dominated by `journal.compose` (1,204/1,223).
3. **Distinctness.** Distinct selected ids ÷ total selections on a telemetry replay. Today: **107 / 5,095 (2.1%)**. The top item was selected in 369 of 411 recalls.
4. **Abstention honesty.**
   - Share of queries with no referent for which recall abstains instead of filling slots.
   - The share of queries that *do* name a referent but still abstain: this is the "index doesn't know the words" signal that the vector verdict watches.
5. **Feed share.** Feed items per recall (today about 3.7 on average, 30% of selections).
6. **Epistemic label coverage.** 100% of rendered items carry speaker/status. AI Town items and metacog digests in default profiles: 0.
7. **Latency.** Per-stage timings from the companion spec. Referent stages must add ≤ 100 ms p95.

**Baseline.** The harness calls `process_recall` in-process against the live read-only stores, so today's pipeline is scored on the same sets. Recall has write side effects during retrieval: active-packet retrieval events, recall-boost persistence, and telemetry writes. The harness must run with these disabled, and Phase 0 must **prove** they are disabled by counting rows before and after a run. Whether the existing flags gate all of them is **UNVERIFIED**; a dry-run flag has missed side effects in this repo before.

**Metric quality gate** (CLAUDE.md), applied to the new ranking signal "Σ idf of shared referents":
1. **Provenance**: `recall_term_stats` is produced by the indexer's df pass, and postings by the indexer's extraction. Both are named in each phase's PR.
2. **Independence**: it replaces the overlap, rare-token and entity-boost terms, which read the same text. It does not sit beside them. The old terms are deleted when it goes primary.
3. **Theory anchor**: inverse document frequency (Spärck Jones 1972) measures how much a term identifies a document within a collection. Here the collection is Orion's own.
4. **Live data**: the df distribution printed above is not flat, and "the"/"feel" sit near zero. The rest state is real: a query with no rare term yields zero, and recall abstains.
5. **Existing mechanism**: this reuses spaCy entities, Falkor edges, worldview edges and the graphify bundle. There is no new model.
6. **Reversibility**: tables are additive, and `RECALL_RANKING_MODE=composite` restores today's ranking.

## Phased plan

| Phase | What | Acceptance checks | Rollback | Cost |
|---|---|---|---|---|
| **0: Automatic eval + baseline** | `services/orion-recall/evals/`: known-item generator (seeded, deterministic), cousin-rate + distinctness + abstention replay over `recall_telemetry`, side-effect-free harness mode with a before/after row-count proof, baseline report | Same numbers on two runs. Baseline report committed with hit@8/recall@8/MRR, cousin rate (≈98%), distinctness (≈2%), feed share, latency. Row counts of `memory_crystallization_retrieval_events`, `recall_telemetry` and the recall-boost target unchanged by a harness run | Delete the eval dir | CPU only, minutes per run. No GPU. No human time |
| **1: Clean the path** | Kill the active-packet Chroma rail and its query embed, and its graphiti rail. Set `RECALL_ENABLE_RDF=false`. Box the context feeds (≤3, separate render block). Exclude AI Town and metacog digests from default profiles. Render speaker/status labels on existing items | 24h live: 0 "chromadb not installed". 0 recall→vector-host calls. Feed items ≤3 per recall. 0 AI Town/metacog items in default-profile telemetry. Eval: known-item not worse than baseline | Revert, plus env flags | Removes ~160 wasted embed calls/day |
| **2: Referent index + direct lookup** | `recall_referent`, `recall_referent_posting`, `recall_term_stats`. Cursor-based indexer (chat, journals, claims/articles, reading turns, filtered crystallizations). Query referent extraction, ranking, abstention, diversity caps, reasons in telemetry. Shadow for 3 days, then primary | Metric gate recorded. Known-item hit@8 ≥ 0.9. Cousin rate ≤ 20% on replay. Distinctness ≥ 10× baseline. Abstain-with-referent rate reported. Indexer lag < 120 s p95. Referent stages ≤ 100 ms p95 | `RECALL_RANKING_MODE=composite` | Postgres rows (tens of thousands, **UNVERIFIED**). No GPU |
| **3: Typed links + artifact referents** | Offline builder from the published graphify bundle + git merge log (PR# → report → files). Curiosity (questions, findings, priors with status rules) from `orion_worldview`. One-hop expansion with `via` reasons | Known-item over PR numbers and file names (auto-generated from git log) hit@8 ≥ 0.8. Every one-hop item carries `via`. Refuted priors: 0 selected. Latency budget held | Disable link expansion flag | Offline build minutes/night. A 108 MB JSON read once per publish, never per request |

No Phase 4. A reranker, vectors and community summaries are all out. Each has a stated condition for being reconsidered (above).

## Missing questions

1. **Aliases:** who may add them? Proposal: Juniper, directly or via a chat correction ("I mean the walkway camera"), and Orion only as a suggestion that shows up in telemetry. Aliases change what Orion recalls, so the default is that they are not self-authored.
2. **graphify staleness:** the published bundle is 18 days old. Should recall's artifact referents follow the published bundle (reviewed, stale) or main's working bundle (fresh, unreviewed)? Recommendation: published, labelled with the build date, plus a note to Juniper that publishing more often directly improves recall.
3. **Orion's own responses as grounding:** Orion's past words in chat are "Orion said X", not "X is true". Should they be rendered with that framing (recommended), or excluded when the query asks about facts?

## Proposed schema / API changes

- `recall_referent(referent_key text primary key, kind text, display text, aliases text[], source text, df int, idf real, updated_at timestamptz)`. `kind` ∈ {pr, file, service, channel, symbol, url, entity, question, prior, article, term}.
- `recall_referent_posting(referent_key text, doc_kind text, doc_id text, ts timestamptz, speaker text, epistemic_status text, via text, primary key (referent_key, doc_kind, doc_id, via))`, with an index on `(doc_kind, doc_id)`.
- `recall_term_stats(term text primary key, df int, n_docs int, snapshot_at timestamptz)` and `recall_indexer_cursor(source text primary key, last_ts timestamptz, last_id text)`.
- `epistemic_status` values: `juniper_words`, `orion_conversation`, `orion_reflection`, `orion_investigation_note`, `orion_hypothesis`, `orion_conclusion`, `external_source`, `external_claim_unverified`, `external_claim_observed`, `code_artifact`, `sensor_summary`, `context_feed`. Each value has a producer (the indexer) and a consumer (the renderer), so none is a free-floating label.
- `MemoryItemV1` (`orion/core/contracts/recall.py`, `extra="forbid"`): optional `recall_reason {referents: [str], via: str, hop: int}` and `epistemic_status: str`. The rollout must be **consumer-first**. Registry update.
- `recall_telemetry` (nullable, `ADD COLUMN IF NOT EXISTS`): `query_referents jsonb`, `abstained bool`, `selected_reasons jsonb`, `ranking_mode text`.
- Env (orion-recall `.env_example`, then run the env sync): `RECALL_RANKING_MODE` (`composite`|`referent_shadow`|`referent`), `RECALL_RARE_TERM_MAX_DF_FRAC`, `RECALL_REFERENT_LINK_HOPS` (0|1), `RECALL_FEEDS_MAX_ITEMS`, `RECALL_EXCLUDE_AITOWN`, `RECALL_EXCLUDE_METACOG_DIGESTS`, `RECALL_REFERENT_GRAPHIFY_PATH` (default `/graphify/published/graphify-out/graph.json`, mounted read-only like cortex-exec).
- No new bus channels, no new services, no new stores outside Postgres.

## Files likely to touch

- Phase 0: `services/orion-recall/evals/` (new: `known_item.py`, `replay_telemetry.py`, `harness.py`, `README.md`, `baseline-2026-09-xx.md`). Fold in or retire `app/recall_eval.py` and `recall_eval_corpus.json`.
- Phase 1: `orion/memory/crystallization/retriever.py`, `services/orion-recall/app/collectors/active_packet.py`, `app/render.py`, `app/fusion.py` (feed box), `orion/recall/profiles/*.yaml`, `services/orion-recall/.env_example`, `settings.py`.
- Phase 2: `services/orion-recall/app/referents/` (new: `extract.py`, `index_store.py`, `indexer.py`, `rank.py`), `app/worker.py` (wiring + abstention), `orion/core/contracts/recall.py`, `orion/schemas/registry.py`, tests.
- Phase 3: `services/orion-recall/app/referents/links.py`, `scripts/build_recall_artifact_referents.py` (graphify + git log, offline), `services/orion-recall/docker-compose.yml` (read-only `/graphify` mount), the worldview reader. Delete `_entity_relatedness_boost` and the overlap/rare-token terms once referent ranking is primary.

## Non-goals

- Vectors, rerankers, GraphRAG community summaries, graphiti, HyDE and LLM query rewriting. Each has a stated reconsideration condition.
- Human-labelled relevance and groundedness evals: deferred, not replaced by LLM judges.
- Fixing the Falkor ChatTurn text-join gap (324/1,916), the Falkor writer lag, or Chroma's lost rows. These are separate tickets; Phase 3 avoids depending on the textless turns.
- Changing what gets written to memory. The only new writes are the derived, rebuildable referent tables and telemetry columns.
- Query formation and fan-out (the companion spec).

## Proposal-mode disclosure

- **Capability change:** Orion recalls the specific PR, service, person, question, article or earlier conversation that a turn names, together with the reason and whose words it is. When a turn names nothing distinctive, Orion is told plainly that nothing matched. This changes what Orion "remembers" in every recalling verb.
- **Data touched:** reads chat, non-metacog journals, world-pulse, reading turns, source-filtered crystallizations, Falkor `orion_recall` (chat.history only) and `orion_worldview`, and the published graphify bundle and git history. Writes only the new referent/posting/term-stats/cursor tables and telemetry columns.
- **Privacy boundary:** AI Town and metacog digests are excluded at index time, so they never enter postings. Existing lane/visibility scoping is applied inside the lookup, before ranking. Orion's guesses about Juniper (priors, investigation notes) are only ever rendered with their hypothesis label, and refuted priors are never recalled. Nothing leaves the host.
- **Trace that proves it:** in `recall_telemetry`, `query_referents` is non-empty for referent-bearing queries, `selected_reasons` names a shared referent for every non-feed item, and `abstained` is set when none exists. Cousin rate and distinctness on the replay move from 98% / 2%.
- **Dangerous failure mode:** (a) A hypothesis recalled as fact: a 0.95-confidence prior about Juniper's behaviour gets stated as truth. Mitigated by mandatory status labels and refuted-prior exclusion, with a test per status. (b) Silent narrowing: a turn that talks around a referent without naming it now gets nothing, where before it got (irrelevant) recent turns. Mitigated by abstention being visible in telemetry and to Orion, and by the abstain-with-referent rate being tracked. (c) A stale graphify bundle makes recall confidently cite code that has changed. Mitigated by the build-date label.
- **Rollback:** `RECALL_RANKING_MODE=composite` restores today's ranking in one env flip. `RECALL_REFERENT_LINK_HOPS=0` disables link expansion. The tables are derived and can be dropped.

## Acceptance checks

1. Phase 0 baseline report committed and reproducible (two identical runs), with side-effect row counts unchanged.
2. Unit: extraction finds `#2287`, `stance_react`, `orion-recall`, `p4` in a sentence, and finds nothing in "how are you feeling".
3. Unit: an AI Town turn and a metacog digest never produce postings.
4. Unit: a refuted prior is never selected. A supported prior renders with status and confidence.
5. Unit: a query with no referent returns `abstained=true`, feeds only, plus ≤3 labelled recent turns.
6. Eval gates per phase, as in the phased plan.
7. Live 24h after Phase 2 goes primary: cousin rate ≤ 20% and distinctness ≥ 10× on that day's telemetry. p95 latency within the companion spec's budget plus 100 ms.

## Recommended next patch

**Phase 0 and Phase 1, as two small PRs in parallel.**
- Phase 0 puts today's failure into reproducible numbers with no human labels: known-item hit rate, a 98% cousin rate, 2% distinctness. It proves the harness has no side effects.
- Phase 1 removes the dead Chroma, graphiti and RDF paths, boxes the feeds, excludes AI Town and metacog digests, and labels whose words each item is. It is useful immediately and changes no ranking logic.

Phase 2 (the referent index) follows once the baseline exists to beat.

## Revision history

- **r1 (earlier 2026-09-29):** a generic hybrid pipeline (dense + BM25 + RRF + cross-encoder + MMR). Rejected by Juniper: embeddings return topical cousins, not the same thing; the design ignored Orion's distinctive vocabulary and its existing typed stores; and it assigned labelling work to her.
- **r2 (this):**
  - Replaced similarity retrieval with referent identity: own-corpus rarity, a referent index, typed links, and reasons.
  - Added the cousin definition with live examples, and a per-source epistemic-status table.
  - Applied Juniper's answers: metacog excluded, AI Town excluded, the T10 measured (5.69 GB worst-case free, 100% utilization peaks, so no reranker), labels deferred.
  - Replaced the LLM-judged eval with automatic known-item, cousin-rate, distinctness and abstention checks.
  - Kept feed boxing, diversity caps, provenance and dead-path removal from r1.

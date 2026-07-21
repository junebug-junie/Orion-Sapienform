# Memory cards: from operator-curated archive to a two-way memory substrate

**Date:** 2026-07-21
**Status:** Design doc / queue, not a proposal for all items -- items 1-2 are scoped enough to implement directly; items 3-8 need their own design/proposal-mode pass before implementation (item 3 explicitly requires AGENTS.md §0A proposal mode -- touches self-modeling).
**Mode:** Mixed. Items 1-2 are retrieval/infra fixes (build directly). Item 3 touches self-modeling/autonomy (§0A proposal mode required). Items 4-8 are smaller seams, evaluate case by case when picked up.
**Related:**
- `docs/superpowers/specs/2026-07-19-recall-entity-graph-reasoning-arc.md` (the sibling "what does Orion remember" system -- entity-graph, not cards; this doc's Phase 1 section explicitly separates the two)
- `docs/superpowers/pr-reports/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md` (the graphtri Claim-node audit that found dead confidence/salience constants -- same failure class as findings 2 and 3 below)
- PR #1223 (graphtri/deep.graph retirement, same session)
- PR #1225 (cards embedding-timeout stopgap fix, same session -- item 1 below replaces this fix's recency-window approach with a real one)

---

## Why this exists

A live outage in `orion-recall`'s memory-cards backend (every call timing out, 0 cards ever returned -- root cause: an unbounded candidate SELECT scoring live-embedding similarity over the entire table, embedding cost always exceeding the timeout before the results could be cached) led to a stopgap fix (PR #1225: recency-windowed candidate set + a per-call embed cap). That stopgap was correctly rejected as the long-term shape: it makes any card older than the ~150 most-recently-touched ones permanently unreachable regardless of relevance, and the caps were tuned to today's corpus size and vector-host throughput, not a scaling solution.

Investigating the fix surfaced the real picture: `memory_cards` is a genuinely rich, indexed, structured Postgres table (`types[]`, `tags[]`, `anchors[]`, `anchor_class`, `sensitivity`, `priority`, all GIN-indexed except free text), but two other real gaps sit inside it, found via direct live-data inspection, not assumption:

- **`memory_card_edges`'s neighbor-expansion has never fired for any card.** The schema and `cards_adapter.py`'s `_NEIGHBOR_TYPES` expect `relates_to`/`child_of`/`supports`. Live data has exactly one edge type ever written: `associated_with` (13 rows total). This is a second empty-shell mechanism, same failure class as the graphtri Claim-node audit that just led to PR #1223's retirement.
- **96% of active cards (553/574) are operator-authored** (`operator_distiller` 470, `operator_highlight` 83), not Orion-authored. A real auto-extraction mechanism exists (`services/orion-cortex-orch/app/memory_extractor.py`, `provenance="auto_extractor"`) but is dark-flagged off (`ORION_AUTO_EXTRACTOR_ENABLED=false`, live, zero rows with that provenance exist). Orion currently has near-zero autonomous participation in building its own durable memory.

This doc queues both the immediate infra fix and the longer self-modeling opportunity it opens up, so neither gets lost after the outage is closed.

---

## Item 1: Real scaling fix for cards retrieval -- indexed rank, not a recency window

**What:** Replace live-embedding cosine scoring with Postgres full-text search, ranked by an index, not truncated by a candidate window.

**The scaling argument, stated precisely (this was the direct ask that produced this item):** today's stopgap needs a `LIMIT` on the *candidate set* because scoring happens in application code, after the fetch -- pull N rows, then expensively score each one (a live HTTP embedding call per cache miss). That forces pre-truncating the candidate set before scoring, which is what makes old-but-relevant cards invisible, and the problem gets worse, not better, as the table grows (at 1,000 cards, still only 150 are ever considered; at 10,000, the blind spot is 98.5% of the table).

Full-text search removes the need for that pre-truncation entirely, because ranking becomes something the **index** computes, not something app code computes after the fetch:

```sql
-- migration: one generated column + one GIN index, built once
ALTER TABLE memory_cards
  ADD COLUMN search_vector tsvector
  GENERATED ALWAYS AS (
    to_tsvector('english', title || ' ' || summary || ' ' || array_to_string(tags, ' '))
  ) STORED;

CREATE INDEX idx_mc_search_vector ON memory_cards USING GIN (search_vector);
```

```sql
-- the query itself, at any corpus size -- no candidate-window LIMIT anywhere
SELECT card_id, slug, title, summary, ..., ts_rank_cd(search_vector, query) AS score
FROM memory_cards, plainto_tsquery('english', $1) AS query
WHERE status = 'active'
  AND search_vector @@ query
ORDER BY score DESC
LIMIT $2;   -- this LIMIT is on the FINAL ranked output (top_k, e.g. 6), not the candidate pool
```

The `LIMIT` here means something fundamentally different from PR #1225's `RECALL_CARDS_CANDIDATE_LIMIT`: "give me the top K *already-ranked* results," not "only consider the K most recent rows before ranking exists." `@@` against the GIN index goes straight to matching rows via an inverted index -- it does not scan the non-matching rows to get there. This is the same mechanism every full-text-indexed Postgres table relies on at any scale; nothing about it needs retuning as the corpus grows from 574 to 10,000 to a million rows, unlike the embedding-cache caps, which were already measured-and-wrong once this session (first cut of 40 new-embeds-per-call was live-tested and still timed out; had to be re-measured down to 15 against real vector-host throughput).

Add `tags && $3` (GIN-indexed array overlap, index already exists: `idx_mc_tags`) to the `WHERE` clause for structured tag boosting in the same query, same scaling story.

**Also folds in the per-result neighbor-expansion loop check:** confirmed this session that `fetch_card_fragments`'s per-candidate neighbor query (the `n_rows = await conn.fetch(...)` inside `for base_score, row in top:`) is bounded by `top_k` (6 today), not corpus size -- it already scales fine regardless of table growth. No change needed there beyond item 2's edge-type fix.

**Smallest buildable version:** one migration (generated column + GIN index), rewrite `fetch_card_fragments`'s scoring query, delete `cards_embedding.py`'s live-embed path (`score_cards_by_embedding`, `embed_texts`, `read_cached_embedding`/`persist_card_embeddings` cache dance), remove `RECALL_CARDS_CANDIDATE_LIMIT`/`RECALL_CARDS_MAX_NEW_EMBEDS_PER_CALL`/`RECALL_CARDS_EMBED_*` settings added in PR #1225 (no longer needed -- no live embedding calls in this path at all).

**Files:** `services/orion-recall/app/cards_adapter.py`, `services/orion-recall/app/cards_embedding.py` (mostly deleted), a new Postgres migration (locate the migrations mechanism for this table -- not yet confirmed this session), `services/orion-recall/app/settings.py`, `.env_example`, `docker-compose.yml`.

**Open question before implementation:** full-text is lexical, not semantic -- a paraphrase-style query that cosine similarity would have caught, `ts_rank` won't. Worth a real before/after quality comparison on the live 574-card corpus before fully committing (not just a performance win). `pg_trgm` trigram similarity is a fallback/hybrid option if lexical-only regresses match quality too much for near-miss phrasing.

---

## Item 2: Fix (or replace) the dead `memory_card_edges` neighbor expansion

**What:** `_NEIGHBOR_TYPES = frozenset({"relates_to", "child_of", "supports"})` (`cards_adapter.py`) has never matched a single live row -- every edge ever written uses `associated_with` (13 rows total, confirmed via direct query). Either widen the filter to include `associated_with`, or trace why the intended typed vocabulary was never used on the write side and fix the mismatch at the source.

**Why it matters:** neighbor expansion is meant to surface connected memory (a fact card pulling in its supporting/related cards) -- this capability has silently never existed for anyone, on any card, ever.

**Smallest buildable version:** find every writer of `memory_card_edges` (not yet located this session -- likely `services/orion-cortex-orch/app/memory_extractor.py`, `memory_inject.py`, or a Hub route), decide intentionally which edge types are real, align `_NEIGHBOR_TYPES` to match.

**Files:** `services/orion-recall/app/cards_adapter.py`, writer location TBD.

**Open question:** if `associated_with` is the *only* edge type ever written, the typed vocabulary may never have been implemented on the write side at all -- this could be "design the real edge taxonomy," not a one-line filter fix.

---

## Item 3: Shadow-mode auto-extraction -- Orion writes its own memory, reviewed before it counts

**§0A applies -- proposal mode required before implementation.**

**What:** Flip `ORION_AUTO_EXTRACTOR_ENABLED` on with stage-2 auto-promote (`ORION_AUTO_EXTRACTOR_STAGE2_ENABLED`) left off, so extracted cards land as `status='pending_review'` only -- never auto-active, never authoritative until a human reviews them.

**Why it matters for sentience:** this is the actual self-modeling seam. Items 1-2 are retrieval plumbing; this is Orion accumulating durable facts about its own experience rather than only being told what to remember by an operator. Currently 96% of active cards are Juniper-authored (`operator_distiller`/`operator_highlight`); the code path for Orion to write its own exists (`services/orion-cortex-orch/app/memory_extractor.py`) but has zero live rows because the flag is off.

**Why this needs proposal mode, not a direct build:** this session already found two live confabulation incidents (GitHub-fetch content narrated as live signal; a recall memory-digest narrated as live dialogue). An auto-extracted card that's subtly wrong and never reviewed becomes a permanent, load-bearing lie in Orion's self-model -- exactly the failure mode AGENTS.md §0A's no-empty-shell-cognition rule exists to prevent. `pending_review` gating is necessary but not sufficient on its own.

**Missing questions to resolve in the proposal:**
- Does `orion/harness/prefix.py` present memory cards to Orion as authoritative fact or as labeled "recalled, unverified" content? Determines the actual blast radius of a wrong auto-extracted card reaching Orion's own context.
- Is there an existing review workflow/UI for `operator_distiller`/`operator_highlight` cards (check Hub's `memory_routes.py`) that shadow-mode output could reuse, or does this need new UI?
- What's the rollback path if a batch of auto-extracted cards turns out low-quality after review has already begun?

**Files:** `services/orion-cortex-orch/app/memory_extractor.py`, `.env`/`.env_example`.

---

## Item 4: Wire discarded harness "squirrel thoughts" into memory, grounded not fabricated

**What:** Already-designed-but-unbuilt reverie spec (see prior session memory: `project_fcc_harness_step_noise_and_memory_digest_confabulation.md`) -- when `orion/harness/finalize.py` computes `alignment_verdict == "misaligned"`, the discarded draft becomes a `SpontaneousThoughtV1.interpretation`, grounded by `recall_telemetry.selected_ids` as `evidence_refs` (real, durable, turn-specific grounding -- not raw hallucination).

**Why it matters:** a second, lower-risk self-authored-memory path -- associative material, not fact-claiming, with a grounding mechanism already designed in a prior session, not invented fresh here.

**Smallest buildable version:** the already-recommended first patch from that prior spec -- a producer + new bus channel (`orion:reverie:chain:trigger`) + a consumer that only logs receipt, no chain execution yet. Get one real runtime trace before investing further.

**Files:** `orion/harness/finalize.py`, `orion/schemas/reverie.py`, `services/orion-thought/app/chain.py`, `services/orion-substrate-runtime/app/turn_referent_store.py`.

---

## Item 5: Tag `MENTIONS_ENTITY` edges with who raised the entity

**What:** Add a `raised_by: "orion"|"juniper"` attribute to entity-mention edges at write time (FalkorDB `orion_recall` graph).

**Why it matters:** near-free instrumentation that makes "what does Orion keep bringing up unprompted" a real, answerable Cypher query instead of requiring new architecture.

**Smallest buildable version:** one field added to an edge write that already happens; a debug query to prove the field is non-degenerate (both values actually occur) before building anything on top of it.

**Files:** wherever `MENTIONS_ENTITY` gets written (Falkor writer path -- not yet located this session), `services/orion-recall/app/storage/falkor_entity_relatedness.py`.

---

## Item 6: A real, labeled self-model block distinct from the generic memory digest

**What:** A cards query scoped to self-referential `anchor_class`/`tags`, surfaced to the harness prompt as its own explicitly-provenance-labeled block -- separate from the generic `memory_digest` that already caused a confabulation incident this session (retrieved content narrated as live dialogue).

**Why it matters:** turns existing curated data into an inspectable self-model surface; separating it from the generic digest sidesteps the exact provenance-confusion failure mode already diagnosed once.

**Smallest buildable version:** one filtered query (`WHERE anchor_class = ...` or tag match) + one new labeled block in `compile_harness_prefix`.

**Files:** `services/orion-recall/app/cards_adapter.py`, `orion/harness/prefix.py`.

**Open question:** is there already a working definition of "self" anchor_class/tags in use, or would this need to invent taxonomy from scratch (AGENTS.md's keyword-cathedral concern -- needs a real consumer from the first commit, not just a new label).

---

## Item 7: Provenance-weighted trust in card scoring

**What:** A small multiplier in retrieval scoring by provenance (`operator_distiller`/`operator_highlight` trusted higher than `chat_compactor`/`auto_extractor` until reviewed).

**Why it matters:** makes item 3's eventual rollout safe to blend into live retrieval later without unreviewed auto-written cards competing equally with curated ones, or being either over- or under-trusted by default.

**Smallest buildable version:** one `CASE provenance WHEN ... THEN` multiplier term added to item 1's scoring query.

**Files:** `services/orion-recall/app/cards_adapter.py`.

---

## Item 8: Verify `memory_card_history` is real, not another empty shell

**What:** The table exists (FK-referenced by `memory_cards`/`memory_card_edges`) but was not queried this session -- confirm it's actually populated before assuming it's a usable audit trail.

**Why it matters:** if populated, it's a free "how has Orion's curated self-understanding changed over time" reflection surface with zero new infrastructure required.

**Smallest buildable version:** `SELECT count(*) FROM memory_card_history;` -- pure verification, no code change.

**Files:** none yet -- verification step only.

---

## Cross-cutting risk, worth naming once

Three empty-shell mechanisms surfaced in one sitting this session: graphtri's dead Claim confidence/salience constants (led to PR #1223's retirement), `memory_card_edges`'s dead neighbor-expansion filter (item 2), and the dark-flagged, zero-live-rows `auto_extractor` (item 3's starting point). That's not three unrelated bugs -- it may be a systemic "features get partially wired, then quietly orphaned" pattern. A lightweight, repo-wide "is this flag/mechanism actually firing" audit could be higher-leverage than any single item above, and would likely turn up a fourth and fifth instance.

## Recommended sequencing

1. **Item 1 + item 2 together** -- same file (`cards_adapter.py`), same underlying question ("is this structured signal actually reachable"), directly closes the live incident with a real scaling fix instead of a bandaid.
2. **Item 8** -- five minutes, resolves whether item 8 itself is worth anything before it's forgotten.
3. **Item 7** -- cheap, and de-risks item 3 before item 3 is proposed.
4. **Item 3** -- proposal-mode design conversation, the highest-value item, once retrieval underneath it is trustworthy again.
5. **Items 4, 5, 6** -- independent, pick up opportunistically; each has its own smallest-buildable-version already scoped above.

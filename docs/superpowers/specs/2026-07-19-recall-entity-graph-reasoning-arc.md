# Entity-graph reasoning: Phase 0 → Phase 1 → Phase 2

**Date:** 2026-07-19
**Status:** Status doc, not a proposal -- Phase 0 and Phase 1 are already built, reviewed, and shipped (merged / open PR respectively). This document narrates the arc and names what Phase 2 needs to decide before it's built.
**Mode:** AGENTS.md §0A applies to Phase 2 (touches recall ranking/fusion, a cognition-adjacent surface) -- Phase 2 is explicitly deferred until that decision is made, not built here.
**Related:**
- `docs/superpowers/plans/2026-07-18-recall-tag-entity-falkor-writer-plan.md` (the writer this whole graph depends on)
- `services/orion-meta-tags/README.md` ("Recall Falkor writer" section -- write-path details)
- `services/orion-recall/README.md` ("Graphtri" and "Entity-graph reasoning primitives" sections -- read-path details)
- PR #1203 (Phase 0), PR #1204 (Phase 1)

---

## Why this exists

This arc started from a direct critique, not a planned roadmap item: does the entity-mention graph FalkorDB already writes (`(:ChatTurn)-[:MENTIONS_ENTITY]->(:Entity)`, built by `services/orion-meta-tags` as part of the RDF→Falkor cutover) support real reasoning, or is it dressed-up keyword matching?

Walking through it live surfaced two separate problems:

1. **Data quality.** The live graph had numeric/temporal NER noise (`"first"`, `"the day"`, `"a moment ago"` as first-class `:Entity` nodes) and unmerged diacritic duplicates (`"Oríon"` vs `"orion"`).
2. **No theory behind the retrieval.** The existing graphtri read-side (`fetch_falkor_graphtri_fragments`) ranked entity mentions by raw CONTAINS-match + fixed score + recency -- no document-frequency weighting, so an entity mentioned in nearly every turn (e.g. `"orion"`, `"juniper"`) would rank identically to one that's genuinely topic-specific. This is the textbook IR failure mode TF-IDF/BM25 exist to fix, and this graph had none of it.

Both had to be fixed before building anything on top -- reasoning over a noisy, unweighted graph just propagates the noise.

---

## Phase 0: data-quality cleanup (PR #1203, merged)

- **Write-path fix:** `services/orion-meta-tags/app/main.py::_named_entities` now filters `doc.ents` by `ent.label_` against an allow-list of nameable spaCy/OntoNotes categories (`PERSON`/`ORG`/`GPE`/`LOC`/`PRODUCT`/`EVENT`/`WORK_OF_ART`/`FAC`/`NORP`/`LAW`/`LANGUAGE`), dropping the numeric/temporal ones (`DATE`/`TIME`/`ORDINAL`/`CARDINAL`/`QUANTITY`/`PERCENT`/`MONEY`). A type-level filter, not a growing word blacklist -- live-verified against the real `en_core_web_trf` model.
- **Identity-key fix:** `app/falkor_recall_writer.py::_normalize_identity_key` strips Unicode combining marks (NFKD, category `Mn`) before lowercasing, merging accent variants without destroying non-Latin-script entity names. The first version of this fix (ASCII-encode, matching an existing but wrong-for-this-case codebase idiom) was itself caught in code review before shipping -- it would have silently dropped any Cyrillic/CJK/Arabic entity to `""`. Checked the actual corpus: no real data was lost, but the bug was live and would have bitten later.
- **One-time reconcile job** (`scripts/backfill_recall_entity_graph_cleanup_reconcile.py`) replayed all 1,750 turns' real text through the fixed pipeline against the live graph: **1,501 → 737 `:Entity` nodes, 3,165 → 1,901 `MENTIONS_ENTITY` edges, 0 errors.**

## Phase 1: reasoning primitives (PR #1204, open)

Three read-only Cypher primitives in `services/orion-recall/app/storage/falkor_entity_relatedness.py`, exposed via debug endpoints:

- **`fetch_related_entities`** -- co-occurrence relatedness ranked by **Jaccard similarity** (`shared / (degree_a + degree_b - shared)`), not raw shared-turn count. This is the direct fix for the "no document-frequency weighting" critique that started this arc. Live-verified: against `"nvidia"`, raw count ties `"athena"` (degree 32, co-occurs with almost everything) with `"tesla"` (degree 7, genuinely nvidia-specific) at shared=4. Jaccard correctly separates them -- tesla 0.154, athena 0.078.
- **`fetch_bridging_turns`** -- direct co-mention first, falling back to a 2-hop bridge via a shared intermediate entity.
- **`fetch_entity_mention_timeline`** -- raw mention timestamps, the "when" dimension.

**What this graph shape can and cannot support**, stated plainly (see `services/orion-recall/README.md`'s Graphtri section for the fuller version): the bipartite `ChatTurn↔Entity` graph, projected onto entity↔entity co-occurrence, supports **associative reasoning only** -- relatedness ranking, multi-hop bridging, temporal drift. It does **not** support typed/predicate reasoning (`"X supports Y"`, `"X caused Y"`) -- there is exactly one edge type, `MENTIONS_ENTITY`. Real predicate reasoning would need the typed-relation shape `services/orion-recall`'s **memory cards** (`app/cards_adapter.py`, `relates_to`/`child_of`/`supports` edges, embedding-scored) already have, at a much smaller, curated scale (584 cards vs. ~2,000 raw entity edges). These are two different tiers of the same "what does Orion remember" question, not competing implementations of the same thing.

**Honestly incomplete, not hidden:** nothing in `worker.py`'s `_query_backends`/`process_recall` fusion pipeline calls any of these three functions. They're reachable only via `/debug/entity-graph/*` and the test suite. The original commit for this PR described the debug endpoints as satisfying the "needs a consumer" bar from AGENTS.md's no-keyword-cathedral rule -- code review correctly pushed back: a debug/UI surface does satisfy the letter of that rule (it's one of several valid options), but it's a materially weaker bar than "wired into what Orion actually retrieves," which is what this repo's existing precedent (`RECALL_FALKOR_IN_CHAT`, `RECALL_FALKOR_GRAPHTRI_IN_CHAT` -- both dark-flagged real call sites inside the fusion pipeline) actually does. The framing was corrected rather than rushing a fusion-weight decision under review pressure.

---

## Phase 2: not built, needs a real decision first

The open question is **how** relatedness should influence recall, not just whether it's plumbed in. Candidate shapes, not yet chosen between:

- **A query-expansion signal**: use `fetch_related_entities` to widen the anchor-term set feeding `fetch_rdf_connected_chatturns`/the graphtri expansion chain, the same role `fetch_graphtri_anchors` already plays.
- **A fusion-weight boost**: when a candidate fragment's source turn shares high-Jaccard entities with the query's own extracted entities, boost its `fusion.py` score -- closer to a relevance re-ranker than a new backend.
- **A new first-class backend**, following the `RECALL_FALKOR_IN_CHAT` pattern exactly: a dark-flagged fragment source in its own right, rendered like graphtri's `Claim: mentions X` fragments are today.

Each has real, different tradeoffs (query-expansion is cheapest and lowest-risk; fusion-weight boosting is more powerful but couples relatedness scoring into an already-complex ranking function; a new backend is the most consistent with existing patterns but adds another render-budget consumer). Picking one needs its own design conversation, grounded in what actually improves recall quality -- not a default chosen to close out a phase number.

**Acceptance checks for whichever shape gets picked:**
- Live before/after evidence that real recall responses change (not just that the function is reachable).
- A regression test locking in the chosen integration point, matching the existing `test_falkor_chat_swap.py`/`test_falkor_graphtri_swap.py` pattern.
- A dark-flagged rollout (ships `false`, flipped live with evidence), matching every other Falkor swap-in this service has done.

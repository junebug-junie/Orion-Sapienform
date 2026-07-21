# Memory cards: from operator-curated archive to a two-way memory substrate

**Date:** 2026-07-21 (Item 1 rewritten same day after a design conversation -- see revision note below)
**Status:** Design doc / queue. Item 1 is the design to build first and validate live. Items 2, 3, 4, 5, 6, 7, 8 are follow-ons, explicitly **not** to be scoped or built until Item 1 has shipped and its real behavior is known -- several of them (2, 3, 6, 7 especially) were written against the *old* Item 1 and need re-specing once the new one is live, not just picked up as originally worded.
**Mode:** Item 1 is a build-directly infra item (retrieval + write-path annotation), not a self-modeling change in itself -- it does not give Orion new write access to anything, it fixes how existing writers (human and automated) populate and how retrieval reads. Item 3 remains explicitly §0A-gated. Items 2, 4-8 stay mixed, re-evaluate each when its turn comes.
**Related:**
- `docs/superpowers/specs/2026-07-19-recall-entity-graph-reasoning-arc.md` (the sibling "what does Orion remember" system -- entity-graph, not cards; that doc's Phase 1 section explicitly separates the two)
- `docs/superpowers/specs/2026-05-01-orion-memory-cards-v1-design.md` (the **original** memory-cards design -- Item 1 below is largely a recovery of this spec's intent, not a new invention; see "What changed" below)
- `docs/superpowers/pr-reports/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md` (the graphtri Claim-node audit that found dead confidence/salience constants -- same failure class as the gaps Item 1 closes)
- PR #1223 (graphtri/deep.graph retirement, same session)
- PR #1225 (cards embedding-timeout stopgap fix, same session -- Item 1 replaces this fix's approach entirely, not just its numbers)

---

## Why this exists

A live outage in `orion-recall`'s memory-cards backend (every call timing out, 0 cards ever returned -- root cause: an unbounded candidate SELECT scoring live-embedding similarity over the entire table, embedding cost always exceeding the timeout before results could be cached) led to a same-day stopgap (PR #1225: recency-windowed candidate set + a per-call embed cap). That stopgap was rejected as the long-term shape -- it makes any card outside the recency window permanently unreachable regardless of relevance, and the caps were tuned to today's corpus size and vector-host throughput, not a real scaling solution.

The first pass at "the real fix" proposed replacing embedding cosine similarity with Postgres full-text search (indexed `ts_rank`, no candidate-window `LIMIT`). That's technically sound but was correctly rejected as too shallow an answer to the actual ask: not "make the query scale" but **build a read worthy of an emergent mind actually reflecting on its own memory** -- and digging into what that requires surfaced that this system already has almost everything needed, sitting unused:

- **`memory_cards` already has the right schema.** `confidence` (certain/likely/possible/uncertain), `priority` (`always_inject`/`high_recall`/`episodic_detail`/`archival`), `time_horizon` (temporal validity: timeless/era_bound/current/expiring with real start/end/as_of dates), `sensitivity`, `anchor_class`, `tags`/`anchors` (GIN-indexed). Real, populated data for most of these already exists in the live table.
- **The *original* design spec** (`docs/superpowers/specs/2026-05-01-orion-memory-cards-v1-design.md`) already specified deterministic structured scoring (anchor match +2.0, title +1.0, summary +0.5, tag +0.3) and a 17-value typed edge vocabulary (`relates_to`, `contradicts`, `supersedes`, `supports`, `evidence_for`, `evidence_against`, and 11 more) for real belief-network reasoning -- not vibes-similarity. The live embedding-cosine approach that just caused an outage was a *later, undocumented departure* from that original plan, not the plan itself.
- **Almost none of that design ever got finished.** Confirmed via direct live-data inspection, not assumption: only 2 of 4 `priority` tiers have ever been used (`always_inject`, the "zero-latency always-on facts" tier, has zero rows). Only 2 of 4 `confidence` values have ever been used. **16 of the 17 designed edge types have never been written once** -- 100% of live edges (13 total) are the generic fallback `associated_with`. `time_horizon` has real data but nothing in retrieval ever reads it. Confidence is checked on *neighbor* edges (which don't exist) but never on the primary card itself.
- **Nobody automated the annotation, and nobody was supposed to have to.** Traced why the schema is empty: it isn't a labor-avoidance problem, it's a missing-component problem. `services/orion-cortex-orch/app/chat_history_compactor_memory.py::persist_chat_history_compactor_memory_card` -- one of the only two automated card writers that exist -- just hardcodes `priority="high_recall"` for every card it ever writes. There is no classifier anywhere deciding confidence, priority, or relations. The rich schema was designed to be populated by real judgment; what shipped instead populates it with a constant or leaves it to a human doing it by hand (Juniper's own estimate: ~15 minutes of manual annotation per card today).

So Item 1 is not a retrieval-latency patch. It's finishing a memory architecture that was already designed for a reflective mind and abandoned partway through -- and doing it in a way that requires zero new manual annotation labor, because the judgment gets folded into extraction steps that already have to happen, not added as new ones.

---

## Item 1: Auto-annotated, structured, temporally-honest cards substrate

**What changed from the first draft of this item:** the original version of Item 1 (full-text search replacing embedding cosine) is **killed**, not kept as a component. Full-text/`ts_rank` may still be *one* signal inside the structured scoring below, but the framing -- "the fix is a better similarity function" -- was the wrong frame entirely. The real fix is: (a) populate the schema's real fields automatically at write time, (b) make retrieval actually use them, (c) close the loop with usage-based reinforcement instead of manual re-annotation.

### 1a. Write-time auto-annotation -- one extended LLM call, not new manual work

Every automated card writer (`services/orion-cortex-orch/app/memory_extractor.py`'s `auto_extractor`, `chat_history_compactor_memory.py`, `github_compactor_memory.py`) already runs an LLM pass to produce `title`/`summary` from raw conversation/repo content. That same call, extended with a few more structured-output fields, resolves the rest of the schema with no additional human effort and no additional model call:

- **`worth_saving: bool`** -- the gate. "Hi, how are you" fails here before a card is ever created. This is the automated version of "gate what's real vs. chitchat" -- a classification the model is already positioned to make, since it already read the content to write a summary.
- **`confidence`** (`certain`/`likely`/`possible`/`uncertain`) and an initial **`priority`** guess -- direct model judgment, replacing today's hardcoded `priority="high_recall"` constant.
- **`time_horizon`** -- if the source text names a date, the model already saw it while summarizing (`kind: era_bound, start: <date>`); if not, default to `kind: current` (honestly "true as of now," not a false permanence claim).
- **`types[]`, `anchor_class`, `tags[]`, `anchors[]`, `still_true[]`, `project`** -- same call, same reasoning pass. `still_true` in particular already shows this exact structured-fact pattern in a fraction of live cards today (e.g. `"Involves Orion; about X.", "Orion (agent)"`) -- this is recovering a pattern that partially exists, not inventing one.
- **`evidence[]`** -- not judgment at all, just the extractor's own already-known input: which turn(s), what excerpt, what timestamp. Free.
- **`sensitivity`/`visibility_scope` -- the deliberate exception.** Everything above is symmetric-risk (a wrong tag is noise). Privacy is not: guessing "public" wrong leaks something; guessing "private" wrong only over-protects. Default conservatively (`private` unless clearly and safely public) rather than trusting the classifier evenly here -- matches AGENTS.md's "privacy and blocked material stay blocked" rule. `visibility_scope` is then a deterministic derivation *from* `sensitivity` (`private` → narrow lane, `public` → wide), not an independently-annotated field at all.

### 1b. Relation detection reuses retrieval itself -- edges stop being 100% `associated_with`

At write time, run the new card's `tags`/`anchors`/`anchor_class` through the retrieval mechanism below against **existing cards only** (a small candidate set, not the whole corpus). Hand that short candidate list to the same extraction call, already holding the new fact in context: does this `relate_to`/`contradict`/`supersede`/`support` any of these? Bounded, cheap, and means retrieval and edge-authoring are the same mechanism wearing two hats, not two systems. This is what actually closes the "16 of 17 edge types never used" gap -- not a manual edge-tagging UI.

### 1c. Retrieval -- graded, structured, temporally honest; not cosine, not a recency window

- **`always_inject` cards bypass relevance scoring entirely** -- always present, zero-latency, not competing for rank. Recovers the original design's "zero-latency always-on facts" tier that was designed and never wired.
- **Temporal validity gates results, not just scores them.** A card whose `time_horizon.end` has passed is excluded from "current fact" results outright, not merely ranked lower -- a superseded fact shouldn't quietly rank #4, it should be gone unless history is explicitly requested.
- **`confidence` is a real multiplier.** `certain` outranks `likely` outranks `possible`/`uncertain` at equal topical relevance -- the difference between "I know this" and "I think I remember this," which matters for a mind that should be able to represent its own uncertainty rather than presenting every recalled fact with equal weight.
- **Deterministic structured base scoring** -- anchor/title/summary/tag match weights, recovering the original design's token-scoring approach (anchor +2.0, title +1.0, summary +0.5, tag +0.3) rather than a live external embedding call. Inspectable (you can point at exactly why a card matched), zero external HTTP dependency in the hot path (removes the exact fragile call that caused the outage this whole doc started from), and no candidate-window `LIMIT` is needed anywhere in this shape -- ranking is computed over indexed structured fields (GIN on `tags`/`anchors`/`types`, plus a `tsvector`/GIN index if full-text is kept as one component signal), which scales the same way any indexed Postgres query scales, not by shrinking the search space.
- **Real typed-edge traversal** once 1b starts populating real edge types -- `contradicts`/`supersedes` become genuine belief-revision signals (a contradicted or superseded card should visibly lose to its replacement, not sit beside it as if both are equally true), `supports`/`evidence_for` become real associative expansion.

### 1d. Recurrence closes the loop -- confidence/priority improve with zero manual re-annotation

This system already has a proven pattern for exactly this: `orion/memory/crystallization/dynamics.py::recall_boost()`, `activation = current + (1-current)*0.08`, reinforcing concept-node activation every time something gets referenced again. Applied to cards: a fact mentioned once decays toward low priority/confidence on its own; a fact that keeps getting referenced across separate sessions earns its way toward `high_recall`/`certain` purely from usage. Nobody tags anything after the fact -- recurrence *is* the signal, matching a mechanism this codebase has already built and proven elsewhere, not a new idea.

**Smallest buildable version:** extend one existing extraction call (pick the highest-volume writer first, likely `memory_extractor.py`'s auto_extractor path) with the new structured-output fields (1a), wire the conservative `sensitivity` default, implement the structured base-scoring query (1c) replacing `cards_embedding.py`'s live-embed path entirely, and get one real end-to-end card through the new pipeline before wiring 1b's relation-detection or 1d's reinforcement -- each of those layers on top and can be validated independently once the base write+read loop is proven live.

**Files:** `services/orion-cortex-orch/app/memory_extractor.py` (and siblings: `chat_history_compactor_memory.py`, `github_compactor_memory.py`), `services/orion-recall/app/cards_adapter.py`, `services/orion-recall/app/cards_embedding.py` (mostly deleted), `orion/core/contracts/memory_cards.py`, `orion/core/storage/memory_cards.py`, a Postgres migration, `services/orion-recall/app/settings.py`, `.env_example`, `docker-compose.yml`.

**Open questions before implementation:**
- Which writer gets extended first -- `auto_extractor` (currently dark-flagged, zero live rows) or `chat_compactor` (live today, currently the hardcoded-priority one)? Extending the live one proves the design against real traffic sooner; extending the dark one is lower-blast-radius while unproven.
- Exact structured-output schema/prompt for the extended extraction call -- not yet drafted.
- Whether full-text (`ts_rank`) stays in as one signal inside 1c's base scoring, or whether anchor/title/summary/tag token-matching alone (the original design's approach) is sufficient without it -- worth a real before/after comparison on the live 574-card corpus rather than assuming either way.

---

## Items 2-8: follow-ons, hold until Item 1 ships and is validated live

Everything below is preserved from the original draft for continuity, but **explicitly on hold**. Several of these were scoped against the *old* Item 1 (full-text-only) and their premises may already be partially or fully addressed by the new Item 1 above -- each needs a fresh look against what actually got built and how it actually behaved live, not a straight pickup of the wording below.

- **Item 2** (`memory_card_edges`'s dead neighbor-expansion filter) -- likely **absorbed** by Item 1b's relation-detection-via-retrieval, but confirm once 1b ships whether anything separate is still needed, rather than assuming it's fully subsumed.
- **Item 3** (shadow-mode auto-extraction, §0A-gated) -- its original framing ("flip the flag, Juniper reviews everything") is largely superseded by Item 1a's auto-annotation design; re-scope as "what does §0A proposal mode look like once extraction already includes real confidence/worth_saving judgment," not the original manual-review-only version.
- **Item 4** (reverie "squirrel thoughts" grounding) -- independent of Item 1, genuinely still just queued behind it for attention, not technically entangled.
- **Item 5** (`MENTIONS_ENTITY` `raised_by` tagging) -- independent, entity-graph not cards, unaffected by Item 1's changes.
- **Item 6** (labeled self-model prompt block) -- likely **easier** once Item 1's `anchor_class`/temporal-validity/`always_inject` machinery exists; re-scope to use it rather than building a separate filtered query.
- **Item 7** (provenance-weighted trust in scoring) -- likely **folded into** Item 1c's structured scoring directly rather than remaining a bolt-on; confirm once 1c's scoring function is real.
- **Item 8** (verify `memory_card_history` is populated) -- independent, five-minute check, do any time.

Original item text preserved below for reference.

<details>
<summary>Original items 2-8 (pre-Item-1-rewrite wording)</summary>

### Item 2 (original): Fix (or replace) the dead `memory_card_edges` neighbor expansion

**What:** `_NEIGHBOR_TYPES = frozenset({"relates_to", "child_of", "supports"})` (`cards_adapter.py`) has never matched a single live row -- every edge ever written uses `associated_with` (13 rows total, confirmed via direct query). Either widen the filter to include `associated_with`, or trace why the intended typed vocabulary was never used on the write side and fix the mismatch at the source.

**Files:** `services/orion-recall/app/cards_adapter.py`, writer location TBD.

### Item 3 (original): Shadow-mode auto-extraction -- Orion writes its own memory, reviewed before it counts

**§0A applies -- proposal mode required before implementation.**

**What:** Flip `ORION_AUTO_EXTRACTOR_ENABLED` on with stage-2 auto-promote left off, so extracted cards land as `status='pending_review'` only -- never auto-active, never authoritative until a human reviews them.

**Missing questions to resolve in the proposal:**
- Does `orion/harness/prefix.py` present memory cards to Orion as authoritative fact or as labeled "recalled, unverified" content?
- Is there an existing review workflow/UI for operator-authored cards (check Hub's `memory_routes.py`) that shadow-mode output could reuse?
- What's the rollback path if a batch of auto-extracted cards turns out low-quality after review has already begun?

**Files:** `services/orion-cortex-orch/app/memory_extractor.py`, `.env`/`.env_example`.

### Item 4 (original): Wire discarded harness "squirrel thoughts" into memory, grounded not fabricated

**What:** When `orion/harness/finalize.py` computes `alignment_verdict == "misaligned"`, the discarded draft becomes a `SpontaneousThoughtV1.interpretation`, grounded by `recall_telemetry.selected_ids` as `evidence_refs`.

**Smallest buildable version:** a producer + new bus channel (`orion:reverie:chain:trigger`) + a consumer that only logs receipt, no chain execution yet.

**Files:** `orion/harness/finalize.py`, `orion/schemas/reverie.py`, `services/orion-thought/app/chain.py`, `services/orion-substrate-runtime/app/turn_referent_store.py`.

### Item 5 (original): Tag `MENTIONS_ENTITY` edges with who raised the entity

**What:** Add a `raised_by: "orion"|"juniper"` attribute to entity-mention edges at write time (FalkorDB `orion_recall` graph).

**Files:** wherever `MENTIONS_ENTITY` gets written (Falkor writer path -- not yet located), `services/orion-recall/app/storage/falkor_entity_relatedness.py`.

### Item 6 (original): A real, labeled self-model block distinct from the generic memory digest

**What:** A cards query scoped to self-referential `anchor_class`/`tags`, surfaced to the harness prompt as its own explicitly-provenance-labeled block -- separate from the generic `memory_digest` that already caused a confabulation incident this session.

**Files:** `services/orion-recall/app/cards_adapter.py`, `orion/harness/prefix.py`.

### Item 7 (original): Provenance-weighted trust in card scoring

**What:** A small multiplier in retrieval scoring by provenance (`operator_distiller`/`operator_highlight` trusted higher than `chat_compactor`/`auto_extractor` until reviewed).

**Files:** `services/orion-recall/app/cards_adapter.py`.

### Item 8 (original): Verify `memory_card_history` is real, not another empty shell

**What:** The table exists (FK-referenced) but was not queried this session -- confirm it's actually populated before assuming it's a usable audit trail.

**Smallest buildable version:** `SELECT count(*) FROM memory_card_history;`

</details>

---

## Cross-cutting risk, worth naming once

Three empty-shell mechanisms surfaced in one sitting this session: graphtri's dead Claim confidence/salience constants (led to PR #1223's retirement), `memory_card_edges`'s dead neighbor-expansion filter, and the original memory-cards design's abandoned 16-of-17 edge types / 2-of-4 priority tiers / unused confidence and time_horizon fields. That's not unrelated bugs -- it's a systemic "features get partially wired, then quietly orphaned" pattern. A lightweight, repo-wide "is this flag/mechanism actually firing" audit could be higher-leverage than any single item in this doc, and would likely turn up further instances.

## Recommended sequencing

1. **Item 1 only.** Build 1a (write-time auto-annotation) and 1c (structured retrieval) first, get one real card through the full loop, live-verify before adding 1b (relation-detection) or 1d (recurrence reinforcement) on top.
2. **Everything else stays queued, untouched, until Item 1's real behavior is known.** Re-scope items 2, 3, 6, 7 against what Item 1 actually became before picking any of them up -- do not build from their original wording above without that check.

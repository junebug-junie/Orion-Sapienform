# Recall follow-ups: concept-relation revision loop, retirement surfacing, saturation gate operationalization

**Date:** 2026-07-13
**Status:** Proposed
**Scope:** Three follow-ups from the "bullshit RAG" investigation, discussed and refined in conversation. Drive/goal-conditioned recall ranking is explicitly **not** covered here — still at the discussion stage, no spec yet.

---

## 1. Concept-relation decision loop (closes the "wasted space" gap in spec item 2)

**Problem:** logging every `maybe_resolve_concept_relation()` decision (proposed, not built) is only useful if something reads it. A write-only log is the same theater pattern this session already killed twice (the old retrieval-events table, the pre-fix contradiction detection).

**Two loops, not one:**

### 1a. Threshold-tuning report (operational)
Periodic (or N-decisions-triggered) job reads new decision log entries since last run, reports: call volume, relation-decision distribution, count of `contradicts`/`refines` decisions that landed *below* the `CONCEPT_RELATION_CONFIDENCE_FLOOR` (0.6) — i.e., near-misses currently invisible. Output is a report, not an auto-applied threshold change — matches "no auto-apply of proposals."

### 1b. Belief-revision digest (the loop that matters for the stated goal)
For every real `contradicts`/`refines`/`same` decision since last run, produce a short structured artifact: subject, prior belief, new belief, relation, confidence. This is a direct instance of AGENTS.md's own named sentience prerequisite ("error correction... coherent action over time") — a genuine trace of Orion revising its own beliefs, not a hand-authored taxonomy. Write as a new `reflection`-kind crystallization (ties to the previously-shelved "self-referential memory kind" brainstorm idea — this gives it a real producer) and/or surface in Hub's Memory tab review queue.

**Files likely to touch:** `orion/memory/crystallization/concept_relation.py` (the missing full-decision log line, spec item 2's original ask), new module for the digest job (mirrors `orion-memory-consolidation`'s existing periodic-loop pattern, e.g. `run_retry_loop` in `services/orion-memory-consolidation/app/main.py`), `orion/memory/crystallization/schemas.py` if `reflection` becomes a new `CrystallizationKind` value (needs a producer/consumer in the same patch — this digest job is that producer).

**Non-goals:** no LLM in the digest job itself (pure aggregation over already-LLM-classified decisions). No auto-supersede — the digest reports revision, doesn't apply it (that already requires human review via the existing link-review pattern).

**Acceptance check:** after a real usage window, the digest produces at least one real belief-revision artifact when a real contradiction/refinement has fired — verified against real data, not synthetic.

---

## 2. Retirement surfacing (elaborates the deferred item from the reinforcement/decay spec)

**Reframing, thinner than originally scoped:** the retirement *action* already exists — `POST /api/memory/crystallizations/{id}/deprecate` (used this session for test cleanup) sets `status=deprecated` and de-projects from recall. What's missing is only *surfacing which crystallizations are candidates* — no new mutation path needed.

**Proposed (passive, no new job):** Hub's crystallization list endpoint/UI computes `decayed_activation(c, now=utcnow())` per row live (same function already used at the ranking read-site) and flags/sorts anything under `should_retire()`'s floor (0.05 default) as a candidate. A human reviewing the Memory tab sees stale items, clicks the existing deprecate endpoint. Zero new schema, zero new scheduled job.

**Deferred further (only if passive isn't enough):** a periodic proactive-notification job, mirroring `orion-memory-consolidation`'s existing retry-loop pattern, if retirement candidates turn out too numerous/important to wait for passive discovery. Hold off building this until the passive version shows real signal — same "prove it before you build more" discipline as the rest of this session.

**Files likely to touch:** `services/orion-hub/scripts/crystallization_routes.py` (list endpoint — add `decayed_activation` computation per row), `services/orion-hub/static/js/` (Memory tab — surface the flag), no `orion/memory/crystallization/` changes needed since `decayed_activation()`/`should_retire()` already exist.

**Non-goal, unchanged from the original spec:** never auto-apply retirement. A human always clicks deprecate.

---

## 3. `scripts/check_activation_saturation.py` operationalization

**Current state:** works, on-demand only (`POSTGRES_URI=... python scripts/check_activation_saturation.py`), has `--fail-above` for one-shot regression checks. Not wired into anything recurring.

**Existing seam to ride, not invent:** `Makefile` already has standalone `make check-inner-state-registry` / `make check-single-consumer-channels` targets — simple wrappers around `python scripts/check_*.py`, explicitly documented as the real, working piece of a promised-but-not-built `agent-check` chain (see `Makefile`'s own comment: `agent-check` itself doesn't exist yet, these standalone targets are what's real). Mirror that pattern exactly:

```makefile
check-activation-saturation:
	@python scripts/check_activation_saturation.py
```

**For the "after a real usage window" comparison the spec's acceptance check 1 actually needs** (not just a single-run gate): the script has no persisted baseline to diff against by design (documented in its own docstring — deliberately not inventing cross-run state). Two options, not a recommendation yet:

- **Manual cadence**: run it by hand periodically (weekly?), eyeball the trend. Zero new infrastructure. Matches this script's stated design intent exactly ("standing check you re-run and read").
- **Scheduled + persisted baseline**: a cron/schedule (this repo has a `schedule` skill for exactly this) that runs it on an interval and writes each run's `ceiling_fraction` (via `--json`) to a small append-only log or table, so trend-over-time is queryable instead of remembered by a human. More infrastructure, only worth it if manual cadence turns out to be forgotten in practice (matches "if Juniper has to repeat a rule twice, turn it into a script" — but this would be turning a *check* into a *scheduled* check, a step up, not needed until the manual version proves insufficient).

**Recommendation:** start with the `make check-activation-saturation` target (zero new infrastructure, rides an existing pattern exactly) and manual cadence. Revisit scheduling only if the manual habit doesn't stick.

**Files likely to touch:** `Makefile` only.

---

## Recommended sequencing

All three are independent of each other and of the still-undecided drive-conditioning discussion. Suggested order: `Makefile` target (trivial, ships immediately) → retirement surfacing (thin, no new job) → concept-relation loop (the most substantive of the three, needs the missing full-decision log line first per spec item 2, then the digest job on top).

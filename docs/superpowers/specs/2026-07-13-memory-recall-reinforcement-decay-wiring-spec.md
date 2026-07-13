# Wire recall reinforcement + decay (Phases 4+5 of the 2026-07-07 dynamics spec)

**Date:** 2026-07-13
**Status:** Proposed — supersedes/extends `2026-07-07-memory-weight-reinforcement-decay-design.md` Phases 4-5 with live-data findings
**Scope:** Recall-time only. Phase 1 (schema), Phase 2 (reinforce-on-dedup) are already live. This spec covers Phase 5 (recall reinforces) and the recall-time half of Phase 4 (decay), and explicitly kills/defers three other candidate ideas that live-data checks disqualified.

---

## Arsonist summary

The 2026-07-07 spec built the right skeleton — `CrystallizationDynamicsV1` (`activation`, `reinforcement_count`, `last_reinforced_at`, `last_recalled_at`, `decay_half_life_days`) and pure functions (`reinforce`, `recall_boost`, `decay`, `decayed_activation`, `should_retire`) in `orion/memory/crystallization/dynamics.py` — and wired half of it. `reinforce()` fires on formation-dedup (Phase 2, confirmed live: `intake_pipeline.py`, `concept_relation.py`, `governor.py`, `formation_executor.py` all call it). `recall_boost()` and `decay()` are written, tested in isolation, and **never called from anywhere in the live path**. Ranking (`active_packet.py:55`) reads the raw stored `.dynamics.activation` directly — not `decayed_activation()` — so decay has never been applied, not even implicitly at read time.

This was flagged as a real risk, not a hypothetical one, by querying the live table before writing this spec:

```
dynamics.activation (100 active crystallizations): 55 at exactly 1.00 (ceiling), 37 at ~0.25, rest scattered
dynamics.reinforcement_count: 100 at 0 — reinforce() has fired via Phase 2 dedup, but the count field itself shows every row untouched by any recall-time mechanism
```

55% already sitting at the activation ceiling with zero recall-time reinforcement ever applied is a live precursor to the exact saturation failure this codebase has already shipped once (homeostatic drives pinned flat at 0.731 because a leaky integrator's soft-saturate went stale — see `project_homeostatic_drives_real_tensions` history). Wiring `recall_boost()` without also wiring `decay()` in the same patch would only grow that ceiling-pinned fraction over time, flattening the one dynamics signal that currently has real variance. **They ship together or not at all.**

Three other candidate ideas from this session's brainstorm were checked against live data and are explicitly out of scope here, not deferred-as-good-ideas:

- **Confidence/dispute markup in rendered recall text** — killed. `confidence` is `"likely"` on all 100 active crystallizations (zero variance); `memory_crystallization_links` contains zero `contradicts`/`supersedes` relations (only `supports`, 16 of them). Rendering either field today would be decorative — a label with no measurable runtime behavior. Blocked until governance actually assigns varying confidence and contradiction-detection (exists in code, wired into `intake_pipeline.py`/`concept_relation.py`, has produced zero live contradictions) actually fires on real conflicting input.
- **Retrieval co-occurrence → implicit graph links** — killed for now, not just risky. `services/orion-recall/` (the real chat-time recall path) writes zero retrieval-event history anywhere (confirmed by grep, no matches). The only writer, `insert_retrieval_event` in Hub's `crystallization_routes.py`, is called from a route real chat doesn't use. There is no real usage data to mine co-occurrence from yet, independent of the popularity-bias / feedback-loop concerns raised separately.
- **Drive/goal-state-conditioned recall ranking** — deferred, not killed. This is a first-time coupling between the motivational/homeostatic subsystem and memory retrieval; per AGENTS.md's proposal-mode rule for changes touching memory/cognition loops, it needs its own design pass, not a bundle-in here.

---

## Current architecture

- **Schema** (`orion/memory/crystallization/schemas.py`): `CrystallizationDynamicsV1` — live, round-trips via `dynamics jsonb` column, `repository.py`.
- **Formation** (`dynamics.py::seed_dynamics`): live — `activation = salience` at creation.
- **Reinforce-on-recurrence** (`dynamics.py::reinforce`, boost=0.2, multiplicative-toward-ceiling): live — called from `intake_pipeline.py`, `concept_relation.py`, `governor.py`, `formation_executor.py` on dedup/approve.
- **Recall boost** (`dynamics.py::recall_boost`, boost=0.08 — deliberately smaller than `reinforce`'s 0.2, since "was retrieved once" is a weaker signal than "recurred independently"): written, tested, **zero call sites**.
- **Decay** (`dynamics.py::decay`, `decayed_activation` — pure half-life read, reuses `orion.core.activation_decay.decay_activation`): written, tested, **zero call sites**. `decayed_activation()` is a pure read — it does not require a scheduled job to take effect; it can be applied at the point ranking reads `activation`.
- **Retirement** (`dynamics.py::should_retire`, floor=0.05 default): written, tested, **zero call sites**. Distinct from decay — this is the heavier, consequential action (drop from active recall entirely), not just deprioritization.
- **Ranking read site** (`orion/memory/crystallization/active_packet.py:55`): reads `c.dynamics.activation` directly (raw stored value), not `decayed_activation(c, now=...)`.
- **Retrieval audit log** (`repository.py::insert_retrieval_event`, `crystallization_routes.py:517`): writes to `memory_crystallization_retrieval_events`, called from the Hub `/api/memory/active-packet` route. Nothing reads it back. Real chat-time recall (`services/orion-recall/`) does not call this route and does not write any retrieval log of its own.

---

## Non-goals

- Confidence/dispute rendering (blocked on real variance existing first — see Arsonist summary).
- Retrieval co-occurrence graph structure (blocked on real retrieval-event data existing first).
- Drive/goal-conditioned recall ranking (separate proposal-mode design pass).
- Automatic hard retirement / auto-archival without governor review — `should_retire()` returning `True` produces a review-queue candidate, never a direct status mutation, matching the existing propose/approve/reject pattern and the 2026-07-07 spec's own non-goal ("no hard deletion of canonical artifacts").
- No change to `reinforce()`'s existing Phase 2 formation-dedup wiring or its boost magnitude — that signal is separate from recall and already live.
- No LLM anywhere in this path (unchanged from the 2026-07-07 spec's Phase 1 non-goal).
- No new taxonomy, no new crystallization kind, no new bus channel required for the recall-boost/decay wiring itself.

---

## Proposed changes

### 1. Decay applied at ranking read time (thinnest possible patch — no new service)

Change `active_packet.py:55`'s ranking key from `c.dynamics.activation` to `decayed_activation(c, now=utcnow())`. `decayed_activation()` is already a pure function reusing the substrate's existing half-life math — this is a one-line read-site change, not new infrastructure. No cron job, no worker, no new failure mode. Decay becomes "live" everywhere ranking happens, immediately.

Retirement (`should_retire()`) is explicitly **not** part of this thin patch — surfacing retirement candidates for governor review is a separate, heavier follow-on (own acceptance check below, own recommended timing).

### 2. `recall_boost()` wired at the point a crystallization is actually returned in a real packet

Call site: wherever `retrieve_active_packet()`'s result is finalized with real (non-empty) `crystallization_refs` — either in `orion/memory/crystallization/retriever.py` after `build_active_packet()`, or immediately after `insert_retrieval_event` in `crystallization_routes.py` for the Hub-route path. Apply `recall_boost()` to every crystallization that actually appears in `packet.crystallization_refs` (not every candidate considered — only what actually made the cut into the render budget, since that's the actual "this got recalled" event).

**Hard invariant, must be enforced in code review, not just documented:** `recall_boost()`/`reinforce()` move `activation` (recall priority) only. Neither may touch `confidence`, `salience`, or any field a reader would interpret as "how true is this." Reinforcement changing what surfaces more often is fine; reinforcement changing how certain Orion sounds about it is not — that's the conformity-bias failure mode named in the brainstorm this spec follows from.

---

## Files likely to touch

- `orion/memory/crystallization/active_packet.py` — ranking read site (decay).
- `orion/memory/crystallization/retriever.py` — `recall_boost()` call site after packet finalization.
- `orion/memory/crystallization/repository.py` — persist the updated `dynamics` block back to Postgres after `recall_boost()` (currently `reinforce()`'s Phase 2 call sites already do this pattern — mirror it, don't invent a new persistence path).
- `services/orion-hub/scripts/crystallization_routes.py` — if the Hub-route path is the chosen call site for item 2, wire alongside the existing `insert_retrieval_event` call at line ~517.
- `tests/test_memory_crystallization_dynamics.py` (exists per 2026-07-07 spec, Phase 1) — extend with the acceptance checks below.
- No schema, bus, or `.env` changes — every field and function this patch wires already exists.

---

## Acceptance checks

1. **Saturation regression gate (the one that matters most).** Query `count(*) where dynamics.activation >= 0.99` before enabling recall_boost+decay, then again after a real usage window (not a synthetic smoke burst — actual chat/API traffic over at least several days). **If the ceiling-saturated fraction increases, that's a fail — revert, don't ship as-is.** This is one SQL query, runnable by anyone, no eval harness required. Make it a standing check, not a one-time manual query — a small script under `scripts/` that any future session can re-run.
2. **Head-to-head recall competition test (proves reinforcement actually changes behavior, not just a stored number).** Seed two crystallizations with equal initial `activation` and overlapping eligibility for the same render-budget bucket. Recall one via real repeated `retrieve_active_packet()` calls (distinct queries, not the same call replayed); leave the other untouched. Assert the reinforced one wins the contested bucket slot in a subsequent ambiguous query where both are plausible candidates. A number moving without a demonstrated behavior change is exactly the "reducer alive, cursor stale" failure mode — this check is what rules that out.
3. **Decay actually reduces rank over time for disused items.** Seed a crystallization, do not touch it, advance the clock (or use a short `decay_half_life_days` in the test) past its half-life, assert `decayed_activation()` at ranking time is measurably lower than the stored value and that it loses a contested bucket slot to a more-recently-recalled equal-salience competitor.
4. **Invariant check.** Property-based or explicit test: after any number of `recall_boost()`/`decay()` calls in any order, `confidence` and `salience` on the crystallization are provably unchanged — only `dynamics.*` fields move.
5. **No regression to Phase 2.** `reinforce()`'s existing formation-dedup call sites and boost magnitude (0.2) are untouched; existing tests for those still pass.

---

## Deferred, not in this patch

- **Retirement wiring** (`should_retire()` → governor review queue). Real action here needs its own acceptance check (a retirement candidate must be reviewable/reversible, never auto-applied) and its own call site (likely a periodic tick, since — unlike decay-at-read-time — surfacing retirement candidates needs to happen even for items nobody is currently querying). Recommend as the very next follow-on once items 1-2 above are live and the saturation gate has run clean for at least one real usage window.
- Confidence/dispute rendering, co-occurrence links, drive-conditioned ranking — per Non-goals above, blocked on prerequisites this patch does not create.

## Recommended next patch

Items 1 + 2 together, exactly as scoped — decay-at-read (one-line ranking change) and recall_boost wiring, shipped in the same PR since shipping reinforcement without decay reproduces a saturation bug this codebase has already hit once. Acceptance check 1 (saturation gate) is the hard ship/no-ship criterion, not a nice-to-have — it should run and pass clean before this is called done, not just before it's called merged.

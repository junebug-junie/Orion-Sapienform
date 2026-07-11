# Reverie semantic lift (v1)

**Date:** 2026-07-07  
**Status:** Approved for implementation planning (brainstorming session)  
**Authors:** Operator + agent  
**Related:** [`2026-07-05-reverie-dream-compaction-weave.md`](../plans/2026-07-05-reverie-dream-compaction-weave.md), [`2026-07-07-drive-attribution-substrate-act-design.md`](./2026-07-07-drive-attribution-substrate-act-design.md)

---

## Executive summary

Reverie today narrates **attention bookkeeping** (`harness_closure:uuid`, stability scores, open-loop ids) because the prompt receives coalition mechanics without **semantic referents**. The coalition signal is real; the cognition layer is wrong — like narrating blood pressure instead of what you're worried about.

This spec adds a **semantic lift** on the reverie path only:

1. Persist turn referents in a typed SQL table when harness surprise is unresolved.
2. Resolve coalition pointers → `ConcernCardV1` human text (deterministic).
3. Gate: resolve-or-skip (no wander fallback in v1).
4. Rewrite `reverie_narrate` prompt + post-LLM gates (referent overlap, banned mechanism vocabulary).
5. Route reverie LLM calls through **metacog GPU lane** on **cortex-exec-background** — evoked `stance_react` stays on chat lane.

**Evoked thought path is untouched.**

---

## Decisions (brainstorming)

| Question | Decision |
|----------|----------|
| Reverie purpose | **Hybrid C** — coalition picks thread; resolver supplies meaning |
| Infra-only coalition | **C → A** — resolve referents; skip tick if none |
| Acceptance bar | **All three:** referent (no infra vocab), voice (inner speech), grounding (traceable to concern card) |
| Referent storage | **Typed SQL table** — not jsonb blob |
| Compute lane | **metacog** route + **background** exec lane |
| Wander fallback | **No** in v1 |

---

## Problem statement

### Observed failure (production, 2026-07-07)

```text
harness post_turn_closure (surprise_unresolved)
  → substrate_turn_referent MISSING (no human text persisted)
  → prediction_error node harness_closure:{corr}
  → attention broadcast selects same loops every tick
  → reverie_narrate(mode=brain, chat lane) narrates coalition mechanics
  → 82 near-identical "two open loops / substrate pressure" thoughts
```

### Root cause

| Layer | Issue |
|-------|-------|
| Input | `build_reverie_context()` sends coalition_projection + open_loop scores, not turn content |
| Prompt | `reverie_narrate.j2` instructs narrating "coalition ids / strains / unresolved nodes" |
| Persistence | Turn text lives in harness finalize chain but is not joinable from broadcast pointers |
| Compute | Reverie competes on chat GPU lane with `mode=brain`; should use metacog inner-voice lane |

### What is NOT the bug

- Coalition IDs and scores are real (verified against `substrate_attention_broadcast_projection`).
- `evidence_refs` hollow guard works.
- Evoked `stance_react` path — different job, different anchor (`user_message`).

---

## Architecture

```text
rung 3 broadcast (pointers)
        │
        ▼
resolve_concern_cards()          ← orion/reverie/semantic_lift.py (deterministic)
        │
   ┌────┴────┐
   │ 0 cards │──► skip (reverie_skipped_no_semantic_referent)
   └────┬────┘
        │ ≥1 ConcernCardV1
        ▼
build_reverie_context(cards, coalition_refs)
        │
        ▼
cortex-exec-background           ← orion:cortex:exec:request:background
  reverie_narrate                ← ctx mode=metacog, llm_route=metacog
        │
        ▼
post-LLM gates (referent overlap, infra vocab, hollow)
        │
        ▼
SpontaneousThoughtV1             ← evidence_refs still coalition ids
```

**Unchanged:** `run_bus_worker`, `stance_react`, `ThoughtEventV1`, hub turn orchestration.

---

## Schema & contracts

### `substrate_turn_referent` (SQL migration)

```sql
-- services/orion-sql-db/manual_migration_substrate_turn_referent_v1.sql

create table if not exists substrate_turn_referent (
    correlation_id          text primary key,
    coalition_ref           text not null,
    user_message_excerpt    text not null default '',
    stance_imperative       text not null default '',
    surprise_unresolved     boolean not null default true,
    thought_event_id        text,
    created_at              timestamptz not null,
    stored_at               timestamptz not null default now()
);

create index if not exists idx_substrate_turn_referent_created_at
    on substrate_turn_referent (created_at desc);

create index if not exists idx_substrate_turn_referent_coalition_ref
    on substrate_turn_referent (coalition_ref);
```

- One row per `correlation_id` (upsert on re-closure).
- Written only when `surprise_unresolved=true`.
- Resolver freshness gate: default max age 24h (`ORION_REVERIE_REFERENT_MAX_AGE_HOURS`).

### `HarnessPostTurnClosureV1` extension

| Field | Type | When |
|-------|------|------|
| `user_message_excerpt` | str, max 300 | `surprise_unresolved` |
| `stance_imperative` | str, max 300 | `surprise_unresolved` |
| `thought_event_id` | str | optional audit |

Finalize chain already has `user_message` + `thought` at `emit_post_turn_closure` — cap and pass. Register schema bump in `orion/schemas/registry.py`.

### `ConcernCardV1` (runtime, `orion/schemas/reverie.py`)

| Field | Notes |
|-------|-------|
| `card_id` | stable hash of `coalition_ref` |
| `coalition_ref` | node id or open-loop id |
| `source_kind` | v1: `harness_turn` only (episode/self_state v1.1) |
| `human_text` | capped 500 chars, deterministic template |
| `thread_label` | capped 80 chars |
| `freshness_sec` | age of source |

**`harness_turn` template (no LLM):**

```text
You said: "{user_message_excerpt}"
I felt I should: {stance_imperative}
(Surprise still open from that turn.)
```

Invalid if both excerpts empty → card dropped.

---

## Resolver (`orion/reverie/semantic_lift.py`)

```python
def resolve_concern_cards(broadcast, *, referent_loader) -> list[ConcernCardV1]
def reverie_semantic_gate(cards) -> Literal["proceed", "skip"]
def enforce_semantic_quality(thought, cards) -> SpontaneousThoughtV1  # post-LLM gates
```

**Resolution order (v1):**

1. Selected open loop + `source_refs` → `harness_closure:{corr}` → SQL row.
2. Other open loops (dedupe by `correlation_id`).
3. Attended nodes not covered.
4. Cap **3 cards** per tick.

**Gate:** ≥1 card with `len(human_text) >= 40` → proceed; else skip. Fail-open on loader errors.

---

## Prompt contract (`reverie_narrate.j2`)

**Primary input:** `concern_cards` (thread_label + human_text).  
**Audit only:** `coalition_refs` for `evidence_refs` — not for narration.

**Job:** inner speech turning over a concern — not reporting attention state.

**Forbidden in interpretation:** coalition, open loop, harness_closure, substrate, stability score, dwell tick, attended node, projection_id.

`stance_react.j2` unchanged.

---

## Post-LLM gates (deterministic)

Applied after `parse_reverie_payload`, before existing `is_hollow()`:

| Gate | Rule | `hollow_reason` |
|------|------|-----------------|
| Referent overlap | ≥2 shared content tokens (len≥4) with card `human_text` OR ≥20% overlap | `unanchored_no_referent_overlap` |
| Infra vocabulary | regex on mechanism terms (structural list only) | `infra_vocabulary` |
| Existing hollow guard | length, evidence_refs ⊆ coalition | (unchanged) |

---

## Compute lane — metacog GPU (reverie only)

Reverie is unprompted inner voice — same **GPU route family** as collapse mirror / daily metacog, not chat turns.

### Routing (two orthogonal axes)

| Axis | Reverie | Evoked `stance_react` |
|------|---------|----------------------|
| **Exec channel** | `orion:cortex:exec:request:background` | `orion:cortex:exec:request` (or `:chat` when lane routing on) |
| **Worker** | `cortex-exec-background` | `cortex-exec-chat` |
| **ctx.mode** | `"metacog"` | `"brain"` / chat modes |
| **llm_route** | `"metacog"` | default chat/quick mapping |
| **Gateway target** | `route=metacog` → atlas-worker-2 (`ATLAS_METACOG_PROFILE_NAME`) | chat/quick lanes |
| **llm_lane options** | `{ llm_lane: "background", allow_chat_fallback: false }` | chat lane defaults |

### `build_reverie_plan_request` changes

```python
plan = build_plan_for_verb("reverie_narrate", mode="metacog")
PlanExecutionArgs(
    extra={
        "llm_profile": "metacog",
        "mode": "metacog",
        "llm_route": "metacog",
        "execution_lane": "background",
    },
)
context={
    "mode": "metacog",
    "llm_route": "metacog",
    "options": {"llm_lane": "background", "allow_chat_fallback": False},
    ...
}
```

### `orion-thought` settings

```text
# Evoked path (unchanged)
CHANNEL_CORTEX_EXEC_REQUEST=orion:cortex:exec:request

# Reverie path (new)
CHANNEL_REVERIE_CORTEX_EXEC_REQUEST=orion:cortex:exec:request:background
```

`CortexExecClient` for reverie uses `CHANNEL_REVERIE_CORTEX_EXEC_REQUEST`. Stance path keeps existing client/channel.

### `orion-cortex-exec` registration

Add `reverie_narrate` to `_BACKGROUND_VERBS` in `app/llm_lane.py` (alongside `log_orion_metacognition`, `dream_cycle`).

### Runtime evidence

Log line must show:

```text
llm_route_selected ... verb=reverie_narrate route=metacog
exec_llm_lane_decision ... llm_lane=background verb=reverie_narrate
```

Smoke: reverie tick does not appear on `cortex-exec-chat` request channel metrics during hub chat load.

---

## Feature flag

```text
ORION_REVERIE_SEMANTIC_LIFT_ENABLED=false   # default off until referent rows exist
ORION_REVERIE_REFERENT_MAX_AGE_HOURS=24
```

When false: existing Phase A behavior (unchanged rollback).

---

## Tests & evals

| Layer | File | Asserts |
|-------|------|---------|
| Unit | `orion/reverie/tests/test_semantic_lift.py` | resolver join, gate skip |
| Unit | same | infra vocab + referent overlap gates |
| Unit | `services/orion-cortex-exec/tests/test_llm_lane_propagation.py` | `reverie_narrate` → background llm_lane |
| Integration | `services/orion-thought/tests/test_reverie_semantic_lift.py` | SQL referent + broadcast → published thought |
| Integration | same | metacog context fields on plan request |
| Eval | `orion/reverie/evals/test_reverie_semantic_quality_eval.py` | labeled good/bad set; all three bars |
| Eval | extend `test_reverie_hollow_guard_eval.py` | meta narration cases fail |

**Good fixture:** referent from real turn → interpretation paraphrases user message, zero banned tokens.  
**Bad fixture:** today's "coalition centers on two open loops…" → `infra_vocabulary`.

---

## Implementation phases

| Step | Ships |
|------|-------|
| 1 | SQL migration `substrate_turn_referent` |
| 2 | `HarnessPostTurnClosureV1` + substrate-runtime writer |
| 3 | `orion/reverie/semantic_lift.py` + schema `ConcernCardV1` |
| 4 | Prompt rewrite + gates + `ORION_REVERIE_SEMANTIC_LIFT_ENABLED` |
| 5 | Metacog/background lane wiring + `_BACKGROUND_VERBS` |
| 6 | Tests + evals |

---

## Non-goals (v1)

- Wander fallback when resolve fails
- Changing `stance_react` or chat lane routing
- Reading spontaneous thought stream for prompt input
- Episode/motif cards in prompt (v1.1)
- Dream / compaction / proposal changes

---

## Ouroboros safety

| Reads | Never reads |
|-------|-------------|
| `substrate_turn_referent`, broadcast, evoked thought at write-time | `SpontaneousThoughtV1` stream for prompt |
| `SelfStateV1` (v1.1) | Own prior interpretations as primary input |

`evidence_refs` remain coalition ids for audit. Type asymmetry preserved.

---

## Acceptance checks

- [ ] `substrate_turn_referent` row written on unresolved harness closure
- [ ] Reverie with semantic lift on produces thought with no banned mechanism vocabulary
- [ ] Thought overlaps referent card text (deterministic gate passes)
- [ ] Infra-only coalition → skip tick, no publish
- [ ] `stance_react` still routes chat lane; unchanged tests pass
- [ ] Reverie exec logs show `route=metacog`, `llm_lane=background`
- [ ] Eval `test_reverie_semantic_quality_eval` passes labeled set

---

## Risks

| Severity | Risk | Mitigation |
|----------|------|------------|
| Med | Referent table empty until new turns | flag default-off; skip until rows exist |
| Med | background worker down | reverie degrades to skip; never blocks chat |
| Low | metacog GPU contends with collapse mirror | low priority + background lane; monitor atlas-worker-2 |

---

## Recommended next patch

Smallest vertical slice: migration + closure writer + resolver + gate + metacog routing behind `ORION_REVERIE_SEMANTIC_LIFT_ENABLED=false`. One integration test proving a referent row → non-meta thought on metacog lane.

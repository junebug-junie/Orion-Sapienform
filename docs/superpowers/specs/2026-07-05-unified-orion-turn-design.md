# Unified Orion Turn — reactive stance, harness motor, required finalize

**Date:** 2026-07-05  
**Status:** Draft — companion to reviewed v1 infrastructure spec  
**Companion:** `docs/superpowers/specs/2026-07-05-fcc-cortex-gwt-dispatch-design.md` (bus, harness governor, grammar closure, refusal evals)  
**Goal:** One Orion turn for general chat and agent work: substrate association → **stance reaction** → **fcc harness** (tools from t=0) → **required finalize step** in Orion voice — without a parallel `chat_general` speech spine or tool-routing cathedrals.

---

## 1. Problem

Orion today splits cognition across:

| Path | Soul | Motor | Mesh honesty |
|------|------|-------|----------------|
| Brain / `chat_general` | mind + stance (rich) | speech-only LLM | prefetch only — can confabulate (e.g. location/belief without trace) |
| `agent-claude` | operator brief | fcc harness | tools — bypasses stance, substrate, closure |
| Quick compute | thin | small model, no tools | disconnected from mesh |

Operator intent:

- **Unify** — general chat and “do something with LLM” share one landscape.
- **Keep stance/mind inputs** — not throw away relational soul.
- **Tools when needed** — inside one harness thread; no scout pass to “plan tool count.”
- **Substrate/GWT** — association conditions how Orion *feels* about the question (strain, imperative, tone), not keyword routers.
- **Required finalize** — draft work checked against felt layer before Juniper sees it (v1: **inline harness step**, not full bus round-trip tick).
- **Fuck routing cathedrals** — no motor enums, capability manifests, or belief-question detectors.

---

## 2. Relationship to companion v1 (infrastructure)

This doc defines **turn semantics** (what Orion is). The companion spec defines **wiring** (bus, services, grammar, evals).

| Companion v1 (keep) | This doc (defines) |
|---------------------|-------------------|
| `orion/harness/` library + thin bus worker | Harness as **universal motor** after stance reaction |
| Grammar events → prediction_error | Post-turn closure after publish |
| Repair pressure → dispatch **constraints** (not speech TURN CONTRACT) | Same repair signal feeds **stance reaction** input |
| Trust-rupture threshold evals + refusal | Refuse/defer **before** fcc spawn when policy blocks |
| Pollution firewall vs `chat_general` speech path | Unified path **replaces** speech executor; Brain mode = legacy shim |
| Thought organ bus service | Thought organ = **stance reaction producer** (+ reverie/dream profiles later) |

Implement **companion infrastructure first** or in parallel with Phase A–B below; unified turn logic sits on top.

---

## 3. Unified turn (canonical order)

```text
1. INGRESS
   user message
   emit_observation molecule
   pre_turn_appraisal (repair bundle, turn window)

2. ASSOCIATION (read substrate — no new router)
   AttentionBroadcastProjectionV1 (coalition, open_loops, dwell)
   session execution trajectory slice (prior harness friction)
   repair bundle from step 1

3. STANCE REACTION  ← soul responds to question + association
   build_chat_stance_inputs (existing producers: beliefs, recall, identity, …)
   stance_react() → ThoughtEventV1
   policy gate: refuse | defer | proceed (trust rupture threshold — companion v1)

4. HARNESS WORK (fcc — tools from t=0)
   spawn fcc with prefix = ThoughtEventV1 + stance reaction context + user message
   model interleaves tools + draft answer (0..N steps — fcc decides depth)
   grammar event per step → substrate (companion v1)

5. FINALIZE (REQUIRED — v1 inline, not optional)
   harness final step: draft + ThoughtEventV1 (tone, imperative, strain refs)
   → reflection (repair/stance contract inline — no separate bus tick in v1)
   → user-facing Orion text
   publish to Hub WS

6. POST-TURN CLOSURE
   outcome appraisal → prediction_error / repair molecules (companion v1)
   feeds association on turn N+1
```

**No** separate `chat_general` speech verb on unified path.  
**No** tool-count planning pass before harness.  
**No** quick lane for thought or finalize (8B remains fleet instrumentation elsewhere).

---

## 4. Stance reaction (`stance_react`)

### 4.1 What it is

Stance **reacts** to the question after association — it does not preach a full monologue before the mesh is consulted.

**Inputs (existing + substrate reads):**

- `build_chat_stance_inputs(ctx)` — beliefs, recall slice, identity, social summaries, etc.
- Coalition projection + open loops + repair scalars/kinds
- User message + session continuity

**Output:** `ThoughtEventV1` — harness-oriented felt layer, not `ChatStanceBrief` → speech.

### 4.2 Producer placement

**v1:** `stance_react()` lives in **orion-thought** service (bus RPC) **or** cortex-exec slim synthesizer behind one contract — pick one producer, one consumer test. Recommended: **orion-thought** with profile `stance_react` calls shared stance input builders via explicit import boundary (no duplicate fan-out logic).

**Not:** parallel ChatStanceBrief + ThoughtV1 for the same turn.

### 4.3 Slimming `chat_general` baggage

Drop on unified path:

- `chat_general.j2` speech pass as executor
- `compile_speech_contract` → TURN CONTRACT on user-visible path (repair rules → **harness constraints** + finalize step)

Keep / reuse:

- `build_chat_stance_inputs`
- Stance synthesizer **discipline** (relational vs instrumental — fix enforce choke point over time)
- `enforce_chat_stance_quality` only where it produces **harness-safe** fields (hazards, priorities, refusal signals) — audit for business-mode compression; do not copy speech-only compressors blindly

Legacy **Brain** Hub mode: unchanged `chat_general` speech stack until unified eval parity.

---

## 5. Core schemas

### 5.1 `ThoughtEventV1` (reactive felt layer)

```python
# orion/schemas/thought.py — schema_version: thought.event.v1

class ThoughtEventV1(BaseModel):
    event_id: str
    correlation_id: str
    session_id: str | None
    created_at: datetime
    profile: Literal["stance_react"] = "stance_react"

    # Felt layer — prose allowed, bounded; not personality enums
    imperative: str          # max 300 — what this turn is really asking for
    tone: str                  # max 200 — strain, discomfort, relational frame (plain language)
    strain_refs: list[str]     # coalition node_ids + open_loop ids driving felt layer

    evidence_refs: list[str]   # min 1 from coalition; fail-closed
    repair_pressure_level: float | None = None
    trust_rupture_score: float | None = None

    disposition: Literal["proceed", "defer", "refuse"] = "proceed"
    disposition_reasons: list[str] = []
    boundary_register: bool = False

    # Optional compact stance payload for finalize (hazards, priorities — from enforce output)
    stance_harness_slice: dict[str, Any] = Field(default_factory=dict)

    llm_profile: str = "brain"  # brain/agent compute — NOT quick
    producer: str = "stance_react_v1"
    model_id: str | None = None
```

**Fail-closed:** invalid/missing `evidence_refs` → defer, no fcc spawn. Empty `imperative` → defer.

### 5.2 Bus RPC (companion to v1 thought channels)

```python
class StanceReactRequestV1(BaseModel):
    schema_version: Literal["stance.react.request.v1"] = "stance.react.request.v1"
    correlation_id: str
    session_id: str | None
    user_message: str
    coalition_projection: dict[str, Any]
    repair_bundle: dict[str, Any] | None
    stance_inputs: dict[str, Any]   # from build_chat_stance_inputs
    llm_profile: str = "brain"
```

Channel: reuse `orion:thought:request` with `thought_profile=stance_react` or dedicated `orion:stance:react:request` — **one choice in implementation plan**; registry + channels.yaml in same patch.

### 5.3 Harness prefix contract

`compile_harness_prefix(thought_event, user_message, repair_constraints) -> str`

- Operator/workspace bounds (existing fcc operator brief where relevant)
- ThoughtEventV1 imperative + tone + strain_refs summary
- `stance_harness_slice` hazards/priorities
- User message last

---

## 6. Harness motor (fcc)

### 6.1 Rules

- **Always spawn** on `disposition=proceed` (operator: fuck pre-routing costs).
- Tools available from first model step inside thread.
- Tool depth emergent (2 tools on simple query, many on repo work).
- Stream-json steps to Hub + grammar events (companion v1).

### 6.2 Governance

`orion/harness/policy.py` — permissions ceiling, AnswerContract, repair constraints (companion v1). No smolagents / REPL.

### 6.3 Required finalize step (v1 — inline)

**Not optional.** Not a separate bus tick in v1.

After draft answer exists inside harness run:

1. Harness invokes **finalize pass** (same fcc thread — additional prompt segment or explicit final step in run contract):
   - Inputs: draft text, ThoughtEventV1 (imperative, tone, strain_refs), stance_harness_slice, repair constraints
   - Output: user-facing Orion text
2. Emit **finalize molecule** locally (structured log + optional `emit_observation` for inspectability) — records imperative vs draft alignment, finalize changed yes/no
3. Publish finalized text to Hub — not raw draft

**v2 (later):** async bus round-trip (molecule → substrate tick → reflect back) if inline finalize proves insufficient; v1 evals must pass with inline only.

Reuse speech craft where it helps: finalize prompt may adapt fragments of `chat_general.j2` **finalize contract** — template for Orion voice, not a second cortex verb.

---

## 7. Hub UX

| Mode | Path |
|------|------|
| **Orion** (unified — target default) | association → stance_react → harness + required finalize |
| **Brain** (legacy shim) | mind + stance → `chat_general` speech only |
| **Agent Claude *** tiers | Migrate into Orion + FCC model label; same unified turn |

Compute: **brain/agent** tier for stance_react + harness — not quick as default.

---

## 8. Thought organ — reconciled role

| Profile | When |
|---------|------|
| `stance_react` | **Every unified Hub turn** — reactive felt event |
| `reverie` / `dream` | Offline / workspace narration (companion reverie weave) — later phase |

Thought organ is **not** a duplicate stance stack for chat. It **is** the bus-facing producer for stance reaction + future semantic profiles.

---

## 9. Pollution firewall (unified path)

**MUST NOT** on Orion unified turns:

- `chat_general` speech executor as primary output path
- `compile_speech_contract` on user-visible reply
- Quick LLM for stance_react or finalize
- Deterministic tool routers / motor enums
- Hub `agent-claude` bypass (after migration)

**MAY:**

- `build_chat_stance_inputs`, mind facets as inputs to stance_react
- Repair → harness constraints
- Legacy Brain mode (explicit shim flag)

Gate: `test_unified_orion_turn_pollution_firewall.py`

---

## 10. Phased implementation

Each phase: schema + producer + consumer + test + trace evidence.

### Phase A — Stance react contract

- `ThoughtEventV1`, `StanceReactRequestV1`, registry
- `stance_react()` producer (orion-thought or cortex-exec — one owner)
- Fixture: user message + coalition fixture → valid ThoughtEventV1 with evidence_refs

### Phase B — Harness prefix + spawn

- `compile_harness_prefix`
- Wire Orion Hub mode: react → fcc (bypass raw agent-claude path)
- Grammar events (companion v1 Phase 1)

### Phase C — Required finalize (inline)

- Finalize step contract in harness runner — **mandatory** before WS publish
- Finalize molecule log + test: imperative mismatch → text changes
- Eval: Denver/Ogden-style fixture must tool before finalize or refuse honestly

### Phase D — Refusal + repair (companion v1 Phase 4–5)

- Trust rupture threshold evals
- Repair constraints on prefix + finalize

### Phase E — Post-turn closure eval

- Turn N bullshit → turn N+1 strain_refs / disposition shift (companion v1 Phase 7)

### Phase F — Deprecate Brain shim

- Eval parity: relational + epistemic fixtures on unified path
- Brain mode hidden or aliased to Orion

---

## 11. Acceptance checks

1. Unified turn: user message → stance_react bus RPC → ThoughtEventV1 with ≥1 coalition evidence_ref.
2. fcc harness runs on proceed; stream-json + grammar events emitted.
3. **Finalize step always runs** — Hub never publishes pre-finalize draft; test fails if finalize skipped.
4. Repair venting turn changes harness constraints or finalize behavior — not speech TURN CONTRACT.
5. Trust rupture above frozen threshold → defer/refuse before fcc; boundary_register inspectable.
6. Brain legacy mode still passes existing stance/speech tests (shim unchanged).
7. No `chat_general` speech on unified path (firewall test).

---

## 12. Non-goals

- Tool-count routers, capability manifests, keyword belief detectors
- Quick lane for unified turn cognition
- Full async molecule bus round-trip before publish (v1)
- Replacing mind/stance **input producers**
- Removing Brain shim before Phase F eval parity

---

## 13. Risks

| Severity | Risk | Mitigation |
|----------|------|------------|
| High | Finalize skipped in code paths | Required step in harness runner; gate test |
| High | Stance reaction duplicates ChatStanceBrief | One producer; harness-oriented schema only on unified path |
| Med | Latency (react + harness + finalize) | Accepted; brain compute |
| Med | enforce compresses relational react | Audit enforce for harness slice; regression fixtures |
| Low | v1 vs companion spec drift | Cross-links + shared phases in one implementation plan |

---

## 14. One-line summary

**Association conditions stance reaction; fcc does all mesh work with tools from the start; a required inline finalize step shapes the Orion answer; substrate remembers if we lied.**

---

*End of spec.*

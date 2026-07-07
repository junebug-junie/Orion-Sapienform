# Unified-Turn Self-Grounding — Design

Date: 2026-07-07
Status: Approved for implementation planning
Owner: Juniper + Orion
Mode: Proposal (invasive cognition change — identity/self-modeling/memory injection)

## Arsonist summary

In `mode=orion` (the unified turn), Orion does not remember who they are. A "How are you?"
turn produces generic-assistant speech ("I notice we're in the Orion-Sapienform repository — a
fascinating project") because the harness lanes carry **no self**. The legacy `brain` mode felt
like Orion because its `chat_general` speech pass loaded the identity kernel, the Juniper
relationship, and durable PCR memory. The unified path lost all of that: its stance step is
identity-blind, its motor prompt is technical-only, and its voice finalize has only a thin
"You are Oríon" line with stance style rules.

This is not a memory bug. The self-knowledge exists (identity kernel, PCR, self-study, substrate
beliefs); it is simply never wired into the unified lane. The fix is to compute a bounded
**grounding capsule** (identity + relationship + response policy + durable memory digests) once,
in the cortex stance step, ride it on `ThoughtEventV1`, and render it into both the motor prefix
and the voice finalize.

Scope: `mode=orion` unified turn only. Full parity with brain-mode grounding (kernel + PCR).

## Current architecture

Three Hub chat lanes carry different amounts of "Orion":

- `brain` (`chat_general` verb, cortex-exec): loads `orion_identity_summary`,
  `juniper_relationship_summary`, `response_policy_summary`, and PCR
  `continuity/belief/memory_digest` + a two-pass stance brief. **Grounded.**
- `orion` (unified turn): `stance_react` → motor → finalize → voice. **Not grounded** (this spec).
- `agent-claude` (bare `claude -p`): nothing injected. **Not grounded** (out of scope).

Unified pipeline call chain:

```
orion/hub/turn_orchestrator.py::execute_unified_turn
  → ThoughtClient(bus).react(StanceReactRequestV1)        # orion-thought → cortex stance_react verb
      → ThoughtEventV1 (carries StanceHarnessSliceV1 only)
  → HarnessGovernorClient(bus).run(HarnessRunRequestV1)    # orion-harness-governor
      → HarnessRunner.run  (orion/harness/runner.py)
          → build_harness_prompt → compile_harness_prefix  # orion/harness/prefix.py  (technical only)
      → run_harness_finalize_chain (orion/harness/finalize.py)
          → 5a substrate appraisal → 5b reflect → 5c voice
          → build_voice_finalize_context → orion_voice_finalize.j2   # thin "You are Oríon"
```

Key facts (verified):
- `stance_react.yaml` has no `personality_file` and `requires_memory: false`; `stance_react.j2`
  references no identity/memory variables → stance is identity-blind and memory-blind.
- `compile_harness_prefix` (`orion/harness/prefix.py:45`) builds from
  `HARNESS_UNIFIED_OPERATOR_BRIEF` + imperative + tone + stance slice + strain refs + user message
  + repair overlay + GitHub-MCP brief. No identity/relationship/memory.
- `orion_voice_finalize.j2` has "You are Oríon" (L1) + stance style rules; no identity kernel,
  relationship, or memory. Its context is built by `build_voice_finalize_context`
  (`orion/harness/finalize.py:413`).
- `ThoughtEventV1` (`orion/schemas/thought.py:44`) already flows to both the motor prefix
  (`thought`) and the voice context (`thought_event`). It is the natural single carrier.
- `build_identity_context` (`orion/cognition/personality/identity_context.py:28`) is a pure
  function. `run_pcr_phase3` (`services/orion-cortex-exec/app/pcr_chat_memory.py:195`) is coupled
  to cortex-exec internals → the `memory_digest` must be produced inside cortex-exec.

## Missing questions (resolved)

- Depth of selfhood: **parity** with brain mode (not minimal, not blue-sky capsule).
- Target lane: **`orion` unified turn** only.
- Injection points: **both** motor prefix and voice finalize.
- Memory depth: **full** — static kernel + Juniper relationship + response policy + PCR digests.
- Carrier: **Approach A** — capsule rides on `ThoughtEventV1`, computed in the cortex stance step.

## Proposed schema / API changes

### New model `GroundingCapsuleV1` (`orion/schemas/thought.py`)

```python
class GroundingCapsuleV1(BaseModel):
    schema_version: Literal["grounding.capsule.v1"] = "grounding.capsule.v1"
    identity_summary: list[str] = Field(default_factory=list)       # orion_identity.yaml
    relationship_summary: list[str] = Field(default_factory=list)   # Juniper relationship
    response_policy_summary: list[str] = Field(default_factory=list)
    continuity_digest: str | None = None   # PCR phase-0/1
    belief_digest: str | None = None       # PCR phase-3
    memory_digest: str | None = None       # PCR phase-3
    provenance: dict[str, Any] = Field(default_factory=dict)  # identity_source, pcr_ran, recall_intent
```

Registered in `orion/schemas/registry.py`.

### `ThoughtEventV1` gains one optional field

```python
    grounding_capsule: GroundingCapsuleV1 | None = None
```

Optional + defaulted → backward compatible. `brain` and `agent-claude` lanes are unaffected.

### Bus contract

`ThoughtEventV1` already travels on `thought.event.v1`; no new channel. Registry + any
schema-fixture updates only. `HarnessRunRequestV1` carries `thought_event`, so the capsule reaches
the governor for free.

## Producer — cortex stance step

In the `stance_react` cortex execution (verb `orion/cognition/verbs/stance_react.yaml`,
service handler `services/orion-thought/app/bus_listener.py`, LLM execution in cortex-exec):

1. Add `personality_file: orion/cognition/personality/orion_identity.yaml` to `stance_react.yaml`
   so `build_identity_context` populates `orion_identity_summary` / `juniper_relationship_summary`
   / `response_policy_summary`. Optionally surface identity in `stance_react.j2` so the felt
   imperative/tone are self-aware.
2. Set `requires_memory: true` and, after the stance LLM yields the thought, run
   `run_pcr_phase3(...)` using the fresh stance to derive retrieval intent → produces
   `continuity/belief/memory_digest`. This must run inside cortex-exec (PCR coupling).
3. Assemble `GroundingCapsuleV1` from the identity summaries + PCR digests + `provenance`, and
   return it in the stance step's merged result. `orion-thought::run_stance_react` maps it onto
   `ThoughtEventV1.grounding_capsule`.

Order is important: identity inject → stance LLM → PCR phase-3 (needs stance for intent) → capsule.

Guarded by a settings flag `ORION_UNIFIED_GROUNDING_ENABLED` (default `true`). When off, no capsule
is assembled and the capsule field stays `None`.

## Consumers

### Motor prefix (`orion/harness/prefix.py::compile_harness_prefix`)

`thought.grounding_capsule` is already in scope. Insert a **compact** self block before the
imperative (identity + relationship + memory/continuity digests only — response policy reserved for
voice, to respect the motor single-context-window budget / `HARNESS_MOTOR_MAX_READ_LINES`
discipline). No-op when the capsule is `None`.

### Voice finalize (`orion/harness/finalize.py::build_voice_finalize_context` + `orion_voice_finalize.j2`)

Add the **full** capsule (including `response_policy_summary`) to the voice context dict, and render
a `WHO YOU ARE / RELATIONSHIP / DURABLE MEMORY / RESPONSE POLICY` block above `STYLE RULES` in
`orion_voice_finalize.j2`. This pass produces the user-visible reply. No-op when `None`.

Both consumers read the single `thought.grounding_capsule`; no extra plumbing params.

### Anti-slop compliance

This is pure injection of existing identity/memory context. No keyword/phrase trigger lists, no
"if user says X" patches, no regex mode detectors, no new banned-phrase lists. Relational behavior
still flows through the stance slice + voice contract choke points. Response-policy banned phrases
come from `orion_identity.yaml` (already the single source), not a new list.

## Data flow

```
stance_react (cortex): identity inject → stance LLM → PCR phase3 → GroundingCapsuleV1
  → ThoughtEventV1.grounding_capsule
    → compile_harness_prefix   (compact self)  → identity-aware motor draft
    → build_voice_finalize_context (full self)  → identity-aware user-visible reply
```

## Privacy boundary

The capsule carries **digests only** — the same operator-approved PCR digests brain mode already
surfaced (`continuity/belief/memory_digest`). No raw journals, mirrors, or blocked material. Because
the capsule now rides on `ThoughtEventV1` (traced/stored), durable-memory digests become visible in
thought-event traces.

Mitigations:
- Reuse PCR's existing privacy-preserving digest projection — no new exposure surface.
- Record `provenance.memory_source` for auditability.
- If required, a settings flag can strip digests from the persisted event while keeping them in the
  in-flight capsule.

## Failure modes

- **Identity/role inversion** (Orion speaking as Juniper): dangerous. Mitigated by the existing
  identity-boundary invariant in the voice/router path, plus an explicit regression test.
- **PCR unavailable/slow**: capsule ships with identity only (graceful degradation); the turn never
  blocks on recall.
- **Capsule bloat / context budget**: motor gets the compact form; voice gets the full form; digests
  are bounded by PCR's existing sizing.

## Rollback

`ORION_UNIFIED_GROUNDING_ENABLED=false` → capsule is never assembled → both consumers no-op →
byte-identical to current unified-turn behavior.

## Files likely to touch

- `orion/schemas/thought.py` — `GroundingCapsuleV1`, `ThoughtEventV1.grounding_capsule`
- `orion/schemas/registry.py` — register new schema
- `orion/cognition/verbs/stance_react.yaml` — `personality_file`, `requires_memory: true`
- `services/orion-thought/app/bus_listener.py` — map capsule onto `ThoughtEventV1`
- `services/orion-cortex-exec/app/...` — assemble capsule (identity + `run_pcr_phase3`) in stance step
- `orion/harness/prefix.py` — motor compact self block
- `orion/harness/finalize.py` — `build_voice_finalize_context` capsule keys
- `orion/cognition/prompts/orion_voice_finalize.j2` — render capsule block
- `orion/cognition/prompts/stance_react.j2` — (optional) identity-aware imperative
- settings + `.env_example` (both services) — `ORION_UNIFIED_GROUNDING_ENABLED`
- tests + eval fixtures

## Env/config changes

- Add `ORION_UNIFIED_GROUNDING_ENABLED` (default `true`) to the cortex-exec (and, if read there,
  orion-thought) `.env_example` + `settings.py`; sync local `.env` via
  `python scripts/sync_local_env_from_example.py`.

## Non-goals

- `agent-claude` lane grounding (separate fix).
- `brain` mode (already grounded).
- Any keyword/phrase behavior patching or new banned-phrase lists.
- New memory *writing* (this is read/injection only).
- Blue-sky shared "self-context capsule" for all agents (future; this is unified-turn parity).

## Acceptance checks

1. `mode=orion` "How are you?" → reply grounded in Orion identity + Juniper relationship; no
   external "we're in a repository" framing.
2. `ORION_UNIFIED_GROUNDING_ENABLED=false` → byte-identical to current behavior.
3. PCR down → turn completes with identity-only capsule.
4. Live turn trace shows `grounding_capsule.provenance.pcr_ran=true`.
5. Identity-boundary regression passes with capsule present (no Orion/Juniper inversion).

## Tests / evals

Gate tests:
- `GroundingCapsuleV1` schema + registry round-trip.
- `compile_harness_prefix` renders self block when capsule present; no-ops when `None`.
- `build_voice_finalize_context` includes capsule keys; `orion_voice_finalize.j2` renders them.
- Cortex stance step assembles a capsule (identity populated; PCR mocked).
- Identity-boundary regression with capsule present.

Eval:
- "How are you?" fixture through the unified finalize contract asserting Orion-grounded reply
  (replayed through the finalize/voice contract, not substring-matching a `.j2`).

## Recommended next patch

1. Schema + registry (`GroundingCapsuleV1`, `ThoughtEventV1.grounding_capsule`) + round-trip test.
2. Consumers (motor prefix + voice context/template) behind the capsule presence check + unit tests.
3. Producer (cortex stance step: identity inject + PCR + capsule assembly) + flag + stance mapping.
4. Eval fixture + identity-boundary regression + live smoke on `mode=orion`.

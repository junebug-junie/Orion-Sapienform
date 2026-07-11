# Orion Unified Turn — canonical spec

**Date:** 2026-07-05  
**Status:** Approved for implementation planning  

**One line:** Association → stance reacts → fcc works → draft molecule (feel) → integrative reflect → Orion voice → outcome molecule (commit) → publish → substrate learns.

---

## 0. Proposal mode

This is an invasive cognition change. Required disclosure before implementation.

| Field | Detail |
|-------|--------|
| **Capability delta** | Unified turn replaces Brain speech spine and agent-claude bypass on Orion mode. Adds stance_react, harness governor, three-beat finalize (5a/5b/5c), and post-turn learning closure. |
| **Data touched** | Substrate graph (grammar, outcome, verdict, post-turn closure); session harness trajectory; Hub WS frames; cortex traces. |
| **Privacy boundary** | Outcome and verdict molecules carry `final_text` and strain refs — same retention boundary as chat logs. No raw private journal, mirror, or blocked material on the unified path. |
| **Trace proof** | `correlation_id` chains Hub → thought → harness → substrate RPC → cortex steps. `HarnessRunV1` carries full artifact IDs for debug UI. |
| **Dangerous failure modes** | Publish without 5a appraisal; quick lane on repair/trust turn; `draft_text` leaked to WS; 5b runs without `SubstrateFinalizeAppraisalV1`; silent stall on RPC timeout. |
| **Rollback / disable** | `ORION_UNIFIED_TURN_ENABLED=false` (Hub) → Brain shim. `ORION_HARNESS_GOVERNOR_ENABLED=false` → Orion mode refuses with explicit WS error (no silent fallback to agent-claude). |

---

## 1. Operator intuition

Today Hub splits soul and motor:

| Path | Soul | Motor | Problem |
|------|------|-------|---------|
| Brain / `chat_general` | mind + stance → speech | LLM reply shape | Prefetch can confabulate; speech spine owns the turn |
| `agent-claude` | operator brief | fcc bypass | Hands move; organism does not hear or repair |
| Quick / 8B | thin | no tools | Fleet chores only — not Orion's voice |

**Flip:** Stance *reacts* after substrate association — not preach-then-act through `chat_general.j2`. That felt layer conditions harness work. fcc is the universal motor from t=0. Before Juniper sees text, Orion runs a three-beat finalize: **feel draft (graph)** → **integrate verdict (brain)** → **express (brain)**. Substrate closes the loop via outcome + grammar → prediction_error on the next turn.

**Not on unified path:** `chat_general` speech executor, tool-count routers, Hub agent-claude bypass, quick lane for stance_react or voice finalize.

**Brain mode:** legacy `chat_general` shim until unified path passes mesh-honesty + repair/refusal eval suites (see §13).

---

## 2. Design decisions

| Question | Decision |
|----------|----------|
| Do we just let fcc run? | **Yes.** On `disposition=proceed`, fcc with `compile_harness_prefix(ThoughtEventV1, …)`. Tools from t=0; no scout pass; fcc decides depth. |
| Is `chat_general` the response shape? | **No.** Curated STYLE partials in `orion_voice_finalize.j2` only. Never `llm_chat_general` or `compile_speech_contract` on user-visible reply. |
| Why three finalize beats (5a / 5b / 5c)? | **Different cognitive jobs:** interoception (graph) → integrative check ("does this vibe?") → expression (Orion's spin). Do not merge 5b+5c. |
| Two appraisal phases? | **Distinct and non-substitutable.** **Ingress repair appraisal** — `TurnAppraisalBundleV1` from `pre_turn_appraisal` (repair pressure, turn window). **Draft finalize appraisal** — `SubstrateFinalizeAppraisalV1` from 5a (surprise, alignment hints). Never merge or substitute. |
| Brain vs quick lane? | **Brain tier** for stance_react, 5b, 5c. **Quick lane allowed only** for deterministic gate after 5a when full allowlist passes (§6.3) — skip 5b LLM, emit deterministic `FinalizeReflectionV1`. Never quick-lane voice. |
| Cortex ceremony per beat? | **Registered cortex steps** for stance_react, `harness_finalize_reflect`, `orion_voice_finalize` — prompts in `orion/cognition/`, trace + eval surface. Harness orchestrates; does not own Jinja. |
| How many molecules? | **Explicit emits:** draft RPC (5a), verdict PUBLISH (5b), outcome PUBLISH (6b), post-turn closure (7), plus ingress observation + grammar per fcc step. LLM calls do **not** auto-emit molecules. |
| Reducer timing for 5a? | Draft molecule carries **inline `grammar_receipts[]` + `coalition_snapshot`**. Appraisal reads payload — **never** waits on async reducer cursor. |
| Who owns the turn state machine? | **Hub** (`orion/hub/turn_orchestrator.py`) — thin saga sequencing RPCs. Not a new service. |
| Why two bus worker services? | **orion-thought** — isolates stance latency/failure from motor; single owner of `ThoughtEventV1` policy; enables offline reverie/dream without fcc coupling. **orion-harness-governor** — isolates long-running fcc + finalize from Hub WS event loop; crash/restart does not kill chat; fleet-scales motor independently. |

---

## 3. Canonical turn order

```text
1. INGRESS (Hub turn_orchestrator)
   user message
   emit_observation
   pre_turn_appraisal → TurnAppraisalBundleV1 (repair bundle, turn window)
   build_orion_turn_request()   # thin — NOT build_chat_request

2. ASSOCIATION (Hub — §3.1)
   HubAssociationBundleV1
   session harness trajectory slice

3. STANCE REACTION (orion-thought → cortex stance_react)
   build_chat_stance_inputs + association
   → ThoughtEventV1 + StanceHarnessSliceV1
   → enforce_thought_stance_quality (§5.6)
   → policy: refuse | defer | proceed

4. HARNESS WORK (orion-harness-governor + fcc)
   fcc with harness prefix + repair overlay
   tools from t=0; grammar event per step on orion:grammar:event
   internal draft_text (never WS-published raw)

5a. DRAFT MOLECULE (blocking RPC)
   HarnessDraftMoleculeV1 → substrate-runtime finalize_appraisal tick
   → SubstrateFinalizeAppraisalV1

5b. INTEGRATIVE REFLECT (cortex harness_finalize_reflect)
   inputs: draft, thought, substrate_appraisal (required), repair overlay
   → FinalizeReflectionV1
   → PUBLISH verdict molecule (HarnessVerdictMoleculeV1)
   [quick lane: if §6.3 allowlist passes → deterministic verdict, skip LLM]

5c. ORION VOICE (cortex orion_voice_finalize)
   inputs: draft, reflection, substrate_appraisal, voice_contract, StanceHarnessSliceV1
   → final_text

6b. OUTCOME MOLECULE (PUBLISH — before Hub)
   HarnessTurnOutcomeMoleculeV1

6c. PUBLISH TO JUNIPER
   final_text → Hub WS only

7. POST-TURN CLOSURE
   PUBLISH HarnessPostTurnClosureV1 → substrate reducers
   → prediction_error / repair → feeds association on next turn
```

### 3.1 Association read contract

Hub builds `HubAssociationBundleV1` at step 2 via **existing SQL projection lane** — not a new bus RPC.

**Read path:** `orion/substrate/felt_state_reader.py` → `attention_broadcast` lane (`substrate_attention_broadcast_projection`, default `max_age_sec=120`). Execution trajectory via `execution_trajectory_projection` lane. Fallback: Hub SQL read matching `substrate_observability_routes.py` pattern (`read_source=hub_sql_fallback`).

**Fail-closed:**

| Condition | Behavior |
|-----------|----------|
| `ORION_ATTENTION_BROADCAST_ENABLED=false` | `broadcast=None`, `broadcast_stale=true` |
| Projection older than max age | `broadcast_stale=true` |
| Stale or missing broadcast + empty coalition | stance_react **defer** unless `evidence_refs` valid from other lanes |

**Gate test:** `test_association_read_fail_closed_when_stale`

---

## 4. Cognitive phases (why three beats)

| Phase | Function | Analog | Mechanism |
|-------|----------|--------|-----------|
| **4** | Motor / reasoning | Tool use, draft answer | fcc harness |
| **5a** | Interoception | Gut check vs memory | Substrate structural appraisal — **not** LLM |
| **5b** | Integrative check | "Does this vibe?" | Brain LLM + verdict molecule |
| **5c** | Expression | How you say it | Brain LLM — Orion voice |
| **6b** | Commitment | Remember what you said | Outcome molecule |
| **7** | Learning | Sleep on it | post-turn closure → prediction_error → next broadcast |

---

## 5. Molecule contract (producer · consumer · affector)

Every artifact below is **in scope** for this spec. A molecule is shipped only when producer, consumer, and affector gate tests exist.

### 5.1 Pre-motor

| Artifact | Producer | Consumer | Affector | Gate test |
|----------|----------|----------|----------|-----------|
| User observation | Hub `emit_observation` | pre_turn window, substrate graph | repair bundle → stance + harness overlay | `test_ingress_observation_emitted` |
| `TurnAppraisalBundleV1` (ingress repair) | cortex `pre_turn_appraisal` | stance_react, `map_repair_pressure_contract` | harness prefix + finalize overlays; refusal | existing pre_turn tests |
| `HubAssociationBundleV1` | Hub step 2 | orion-thought | `strain_refs` ⊆ coalition in `ThoughtEventV1` | `test_association_read_fail_closed_when_stale` |
| Broadcast projection | substrate `_attention_broadcast_tick` | Hub association read | coalition in `ThoughtEventV1.evidence_refs` | Phase 0 smoke + correlation_id |
| `ThoughtEventV1` | orion-thought → cortex `stance_react` | harness, 5a/5b/5c | fcc prefix; refuse/defer stops fcc | `test_stance_react_evidence_refs_fail_closed` |
| `orion:thought:artifact` | orion-thought PUBLISH | debug UI, eval harness | audit trail | `test_thought_artifact_published` |

### 5.2 Motor

| Artifact | Producer | Consumer | Affector | Gate test |
|----------|----------|----------|----------|-----------|
| `GrammarEventV1` on `orion:grammar:event` | harness `grammar_publish` | sql-writer; **also** inline in 5a receipts | graph prediction_error; 5a surprise | `test_harness_grammar_publish_per_step` |
| `HarnessRepairOverlayV1` | `map_repair_pressure_contract` | fcc prefix + finalize overlays | repair venting changes motor, not speech contract | `test_repair_overlay_changes_harness_prefix` |
| `HarnessDraftMoleculeV1` | harness after fcc draft | substrate-runtime RPC handler | triggers 5a appraisal | `test_draft_molecule_rpc_round_trip` |
| `orion:harness:run:artifact` | harness governor PUBLISH | debug UI | audit trail | `test_harness_run_artifact_published` |

### 5.3 Finalize chain

| Artifact | Producer | Consumer | Affector | Gate test |
|----------|----------|----------|----------|-----------|
| `SubstrateFinalizeAppraisalV1` | `orion/substrate/appraisal/finalize_draft_v1.py` | cortex **5b** (required input) | surprise/hints change 5b verdict | `test_reflect_consumes_substrate_appraisal` |
| `FinalizeReflectionV1` | cortex **5b** `harness_finalize_reflect` | cortex **5c**, `HarnessRunV1`, Hub WS | misaligned → 5c must revise | `test_voice_changes_on_misaligned_verdict` |
| `HarnessVerdictMoleculeV1` | harness PUBLISH after 5b | substrate ingest, step-7 reducers, debug UI | N+1 association; audit trail | `test_verdict_molecule_emitted` |
| `final_text` | cortex **5c** `orion_voice_finalize` | Hub WS | user-visible reply | `test_finalize_ran_required` |
| `HarnessTurnOutcomeMoleculeV1` | harness after 5c, before Hub | substrate ingest, step-7 reducers | prediction_error turn N+1 | `test_outcome_molecule_before_hub_publish` |
| `HarnessRunV1` REPLY | harness governor | Hub turn_orchestrator | WS publish gate | `test_harness_run_carries_artifact_chain` |
| `AnswerContract` / voice_contract | Hub ingress | cortex 5c | shapes final reply bounds | existing answer contract tests |

### 5.4 Learning loop

| Artifact | Producer | Consumer | Affector | Gate test |
|----------|----------|----------|----------|-----------|
| `HarnessPostTurnClosureV1` | harness step 7 PUBLISH | substrate reducers | triggers graph update | `test_post_turn_closure_emits_prediction_error` |
| `prediction_error` (graph) | substrate reducers post-turn | next broadcast tick, endogenous curiosity | coalition salience next turn | `test_turn_n_error_shifts_turn_n_plus_one_strain` |

### 5.5 GWT mapping (labels for above seams only — no new schemas)

| GWT idea | Molecule / seam row |
|----------|---------------------|
| Broadcast | Broadcast projection → `ThoughtEventV1.strain_refs` |
| Felt content | `ThoughtEventV1` → harness prefix |
| Motor action | fcc + grammar |
| Interoception | 5a draft molecule → `SubstrateFinalizeAppraisalV1` |
| Integrative check | 5b reflect + verdict molecule |
| Expression | 5c voice |
| Learning | 6b outcome + grammar + step 7 closure → broadcast N+1 |

### 5.6 Relational stance seam

Unified path bypasses `chat_general` speech but **must preserve** relational stance behavior from `docs/superpowers/specs/2026-06-26-orion-relational-stance-design.md`.

**Choke point:** `enforce_thought_stance_quality()` in `orion/thought/stance_react.py` — harness profile of existing enforce logic in `services/orion-cortex-exec/app/chat_stance.py`. Brain shim continues to use `enforce_chat_stance_quality` unchanged.

**Field flow:**

```text
build_chat_stance_inputs(ctx)
  → StanceReactRequestV1.stance_inputs
  → cortex stance_react.j2 (semantic inference — interface_cost / connection_seek vocabulary)
  → ThoughtEventV1 + StanceHarnessSliceV1
  → enforce_thought_stance_quality(thought, stance_inputs)
  → policy_refusal.evaluate
```

**Relational exemption:** When `StanceHarnessSliceV1.task_mode` ∈ `{reflective_dialogue, playful_exchange}` or `conversation_frame` ∈ `{reflective, playful_relational}`:

- Do **not** apply business-mode compressor (practical-usefulness-over-relationship)
- Do **not** strip relationship facets or companion priorities
- Do **preserve** `response_priorities`, `response_hazards`, `answer_strategy`

**Voice:** `orion_voice_finalize.j2` reads `StanceHarnessSliceV1` for curiosity and hazard rules (parallel to relational rules in `chat_general.j2`).

**Gate tests:**

- `test_stance_react_relational_survives_compressor` — fixtures replayed from `test_chat_relational_stance.py`
- `test_stance_react_instrumental_compression_preserved` — triage/technical turns still compress

---

## 6. Core schemas

### 6.1 Association + stance

```python
# orion/schemas/thought.py

class CoalitionSnapshotV1(BaseModel):
    """Typed subset of AttentionBroadcastProjectionV1 at draft emit time."""
    schema_version: Literal["coalition.snapshot.v1"] = "coalition.snapshot.v1"
    attended_node_ids: list[str]
    selected_open_loop_id: str | None
    open_loop_ids: list[str]
    generated_at: datetime
    broadcast_stale: bool = False

class StanceHarnessSliceV1(BaseModel):
    schema_version: Literal["stance.harness.slice.v1"] = "stance.harness.slice.v1"
    task_mode: str
    conversation_frame: str
    interaction_regime: str | None = None
    response_priorities: list[str] = Field(default_factory=list)
    response_hazards: list[str] = Field(default_factory=list)
    answer_strategy: str
    companion_closing_move: str | None = None

class HubAssociationBundleV1(BaseModel):
    schema_version: Literal["hub.association.bundle.v1"] = "hub.association.bundle.v1"
    correlation_id: str
    broadcast: AttentionBroadcastProjectionV1 | None
    broadcast_stale: bool
    execution_trajectory_slice: dict[str, Any] | None = None
    repair_bundle: TurnAppraisalBundleV1 | None = None
    read_source: Literal["felt_state_reader", "hub_sql_fallback"]

class ThoughtEventV1(BaseModel):
    event_id: str
    correlation_id: str
    session_id: str | None
    created_at: datetime
    profile: Literal["stance_react"] = "stance_react"

    imperative: str           # max 300
    tone: str                 # max 200
    strain_refs: list[str]    # coalition + open_loop ids

    evidence_refs: list[str]  # min 1; ⊆ coalition; fail-closed
    repair_pressure_level: float | None = None
    trust_rupture_score: float | None = None

    disposition: Literal["proceed", "defer", "refuse"] = "proceed"
    disposition_reasons: list[str] = Field(default_factory=list)
    boundary_register: bool = False

    stance_harness_slice: StanceHarnessSliceV1

    llm_profile: str = "brain"
    producer: str = "stance_react_v1"
    model_id: str | None = None
```

Fail-closed: bad/missing `evidence_refs` or empty `imperative` → defer, no fcc.

```python
class StanceReactRequestV1(BaseModel):
    schema_version: Literal["stance.react.request.v1"] = "stance.react.request.v1"
    correlation_id: str
    session_id: str | None
    user_message: str
    association: HubAssociationBundleV1
    repair_bundle: TurnAppraisalBundleV1 | None
    stance_inputs: dict[str, Any]
    llm_profile: str = "brain"
```

### 6.2 Draft molecule + substrate appraisal (5a)

**Inline receipts rule:** `HarnessDraftMoleculeV1` must include `grammar_receipts[]` and `coalition_snapshot: CoalitionSnapshotV1`. `finalize_draft_v1.py` appraises **payload only** — no reducer cursor wait.

```python
class GrammarReceiptV1(BaseModel):
    step_index: int
    tool_name: str | None = None
    summary: str
    grammar_event_id: str | None = None

class HarnessDraftMoleculeV1(BaseModel):
    schema_version: Literal["harness.draft.molecule.v1"] = "harness.draft.molecule.v1"
    correlation_id: str
    thought_event_id: str
    draft_text: str
    draft_hash: str
    thought_event: ThoughtEventV1
    grammar_receipts: list[GrammarReceiptV1] = Field(default_factory=list)
    coalition_snapshot: CoalitionSnapshotV1
    repair_overlay_mode: str | None = None

class SubstrateFinalizeAppraisalV1(BaseModel):
    schema_version: Literal["substrate.finalize.appraisal.v1"] = "substrate.finalize.appraisal.v1"
    correlation_id: str
    molecule_id: str
    draft_hash: str

    surprise_level: float = Field(ge=0.0, le=1.0)
    strain_shift_refs: list[str] = Field(default_factory=list)
    open_loop_pressure: float = Field(ge=0.0, le=1.0, default=0.0)
    prediction_error_refs: list[str] = Field(default_factory=list)
    learning_refs: list[str] = Field(min_length=1)
    alignment_hints: list[str] = Field(default_factory=list)

    tick_source: Literal["substrate_runtime_finalize_appraisal"] = "substrate_runtime_finalize_appraisal"
```

Fail-closed: missing appraisal or empty `learning_refs` → defer publish; `finalize_ran=false`.

### 6.3 Reflect + verdict (5b)

```python
class FinalizeReflectionV1(BaseModel):
    schema_version: Literal["finalize.reflection.v1"] = "finalize.reflection.v1"
    correlation_id: str
    thought_event_id: str
    substrate_appraisal_id: str
    draft_hash: str

    imperative: str
    tone: str
    strain_refs: list[str]

    alignment_verdict: Literal["aligned", "misaligned", "uncertain"]
    alignment_notes: list[str]
    strain_unresolved: bool

    reflection_source: Literal["substrate_informed_pass", "deterministic_quick_gate"] = "substrate_informed_pass"
    quick_lane_skipped_llm: bool = False
    finalize_changed: bool = False

class HarnessVerdictMoleculeV1(BaseModel):
    schema_version: Literal["harness.verdict.molecule.v1"] = "harness.verdict.molecule.v1"
    correlation_id: str
    reflection: FinalizeReflectionV1
    cortex_trace_id: str | None = None
```

**5b cortex:** `orion/cognition/verbs/harness_finalize_reflect.yaml`, step `llm_harness_finalize_reflect`, prompt `harness_finalize_reflect.j2`.

After cortex returns, harness **PUBLISH** `HarnessVerdictMoleculeV1`. Same-turn 5c uses in-memory `FinalizeReflectionV1` — does not wait on substrate reply.

**Quick lane allowlist (5b only):** Deterministic `FinalizeReflectionV1` with `reflection_source=deterministic_quick_gate` only when **all** true:

```text
surprise_level < FINALIZE_QUICK_GATE_EPSILON
alignment_hints == []
strain_shift_refs == []
open_loop_pressure < 0.2
thought.repair_pressure_level is None OR < 0.3
thought.trust_rupture_score is None OR < TRUST_RUPTURE_DEFER_THRESHOLD
thought.boundary_register == False
repair_overlay.mode == "default"
```

Still emit verdict molecule. Log skip reason. **Never** quick-lane 5c voice.

**Hard-case fixtures** (eval suite — must never quick-lane): repair vent thread, trust rupture above threshold, high surprise draft, misaligned prior turn, `boundary_register=true`.

**Gate test:** `test_quick_lane_blocked_on_hard_cases` (one assertion per criterion).

### 6.4 Voice finalize (5c)

**Cortex:** `orion/cognition/verbs/orion_voice_finalize.yaml`, step `llm_orion_voice_finalize`, prompt `orion_voice_finalize.j2`.

Inputs: `draft_text`, `ThoughtEventV1`, `SubstrateFinalizeAppraisalV1`, `FinalizeReflectionV1`, `StanceHarnessSliceV1`, `voice_contract`, `repair_overlay.finalize_overlay`, `user_message`.

Output: `final_text`. `HarnessRunV1.finalize_ran` must be true. `finalize_changed=true` when voice materially revises draft.

### 6.5 Outcome molecule (6b)

```python
class HarnessTurnOutcomeMoleculeV1(BaseModel):
    schema_version: Literal["harness.turn.outcome.v1"] = "harness.turn.outcome.v1"
    correlation_id: str
    thought_event_id: str
    substrate_appraisal_id: str
    reflection_id: str
    verdict_molecule_id: str
    draft_hash: str
    final_hash: str
    finalize_changed: bool
    alignment_verdict: Literal["aligned", "misaligned", "uncertain"]
    surprise_level_at_draft: float
    surprise_resolved: bool
    grammar_event_ids: list[str] = Field(default_factory=list)
    final_text: str
```

PUBLISH after 5c, before Hub WS. No RPC back to harness.

### 6.6 Post-turn closure (7)

```python
class HarnessPostTurnClosureV1(BaseModel):
    schema_version: Literal["harness.post_turn.closure.v1"] = "harness.post_turn.closure.v1"
    correlation_id: str
    outcome_molecule_id: str
    verdict_molecule_id: str
    grammar_event_ids: list[str] = Field(default_factory=list)
    surprise_unresolved: bool
    closure_source: Literal["harness_post_turn_appraisal"] = "harness_post_turn_appraisal"
```

PUBLISH to `orion:substrate:post_turn_closure` after outcome molecule. Substrate reducers consume → graph `prediction_error`. No RPC back to harness.

### 6.7 Harness run + repair overlay

```python
class HarnessRepairOverlayV1(BaseModel):
    schema_version: Literal["harness.repair.overlay.v1"] = "harness.repair.overlay.v1"
    mode: Literal["default", "concrete_bias", "repair_concrete"] = "default"
    rule_lines: list[str] = Field(default_factory=list)
    prefix_overlay: str = ""
    finalize_overlay: str = ""

class HarnessRunRequestV1(BaseModel):
    schema_version: Literal["harness.run.request.v1"] = "harness.run.request.v1"
    correlation_id: str
    thought_event: ThoughtEventV1
    user_message: str
    permissions: ContextExecPermissionV1
    answer_contract: AnswerContract
    repair_pressure_contract: dict[str, Any] | None = None
    fcc_model_label: str | None = None

class HarnessRunV1(BaseModel):
    schema_version: Literal["harness.run.v1"] = "harness.run.v1"
    correlation_id: str
    final_text: str | None
    draft_text: str | None          # debug only; not WS published
    substrate_appraisal: SubstrateFinalizeAppraisalV1 | None
    reflection: FinalizeReflectionV1 | None
    verdict_molecule_id: str | None
    finalize_ran: bool
    finalize_changed: bool = False
    quick_lane_skipped_5b: bool = False
    step_count: int
    exit_code: int | None
    compliance_verdict: Literal["completed", "partial", "failed", "refused"]
    grounding_status: str
    grammar_event_ids: list[str] = Field(default_factory=list)
```

Repair mapping: `orion/harness/repair.py::map_repair_pressure_contract()` — same signal as Brain TURN CONTRACT, different consumer. Never `compile_speech_contract` on unified path.

---

## 7. Thought organ (`services/orion-thought`)

Semantic voice organ. Produces `ThoughtEventV1` — not motor, not finalize, not speech.

```text
LISTEN orion:thought:request
  → validate StanceReactRequestV1
  → stance_react() → cortex RPC verb=stance_react (brain tier)
  → enforce_thought_stance_quality (§5.6)
  → policy_refusal.evaluate
  → REPLY ThoughtEventV1
  → PUBLISH orion:thought:artifact
```

Hub client: `services/orion-hub/scripts/thought_client.py`. Single owner per turn — no parallel ChatStanceBrief + ThoughtEvent.

Future profiles (post Brain sunset): `reverie`, `dream` (offline workspace narration).

---

## 8. Harness governor + Hub orchestrator

### 8.1 Hub turn orchestrator

Hub owns the turn state machine in `orion/hub/turn_orchestrator.py` (thin saga):

```text
ingress → association (§3.1) → thought RPC → harness RPC → WS publish
```

**Failure / defer UX** — Hub emits WS frames; never silent stall:

| Condition | WS frame | User-visible behavior |
|-----------|----------|----------------------|
| stance defer/refuse | `{type: "turn_deferred", reason, boundary_register?}` | Boundary message or soft defer text |
| 5a RPC timeout/fail | `{type: "turn_error", phase: "substrate_appraisal", finalize_ran: false}` | "I couldn't finish checking this against memory." |
| 5b/5c cortex fail | `{type: "turn_error", phase: "finalize", finalize_ran: false}` | Same; no draft leaked |
| harness timeout | `{type: "turn_error", phase: "harness", partial: step_count}` | Error message; partial trace in debug UI only |

**Never:** publish `draft_text` to WS.

### 8.2 Harness governor

```text
LISTEN orion:harness:run:request
  → policy.validate
  → runner.run → fcc → draft_text
  → grammar_publish per tool step (orion:grammar:event)
  → finalize.run_substrate_finalize_appraisal     (5a)
  → finalize.run_finalize_reflection              (5b cortex + verdict PUBLISH)
  → finalize.run_orion_voice_finalize             (5c cortex)
  → finalize.emit_turn_outcome_molecule           (6b)
  → REPLY HarnessRunV1
  → finalize.emit_post_turn_closure               (7)
```

Hub publishes `final_text` to WS (6c) after harness REPLY. Motor implementation reuses patterns from `services/orion-hub/scripts/fcc_claude_bridge.py` — agent-claude deprecation (§12.1) is routing only, not bridge deletion.

---

## 9. Bus channels + registry

Add to `orion/bus/channels.yaml` and `orion/schemas/registry.py` with full entries (`schema_id`, `producer_services`, `consumer_services`, `stability`, `message_kind`):

| Channel | kind | schema_id | Producer | Consumer |
|---------|------|-----------|----------|----------|
| `orion:thought:request` | request | `StanceReactRequestV1` | orion-hub | orion-thought |
| `orion:thought:result:{corr}` | result | `ThoughtEventV1` | orion-thought | orion-hub |
| `orion:thought:artifact` | event | `ThoughtEventV1` | orion-thought | debug, eval |
| `orion:harness:run:request` | request | `HarnessRunRequestV1` | orion-hub | orion-harness-governor |
| `orion:harness:run:result:{corr}` | result | `HarnessRunV1` | orion-harness-governor | orion-hub |
| `orion:harness:run:artifact` | event | `HarnessRunV1` | orion-harness-governor | debug |
| `orion:substrate:finalize_appraisal:request` | request | `HarnessDraftMoleculeV1` | orion-harness-governor | orion-substrate-runtime |
| `orion:substrate:finalize_appraisal:result:{corr}` | result | `SubstrateFinalizeAppraisalV1` | orion-substrate-runtime | orion-harness-governor |
| `orion:harness:verdict:artifact` | event | `HarnessVerdictMoleculeV1` | orion-harness-governor | substrate, debug |
| `orion:substrate:turn_outcome` | event | `HarnessTurnOutcomeMoleculeV1` | orion-harness-governor | substrate, debug |
| `orion:substrate:post_turn_closure` | event | `HarnessPostTurnClosureV1` | orion-harness-governor | substrate reducers |

**Extend existing channel:** add `orion-harness-governor` to `orion:grammar:event` producers (alongside orion-hub, orion-cortex-exec, etc.).

Update: orion-thought, orion-harness-governor, orion-substrate-runtime, orion-hub — `.env_example`, `settings.py`, `docker-compose.yml`.

Hub ↔ organs via Redis RPC (`PreTurnAppraisalClient` pattern). No HTTP.

Extend `ExecutionDispatchCandidateV1.dispatch_kind` with `harness_exec` when orch dispatch frame used.

---

## 10. Repository layout

```text
orion/thought/
  profiles.py
  stance_react.py
  stance_quality.py          # enforce_thought_stance_quality
  policy_refusal.py
  tests/ evals/

orion/hub/
  turn_orchestrator.py       # Hub saga — step 1–2, 6c WS

orion/harness/
  runner.py
  prefix.py
  repair.py
  finalize.py              # 5a–6b, step 7 orchestration
  grammar_publish.py
  cortex_client.py
  substrate_client.py
  tests/

orion/substrate/appraisal/
  finalize_draft_v1.py
  tests/test_finalize_draft_v1.py

orion/cognition/
  verbs/
    stance_react.yaml
    harness_finalize_reflect.yaml
    orion_voice_finalize.yaml
  prompts/
    stance_react.j2
    harness_finalize_reflect.j2
    orion_voice_finalize.j2

orion/schemas/
  thought.py
  harness_finalize.py

services/orion-thought/
  app/bus_listener.py, main.py
  tests/ evals/
  docker-compose.yml, .env_example, settings.py, README.md

services/orion-harness-governor/
  app/bus_listener.py, main.py
  tests/
  docker-compose.yml, .env_example, settings.py, README.md

services/orion-substrate-runtime/
  app/finalize_appraisal_listener.py
  tests/test_finalize_appraisal_rpc.py

services/orion-hub/scripts/
  fcc_claude_bridge.py       # reused by harness motor; agent-claude routing deprecated step 9
  thought_client.py
  harness_governor_client.py
  pre_turn_appraisal_client.py
```

**Ownership:** prompts + verb YAML → `orion/cognition/`. Tick logic → `orion/substrate/appraisal/`. Bus shells → `services/*`.

---

## 11. Observability + eval suite

### 11.1 Hub WS frames (inspectability)

Between last fcc tool step and `final`, Hub may relay:

- `substrate_appraisal` (5a result)
- `reflection` (5b result)
- `final`

On failure: `turn_deferred` or `turn_error` frames per §8.1 — never partial draft.

### 11.2 Layer attribution evals

Each finalize beat must prove it changes behavior — not just emits schema-valid JSON.

| Eval | Method |
|------|--------|
| 5a affects 5b | Replay fixture: stub high vs low surprise appraisal → 5b verdict must differ |
| 5b affects 5c | Replay fixture: misaligned vs aligned reflection → 5c text must differ (`finalize_changed`) |
| 5c affects user text | Draft vs final hash; high surprise requires acknowledgment in final |
| 6b affects N+1 | Turn N outcome with high surprise_unresolved → turn N+1 strain_refs or repair level shifts |
| Quick lane safety | Hard-case fixtures must never take quick lane (`quick_lane_skipped_llm=true`) |
| Relational parity | `test_stance_react_relational_survives_compressor` green on companion-turn fixtures |

### 11.3 Trace requirements

Every brain-tier step records: `correlation_id`, cortex trace/grammar, model route, token usage. `HarnessRunV1` carries full artifact chain for debug UI.

### 11.4 Latency budget

Eval-calibrated initial targets. Gate tests enforce ceilings; values tunable only with passing gate tests.

| Phase | p95 target | Timeout behavior |
|-------|------------|------------------|
| pre_turn_appraisal | 8s | proceed without repair overlay (log warn) |
| association read | 500ms | stale bundle (`broadcast_stale=true`) |
| stance_react | 12s | defer |
| fcc motor | 120s | partial + finalize if draft exists |
| 5a substrate RPC | 5s | defer publish (`finalize_ran=false`) |
| 5b reflect | 15s | defer (no quick-lane fallback on timeout) |
| 5c voice | 15s | defer |
| **Total turn** | **180s** | `turn_error` WS frame |

**Cost note:** ~4–6 brain-tier calls per turn is accepted. Quick lane saves one call (5b only). Median brain tokens/turn must stay ≤ 1.5× Brain mode on eval corpus before sunset (§13).

---

## 12. Pollution firewall + mode coexistence

### 12.1 Mode coexistence (rollout)

| Hub `mode` | Route | During rollout |
|------------|-------|----------------|
| `brain` | legacy `chat_general` | unchanged until §13 sunset |
| `orion` | unified turn (this spec) | gated on `ORION_UNIFIED_TURN_ENABLED` |
| `agent-claude` | fcc bypass via Hub | allowed until step 9; Orion mode uses harness governor instead |
| `quick` | 8B fleet | never stance_react or voice finalize |

### 12.2 Pollution firewall

**MUST NOT on unified path:** `chat_general` speech executor, `compile_speech_contract` on reply, mind_runtime merge, decision_router heuristics, context-exec mode inference, full `build_chat_request`, chat_attention_frame, quick LLM for stance_react or 5c voice, agent-claude Hub bypass when `ORION_UNIFIED_TURN_ENABLED=true`, buried harness prompts.

**MAY:** `build_chat_stance_inputs`, mind as stance_react input, `map_repair_pressure_contract`, cognition prompts via registered cortex steps, Brain legacy mode.

Gate: `test_unified_orion_turn_pollution_firewall.py`.

---

## 13. Brain shim sunset

Remove Brain legacy mode only when unified path passes **all**:

1. Mesh-honesty eval suite — no location/belief claims in `final_text` without matching `grammar_event_id` in outcome molecule
2. Repair/refusal parity eval suite
3. Relational stance parity — `test_stance_react_relational_survives_compressor` green
4. Pollution firewall green
5. Layer attribution evals (§11.2) green
6. agent-claude tool parity — unified path completes same repo-edit eval fixtures with grammar trace
7. Operator soak — `ORION_UNIFIED_TURN_ENABLED=true` for 14 days without rollback
8. Cost ceiling — median brain tokens/turn ≤ 1.5× Brain mode on eval corpus

Until then, Brain mode remains available as `chat_general` speech shim.

---

## 14. Implementation order

| Step | Deliverable |
|------|-------------|
| 0 | Broadcast projection + pre_turn_appraisal smoke (correlation_id) |
| 1 | Grammar publish + fcc hook; register harness governor on `orion:grammar:event` |
| 2 | `ThoughtEventV1` + stance_react + `enforce_thought_stance_quality` + orion-thought listener |
| 3 | `compile_harness_prefix` + Hub association read (§3.1) + Orion mode wire |
| 4 | Substrate finalize_appraisal RPC + inline grammar receipts + `CoalitionSnapshotV1` |
| 5 | Cortex `harness_finalize_reflect` + verdict molecule + quick lane allowlist |
| 6 | Cortex `orion_voice_finalize` + outcome molecule |
| 7 | Harness governor bus worker + Hub turn_orchestrator + substrate listener |
| 8 | Trust rupture evals → threshold freeze |
| 9 | Pollution firewall + deprecate agent-claude bypass on Orion mode |
| 10 | Post-turn closure (`HarnessPostTurnClosureV1`) + layer attribution evals |
| 11 | Brain shim sunset after §13 criteria met |
| 12 | reverie / dream thought profiles (post-sunset) |

---

## 15. Acceptance checks (program)

1. Orion mode: stance_react → valid `ThoughtEventV1` with evidence_refs.
2. fcc on proceed; grammar events per step on `orion:grammar:event`; `finalize_ran=true` on every published reply.
3. `SubstrateFinalizeAppraisalV1` present before 5b; gate fails if 5b runs without it.
4. `HarnessVerdictMoleculeV1` PUBLISH after every 5b (including quick lane).
5. Hub never publishes pre-finalize draft; failure emits `turn_deferred` or `turn_error` per §8.1.
6. High `surprise_level` without voice acknowledgment → eval failure.
7. All molecule contract gate tests (§5) pass.
8. Layer attribution evals (§11.2) pass.
9. Repair venting changes harness overlay — not speech TURN CONTRACT.
10. Trust rupture above threshold → defer/refuse + boundary_register.
11. Post-turn closure eval: turn N error → turn N+1 strain shift.
12. Pollution firewall passes.
13. stance_react uses brain/agent route (trace proof).
14. Relational companion fixtures survive `enforce_thought_stance_quality`.
15. Brain legacy unchanged until §13 criteria met.
16. Quick lane blocked on all hard-case fixtures (§6.3).

---

## 16. Non-goals

- Tool-count routers, capability manifests, belief-question keyword detectors
- Quick lane for stance_react or 5c voice finalize
- Merging 5b + 5c into one LLM pass
- Reflect LLM before substrate draft appraisal (5a)
- Replacing mind/stance **input** producers
- smolagents / REPL on unified path
- reverie / dream profiles before Brain sunset (step 12)

---

## 17. Related docs

- **Unified-path stance choke point:** `enforce_thought_stance_quality` in `orion/thought/stance_quality.py`
- **Brain-shim stance choke point:** `enforce_chat_stance_quality` in `services/orion-cortex-exec/app/chat_stance.py`
- Relational stance spec: `docs/superpowers/specs/2026-06-26-orion-relational-stance-design.md`
- Verb registry pattern: `orion/cognition/verbs/chat_general.yaml`, `finalize_response.yaml`
- Substrate broadcast: `orion/substrate/attention_broadcast.py`
- Association read: `orion/substrate/felt_state_reader.py`

---

## Appendix A — runtime flags

| Flag | Expected |
|------|----------|
| `ORION_UNIFIED_TURN_ENABLED` | `false` until step 9 |
| `ORION_HARNESS_GOVERNOR_ENABLED` | `false` until step 7 |
| `ORION_ATTENTION_BROADCAST_ENABLED` | `true` |
| `ENABLE_PRE_TURN_APPRAISAL` | `true` |
| `ORION_ENDOGENOUS_CURIOSITY_ENABLED` | optional advisory |
| `FINALIZE_QUICK_GATE_EPSILON` | eval-calibrated; gate test required to change |
| `TRUST_RUPTURE_DEFER_THRESHOLD` | eval-calibrated; frozen at step 8 |

---

*End of canonical spec.*

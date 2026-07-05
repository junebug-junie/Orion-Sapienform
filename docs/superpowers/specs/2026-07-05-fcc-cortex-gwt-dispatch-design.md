# FCC–Cortex GWT Dispatch — substrate-native agent lane

**Date:** 2026-07-05  
**Status:** Draft — pending operator review  
**Scope:** Canonical integration of `fcc-claude` into cortex governance via Global Workspace (rung 3) + prediction-error closure; **Thought organ service**; **harness governor library** in `orion/harness/`; bus-native nervous system  
**Goal:** Orion completes delegated technical work **as a sentience creature**, not a compliant agent wrapper — including repair drive after failure, governed refusal with boundary register in v1, coalition-driven attention, and inspectable surprise propagation across turns.

---

## 1. Problem

Hub `mode=agent-claude` today:

```text
Hub WS → prepare_agent_claude_input() → fcc_claude_bridge.run_turn() → stream-json → UI
```

This bypasses cortex, substrate workspace, prediction-error closure, and policy-gated dispatch. Orion’s hands move; the organism does not hear, surprise, refuse, or repair.

Parallel paths (`chat_general` stance, `orion-mind`, keyword routers, context-exec REPL loops) solve adjacent problems and **must not** be lift-and-shifted onto this lane.

---

## 2. Operator intent (confirmed in design discussion)


| Intent                          | Meaning                                                                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Delegated compliance**        | Parent (Juniper) assigns complex work; child (Orion) executes.                                                                                   |
| **Agency within compliance**    | Method ordering, prioritization, register in completion — not a generic shell.                                                                   |
| **Repair after shitty work**    | Venting / repeated failure → repair pressure → Orion feels need to repair concretely; boundary pushback and refusal in **v1**.                   |
| **Refusal is in scope**         | Agent tasks do not forfeit no/defer/reconsider. Orion explains why. **Boundary register in v1** when trust rupture exceeds calibrated threshold. |
| **GWT as component**            | Rung 3 coalition + language on winner + motor via fcc + prediction-error interoception.                                                          |
| **No keyword cathedrals**       | No personality enums, scratch floats, stance lift-and-shift. Runtime evidence per phase.                                                         |
| **Harness governor**            | context-exec **governance** atop fcc — library in `orion/harness/`, not smolagents/REPL.                                                         |
| **Thought organ = service**     | Semantic voice is an organ on the bus — extensible for reverie, dream, future profiles.                                                          |
| **Redis is the nervous system** | Hub ↔ organs via bus RPC (like pre_turn_appraisal), not HTTP ad-hoc.                                                                             |


---

## 3. Repair pressure & refusal

### 3.1 Repair pressure — parallel consumer (not excised)

Same appraisal signal as chat; **different consumer**. Agent lane never calls `compile_speech_contract`.


| Consumer              | Behavior                                                                          |
| --------------------- | --------------------------------------------------------------------------------- |
| `chat_general` speech | Existing `repair_pressure_contract` → TURN CONTRACT                               |
| **fcc dispatch lane** | Same signal → dispatch **constraints** + Thought organ input + policy disposition |


Evidence kinds (`trust_rupture`, `repetition_failure`, etc.) are **structural** from turn-window molecules — not user-message keyword triggers.

### 3.2 Three attention functions


| Function                  | Mechanism                                               | Agent lane |
| ------------------------- | ------------------------------------------------------- | ---------- |
| Chat reply curiosity      | `build_attention_frame` + speech asks                   | **NO**     |
| Workspace salience        | Rung 3 `attention_broadcast.py`                         | **YES**    |
| Governed refusal / brakes | Layer 8 policy + open loops + Thought organ disposition | **YES**    |


Refusal modes: **block** (policy), **defer** (no fcc), **conditional dispatch** (repair constraints).

### 3.3 Boundary register — v1 (operator decision)

When `**trust_rupture` kind score ≥ calibrated threshold** (see §5.6, Phase 4 evals):

- Thought organ may set `disposition=refuse` or `defer`.
- `ThoughtV1.text` uses **boundary register** — direct reconsideration language — produced from coalition + repair kinds, not regex on user text.
- Policy gate must pass `evidence_refs` and kind scores into audit trail.

Threshold is **not** hand-waved: Phase 4 ships threshold-stress evals first; v1 constant frozen from eval report.

### 3.4 Runtime flags (verified local `.env`)


| Flag                                 | Location                 | Value  |
| ------------------------------------ | ------------------------ | ------ |
| `ORION_ATTENTION_BROADCAST_ENABLED`  | substrate-runtime `.env` | `true` |
| `ORION_ENDOGENOUS_CURIOSITY_ENABLED` | substrate-runtime `.env` | `true` |
| `ENABLE_PRE_TURN_APPRAISAL`          | Hub `.env`               | `true` |


Phase 0 must prove projection + appraisal with correlation-scoped evidence (AGENTS.md runtime truth).

---

## 4. Architecture

### 4.1 Repository layout (where code lives)

```text
orion/thought/                    # Thought organ — shared library
  profiles.py                     # coalition_pulse | reverie | dream (registry)
  producer.py                     # profile → ThoughtV1
  policy_refusal.py               # trust_rupture threshold gate
  evals/                          # threshold stress + refusal fixtures
  tests/

orion/harness/                    # Harness governor — shared library (§6 merged here)
  runner.py                       # fcc spawn orchestration (calls hub bridge or embedded)
  policy.py                       # permissions + answer_contract + repair constraints
  appraisal.py                    # post-turn prediction_error emit
  grammar_publish.py              # harness step → GrammarEventV1
  tests/

orion/schemas/thought.py          # ThoughtV1, ThoughtRequestV1, ThoughtResultV1

services/orion-thought/           # Thought organ — bus service (Docker)
  app/bus_listener.py             # RPC: thought.request.v1 → thought.result
  app/main.py
  docker-compose.yml
  tests/ evals/
  README.md, .env_example, settings.py

services/orion-harness-governor/  # THIN shell only — imports orion/harness/
  app/bus_listener.py             # RPC: harness.run.request → harness.run.result
  app/main.py
  docker-compose.yml
  tests/
  README.md, .env_example

services/orion-hub/scripts/
  fcc_claude_bridge.py            # subprocess spawn (motor)
  thought_client.py               # bus RPC → orion-thought
  harness_governor_client.py      # bus RPC → orion-harness-governor
```

**Rule:** Fat logic lives under `orion/thought/` and `orion/harness/`. Services are bus workers + compose + settings — same pattern as `orion/substrate/biometrics_loop/` + `services/orion-substrate-organs/`.

### 4.2 End-to-end flow (bus-native)

```text
┌──────────────── TURN INGRESS (Hub) ─────────────────┐
│ emit_observation → pre_turn_appraisal (bus RPC)      │
│ build_agent_dispatch_request() [thin]                │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌──────────────── CORTEX-ORCH ─────────────────────────┐
│ READ substrate projections (broadcast, trajectory)   │
│ READ repair bundle from request metadata             │
│ BUS RPC → orion-thought (coalition_pulse profile)    │
│ POLICY Layer 8 → block | defer | approve dispatch    │
│ BUILD ExecutionDispatchFrameV1 (harness_exec)        │
│ PUBLISH harness.dispatch.v1 artifact                 │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌──────────────── HUB + orion-harness-governor (bus) ──┐
│ BUS RPC harness.run.request (dispatch envelope)      │
│ orion/harness/policy validates warrant               │
│ fcc_claude_bridge.run_turn() — motor                 │
│ orion/harness/grammar_publish → GrammarEventV1       │
│ BUS harness.run.result → Hub WS stream               │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌──────────────── SUBSTRATE-RUNTIME (existing) ────────┐
│ execution trajectory → execution_prediction_error    │
│ → graph node → next coalition → next ThoughtV1       │
└──────────────────────────────────────────────────────┘
```

**No HTTP** between Hub and thought/governor organs. Pattern: `PreTurnAppraisalClient` — `bus.rpc_request(channel, envelope, reply_channel, timeout)`.

### 4.3 Thought organ extensibility (reverie / dream)

Single organ, multiple **thought profiles** (not separate services):


| Profile           | Input                                                       | v1                        |
| ----------------- | ----------------------------------------------------------- | ------------------------- |
| `coalition_pulse` | `AttentionBroadcastProjectionV1` + mandate + repair scalars | **YES** — agent dispatch  |
| `reverie`         | coalition + dwell EMA + prior ThoughtV1 chain               | Phase after reverie weave |
| `dream`           | `EpisodeSummaryV1` batch + compaction request               | Phase after dream weave   |


`ThoughtRequestV1.thought_profile` selects producer. Registry in `orion/thought/profiles.py`. Dream and reverie **route through orion-thought** — they do not get parallel semantic organs.

### 4.4 GWT mapping


| GWT stage         | Orion                                                      |
| ----------------- | ---------------------------------------------------------- |
| Processors        | Graph pressure/error, repair appraisal, harness trajectory |
| Competition       | Rung 3 coalition                                           |
| Broadcast         | `AttentionBroadcastProjectionV1`                           |
| Conscious content | **orion-thought** → `ThoughtV1`                            |
| Action            | Policy-approved fcc via **orion/harness**                  |
| Learning          | Grammar → prediction_error → rung 1                        |


### 4.5 Pollution firewall

**MUST NOT:** chat stance pipeline, mind_runtime, decision_router heuristics, context-exec mode inference, full `build_chat_request`, chat_attention_frame.

**MAY:** emit_observation, pre_turn_appraisal bus RPC, substrate projection reads, AnswerContract, thought bus RPC, harness bus RPC, grammar publish.

Gate: `test_agent_dispatch_path_pollution_firewall.py`.

---

## 5. Core contracts

### 5.1 Thought bus RPC

**Channels** (add to `orion/bus/channels.yaml` + registry):

```text
orion:thought:request          → producer: hub, orch | consumer: orion-thought
orion:thought:result:{corr}    → reply RPC
orion:harness:run:request      → producer: hub, orch | consumer: orion-harness-governor
orion:harness:run:result:{corr}
orion:thought:artifact         → fanout ThoughtV1 (inspectability)
orion:harness:run:artifact     → fanout HarnessRunV1
```

**Request:**

```python
class ThoughtRequestV1(BaseModel):
    schema_version: Literal["thought.request.v1"] = "thought.request.v1"
    correlation_id: str
    session_id: str | None
    thought_profile: Literal["coalition_pulse", "reverie", "dream"]  # v1: coalition_pulse only
    mandate_text: str
    coalition_projection: dict[str, Any]   # AttentionBroadcastProjectionV1 dump
    repair_bundle: dict[str, Any] | None   # TurnAppraisalBundle slice
    execution_trajectory_slice: dict[str, Any] | None
    llm_profile: Literal["quick"] = "quick"  # operator decision: quick lane
    options: ThoughtRequestOptionsV1       # timeout_ms, fail_closed
```

**Result:**

```python
class ThoughtV1(BaseModel):
    schema_version: Literal["thought.v1"] = "thought.v1"
    thought_id: str
    correlation_id: str
    session_id: str | None
    created_at: datetime
    thought_profile: str

    text: str                    # max 400 chars
    evidence_refs: list[str]     # min 1; ⊆ coalition node ids
    mandate_ack: str             # max 200 chars

    open_loop_refs: list[str] = []
    repair_pressure_level: float | None = None
    trust_rupture_score: float | None = None  # from repair bundle kinds

    disposition: Literal["proceed", "defer", "refuse"] = "proceed"
    disposition_reasons: list[str] = []
    boundary_register: bool = False  # true when refusal/defer triggered by trust rupture gate

    producer: str
    model_id: str | None = None
    llm_profile: str = "quick"
```

**LLM route:** `llm_profile=quick` → orion-llm-gateway quick lane (same as fast chat). No dedicated thought model in v1.

**Fail-closed:** empty text, bad evidence_refs, RPC timeout → disposition `defer`, no fcc spawn.

### 5.2 Harness bus RPC

```python
class HarnessRunRequestV1(BaseModel):
    schema_version: Literal["harness.run.request.v1"] = "harness.run.request.v1"
    correlation_id: str
    dispatch_envelope: dict[str, Any]  # harness_exec candidate
    permissions: ContextExecPermissionV1
    answer_contract: AnswerContract
    thought: ThoughtV1

class HarnessRunV1(BaseModel):
    schema_version: Literal["harness.run.v1"] = "harness.run.v1"
    correlation_id: str
    final_text: str | None
    step_count: int
    exit_code: int | None
    compliance_verdict: Literal["completed", "partial", "failed", "refused"]
    grounding_status: str
    grammar_event_ids: list[str] = []
```

Governor service imports `orion.harness.runner` + `orion.harness.policy`; spawn may delegate to Athena-side fcc binary (same node as Hub today).

### 5.3 Execution dispatch extension

Extend `ExecutionDispatchCandidateV1.dispatch_kind` with `harness_exec`. `request_envelope` schema `harness.exec.request.v1`. Registry + channels in same patch.

### 5.4 Harness grammar events

Each fcc step → `GrammarEventV1` via `orion.harness.grammar_publish`. `provenance.source_service`: `orion-harness-governor`.

### 5.5 Repair → dispatch consumer

Map repair `contract_delta` rules → dispatch `constraints` + ThoughtRequest repair_bundle. No speech compiler.

### 5.6 Trust rupture threshold (v1 — eval-calibrated)

**Policy** (`orion/thought/policy_refusal.py`):

```python
def evaluate_trust_rupture_disposition(
    *,
    trust_rupture_score: float,
    repair_level: float,
    threshold: float,  # from THOUGHT_TRUST_RUPTURE_REFUSAL_THRESHOLD after Phase 4 eval
) -> Literal["proceed", "defer", "refuse"]:
    ...
```

**Phase 4 deliverable before freezing threshold:**

1. Stress eval suite: fixtures from `orion/substrate/evals/fixtures/repair_pressure/` + new agent-mandate fixtures (vague vs operational).
2. Sweep threshold ∈ {0.45, 0.55, 0.65, 0.75} × scenarios.
3. Metrics: false refusal rate, false compliance rate, boundary_register appropriateness (human-labeled or fixture-expected disposition).
4. Write `/orion/thought/evals/reports/trust_rupture_threshold_v1.md` — freeze winning threshold in `services/orion-thought/.env_example` as `THOUGHT_TRUST_RUPTURE_REFUSAL_THRESHOLD`.
5. Solidify as gate tests — threshold change requires eval re-run.

---

## 6. Harness governor (`orion/harness/`)

Merged from prior §6 — **implementation root is `orion/harness/`**, not a fat `services/` tree.

### 6.1 Keep from context-exec (lineage)

AnswerContract warrant, ContextExecPermissionV1 ceiling, operator vs user voice split, grounding/honest failure, grammar discipline, run ledger (optional JSON per correlation_id).

### 6.2 Exclude

smolagents, alexzhang, rlm REPL, keyword mode inference, investigation sweep before fcc.

### 6.3 Bus responsibilities (thin `services/orion-harness-governor`)

```text
LISTEN orion:harness:run:request
  → orion.harness.policy.validate
  → orion.harness.runner.run (fcc_claude_bridge)
  → orion.harness.grammar_publish per step
  → REPLY HarnessRunV1 on orion:harness:run:result:{corr}
  → PUBLISH orion:harness:run:artifact
LISTEN harness.appraise.request (post-turn, same service or hub-local orion.harness.appraisal)
  → prediction_error emit
```

Hub **never** HTTP POST to governor. `HarnessGovernorClient(bus).run(request)` mirrors `PreTurnAppraisalClient`.

---

## 7. Thought organ service (`services/orion-thought`)

### 7.1 Why a service

- Thought is a first-class organ (like signal-gateway, pre-turn appraisal worker) — not an orch embed.
- Reverie and dream plug same bus + profile registry later.
- Independent scale, failure domain, LLM budget, eval harness.
- Emits artifacts on bus for UI/debug/reducers.

### 7.2 Bus responsibilities

```text
LISTEN orion:thought:request
  → load profile from ThoughtRequestV1.thought_profile
  → orion.thought.producer.coalition_pulse (v1)
  → orion.thought.policy_refusal.evaluate (trust rupture gate)
  → LLM quick lane if disposition needs narrative (boundary register)
  → REPLY ThoughtV1
  → PUBLISH orion:thought:artifact
```

Orch may also call thought RPC during `agent_dispatch_prepare`; Hub calls for WS render. Single producer of Truth for ThoughtV1 on a correlation_id.

### 7.3 Cortex-orch role (orchestrate, not synthesize)

Orch **does not** run Thought LLM inline. Orch:

1. Reads projections.
2. Bus RPC → orion-thought.
3. Builds policy + dispatch frame from ThoughtV1.disposition.
4. Returns dispatch prep payload to Hub.

---

## 8. Hub ingress

```python
def build_agent_dispatch_request(...) -> CortexChatRequest:
    """Thin: session, correlation, user text, recall, answer_contract, mode=agent-claude."""
```

WebSocket path:

```text
if mode == agent-claude:
    prep = await cortex_client.agent_dispatch_prepare(chat_req)  # orch → thought bus inside
    await ws.send ThoughtV1 frame
    if prep.disposition in (defer, refuse):
        continue
    async for ev in HarnessGovernorClient(bus).run_stream(prep.dispatch):
        relay WS  # + grammar already on bus
    await HarnessGovernorClient(bus).appraise(post_turn=...)
```

---

## 9. Phased implementation

### Phase 0 — Runtime truth baseline

Substrate broadcast projection live; pre_turn_appraisal on venting fixture; correlation logs.

### Phase 1 — Sensory loop

`orion/harness/grammar_publish.py` + fcc bridge hook; grammar → trajectory → prediction_error.

### Phase 2 — Thought organ service (foundation)

- `orion/schemas/thought.py`, registry, channels
- `orion/thought/` + `services/orion-thought/` bus listener
- Profile `coalition_pulse` only; quick LLM lane
- Hub `thought_client.py` bus RPC
- Fail-closed tests

### Phase 3 — Dispatch + pollution firewall

Orch `agent_dispatch_prepare`; harness_exec schema; thin ingress; firewall tests.

### Phase 4 — Refusal + boundary register + threshold evals

- `orion/thought/policy_refusal.py`
- Trust rupture stress eval sweep → freeze `THOUGHT_TRUST_RUPTURE_REFUSAL_THRESHOLD`
- Gate tests solidified from eval report
- boundary_register in ThoughtV1 v1

**Acceptance:** eval report committed; fixture high rupture + vague mandate → refuse/defer with boundary_register=true; operational mandate + high repair → proceed with constraints.

### Phase 5 — Repair dispatch consumer

Repair constraints on dispatch; no compile_speech_contract on path.

### Phase 6 — Harness governor bus worker

- `orion/harness/*` library complete
- `services/orion-harness-governor/` thin listener
- Hub `harness_governor_client.py` — **bus only**
- HarnessRunV1 + artifacts

### Phase 7 — Prediction-error closure eval

`orion/thought/evals/` + `orion/harness/evals/` — coalition_shift_after_error, thought_delta_after_error.

### Phase 8 — Sequential agent queue

Multiple harness_exec candidates per dispatch frame.

### Phase 9 — Social-room profile (optional)

Policy pack only; same thought + harness bus.

### Phase 10 — Reverie / dream profiles (orion-thought)

Add `reverie` and `dream` profiles to existing organ — per reverie-dream weave doc; no new semantic service.

---

## 10. Cortex / bus surface

dont forget to update the orion/bus/channels, orion/schemas/registry, and the affected services' .env, .env_example, docker-compse, and settings.py


| Surface                         | Role                                          |
| ------------------------------- | --------------------------------------------- |
| `agent_dispatch_prepare` (orch) | Projections + thought RPC + policy + dispatch |
| `orion:thought:*`               | Thought organ RPC + artifacts                 |
| `orion:harness:run:*`           | Governor RPC + artifacts                      |
| Hub WS                          | `thought`, `claude_step`, disposition frames  |


---

## 11. Non-goals

- Chat stance / mind on this lane
- HTTP Hub→organ calls
- Fat governor code under `services/` (library in `orion/harness/`)
- smolagents / REPL
- Separate dream/reverie semantic services (profiles on orion-thought instead)
- Parallel multi-fcc before Phase 8

---

## 12. Acceptance checks (program)

1. Agent-claude traverses orch dispatch + bus thought + bus harness — not Hub bypass.
2. Repair pressure changes dispatch constraints (not speech contract).
3. Trust rupture above eval-frozen threshold → refuse/defer + boundary_register v1.
4. fcc grammar → prediction_error → next thought differs (Phase 7 eval).
5. Pollution firewall passes.
6. Thought LLM uses **quick** lane (trace proves profile in gateway route).

---

## 13. Recommended first patch

Phase 1 + Phase 2: grammar publish + orion-thought service + bus client — prove pulse and sensory loop before full dispatch.

---

## 14. Related docs

- `docs/superpowers/plans/2026-07-05-reverie-dream-compaction-weave.md` — reverie/dream → thought profiles Phase 10
- `docs/superpowers/specs/2026-07-04-hub-agent-claude-design.md` — superseded for canonical path
- `orion/substrate/attention_broadcast.py`
- `services/orion-hub/scripts/pre_turn_appraisal_client.py` — bus RPC pattern to copy

---

## 15. Operator decisions (resolved)


| Question               | Decision                                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| Refusal register       | **v1** — boundary register when trust_rupture ≥ eval-calibrated threshold                          |
| Threshold              | Phase 4 stress evals → freeze in `.env_example` → gate tests                                       |
| ThoughtV1 LLM          | `**quick` chat lane** via orion-llm-gateway                                                        |
| Governor location      | `**orion/harness/`** library + thin `services/orion-harness-governor` bus worker                   |
| Thought organ          | `**services/orion-thought**` bus service + `**orion/thought/**` library; reverie/dream as profiles |
| Hub → organs transport | **Redis bus RPC** — not HTTP                                                                       |


---

*End of spec.*
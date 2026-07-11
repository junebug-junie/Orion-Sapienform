# Repair Pressure v2 + Pre-Turn Appraisal Rail

**Status:** Draft (2026-07-03)  
**Operator contract:** Follow `AGENTS.md` for implementation (thin seams, tests/evals, bus/schema updates, env parity, no keyword cathedrals).  
**Builds on:** `docs/superpowers/specs/2026-07-03-repair-pressure-speech-wiring-design.md` (speech wiring — shipped)  
**Supersedes detector layer of:** `docs/plans/substrate/2026-05-23-repair-pressure-v1.md` (phrase_match_v1 only; signal/contract shapes largely retained)

---

## Problem

Repair pressure v1 proved the causal chain (observation → evidence → appraisal → signal → contract → speech) but the **detector is fake**:

- `phrase_match_v1` in `orion/substrate/appraisal/evidence.py` — substring tables, not semantic read.
- Hub `substrate_effect_pipeline` ingests **user text only**; assistant turns never enter the window.
- Seven `EvidenceKind` enums are registered and summed into `level`, but kinds do **not** route behavior — only aggregate `level` + `confidence` gate `apply_repair_pressure_contract`.
- Hand-tuned formula weights; substrate gradient terms are always zero on Hub observations.
- E2E proof uses stacked fixture phrases, not real ops-frustration threads.

Speech wiring (v1) is correct: when mode flips, `repair_pressure_contract` metadata reaches `compile_speech_contract`. Live failure is upstream sensemaking, not speech merge.

---

## Goal

Make repair pressure **real** on the same turn:

1. **Paired turn window** (user + assistant from `messages[]`).
2. **Logprob-calibrated evidence** — no self-reported JSON floats, no phrase tables.
3. **Seven kinds behaviorally real** — semantic probes, kind-assembled contract rules, eval-gated orthogonality (grounding chastisement must not trip repair).
4. **Extensible rail** — `pre_turn_appraisal` generic bus seam; `repair_pressure` is the first paradigm plugin.

Hub stays a **dumb orchestrator**. Cognition lives in cortex. **Bus is the electrical tie.**

---

## Non-goals

- Lowering `_LEVEL_MID` (0.45) or `_LEVEL_HIGH` (0.75) thresholds to force fires on harsh grounding turns.
- Grounding/honesty/capability hallucination inside repair pressure (separate future paradigm).
- New schema-kernel atoms, gradients, or molecule kinds.
- Hub calling LLM gateway directly.
- Async one-turn-lag appraisal worker (v2 is same-turn).
- Collapsing the seven kinds into a single scalar without per-kind contract contribution.
- Keyword patches to `chat_general.j2` or stance prompts as substitute for appraiser fixes.

---

## AGENTS.md alignment

| Mandate | How this spec complies |
|---------|-------------------------|
| No keyword cathedrals | Kill `phrase_match_v1`; paradigms use logprob probes, not phrase tables. Registry is explicit plugin list, not a growing YAML taxonomy. |
| Event substrate first | `PreTurnAppraisalRequestV1` / `TurnAppraisalBundleV1` bus contract; paradigm slices are typed payloads with tests. |
| Runtime truth beats config truth | Acceptance = transcript eval fixtures + live smoke; not substring in `.j2`. |
| Thin seams | Hub: window + RPC + attach. Cortex: probe runner + paradigm plugins. Speech wiring unchanged. |
| Service boundaries | Library in `orion/substrate/appraisal/`; RPC handler in `orion-cortex-exec`; Hub orchestrates only. |
| Deterministic gates | Fail-closed on probe timeout / missing logprobs; unit tests on reducer + contract assembly. |

Implementation MUST follow `AGENTS.md` implementation mode: branch, tests, evals, env sync, review, PR report.

---

## Architecture

### Electrical tie (bus)

```text
Hub handle_chat_request / websocket chat
  │
  ├─1─► bus RPC  orion:cortex:pre_turn_appraisal:request
  │         producer: orion-hub
  │         consumer: orion-cortex-exec
  │         payload: PreTurnAppraisalRequestV1
  │         reply:   orion:cortex:pre_turn_appraisal:result:{correlation_id}
  │         returns: TurnAppraisalBundleV1
  │
  ├─2─► attach bundle.metadata_attachments → CortexChatRequest.metadata
  │         (includes repair_pressure_contract when mode changed)
  │
  └─3─► bus RPC  orion:cortex:gateway:request  (unchanged)
            → orch → exec: collect_metacog → stance LLM → speech LLM
            speech reads repair_pressure_contract from ctx.metadata
```

Same `correlation_id` for steps 1–3. Step 1 failure is **fail-open** for chat (empty bundle, no contract attach).

### Hub responsibilities (dumb)

| Does | Does not |
|------|----------|
| Build `turn_window` from `messages[]` | Run phrase_match or probe templates |
| Call pre_turn_appraisal RPC with `paradigms_requested` from settings | Compute level, logprobs, or weights |
| Copy `metadata_attachments` onto chat request | Import paradigm plugins |
| Publish `grammar_scalars` from bundle to grammar emit | Own LLM routes |

Remove or bypass Hub-local `phrase_match` in `substrate_effect_pipeline` when v2 enabled.

### Cortex-exec responsibilities (cognition)

| Component | Role |
|-----------|------|
| `pre_turn_appraisal` RPC handler | Dispatch paradigms, merge bundle, enforce timeout |
| `logprob_probe_runner` | Shared native-completion + logprob extraction |
| `paradigms/repair_pressure_v2` | First plugin: template, evidence, appraise, contract_delta |
| Existing speech path | Unchanged: `executor.py` reads `repair_pressure_contract` metadata |

### Extensibility (paradigm plugin)

```text
orion/substrate/appraisal/
  paradigms/
    base.py              # AppraisalParadigm protocol
    registry.py          # explicit dict — no auto-discovery cathedral
    repair_pressure_v2.py
  probe/
    logprob_runner.py    # shared scoring: P(YES), margin → score/confidence
  turn_window.py         # messages[] → role-tagged molecules
```

Future paradigms (e.g. `grounding_boundary`) register in `PARADIGM_REGISTRY`, add eval fixtures, add `metadata_attachments` key — no Hub rewrite.

---

## Generic contracts

### PreTurnAppraisalRequestV1

```json
{
  "correlation_id": "uuid",
  "session_id": "string",
  "turn_window": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "paradigms_requested": ["repair_pressure"],
  "contract_before": { "mode": "default" },
  "options": {
    "fail_closed": true,
    "timeout_ms": 800,
    "max_turns": 8
  }
}
```

### TurnAppraisalBundleV1

```json
{
  "correlation_id": "uuid",
  "paradigms": {
    "repair_pressure": {
      "appraisal_kind": "repair_pressure",
      "level": 0.82,
      "confidence": 0.71,
      "dimensions": { "specificity_demand": 0.91, "...": 0.0 },
      "evidence": [
        {
          "evidence_kind": "specificity_demand",
          "detector": "logprob_probe_v2",
          "score": 0.91,
          "confidence": 0.78,
          "features": { "logprob_yes": -0.12, "logprob_no": -2.4, "margin": 2.28 }
        }
      ],
      "contract_delta": { "mode": "repair_concrete", "rules": ["..."] },
      "notes": []
    }
  },
  "metadata_attachments": {
    "repair_pressure_contract": { "mode": "repair_concrete", "rules": ["..."] }
  },
  "grammar_scalars": {
    "repair_pressure": { "level": 0.82, "confidence": 0.71 }
  },
  "failed_paradigms": []
}
```

Register schemas in `orion/schemas/registry.py` and channels in `orion/bus/channels.yaml` in the same changeset as the producer/consumer.

---

## Repair pressure v2 paradigm

### v1 kills

| Remove | Replace with |
|--------|--------------|
| `phrase_match_v1` / `_PHRASES` | `logprob_probe_v2` per kind |
| User-only Hub observation buffer | `turn_window` from full `messages[]` |
| Hand weights 0.22/0.20/… | `config/substrate/repair_pressure_weights.v2.yaml` (eval-calibrated) |
| Dead gradient terms in level formula | Dropped from v2 formula |
| Intentional double-count overlap ("you keep" ×2) | One probe per kind per window |
| `coherence_gap` phrase "making shit up" | Redefined probe semantics (thread drift, not grounding lies) |

### Scoring (Approach 1 — logprobs on evidence, Python owns level)

One batched **native completion** per paradigm (via LLM gateway: `return_logprobs=true`, `logprob_probe_mode=native_completion`).

**Completion shape** (seven lines, YES/NO only):

```text
specificity_demand: YES
trust_rupture: YES
coherence_gap: NO
repetition_failure: YES
operational_block: YES
explicit_repair_command: NO
assistant_accountability_demand: NO
```

**Per kind:**

- `score = sigmoid(logprob_YES − logprob_NO)`
- `confidence = top1_margin` at answer token
- Never use model-supplied numeric JSON for score/confidence.

**Level:**

```text
level = Σ weight[kind] × score[kind]     # weights from v2 yaml, calibrated by eval
appraisal.confidence = min(confidence[kind] for kinds with score > 0.5)
```

Keep existing gates in `apply_repair_pressure_contract`:

- `level ≥ 0.75` and `confidence ≥ 0.60` → `repair_concrete`
- `0.45 ≤ level < 0.75` → `concrete_bias`
- else unchanged

### Seven kinds — semantic definition + contract rule

Grounding lies / capability hallucination are **explicitly out of scope** for repair pressure.

| Kind | Probe meaning | Rule when score ≥ 0.65 |
|------|---------------|-------------------------|
| `specificity_demand` | User demands concrete implementation detail (files, steps, boundaries) | include file/module boundaries |
| `trust_rupture` | Prior assistant directions/specs were wrong or unusable | acknowledge correction briefly |
| `coherence_gap` | Assistant stance drifted or contradicted across turns | answer with one concrete operational path |
| `repetition_failure` | User re-asks something not addressed | address the repeated ask directly |
| `operational_block` | User needs handoff-ready output for another builder | include tests/acceptance checks; do not build section |
| `explicit_repair_command` | User imposed explicit style/mode constraint | obey constraint (span in evidence audit) |
| `assistant_accountability_demand` | User holds assistant accountable for prior **build/spec** work | show assumptions |

Active kind rules **union** into `contract_delta.rules`. Level still gates mode. Kinds are real when measured (logprobs), routed (rules), and eval-separated from grounding anger.

### LLM call (plain)

Before the main chat reply, cortex runs one small classification call on the paired thread. It does not answer the user. It answers seven YES/NO questions. Logprobs at each answer become evidence scores. Python computes level and assembles contract rules. Hub only attaches the result.

### Fail-closed

| Condition | Result |
|-----------|--------|
| RPC timeout | `failed_paradigms: ["repair_pressure"]`, no metadata attach |
| Parse miss / no YES-NO tokens | `no_repair_evidence`, level 0 |
| `llm_uncertainty.available=false` | Same as empty evidence |
| Chat | Always proceeds |

---

## Speech + grammar (unchanged seam)

- **Metadata key:** `repair_pressure_contract` (`REPAIR_PRESSURE_CONTRACT_METADATA_KEY`) — attach only when `contract_before.mode != contract_after.mode`.
- **Consumer:** `compile_speech_contract(brief, repair_contract=...)` in `chat_stance.py` — repair_concrete wins; concrete_bias blends.
- **Grammar:** Hub `grammar_emit` continues scalar `repair_signal` atom from `grammar_scalars.repair_pressure` — grammar does not carry seven kinds.

---

## Bus / schema changes

### Add

| Surface | Entry |
|---------|-------|
| `orion/bus/channels.yaml` | `orion:cortex:pre_turn_appraisal:request` (hub → cortex-exec) |
| `orion/bus/channels.yaml` | `orion:cortex:pre_turn_appraisal:result:*` |
| `orion/schemas/` | `PreTurnAppraisalRequestV1`, `TurnAppraisalBundleV1`, paradigm slice models |
| `orion/schemas/registry.py` | Register new message kinds |

### Unchanged

- `graph_cognition` / `repair_pressure` signal kind (dimensions may gain `detector=logprob_probe_v2` in evidence audit only).
- Speech wiring metadata key and cortex-exec merge logic.
- Schema kernel atoms/gradients/molecule kinds.

---

## Config / env

| Key | Service | Meaning |
|-----|---------|---------|
| `ENABLE_PRE_TURN_APPRAISAL` | hub | Master gate for RPC step |
| `PRE_TURN_APPRAISAL_PARADIGMS` | hub | e.g. `repair_pressure` |
| `PRE_TURN_APPRAISAL_TIMEOUT_MS` | hub | RPC timeout |
| `ENABLE_REPAIR_PRESSURE_V2` | cortex-exec | Use logprob paradigm vs legacy phrase_match fallback |
| `REPAIR_PRESSURE_WEIGHTS_V2_PATH` | cortex-exec | Calibrated weights yaml |
| `REPAIR_PRESSURE_PROBE_ROUTE` | cortex-exec | LLM gateway route (small/fast) |
| `LLM_LOGPROB_*` | llm-gateway | Existing logprob rails (must be enabled for probe) |

Update `.env_example` per `AGENTS.md`; sync local `.env` via `python scripts/sync_local_env_from_example.py`.

---

## Files likely to touch

| Path | Why |
|------|-----|
| `orion/schemas/pre_turn_appraisal.py` | Request/bundle models |
| `orion/bus/channels.yaml` | New RPC channels |
| `orion/schemas/registry.py` | Schema registration |
| `orion/substrate/appraisal/paradigms/` | Plugin rail + repair v2 |
| `orion/substrate/appraisal/probe/logprob_runner.py` | Shared logprob scoring |
| `orion/substrate/appraisal/turn_window.py` | Paired turns |
| `orion/substrate/appraisal/contract.py` | Kind-aware rule assembly |
| `config/substrate/repair_pressure_weights.v2.yaml` | Calibrated weights |
| `services/orion-cortex-exec/app/pre_turn_appraisal.py` | RPC handler |
| `services/orion-hub/scripts/pre_turn_appraisal_client.py` | Bus client |
| `services/orion-hub/scripts/api_routes.py` | Replace phrase pipeline hook |
| `services/orion-hub/scripts/websocket_handler.py` | Same |
| `services/orion-hub/scripts/grammar_emit.py` | Read grammar_scalars from bundle |
| `orion/substrate/evals/` | Transcript fixtures + harness |
| `tests/test_repair_pressure_*.py` | Update/remove phrase-only tests |

---

## Eval acceptance (definition of “real”)

Harness: `orion/substrate/evals/repair_pressure_v2_eval.py`  
Fixtures: `orion/substrate/evals/fixtures/repair_pressure/`

| Class | Expectation |
|-------|-------------|
| **Positive** (ops frustration threads) | `level ≥ 0.75`, `confidence ≥ 0.60`, `mode = repair_concrete`, ≥3 kinds active with margin ≥ 0.5 |
| **Negative** (grounding/honesty chastisement) | `level < 0.45`, mode unchanged, `trust_rupture` and `coherence_gap` each `< 0.55` |
| **Neutral** | `no_repair_evidence` or `level < 0.25` |

Live smoke (non-fixture): one organic ops-frustration turn trips chip + speech overlay.

---

## Related keyword paths (out of scope for v2, same debt class)

Document for follow-on paradigms; do not fix in this changeset:

1. `orion/substrate/attention/detectors/legacy_regex.py` — feeds stance via attention frame
2. `orion/substrate/attention/scoring.py` — emotion/plan regex
3. `chat_stance.py` `_TECHNICAL_TURN_PATTERNS` — duplicated in `social.py` adapter
4. `delivery_grounding` / supervisor `_grounding_verdict` — separate grounding stack
5. `social_repair_signal` — naming collision, different organ

---

## Risks

| Risk | Mitigation |
|------|------------|
| +latency before chat | Tight timeout (800ms), small model route, fail-closed |
| Two Hub RPCs | Accept for visible seam; or later fold into gateway batch without moving cognition to Hub |
| Logprobs unavailable on route | Fail-closed; do not fall back to phrase_match in production when v2 enabled |
| Kind/rule explosion | Only seven kinds; rules are fixed map per kind; eval gates orthogonality |

---

## Acceptance checks

- [ ] Hub no longer runs `phrase_match_v1` when v2 enabled
- [ ] Pre-turn appraisal RPC registered and smoke-tested on bus
- [ ] Paired turn window includes assistant messages
- [ ] Evidence scores derived from logprobs only (unit test with fixture logprob payload)
- [ ] Kind-active rules appear in `repair_pressure_contract` when mode flips
- [ ] Speech wiring still merges contract on same turn (regression tests pass)
- [ ] Positive/negative/neutral eval fixtures pass
- [ ] `make agent-check` / schema + channel registry checks pass
- [ ] PR report references `AGENTS.md` completion checklist

---

## Recommended implementation order

1. Schemas + bus channels + registry (contract-only PR slice)
2. `logprob_probe_runner` + `repair_pressure_v2` paradigm (library + unit tests)
3. Cortex-exec RPC handler
4. Hub client + wire into chat handlers; deprecate phrase pipeline
5. Eval fixtures + weight calibration yaml
6. Remove phrase_match from hot path; keep legacy behind flag for rollback

Terminal step per brainstorming: invoke `writing-plans` skill to produce `docs/superpowers/plans/2026-07-03-repair-pressure-v2-pre-turn-appraisal.md`.

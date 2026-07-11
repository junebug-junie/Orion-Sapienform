# Design: AutonomyStateV2 Reducer

**Date:** 2026-05-01  
**Status:** Approved for implementation planning  
**Scope:** `orion/autonomy/` + `services/orion-cortex-exec/` + `services/orion-hub/scripts/autonomy_payloads.py`

---

## What this is

A deterministic, flagged, auditable state reducer that takes explicit evidence signals + a prior
autonomy state + action outcomes and produces a versioned `AutonomyStateV2` and a
`AutonomyStateDeltaV1`.

The goal is to give Orion a causal state spine:

```
evidence/events
  → appraisal / reduction
  → AutonomyStateV2
  → attention / impulse / inhibition
  → exported metadata for chat / hub
  → outcome feedback later
```

## What this is NOT

- Not sentience proof or a consciousness metric.
- Not a phi/spark replacement. Phi, spark, and similar proxy telemetry are low-confidence
  signals that can enter as evidence with `kind = "proxy_telemetry"` but must never be treated
  as canonical state. The reducer enforces this via an inhibited impulse and a confidence penalty
  when proxy-only evidence is all that is present.
- Not a new neural net or inner-state ontology.
- Not a new Hub dashboard (UI work is deferred).
- Not a persistence layer. V2 state lives in the turn context only; nothing is written to the
  graph. This means `previous_state` is always reconstructed each turn from the V1 graph lookup.

---

## Architecture

### Causal loop

```
Graph (V1 state)
  → upgrade_autonomy_state_v1_to_v2
  → AutonomyReducerInputV1 (prior + evidence + outcomes)
  → reduce_autonomy_state
  → AutonomyStateV2 + AutonomyStateDeltaV1
  → chat stance inputs["autonomy"]["state_v2"] / ["delta"]
  → router metadata: autonomy_state_v2_preview, autonomy_state_delta
  → Hub: extracted via autonomy_payloads.py
  → (future) outcome feedback written back as ActionOutcomeRefV1
```

### Environment gate

Disabled by default. Enable with:

```
AUTONOMY_STATE_V2_REDUCER_ENABLED=true
```

When unset or false: existing V1 behavior is entirely unchanged. The reducer code path is
skipped. No new context keys are written.

---

## Module map

| File | Change |
|------|--------|
| `orion/autonomy/models.py` | Add new V1-named Pydantic models + `AutonomyStateV2` + `upgrade_autonomy_state_v1_to_v2`. V1 models untouched. |
| `orion/autonomy/reducer.py` | New module. `AutonomyReducerInputV1`, `AutonomyReducerResultV1`, `reduce_autonomy_state`. |
| `orion/autonomy/summary.py` | Widen type signature to `AutonomyStateV1 | AutonomyStateV2 | None`. Preserve all V1 branches verbatim. Add V2-only extras. |
| `services/orion-cortex-exec/app/chat_stance.py` | Env-gated sidecar after `_load_autonomy_state`. No rewrite. |
| `services/orion-cortex-exec/app/router.py` | Add optional `autonomy_state_v2_preview` and `autonomy_state_delta` to `_autonomy_payload_from_ctx`. |
| `services/orion-hub/scripts/autonomy_payloads.py` | Add `autonomy_state_v2_preview` and `autonomy_state_delta` to the forwarding whitelist. |
| `orion/autonomy/tests/test_autonomy_state_v2_*.py` | New unit tests (schema, reducer, summary). |
| `services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2*.py` | New integration tests (stance + router). |
| `docs/autonomy_state_v2_reducer.md` | Operator-facing doc. |

---

## 1. Schema additions (`orion/autonomy/models.py`)

All existing V1 models are untouched (`extra="forbid"` preserved).

### New models

**`AutonomyEvidenceRefV1`**
```
evidence_id: str                          # stable, deterministic (see §Evidence ID rules)
source: str                               # e.g. "graph", "user_message", "infra", "proxy_telemetry"
kind: str                                 # e.g. "identity_snapshot", "drive_audit", "goal_ref",
                                          #      "user_turn", "infra_health", "proxy_telemetry"
summary: str | None = None
confidence: float = Field(default=0.5, ge=0.0, le=1.0)
observed_at: datetime | None = None
```

**`AttentionItemV1`**
```
item_id: str                              # stable: hash(subject + kind + evidence_id or source)
summary: str
source: str
salience: float = Field(ge=0.0, le=1.0)
drive_links: list[str] = Field(default_factory=list)
tension_links: list[str] = Field(default_factory=list)
unresolved: bool = True
evidence_refs: list[str] = Field(default_factory=list)
```

**`CandidateImpulseV1`**
```
impulse_id: str
kind: str
summary: str
drive_origin: str | None = None
expected_effect: str | None = None
confidence: float = Field(default=0.5, ge=0.0, le=1.0)
evidence_refs: list[str] = Field(default_factory=list)
```

**`InhibitedImpulseV1`**
```
impulse_id: str
kind: str
summary: str
inhibition_reason: str
risk: str | None = None
evidence_refs: list[str] = Field(default_factory=list)
```

**`ActionOutcomeRefV1`**
```
action_id: str
kind: str
summary: str
success: bool | None = None
surprise: float = Field(default=0.0, ge=0.0, le=1.0)
observed_at: datetime | None = None
```

**`AutonomyStateDeltaV1`**
```
subject: str
changed_fields: list[str] = Field(default_factory=list)
  # Semantics: fields that differ vs the V1-upgrade baseline, NOT vs a prior turn.
  # Accurate for surface-level diffs (e.g. new tensions); use drive_deltas for pressure changes.
drive_deltas: dict[str, float] = Field(default_factory=dict)
new_tensions: list[str] = Field(default_factory=list)
resolved_tensions: list[str] = Field(default_factory=list)
new_attention_items: list[str] = Field(default_factory=list)
new_impulses: list[str] = Field(default_factory=list)
new_inhibitions: list[str] = Field(default_factory=list)
confidence_delta: float = 0.0
notes: list[str] = Field(default_factory=list)
```

**`AutonomyStateV2`**  
All V1 fields preserved, plus:
```
schema_version: str = "autonomy.state.v2"
evidence_refs: list[AutonomyEvidenceRefV1] = Field(default_factory=list)
freshness: dict[str, str] = Field(default_factory=dict)
confidence: float = Field(default=0.5, ge=0.0, le=1.0)
unknowns: list[str] = Field(default_factory=list)
attention_items: list[AttentionItemV1] = Field(default_factory=list)
candidate_impulses: list[CandidateImpulseV1] = Field(default_factory=list)
inhibited_impulses: list[InhibitedImpulseV1] = Field(default_factory=list)
last_action_outcomes: list[ActionOutcomeRefV1] = Field(default_factory=list)
previous_state_ref: str | None = None
```

### Evidence ID rules (upgrade helper)

To ensure dedup works across repeated upgrade cycles (V1 is re-fetched every turn, V2 is never
persisted), synthetic evidence IDs in `upgrade_autonomy_state_v1_to_v2` must be deterministic:

```
identity_snapshot:  f"identity_snapshot:{latest_identity_snapshot_id}"
drive_audit:        f"drive_audit:{latest_drive_audit_id}"
goal_ref:           f"goal_ref:{goal_id}"   (one per entry in latest_goal_ids)
```

This means merging the same V1 state twice produces no duplicates.

### `upgrade_autonomy_state_v1_to_v2`

- Preserve all V1 fields.
- `schema_version = "autonomy.state.v2"`.
- `confidence = 0.55` (V1 present = some real state; not 0.25 baseline, not high confidence).
- Create evidence refs for snapshot/audit/goal IDs when present (stable IDs above).
- Add `unknown: "no_action_outcome_history"` (action outcomes never come from V1 graph).
- Add `unknown: "evidence_from_graph_only"` (we have graph state but no turn-level evidence yet).
- Create at least one attention item if `dominant_drive` or `tension_kinds` is non-empty, using
  `item_id = hashlib.sha256(f"{subject}:{kind}:{dominant_drive or ''}".encode()).hexdigest()[:16]`.

---

## 2. Reducer (`orion/autonomy/reducer.py`)

### Input / Output

```
AutonomyReducerInputV1:
  subject: str = "orion"
  previous_state: AutonomyStateV1 | AutonomyStateV2 | None = None
  evidence: list[AutonomyEvidenceRefV1] = Field(default_factory=list)
  action_outcomes: list[ActionOutcomeRefV1] = Field(default_factory=list)
  now: datetime | None = None   # if None, reducer uses datetime.utcnow(); inject in tests

AutonomyReducerResultV1:
  state: AutonomyStateV2
  delta: AutonomyStateDeltaV1
```

`reduce_autonomy_state(input: AutonomyReducerInputV1) -> AutonomyReducerResultV1`

Deterministic for fixed inputs **including** `now`. No LLM calls. No I/O.

### Baseline (previous_state is None)

Use `SUBJECT_BINDINGS` from `orion.autonomy.repository` to resolve `model_layer` and
`entity_id`. Keys: `"orion"`, `"juniper"`, `"relationship"`. Unknown subjects fall back to
`model_layer = "unknown"`, `entity_id = subject`.

```
confidence = 0.25
unknowns = ["no_previous_state"]
source = "reducer"
generated_at = now
```

### Evidence ID dedup

When previous_state is a V2 and new evidence arrives, merge evidence lists deduping on
`evidence_id`. Apply the same dedup to action outcomes on `action_id`. After merge, trim to
bounds:

```
evidence_refs      max 20  (drop oldest by observed_at, then by insertion order)
attention_items    max 8
candidate_impulses max 8
inhibited_impulses max 8
last_action_outcomes max 12
unknowns           max 12
```

### Heuristic pressure rules

Evidence text = `kind + " " + (summary or "")` lowercased.

**Important:** patterns are polarity-blind — they fire on substring match regardless of
negation. This is a known limitation. Confidence is set conservatively to compensate.
Do not raise any drive by more than `+0.15` from a single evidence item.

| Drive | Pattern tokens (any match in evidence text) | Delta |
|-------|----------------------------------------------|-------|
| coherence | `contradiction`, `inconsistency`, `failure`, `bug`, `broken`, `drift`, `confusion` | +0.12 |
| continuity | `memory`, `recall`, `thread`, `history`, `stale`, `missing context` | +0.10 |
| relational | `frustration`, `repair`, `trust`, `relationship`, `social`, `apology` | +0.10 |
| autonomy | `proposal`, `self-modification`, `workflow`, `autonomous`, `action` | +0.08 |
| capability | `tool failure`, `missing capability`, `timeout`, `unavailable`, `error` | +0.10 |
| predictive | `surprise`, `unexpected`, `regression`, `mismatch` | +0.08 |

Evidence with `source = "user_message"` or `kind = "infra_health"` has **no drive pressure
effect**. These kinds carry contextual signal but are not evidence of Orion's internal state.

Evidence with `kind = "proxy_telemetry"` has drive effect at **50% weight** (multiplied before
applying delta). It is not treated as canonical state.

Clamp all pressures to [0, 1]. Also merge in pressures from previous V2 state (carry forward),
before applying new evidence deltas.

`relational_stability` key in drive_pressures is merged into `relational` via `max()`, same as
`summary.py`.

### Dominant drive and active drives

- `dominant_drive` = drive with highest pressure if >= 0.15, else carry forward from previous state.
- `active_drives` = top 3 drives with pressure >= 0.12.

### Tension derivation

| Tension kind | Trigger |
|--------------|---------|
| `tension.coherence_break.v1` | coherence drive pressure >= 0.25, OR evidence text contains `contradiction`, `inconsistency`, `bug`, `broken`, `confusion` |
| `tension.continuity_gap.v1` | continuity >= 0.25, OR evidence text contains `stale`, `missing context`, `recall`, `memory failure` |
| `tension.capability_gap.v1` | capability >= 0.25, OR evidence text contains `unavailable`, `timeout`, `error`, `tool failure` |
| `tension.relational_repair.v1` | relational >= 0.25, OR evidence text contains `frustration`, `trust`, `apology`, `repair` |
| `tension.drive_competition.v1` | top two drive pressures both >= 0.25 AND differ by < 0.08 |

**Drive competition ownership:** `tension.drive_competition.v1` is **derived here** (reducer is
authoritative). `summary.py`'s `_analyze_drive_competition` still runs for the `drive_competition`
structured field on `AutonomySummaryV1` but should not re-add the tension kind string when it is
already present in `tension_kinds` from the state. This is enforced by the existing guard in
`summary.py`: `if drive_competition and _PRESSURE_COMPETITION_KIND not in tension_sources`.

### Impulse generation (candidate_impulses)

One impulse per trigger at most. Impulses should not be duplicated if already present in
previous state (dedup by `kind`).

| Condition | Impulse kind |
|-----------|-------------|
| coherence pressure >= 0.35 | `synthesize_or_reduce` |
| capability pressure >= 0.35 | `triage_capability_gap` |
| continuity pressure >= 0.35 | `recover_context` |
| relational pressure >= 0.35 | `repair_or_acknowledge` |
| autonomy pressure >= 0.35 | `propose_bounded_action` |

### Inhibition generation (inhibited_impulses)

Inhibitions represent single-layer vetoes. Three canonical reasons, each checked independently.
When an inhibition fires, move the impulse (if generated) from `candidate_impulses` to
`inhibited_impulses`. Do not generate both.

| Condition | Inhibition reason | Note |
|-----------|------------------|------|
| autonomy pressure >= 0.35 AND confidence < 0.45 | `low_confidence_for_autonomous_action` | Autonomy impulse is inhibited, not removed |
| capability pressure >= 0.35 AND evidence includes `timeout` or `unavailable` | `dependency_unavailable` | Capability impulse is inhibited |
| ALL evidence has `kind = "proxy_telemetry"` AND confidence < 0.6 | `proxy_signal_not_canonical_state` | No proxy → canonical state inference |

**Hedge consolidation:** The `proxy_signal_not_canonical_state` inhibition is the canonical
place for the proxy hedge. Do not also add a response hazard for this in `summary.py` unless
the inhibition is present on the state. The confidence penalty for proxy evidence (below) is
separate and non-redundant.

### Confidence rules

Starting point: previous confidence or baseline (0.25 if no prior state, 0.55 if V1 upgrade).

| Event | Effect |
|-------|--------|
| High-confidence direct evidence (confidence >= 0.7, kind not proxy) | +0.05 per item, max +0.10 total |
| All evidence is `proxy_telemetry` | -0.10 |
| Any unknown is `no_previous_state` | -0.05 |
| Evidence includes `stale` or `missing context` | -0.05 |
| Evidence includes `timeout` or `unavailable` | -0.05 |
| Action outcome with `surprise >= 0.7` | -0.08 per item, max -0.12 total |

Clamp to [0, 1].

### Unknowns

| Condition | Unknown token |
|-----------|--------------|
| `previous_state is None` | `"no_previous_state"` |
| No fresh evidence (empty evidence list after merge) | `"no_fresh_evidence"` |
| All evidence is `proxy_telemetry` | `"proxy_only_evidence"` |
| No action outcomes in state | `"no_action_outcome_history"` |
| `latest_identity_snapshot_id` is None | `"no_identity_snapshot"` |
| `latest_drive_audit_id` is None | `"no_drive_audit"` |

### Freshness

```python
freshness["state_generated_at"] = now.isoformat()
# For evidence categories where observed_at is present:
freshness["latest_direct_evidence_at"] = most_recent observed_at among non-proxy evidence
freshness["latest_proxy_evidence_at"] = most_recent observed_at among proxy evidence
```

### Delta

Compare produced V2 state against the pre-reduction baseline (upgraded V1 or prior V2):

- `changed_fields`: list of top-level field names that differ. **Document in code** that this
  compares against the upgrade baseline, not a prior turn's persisted state.
- `drive_deltas`: per-drive delta from baseline pressures to new pressures (non-zero only).
- `new_tensions`: tension kinds added vs baseline.
- `resolved_tensions`: tension kinds in baseline not in new state.
- `new_attention_items`: item_ids added.
- `new_impulses`: impulse_ids in candidate_impulses added.
- `new_inhibitions`: impulse_ids in inhibited_impulses added.
- `confidence_delta`: new confidence - baseline confidence.
- `notes`: human-readable reducer notes for debugging (max 5).

---

## 3. Summary widening (`orion/autonomy/summary.py`)

Signature change only: `def summarize_autonomy_state(state: AutonomyStateV1 | AutonomyStateV2 | None)`.

All existing V1 branches are untouched. A V2 is duck-typed — it has all V1 fields, so no
conditional is needed for those.

**V2-only extras (only when `isinstance(state, AutonomyStateV2)`):**

Response hazards:
- If `confidence < 0.4`: append `"avoid overconfident inner-state claims"`
- If `unknowns` non-empty: append `"surface uncertainty when state evidence is thin"`
- If any `inhibited_impulse.inhibition_reason == "proxy_signal_not_canonical_state"`: append
  `"do not treat proxy telemetry as canonical state"`

Proposal headlines: if `state.goal_headlines` is empty AND `state.attention_items` has entries,
use top 3 attention item summaries as `proposal_headlines`.

Drive competition: existing `_analyze_drive_competition` still runs. The guard
`if _PRESSURE_COMPETITION_KIND not in tension_sources` already prevents double-adding.

---

## 4. Chat stance integration (`services/orion-cortex-exec/app/chat_stance.py`)

In `build_chat_stance_inputs`, after `autonomy = _load_autonomy_state(ctx)`:

```python
if os.getenv("AUTONOMY_STATE_V2_REDUCER_ENABLED", "").strip().lower() == "true":
    try:
        v2_result = _run_autonomy_reducer(ctx, autonomy)
        ctx["chat_autonomy_state_v2"] = v2_result.state.model_dump(mode="json")
        ctx["chat_autonomy_state_delta"] = v2_result.delta.model_dump(mode="json")
        inputs["autonomy"]["state_v2"] = ctx["chat_autonomy_state_v2"]
        inputs["autonomy"]["delta"] = ctx["chat_autonomy_state_delta"]
    except Exception as exc:
        logger.warning("autonomy_reducer_v2_failed error=%s", exc)
```

`_run_autonomy_reducer` builds:

```python
evidence = []
# User message — kind="user_turn", source="user_message"
#   NO drive pressure effect (source="user_message" is excluded from pressure heuristics)
msg = ctx.get("user_message") or ctx.get("message") or ""
if msg:
    evidence.append(AutonomyEvidenceRefV1(
        evidence_id=f"user_turn:{hashlib.sha256(msg[:200].encode()).hexdigest()[:16]}",
        source="user_message",
        kind="user_turn",
        summary=msg[:200],
        confidence=0.9,
    ))

# Infra health — kind="infra_health", source="infra"
#   NO drive pressure effect (kind="infra_health" excluded)
avail = autonomy["debug"].get("orion", {}).get("availability", "")
if avail:
    evidence.append(AutonomyEvidenceRefV1(
        evidence_id=f"infra_health:autonomy_graph:{avail}",
        source="infra",
        kind="infra_health",
        summary=f"autonomy graph availability={avail}",
        confidence=1.0,
    ))

# Reasoning fallback — kind="reasoning_quality", source="reasoning"
#   This DOES affect drive pressure if summary indicates issues
if ctx.get("chat_reasoning_summary", {}).get("fallback_recommended"):
    evidence.append(AutonomyEvidenceRefV1(
        evidence_id="reasoning:fallback_recommended",
        source="reasoning",
        kind="reasoning_quality",
        summary="reasoning fallback recommended",
        confidence=0.6,
    ))

# Social bridge hazards — kind="relational_signal", source="social_bridge"
for hazard in (ctx.get("chat_social_bridge_summary") or {}).get("hazards") or []:
    evidence.append(AutonomyEvidenceRefV1(
        evidence_id=f"social_bridge:{hashlib.sha256(str(hazard)[:80].encode()).hexdigest()[:12]}",
        source="social_bridge",
        kind="relational_signal",
        summary=str(hazard)[:200],
        confidence=0.6,
    ))
```

Existing `chat_autonomy_state`, `chat_autonomy_summary`, `chat_autonomy_debug`,
`chat_autonomy_backend`, `chat_autonomy_selected_subject`, `chat_autonomy_repository_status`
and `inputs["autonomy"]["state"]` / `["summary"]` / `["debug"]` are **not modified**.

---

## 5. Router export (`services/orion-cortex-exec/app/router.py`)

In `_autonomy_payload_from_ctx`, after existing keys:

```python
v2_state = ctx.get("chat_autonomy_state_v2")
if isinstance(v2_state, dict):
    payload["autonomy_state_v2_preview"] = {
        "schema_version": v2_state.get("schema_version"),
        "dominant_drive": v2_state.get("dominant_drive"),
        "active_drives": (v2_state.get("active_drives") or [])[:3],
        "confidence": v2_state.get("confidence"),
        "unknowns": (v2_state.get("unknowns") or [])[:5],
        "top_attention_summaries": [
            item["summary"] for item in (v2_state.get("attention_items") or [])[:3]
            if isinstance(item, dict)
        ],
        "top_inhibition_reasons": [
            item["inhibition_reason"] for item in (v2_state.get("inhibited_impulses") or [])[:3]
            if isinstance(item, dict)
        ],
    }
delta = ctx.get("chat_autonomy_state_delta")
if isinstance(delta, dict):
    payload["autonomy_state_delta"] = delta
```

Existing keys are never mutated.

---

## 6. Hub forwarding (`services/orion-hub/scripts/autonomy_payloads.py`)

Add to the key whitelist in `extract_autonomy_payload`:
```
"autonomy_state_v2_preview"
"autonomy_state_delta"
```

---

## 7. Tests

### A. Schema / upgrade (`orion/autonomy/tests/`)

- V1 → V2 upgrade preserves all V1 field values.
- Upgrade creates evidence refs with stable deterministic IDs for snapshot/audit/goal IDs.
- Upgrading same V1 twice and merging produces no duplicate evidence refs.
- V1 with no snapshot/audit IDs creates unknowns `no_identity_snapshot`, `no_drive_audit`.
- V1 with dominant_drive creates at least one attention item.

### B. Reducer

- Baseline (no previous state): confidence == 0.25, unknowns contains `no_previous_state`,
  entity_id matches SUBJECT_BINDINGS for "orion", "juniper", "relationship".
- Evidence kind="infra_health" or source="user_message" produces no drive pressure change.
- Evidence text "GraphDB timeout unavailable" → capability pressure increases, tension
  `tension.capability_gap.v1` present.
- Evidence text "contradiction / broken / confusion" → coherence pressure increases, tension
  `tension.coherence_break.v1` present.
- Evidence negation: "no contradiction" — document expected behavior (polarity-blind; pressure
  still increases). Test asserts this and confirms the limitation in the test name.
- All-proxy evidence → `proxy_signal_not_canonical_state` inhibition, `proxy_only_evidence`
  unknown, confidence penalty applied; no proxy claim in candidate_impulses.
- High-surprise outcome reduces confidence and increases predictive pressure.
- Determinism: same inputs (fixed `now`) produce byte-identical output on two calls.
- Bounded lists: reducing 25 evidence items produces state with max 20 evidence_refs.

### C. Summary

- V2 confidence < 0.4 → hazard `"avoid overconfident inner-state claims"` present.
- V2 with proxy inhibition → hazard `"do not treat proxy telemetry as canonical state"` present.
- V2 with no proxy inhibition → hazard absent.
- V1 paths: all existing `test_chat_stance_brief.py` and summary-touching tests pass unchanged.

### D. Chat stance integration

- `AUTONOMY_STATE_V2_REDUCER_ENABLED=true` → `chat_autonomy_state_v2` and
  `chat_autonomy_state_delta` in ctx after `build_chat_stance_inputs`.
- `inputs["autonomy"]` contains `state_v2` and `delta` keys.
- Flag unset → neither key present; existing ctx keys unchanged.
- Reducer raises → warning logged; build_chat_stance_inputs returns normally; existing keys intact.

### E. Router / Hub export

- Existing `test_router_autonomy_payload_export.py` tests pass unchanged.
- When `chat_autonomy_state_v2` in ctx → metadata contains `autonomy_state_v2_preview` with
  correct shape.
- When absent → `autonomy_state_v2_preview` not in metadata.
- `extract_autonomy_payload` forwards `autonomy_state_v2_preview` and `autonomy_state_delta`
  when present in cortex result metadata.

---

## 8. Operator doc

`docs/autonomy_state_v2_reducer.md` — covers: why this exists, what it is not,
causal loop diagram, phi/spark treatment, env var, known limitations (polarity-blind
heuristics; no durable state; changed_fields compares against upgrade baseline not prior turn;
evidence ID dedup depends on stable IDs in upgrade helper).

---

## Known limitations

1. **Polarity-blind heuristics.** Pattern matching fires on "no contradiction" as well as
   "contradiction." Confidence caps compensate but do not eliminate false signal.
2. **No durable state.** `previous_state` is always rebuilt from V1 each turn. `changed_fields`
   in delta is relative to the upgrade baseline, not a persisted prior turn.
3. **Evidence sources are mixed.** User turn messages and infra health signals are excluded from
   drive pressure by kind/source rule, but they occupy slots in the evidence list and can affect
   confidence scoring. This is intentional but operators should be aware.
4. **entity_id is driven by SUBJECT_BINDINGS.** If new subjects are added to the repo, the
   reducer inherits them automatically. Unknown subjects get a safe fallback.
5. **Drive competition ownership.** Reducer is authoritative for `tension.drive_competition.v1`
   in `tension_kinds`. `summary.py` derives the structured `drive_competition` field
   independently and guards against double-adding the kind string.

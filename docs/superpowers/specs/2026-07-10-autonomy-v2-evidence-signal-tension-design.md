# Design: AutonomyStateV2 evidence redesign via signal_tension

**Date:** 2026-07-10  
**Status:** Approved for implementation planning  
**Scope:** `orion/autonomy/` + `config/autonomy/signal_drive_map.yaml` + `services/orion-cortex-exec/app/chat_stance.py` (+ focused tests/evals)  
**Supersedes (partial):** keyword pressure path in `orion/autonomy/reducer.py`; does not replace the 2026-05-01 AutonomyStateV2 reducer design — it repairs its upstream evidence contract and drive-pressure math.

---

## Arsonist summary

AutonomyStateV2 looks like live appraisal but is structurally inert on the only channel that can move `drive_pressures`: relational evidence is read from `ctx` before it is written; reasoning quality fires as constant fallback theater against an empty repository; keyword substring matching has zero overlap with real hazard vocabulary. Turning the env flag on today would ship empty-shell cognition.

This design replaces that theater with a typed, omit-when-empty evidence contract whose pressure math goes through the same closed surface as endogenous drives (`signal_drive_map` + a direct `signal_tension` adapter). Phi / SelfState / homeostatic DriveEngine stay hard-isolated. The flag stays off until a same-PR movement eval proves `dominant_drive` / pressures actually change.

---

## Problem (verified in tree)

Single producer: `_build_autonomy_reducer_evidence` in `services/orion-cortex-exec/app/chat_stance.py`.

| Evidence | Issue |
|----------|--------|
| `user_turn` | Fine; excluded from pressure math by design |
| `infra_health` | Fine; excluded from pressure math by design |
| `reasoning_quality` | Emits when `fallback_recommended`; live path never writes `reasoning_repository` / non-empty `reasoning_artifacts`, so empty-repo compile almost always recommends fallback → constant, not signal |
| `relational_signal` | Reads `ctx["chat_social_bridge_summary"]` ~37 lines before that key is written → empty or stale every turn |
| Keyword pressures | Tokens like `frustration` / `repair` / `contradiction` do not appear in real hazards (`cooldown_active`, `duplicate_message`, `self_message_loop`, `context_excluded:*`, …) |
| `observed_at` | Never set → freshness fields never populate |
| `proxy_telemetry` guard | No producer emits that kind → untested dead rail |

Phi multicollinearity: **no coupling today**. AutonomyStateV2 is request-scoped only. Risk is future: same-named drives (`coherence`, etc.) from independent unreliable estimators. Pending idea “bridge drives → self-state” is explicitly rejected by this design.

---

## Goals

1. Typed evidence contract: emit a kind only when upstream is proven non-empty.
2. One drive-weight table: extend `config/autonomy/signal_drive_map.yaml`; delete keyword pressure matching.
3. Unify pressure application with the `signal_tension` family via a **direct** chat adapter (no DeviationGate / EWMA).
4. Hard ban: AutonomyStateV2 must not feed phi features, SelfState channels, or homeostatic `DriveEngine`.
5. Trace-proven movement in the same PR before calling enable “safe”; default remains flag-off.

## Non-goals

- Graph persistence of AutonomyStateV2
- Confidence recalibration beyond omit-when-empty (kind-literal confidences stay documented as uncalibrated)
- DeviationGate / EWMA for chat evidence
- Wiring chat tensions into the endogenous tick DriveEngine loop (shared map + adapters only)
- Phi / SelfState merge or namespaced drive bridge
- Hub UI redesign
- Building a live reasoning repository producer (only: stop emitting theater until one exists)

---

## Architecture

```text
chat locals (social/hazards, reasoning repo|artifacts, infra avail, user msg)
  → AutonomyEvidenceCompiler (proven-non-empty gates only)
  → AutonomyEvidenceRefV1 (+ optional signal_kind, dimension, value, observed_at)
  → chat_evidence_to_tension()   // same family as failure_to_tension
  → SignalDriveMap weights
  → fold TensionEventV1 into AutonomyStateV2.drive_pressures
  → existing attention / impulse / inhibition / delta tail
  → ctx + Hub preview only
  ✗ never → phi / SelfState / DriveEngine
```

### Why direct tension, not `signal_to_tension`

Chat hazards and reasoning fallback are discrete events. `signal_to_tension` requires `DeviationGate` warm-up and would swallow first occurrences. Follow `failure_to_tension`: map lookup + severity/value → `drive_impacts`, no EWMA.

---

## Components

### 1. `AutonomyEvidenceCompiler` (`orion/autonomy/`)

Called from `build_chat_stance_inputs` with **explicit locals** (`social`, `social_bridge`, reasoning compile result + whether repo/artifacts were non-empty, autonomy debug availability, user message, `now`). Must not read `ctx["chat_social_bridge_summary"]` for this turn’s evidence.

| Kind | Proven-non-empty rule | Pressure-eligible |
|------|----------------------|-------------------|
| `user_turn` | non-empty user message | No |
| `infra_health` | autonomy debug `availability` ∈ `{available, degraded, empty, unavailable}` | No |
| `reasoning_quality` | repository or `reasoning_artifacts` had artifacts this turn **and** compiler produced a usable quality signal (e.g. `fallback_recommended` with non-empty upstream). Empty-repo cold compile → **omit** | Yes |
| `relational_signal` | hazards from merged `social` + `social_bridge` locals | Yes when mapped (`signal_kind=chat_social_hazard`) |

Every emitted ref sets `observed_at`. Confidence remains kind-literal constants for v1 (document as uncalibrated).

### 2. Schema: `AutonomyEvidenceRefV1`

Add optional fields (backward compatible defaults):

- `signal_kind: str | None`
- `dimension: str | None`
- `value: float | None` (0–1)

Audit evidence without these fields is allowed; it never moves pressures. Unmapped hazards may still be recorded as evidence with summary only.

### 3. `chat_evidence_to_tension()` (`orion/autonomy/signal_tension.py`)

- Input: pressure-eligible `AutonomyEvidenceRefV1` with `signal_kind` + `dimension` + `value`
- Lookup: `SignalDriveMap.match(signal_kind, dimension)`
- Unmapped / missing fields → `None` (no keyword rescue)
- Build `TensionEventV1` via existing `_build_tension` helper
- Never raises; degrade to `None`

### 4. `signal_drive_map.yaml` growth

Pinned `signal_kind` names for this patch:

| signal_kind | dimension(s) | Intent |
|-------------|--------------|--------|
| `chat_social_hazard` | exact keys: `cooldown_active`, `duplicate_message`, `self_message_loop` | relational (+ capability where loop/cooldown implies stuck channel) |
| `chat_reasoning_quality` | `fallback` | coherence / predictive pressure when real artifacts still recommend fallback |

Prefix hazards like `context_excluded:*` / `context_softened:*` are **audit-only in v1** unless an exact or `*` suffix rule is added with a test. No prose matching.

Unlisted hazard strings → evidence ok, pressure zero. Growth requires YAML entry + producer + test (existing map contract).

### 5. Reducer (`orion/autonomy/reducer.py`)

- **Delete** `_apply_single_evidence_pressures` keyword matching (and related token helpers used only for that path).
- For each incoming evidence: mint tension via `chat_evidence_to_tension`; fold `magnitude * drive_impacts[drive]` into `drive_pressures` (clamped).
- **Also in scope:** `_derive_tension_kinds` and confidence adjustments that currently OR in keyword hits on an evidence text blob must stop using those keyword OR-clauses. Tension kinds derive from pressure thresholds (and typed evidence if needed); confidence blob keyword penalties (`"stale" in blob`, etc.) are removed or replaced with typed evidence/outcome fields only.
- Keep attention / candidate impulses / inhibition / freshness / delta logic, adjusted so freshness uses `observed_at`.
- Do not call endogenous `SignalTensionSource` rate limiter or DeviationGate from the chat turn path.

### 6. Stance wiring (`chat_stance.py`)

- Replace `_build_autonomy_reducer_evidence` with compiler call using locals available **before** the reducer runs (same function already has `social` / `social_bridge`).
- Pass compiler debug into ctx for observability when flag is on.
- Env gate unchanged: `AUTONOMY_STATE_V2_REDUCER_ENABLED` default off.

---

## Isolation constraint (hard ban)

This patch must not:

- Import or consume AutonomyStateV2 in `services/orion-spark-introspector` / phi corpus builders
- Pass `drive_pressures` into `build_self_state` / SelfState channel maps
- Feed chat tensions into homeostatic `DriveEngine.update`

Explicitly reject the pending brainstorm idea “Bridge autonomy drive pressures → self-state” until a separate proposal with damping analysis is approved.

Acceptance: focused test or import-graph check that AutonomyStateV2 is not referenced from phi/self-state builder modules.

---

## Tracing & enable criteria

Structured debug (ctx and/or delta-adjacent fields) must record:

- evidence kinds emitted vs omitted + omit reason
- tensions minted (kind, signal_kind, dimension, drives)
- `dominant_drive` / pressures before vs after
- `tension_kinds` new / resolved counts

**Enable bar (same PR):** fixture eval or integration test that feeds real mapped hazards and non-empty reasoning artifacts and asserts non-zero pressure movement and/or `dominant_drive` change vs baseline. Docs state: do not flip the flag in production until that proof is green; default remains false.

---

## Error handling

- Compiler and `chat_evidence_to_tension`: degrade to omit / `None`, never raise into chat.
- Existing stance `try/except` around reducer remains; failure leaves V1 path intact.
- Unmapped hazard: evidence without pressure, not an error.

---

## Files likely to touch

| Path | Change |
|------|--------|
| `orion/autonomy/models.py` | Optional `signal_kind` / `dimension` / `value` on evidence |
| `orion/autonomy/evidence_compiler.py` (new) | Proven-gate compiler |
| `orion/autonomy/signal_tension.py` | `chat_evidence_to_tension` |
| `orion/autonomy/reducer.py` | Map/tension fold; remove keyword pressures |
| `config/autonomy/signal_drive_map.yaml` | Chat signal_kind rows |
| `services/orion-cortex-exec/app/chat_stance.py` | Locals → compiler → reducer |
| `docs/autonomy_state_v2_reducer.md` | Operator notes: evidence contract + enable bar |
| Tests under `orion/autonomy/tests/` and `services/orion-cortex-exec/tests/` | Gates, map, movement, isolation |
| Eval or fixture under `orion/autonomy/evals/` or cortex tests | Trace-proven movement |

---

## Acceptance checks

1. Empty reasoning repo does **not** emit `reasoning_quality` evidence.
2. Hazards from `social`/`social_bridge` locals appear as evidence when present (ordering bug fixed).
3. Mapped hazard bumps expected drive(s) via `signal_drive_map`; unmapped hazard does not.
4. Keyword tokens no longer appear in reducer pressure path, `_derive_tension_kinds`, or confidence blob penalties.
5. `observed_at` set → freshness populates when evidence exists.
6. Movement fixture/eval passes with flag on in test.
7. Isolation check: no AutonomyStateV2 → phi/self-state wire.
8. Default env remains reducer disabled; operator doc updated.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Dual pressure pipelines (chat reducer vs endogenous tick) still exist | Shared map + shared tension helpers; DriveEngine wiring explicitly out of scope |
| Sparse map → pressures still often frozen | Movement eval uses mapped fixtures; omit-empty is preferred over fake motion |
| Hazard string drift vs YAML keys | Exact/suffix rules only; no prose matching; tests pin vocabulary |
| Future phi merge temptation | Hard ban + rejection of self-state bridge in this spec |

---

## Recommended next patch

Implement per this design (Approach: full `signal_tension`-family unify for chat pressure math, direct adapter). Start with failing tests for empty-repo omit + ordering + map-driven pressure movement, then compiler → tension → reducer → stance wiring → docs.

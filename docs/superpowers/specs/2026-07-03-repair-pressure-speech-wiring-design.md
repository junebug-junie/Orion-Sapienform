# Repair Pressure → Chat Speech Wiring (v1)

**Status:** Approved (2026-07-03)  
**Related:** `docs/plans/substrate/2026-05-23-repair-pressure-v1.md`, `docs/superpowers/specs/2026-06-26-orion-relational-stance-design.md`

---

## Problem

Hub runs `run_substrate_effect_pipeline` on every chat turn and computes `contract_after` via `apply_repair_pressure_contract`. Today:

- The Substrate Effect chip shows `changed_behavior` when repair pressure crosses thresholds.
- `contract_after` is **discarded** before the cortex call.
- `compile_speech_contract` in cortex-exec has **no** repair branch — replies ignore repair modes even when the chip says `repair_concrete`.

Curiosity/endogenous paths already consume repair pressure at level ≥ 0.25. Chat replies do not.

---

## Goal

When repair pressure forces `repair_concrete` or `concrete_bias`, the **same turn's** Orion reply follows those rules via the existing TURN CONTRACT block in `chat_general.j2`.

---

## Non-goals (v1)

- Session carryforward of repair mode across turns
- Lowering appraiser thresholds or adding phrase detectors
- New bus events or schema registry entries
- Keyword patches to `chat_general.j2`
- LLM-based repair detection

---

## Architecture

```text
User message (Hub HTTP/WS)
  → run_substrate_effect_pipeline
  → contract_after = apply_repair_pressure_contract({"mode":"default"}, signal)
  → attach contract_after to CortexChatRequest.metadata.repair_pressure_contract
  → Gateway → Orchestrator → cortex-exec ctx.metadata
  → compile_speech_contract(brief, repair_contract=...)
  → ctx["speech_contract"] → chat_general.j2 TURN CONTRACT
  → speech pass
```

### Choke points (two only)

| # | Service | File | Function |
|---|---------|------|----------|
| 1 | Hub | `api_routes.py`, `websocket_handler.py` | Attach snapshot `contract_after` to outgoing `CortexChatRequest.metadata` when behavior changed |
| 2 | Cortex-exec | `chat_stance.py`, `executor.py` | Extend `compile_speech_contract`; read metadata in executor before speech compile |

No changes to `enforce_chat_stance_quality`, stance LLM prompts, or appraiser logic.

---

## Precedence (relational vs repair)

| Mode | Condition | Speech behavior |
|------|-----------|-----------------|
| `repair_concrete` | level ≥ 0.75 AND confidence ≥ 0.60 | **Repair wins** — replace regime contract with repair rules prose |
| `concrete_bias` | 0.45 ≤ level < 0.75 | **Blend** — regime contract + appended bias rules |
| `default` | otherwise | Regime only (unchanged) |

Rationale: HIGH repair is an explicit correction signal; MEDIUM nudges without killing companion tone entirely.

---

## Metadata contract

**Key:** `repair_pressure_contract` (constant `REPAIR_PRESSURE_CONTRACT_METADATA_KEY` in `orion/substrate/appraisal/contract.py`)

**Shape:** Reuse `contract_after` dict from pipeline — no new taxonomy:

```json
{
  "mode": "repair_concrete",
  "rules": ["no broad architecture wandering", "..."],
  "repair_pressure": {
    "level": 0.86,
    "confidence": 0.82,
    "mode_applied": "repair_concrete",
    "evidence_kinds": ["specificity_demand", "trust_rupture"],
    "causal_molecule_ids": ["..."],
    "source_event_id": "..."
  }
}
```

**When attached:** Only when `contract_before.mode != contract_after.mode` (matches `substrate_effect_summary.changed_behavior == true`). Fail-open: pipeline failure → key absent → speech unchanged.

**Per-turn baseline:** `contract_before={"mode": "default"}` always in v1.

---

## Speech compiler

Extend `compile_speech_contract(brief, *, repair_contract: dict | None = None) -> str`:

1. Compute regime contract (existing logic — instrumental / relational / minimal).
2. If `repair_contract.mode == "repair_concrete"` → return repair overlay only.
3. If `repair_contract.mode == "concrete_bias"` → return `regime + " " + overlay`.
4. Else → regime only.

Overlay prose is deterministic Python from `rules` list — no LLM. Rule strings originate from `apply_repair_pressure_contract` (single source of truth in `contract.py`).

---

## Feature flag

| Env key | Service | Default |
|---------|---------|---------|
| `ENABLE_REPAIR_PRESSURE_SPEECH_WIRING` | Hub | `true` |
| `ENABLE_REPAIR_PRESSURE_SPEECH_WIRING` | Cortex-exec | `true` |

When `false`: Hub does not attach metadata; cortex ignores key. Rollback without redeploying appraiser.

---

## Error handling

- Pipeline exception → no metadata → speech unchanged (existing fail-open).
- Invalid/missing metadata → ignore, regime only.
- Flag off → no attach, no read.

---

## Testing

| Test | Proves |
|------|--------|
| Hub: high-pressure fixture → `metadata.repair_pressure_contract.mode == "repair_concrete"` | Hub seam |
| Hub: benign turn → key absent | No false positives |
| `compile_speech_contract`: repair_concrete overrides relational brief | Precedence |
| `compile_speech_contract`: concrete_bias appends to instrumental | Blend |
| Executor unit: metadata → speech_contract contains rule substring | End-to-end compile path |

High-pressure fixture text (from spec §17 / existing tests):

```text
you gave me garbage directions — stop, build me a design spec for claude,
arsonist pov only, nuts and bolts
```

---

## Files touched

| File | Change |
|------|--------|
| `orion/substrate/appraisal/contract.py` | Export metadata key constant |
| `services/orion-hub/scripts/repair_pressure_wiring.py` | **New** — attach helper |
| `services/orion-hub/scripts/api_routes.py` | Capture snapshot; attach before cortex |
| `services/orion-hub/scripts/websocket_handler.py` | Attach after pipeline, before cortex call |
| `services/orion-hub/app/settings.py` | Flag |
| `services/orion-hub/.env_example` | Flag |
| `services/orion-cortex-exec/app/chat_stance.py` | Extend compiler |
| `services/orion-cortex-exec/app/executor.py` | Read metadata; pass to compiler |
| `services/orion-cortex-exec/app/settings.py` | Flag |
| `services/orion-cortex-exec/.env_example` | Flag |
| Hub + cortex-exec tests | Regression coverage |

---

## Acceptance checks

1. High-pressure test utterance: Substrate Effect chip `changed_behavior=true` **and** debug shows repair rules in `speech_contract`.
2. Casual turn ("how are things"): chip unchanged, no repair overlay in speech contract.
3. Relational stance brief + HIGH repair: repair rules win over companion closers.
4. Flag off: behavior identical to pre-patch.

---

## Outcome moved

Repair pressure appraisal becomes **behaviorally inspectable** in chat replies — not only in substrate observability UI and curiosity seeds.

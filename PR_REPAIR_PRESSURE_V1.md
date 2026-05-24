# repair_pressure_v1 — first real substrate-derived organ signal

## Summary

Build one vertical proving loop:

```
raw chat turn
  → substrate observation molecule
  → repair evidence (deterministic phrase match)
  → repair pressure appraisal (explicit formula)
  → OrionSignalV1 graph_cognition/repair_pressure
  → response contract mode = repair_concrete
  → inspectable causal chain
```

If the chain does not change the next response, it is not cognition. It is journaling. This plan makes the chain change the next response, end to end, with tests.

## Arsonist constraints respected

- **No new substrate gradients.** `salience, contradiction, novelty, coherence` only.
- **No new schema-kernel atoms.** All 12 stay.
- **No new substrate molecule kinds.** Chat turn uses the existing `observation` kind via `orion.mind.substrate_emit.emit_observation`.
- **Strict causal layering.** Evidence is a separate pydantic model, not a molecule; appraisal is a separate pydantic model, not a signal; signal is `OrionSignalV1`; contract is a dict.
- **Payload is not the machine contract.** Consumers read `OrionSignalV1.dimensions` and the `evidence_kinds[*]` debug array — not phrase spans or feature dicts.
- **Fail closed.** Empty / single-weak evidence forces `level=0.0` (empty) or caps `confidence ≤ 0.45` (single weak), with explicit notes.

## What's inside (planned)

### `orion/substrate/appraisal/` (new package)

| File | Responsibility |
|---|---|
| `models.py` | `RepairEvidenceV1`, `RepairPressureAppraisalV1` (pydantic, `extra="forbid"`). |
| `evidence.py` | Deterministic phrase-match detector covering 7 evidence kinds. |
| `repair_pressure.py` | Explicit-formula reducer (spec §9.3 / §9.4). |
| `signal_bridge.py` | `repair_appraisal_to_signal` → `OrionSignalV1(graph_cognition/repair_pressure)`. |
| `windowing.py` | `select_recent_chat_molecules` — source_id filter, age cap, count cap. |
| `contract.py` | `apply_repair_pressure_contract` — pure function flipping response mode. |

### `orion/signals/registry.py` (additive edit)

`graph_cognition` organ entry gains:

- `signal_kinds += ["repair_pressure"]`
- `canonical_dimensions += ["level", "specificity_demand", "trust_rupture", "coherence_gap", "repetition_failure", "operational_block", "explicit_repair_command", "assistant_accountability_demand"]`

No causal-parent edges change. No other organ touched.

### Tests

| Test file | Coverage |
|---|---|
| `tests/test_repair_pressure_models.py` | Pydantic shape, `extra="forbid"`, bounds. |
| `tests/test_repair_pressure_evidence.py` | All 6 spec §14.1 phrase thresholds, span auditing, neutral-text empty-result. |
| `tests/test_repair_pressure_appraisal.py` | High / low / fail-closed / single-weak / dimensions present. |
| `tests/test_repair_pressure_signal_bridge.py` | Registry shape + bridge organ/kind/dimensions/causal_parents/deterministic id. |
| `tests/test_repair_pressure_windowing.py` | source_id filter, age cap, count cap, sort order. |
| `tests/test_repair_pressure_behavior_contract.py` | repair_concrete / concrete_bias / no-op / low-confidence-guard / debug metadata / base immutability. |
| `tests/test_repair_pressure_e2e.py` | Full causal chain (spec §17 Definition of Done). |

## Acceptance audit

All 12 spec §15 criteria are enforced by tests or by structural absence-of-diff against the architecture-locked files (`atom.py`, `gradient.py`, `registry.py`, `molecules.py`). See the audit table in `docs/plans/substrate/2026-05-23-repair-pressure-v1.md`, Task 9.

## What this plan deliberately avoids

- No LLM scoring. No embeddings. No GraphDB. No new substrate fields.
- No chat pipeline rewrite. The contract consumer is a pure function callers opt into.
- No service extraction. Phase 5 (`services/orion-substrate-appraiser`) stays unbuilt until the library proves useful.
- No emotion classifier. This is not sentiment.

## Files to be touched during implementation

```
orion/signals/registry.py                              (additive edit)
orion/substrate/appraisal/__init__.py                  (new)
orion/substrate/appraisal/models.py                    (new)
orion/substrate/appraisal/evidence.py                  (new)
orion/substrate/appraisal/repair_pressure.py           (new)
orion/substrate/appraisal/signal_bridge.py             (new)
orion/substrate/appraisal/windowing.py                 (new)
orion/substrate/appraisal/contract.py                  (new)
tests/test_repair_pressure_models.py                   (new)
tests/test_repair_pressure_evidence.py                 (new)
tests/test_repair_pressure_appraisal.py                (new)
tests/test_repair_pressure_signal_bridge.py            (new)
tests/test_repair_pressure_windowing.py                (new)
tests/test_repair_pressure_behavior_contract.py        (new)
tests/test_repair_pressure_e2e.py                      (new)
```

## What's in this PR right now

Only the implementation plan markdown at `docs/plans/substrate/2026-05-23-repair-pressure-v1.md`. The plan is structured for `superpowers:subagent-driven-development` or `superpowers:executing-plans` — each task is bite-sized TDD with exact code in every step. Calibration math (per-kind score formula, phrase weights, level formula) was checked against spec §14.1 and §14.2 thresholds during plan authorship.

## Test plan (for the implementation PR that follows)

- [ ] `pytest tests/test_repair_pressure_models.py -v` passes
- [ ] `pytest tests/test_repair_pressure_evidence.py -v` passes
- [ ] `pytest tests/test_repair_pressure_appraisal.py -v` passes
- [ ] `pytest tests/test_repair_pressure_signal_bridge.py -v` passes
- [ ] `pytest tests/test_repair_pressure_windowing.py -v` passes
- [ ] `pytest tests/test_repair_pressure_behavior_contract.py -v` passes
- [ ] `pytest tests/test_repair_pressure_e2e.py -v` passes
- [ ] `git diff main -- orion/schema_kernel/atom.py orion/schema_kernel/gradient.py orion/schema_kernel/registry.py orion/substrate/molecules.py` is empty
- [ ] Full repo `pytest tests/ -x -q` introduces no new failures

🤖 Generated with [Claude Code](https://claude.com/claude-code)

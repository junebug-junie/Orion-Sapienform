# Orion Vision Council

Consumes visual window summaries from the bus, calls the LLM gateway for scene interpretation, enforces evidence grounding, and publishes structured vision events.

## V2 pipeline

```
VisionWindowPayload → VisionSceneInterpretationV1 → enforce_evidence_grounding → VisionEventPayload
```

1. **Intake** — `VisionWindowPayload` arrives on the council intake channel (or via RPC request).
2. **Interpretation** — `build_interpretation_prompt` shapes the LLM prompt (includes `summary.evidence` rules); the response is parsed into `VisionSceneInterpretationV1` via strict validation, salvage/coercion, then legacy fallback.
3. **Grounding** — `CouncilService._finalize_interpretation()` calls `enforce_evidence_grounding()` on **both** intake and RPC paths:
   - Drop person/activity claims when `person` ∉ `summary.evidence.hard_labels`
   - Cap activity confidence when only captions support the claim
   - On parse failure + `edge_person_hits > 0`: deterministic `person_presence` fallback (no YouTube hallucination)
4. **Projection** — `project_interpretation_to_events` maps grounded `event_candidates` to `VisionEventBundleItem` entries.
5. **Publish** — the event bundle is published on `orion:vision:events` (and returned on RPC reply when applicable).

## Evidence grounding rules

| Condition | Action |
|-----------|--------|
| Narrative mentions person/activity; `person` not in `hard_labels` | Drop event candidate |
| Activity verb without hard `person` | Drop event candidate |
| Activity claim with hard `person` but caption-only support | Cap confidence at 0.4, tag `caption_inferred` |
| LLM parse fails; `edge_person_hits > 0` | Emit `person_presence` fallback (`parse_mode=edge_fallback`) |

Choke point: `services/orion-vision-council/app/evidence_grounding.py`, wired via `_finalize_interpretation()` in `main.py`.

## Debug endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness check (`{"ok": true}`) |
| `GET /debug/last-interpretation` | Most recent `VisionSceneInterpretationV1` (in-memory ring buffer) |
| `GET /debug/recent-interpretations?limit=10` | Last N interpretations (max 20); each record includes Council-local `parse_mode` and optional `salvage_warnings` |

Interpretations are retained in an in-memory ring buffer (max 20 items) for local debugging only; they are not persisted. Debug endpoints are unauthenticated — restrict network exposure in production.

## Tests

From repo root:

```bash
PYTHONPATH=services/orion-vision-council:. pytest services/orion-vision-council/tests -q
```

Key suites: `test_evidence_grounding.py`, `test_main_grounding_wiring.py` (intake/RPC parity via `_finalize_interpretation`).

Design spec: `docs/plans/vision/2026-07-02-vision-grounded-pipeline-design.md`

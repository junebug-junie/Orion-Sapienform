## Summary

Upgrades Vision Council from flat event extraction to typed scene interpretation while preserving the existing `VisionEventPayload` downstream contract.

Adds `VisionSceneInterpretationV1` and Council-side parsing/projection:

- `VisionWindowPayload` → `VisionSceneInterpretationV1`
- `event_candidates` → `VisionEventPayload`
- existing `orion:vision:events` / Scribe persistence path remains unchanged

## Changes

- Added typed vision scene interpretation schemas (`VisionSceneInterpretationV1` + 8 helper submodels)
- Registered all new schemas in `orion/schemas/registry.py`
- Added `interpretation.py` with V2 prompt builder, strict JSON parser, legacy fallbacks, and event projection
- Refactored `main.py` to run interpretation → projection pipeline with in-memory debug ring buffer
- Added 12 unit tests (parsing, projection, wrapper unwrap, hybrid `events` coercion, markdown fences, fallbacks, evidence refs, registry)
- Updated service README and `docs/vision_services.md`

## Code review fixes (post-review)

- **Hybrid legacy `events` key:** V2 payloads with `events` but no `event_candidates` now coerce via `_coerce_legacy_events_field` (previously Pydantic silently dropped `events`, yielding empty bundles)
- **`raw_model_output`:** populated on successful parse for operator inspection
- **`interpretation` wrapper:** explicit test coverage for `{"interpretation": {...}}` unwrap
- **Debug endpoint note:** README documents unauthenticated local-only inspection endpoints

## Pipeline

```text
VisionWindowPayload
→ VisionSceneInterpretationV1 (internal, typed)
→ event_candidates projected to VisionEventBundleItem[]
→ VisionEventPayload on orion:vision:events
→ orion-vision-scribe (unchanged)
```

## Debug endpoints (operator inspection only)

- `GET /health`
- `GET /debug/last-interpretation`
- `GET /debug/recent-interpretations?limit=10`

## Validation

- [x] `PYTHONPATH=services/orion-vision-council:. pytest services/orion-vision-council/tests -q` — **12 passed**
- [x] `cd services/orion-vision-scribe && PYTHONPATH=.:../.. pytest tests -q` — **10 passed**
- [x] `python3 -m compileall services/orion-vision-council/app orion/schemas/vision.py orion/schemas/registry.py` — **exit 0**

## Out of scope

- Grammar projection / `orion:grammar:event` publication
- Vector persistence
- Memory card creation
- Scribe SQL/RDF changes
- Vision Host / Frame Router / Window projection changes
- New perception service
- `vision-edge` behavior changes

## Notes

- No `.env_example` changes (no new env vars).
- Malformed V2 LLM output falls back to legacy `{events: [...]}` / raw list parsing; consumer loop does not crash on parse failure.
- Empty `event_candidates` still skip publish (matches pre-V2 behavior).

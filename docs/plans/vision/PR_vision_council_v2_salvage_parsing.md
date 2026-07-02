## Summary

Hardens Vision Council V2 parsing against real LLM output shapes observed in runtime.

The Council V2 architecture is live, but strict Pydantic validation was discarding useful model output when one nested field had a non-canonical shape. This adds a salvage/coercion pass before legacy fallback.

Fixes live failure modes:

- `salient_observations` with `{event_type, narrative}` now coerces to `{observation}`
- `uncertainties: ["door", "screen"]` now coerces to uncertainty objects
- `{uncertainty_type: ...}` now maps to `{uncertainty: ...}`
- good `scene_summary` and valid `event_candidates` survive partial nested validation errors
- missing `scene_summary` is derived from window captions during salvage (not misrouted to legacy)
- event candidates can be synthesized from salient observations when missing
- top-level `event_type`/`narrative` promote to `event_candidates` when salvage runs with empty events
- flat legacy event dicts still route through legacy path (not mislabeled `salvaged_v2`)

## Changes

- Added `salvage_interpretation_dict` lenient V2 normalization pass
- Added `InterpretationParseOutcome` with parse modes: `strict_v2`, `salvaged_v2`, `legacy_events_dict`, `legacy_events_list`, `parse_failed`
- Added salvage warnings to Council debug ring buffer (`parse_mode`, `salvage_warnings`)
- Added concise parse-mode logging (`interpretation_parse`, `interpretation_strict_failed`, `interpretation_salvage_failed`)
- Added `_is_flat_legacy_event_dict` guard so flat event shapes skip salvage
- Added 23 unit tests covering live malformed output shapes and legacy regression guards

## Code review fixes

- Removed overly strict salvage acceptance gate that dropped scene-summary-only recoveries into legacy
- Added `scene_summary` to flat-legacy V2 marker set so hybrid payloads keep summary through salvage
- Added top-level event field promotion during salvage when `event_candidates` empty

## Validation

- [x] `PYTHONPATH=services/orion-vision-council:. pytest services/orion-vision-council/tests -q` — **23 passed**
- [x] `cd services/orion-vision-scribe && PYTHONPATH=.:../.. pytest tests -q` — **10 passed**
- [x] `python3 -m compileall services/orion-vision-council/app orion/schemas/vision.py orion/schemas/registry.py` — **exit 0**
- [ ] Runtime debug buffer shows real scene summaries with `parse_mode=strict_v2|salvaged_v2` (post-deploy)

## Out of scope

- Grammar projection
- Vector persistence
- Memory cards
- Scribe changes
- SQL/RDF changes
- Vision Host / Frame Router / Window changes
- New perception service
- `vision-edge` behavior changes

## Base branch

Stacks on `feat/vision-council-scene-interpretation-v1` (Council V2 scene interpretation).

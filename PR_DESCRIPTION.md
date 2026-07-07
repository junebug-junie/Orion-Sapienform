## Summary

- Restores the Hub inspect modal's **Recall** tab on the default "orion" (unified turn) path, which regressed to empty.
- Root cause was a plumbing gap, not missing recall: recall runs via PCR phase-3 inside cortex-exec's `stance_react` step and its content already rides to the harness governor as `request.thought_event.grounding_capsule`, but `HarnessRunV1` had no field to carry it out and the unified `final` WS frame never emitted it.
- Adds `recall_debug`/`memory_digest` to `HarnessRunV1`, populates them in the governor from the grounding capsule, and emits them on the unified `final` frame (the browser already reads them).
- No new recall wiring — surfaces already-available data (Approach A).

## Outcome moved

Unified-turn replies now carry recalled-memory content to the Hub inspect modal Recall tab, so the operator no longer has to vibe the correlation ID to see what memory grounded a turn.

## Current architecture

Default "orion" mode runs the unified turn (`orion/hub/turn_orchestrator.py` -> harness governor -> `HarnessRunner` FCC motor + finalize chain). Recall executes in cortex-exec's stance step (PCR phase-3) and is threaded onto `thought.grounding_capsule` (committed to `main` in `d096a6e8`). The `read_recall` permission on the harness request is dead code; the FCC motor (`fcc_motor.py`, a `claude` subprocess) emits no recall. Previously `HarnessRunV1` dropped the capsule's recalled content, so the modal's Recall tab was empty on this path (the classic chat lane still populated it via `PlanExecutionResult.recall_debug`).

## Architecture touched

- Shared schema: `HarnessRunV1` (additive optional fields).
- Service `orion-harness-governor`: populates recall fields at all four run-construction sites.
- `orion/hub` unified turn: emits recall fields on the `final` frame. No JS change (browser already reads `d.recall_debug`/`d.memory_digest`).

## Files changed

- `orion/schemas/harness_finalize.py`: add `recall_debug: dict|None` + `memory_digest: str|None` to `HarnessRunV1` (default None, backward compatible).
- `services/orion-harness-governor/app/bus_listener.py`: `_recall_fields_from_thought()` derives recall fields from `thought.grounding_capsule` (returns `(None, None)` when no capsule or no real content — no-empty-shell); wired into refusal/motor-fail/success/error `HarnessRunV1(...)` sites; explicit `request` sentinel guard on the error path.
- `orion/hub/turn_orchestrator.py`: `_success_frames` conditionally emits `recall_debug`/`memory_digest` on the `final` frame.
- `services/orion-harness-governor/tests/test_harness_governor_rpc.py`: recall coverage across success/refusal/motor-fail/no-capsule + both error-path branches.
- `tests/test_unified_turn_schemas.py`: schema default-None + round-trip.
- `services/orion-hub/tests/test_turn_orchestrator_ws_frames.py`: final-frame present/absent emission.

## Schema / bus / API changes

- Added: `HarnessRunV1.recall_debug: dict|None`, `HarnessRunV1.memory_digest: str|None`.
- Removed / Renamed: none.
- Behavior changed: unified `final` WS frame gains optional `recall_debug`/`memory_digest` keys (only when populated).
- Compatibility: fully backward compatible; older payloads/consumers unaffected; new frame keys appear only when recall content exists.

## Env/config changes

None. No `.env_example` change, nothing to sync.

## Tests run

```text
pytest services/orion-harness-governor/tests/test_harness_governor_rpc.py \
       services/orion-hub/tests/test_turn_orchestrator_ws_frames.py \
       tests/test_unified_turn_schemas.py -q
-> 53 passed
```

## Review findings fixed

- Finding: error-path recall population untested (only success path covered).
  - Fix: added `_handle_bus_message` tests locking in both guard branches (capsule-present raise -> recall carried; validation-fail -> recall None).
  - Evidence: `test_handle_bus_message_error_path_carries_recall_from_capsule`, `test_handle_bus_message_error_path_recall_none_when_request_unbound`.
- Finding: opaque `locals().get("request")` idiom on the error path.
  - Fix: replaced with explicit `request: HarnessRunRequestV1 | None = None` sentinel.
  - Evidence: `services/orion-harness-governor/app/bus_listener.py` except-handler.

## Restart required

```bash
sudo docker compose --env-file .env --env-file services/orion-harness-governor/.env -f services/orion-harness-governor/docker-compose.yml up -d --build
sudo docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low. Recall shows only when PCR ran (capsule has digests); an identity-only degraded capsule leaves the tab empty (accurate, not a bug).
- Follow-up (Approach B): the rich per-item `recall_debug` (classic-lane object) is still discarded on the unified path — cortex-exec throws away PCR's `_debug`. Surfacing it would need a thin cortex-exec + orion-thought change.

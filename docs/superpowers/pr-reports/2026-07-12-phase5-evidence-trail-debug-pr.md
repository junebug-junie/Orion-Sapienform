# PR: Phase 5 — self-state evidence-trail debug endpoint + schema-drift fix

## Summary

- Per `docs/superpowers/plans/2026-07-12-self-state-mesh-substrate-redesign.md`'s Phase 5 goal ("a human, or Orion, can pull one self-state tick's full evidence trail, node-attributed, through the debug surface"): two new endpoints on `orion-hub`'s `substrate_self_state_routes.py` joining a self-state row's per-dimension evidence with the raw upstream field-state data (`node_vectors`, `capability_vectors`, `capability_provenance`) it was built from.
- Also fixes the same schema-drift bug class from the prior incident PR (`docs/superpowers/pr-reports/2026-07-12-substrate-schema-drift-incident-pr.md`), found while reading this file to scope the new endpoints: `_load_latest_self_state()` had the identical unguarded `model_validate()` that crashed 4 other services in that incident.

## Outcome moved

The existing `/latest` endpoint only ever returned the already-summarized top-3-per-dimension `dominant_evidence` strings. The new endpoints expose the full raw field data a tick's evidence was built from, closing the redesign's stated observability goal.

## Files changed

- `services/orion-hub/scripts/substrate_self_state_routes.py`:
  - `_load_latest_self_state()`: now catches `ValidationError`, degrades to `None` (existing 404 path), instead of 500ing on a legacy row
  - New `_load_self_state_by_id(self_state_id)`, `_load_field_state_for_tick(tick_id)` — same graceful-degradation pattern
  - New `_build_evidence_trail(state)` — joins a `SelfStateV1` with its source `FieldStateV1` (by `source_field_tick_id`)
  - New routes: `GET /api/substrate/self-state/latest/evidence-trail` and `GET /api/substrate/self-state/{self_state_id}/evidence-trail` — the literal `/latest/evidence-trail` route is registered *before* the parameterized `{self_state_id}` route so FastAPI's registration-order matching doesn't let the parameterized route swallow the literal `"latest"` segment
- `services/orion-hub/tests/test_substrate_self_state_debug_api.py`: new tests for the schema-drift fix (reusing the exact legacy-payload fixture shape from the incident PR) and both new endpoints, including graceful degradation when the self-state exists but its field-state row doesn't (or vice versa)

## Response shape

```json
{
  "self_state_id": "...",
  "source_field_tick_id": "...",
  "self_state": { /* full SelfStateV1 dump */ },
  "field_state_available": true,
  "field_state": {
    "tick_id": "...", "generated_at": "...",
    "node_vectors": {...}, "capability_vectors": {...}, "capability_provenance": {...}
  }
}
```

Chosen as a flat join rather than a per-dimension-merged shape: the channel→dimension mapping logic already lives in `orion/self_state/builder.py`, tied to policy config — reimplementing it here would be an unnecessary abstraction for a debug endpoint. `field_state_available: false` / `field_state: null` covers the missing-or-incompatible case without failing the whole request.

## Schema / bus / API changes

- Added: 2 new debug HTTP routes (read-only, no auth beyond whatever the router already has, no new bus/schema contract)
- Behavior changed: `/latest` no longer 500s on a schema-incompatible legacy row (now 404s, same as "not found")

## Env/config changes

None.

## Tests run

```text
pytest services/orion-hub/tests/test_substrate_self_state_debug_api.py -q
→ 8 passed
```

Verified directly by the orchestrator on the actual committed branch (not just the implementing agent's report). `git diff --check`: clean.

## Evals run

None applicable — debug API only.

## Docker/build/smoke checks

Not run this session (no live rebuild). Restart command below for when this merges.

## Review findings fixed

The implementing agent ran the code-review skill against its own staged diff (excluding the parallel Phase 4 agent's untouched files) — no blocking findings. One low-severity nit noted and left as-is: each evidence-trail request opens two separate `_engine()` connections instead of sharing one — a pre-existing per-call pattern in this file already, not worth blocking a low-traffic debug endpoint on.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
  Concern: two DB round-trips per evidence-trail request (self-state lookup, then field-state lookup) instead of a single joined query.
  Mitigation: acceptable for a low-traffic debug endpoint; not fixed, flagged only.

## PR link

<!-- filled in after push -->

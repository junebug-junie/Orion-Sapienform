# PR: /projections/chat_session and /projections/route_arbitration debug endpoints

Branch: `feat/substrate-projection-debug-endpoints` → `main`

## Summary

- Adds `GET /projections/chat_session` and `GET /projections/route_arbitration` to `orion-substrate-runtime`, mirroring the existing `GET /projections/execution_trajectory` endpoint's exact response contract.
- Documents all three endpoints in the service README — `execution_trajectory`'s was never documented either, fixed while touching this section.

This is the smaller, unambiguous half of the "is there a consumer/UI surface for route_grammar and stance_disposition" investigation from earlier today. The bigger half (whether/how `stance_disposition` should ever compose into `SelfStateV1`) was deliberately left undecided — see `docs/superpowers/specs/2026-07-13-stance-disposition-inner-state-path.md` — and this patch doesn't touch that question at all. It only makes projections that already exist and already work inspectable over HTTP, the same way `execution_trajectory` already was.

## Outcome moved

`chat_session` and `route_arbitration` projections go from "only inspectable via direct Postgres query" to "one curl away," matching the existing precedent. No composition decision required to ship this — safe regardless of which path the stance-disposition doc's three candidates eventually take.

## Current architecture

`services/orion-substrate-runtime/app/main.py` had one projection read endpoint (`execution_trajectory`, since `0c5e2a23`). `chat_session` (chat_grammar reducer) and `route_arbitration` (route_grammar reducer) have both been running correctly in production since earlier today with zero HTTP visibility.

## Architecture touched

`orion-substrate-runtime` only — `app/main.py` (two new handlers) and its README.

## Files changed

- `services/orion-substrate-runtime/app/main.py` — two new `@app.get` handlers, byte-for-byte identical contract shape to `execution_trajectory`. No changes to `store.py`/`worker.py`/reducers — both store methods (`load_chat_session_projection`, `load_route_arbitration`) and projection-id constants already existed, already used internally by `_chat_tick`/`_route_tick`. Pure wiring.
- `services/orion-substrate-runtime/tests/test_chat_session_route_arbitration_endpoints.py` (new) — 4 tests (projection-present / no-projection × 2 endpoints), mirroring the existing `test_execution_trajectory_endpoint.py`'s `httpx.AsyncClient` + `ASGITransport` + mocked-store pattern exactly.
- `services/orion-substrate-runtime/README.md` — new "Projection debug reads" subsection listing all three endpoints (execution_trajectory's was undocumented before this patch too).

## Schema / bus / API changes

- Added: `GET /projections/chat_session`, `GET /projections/route_arbitration` (internal debug endpoints, no auth, matching `execution_trajectory`'s existing posture — not exposed publicly).
- Compatibility notes: purely additive, no existing endpoint's behavior changed.

## Env/config changes

None.

## Tests run

```
cd /mnt/scripts/Orion-Sapienform-projection-debug-endpoints
/tmp/orion-test-venv/bin/python -m pytest services/orion-substrate-runtime/tests -k "projection or main or execution_trajectory" --ignore=services/orion-substrate-runtime/tests/test_grammar_consumer_integration.py -q
→ 7 passed (4 new + 3 pre-existing execution_trajectory tests)
```
`test_grammar_consumer_integration.py` excluded — pre-existing `ModuleNotFoundError: No module named 'app.models'` collection failure, confirmed identical on unmodified `main` via `git stash`, unrelated to this patch.

## Evals run

Not applicable — dev-visibility endpoints only, no cognition/model component.

## Docker/build/smoke checks

Not deployed as part of this PR. No env/compose changes to validate.

## Review findings fixed

A focused review pass (correctness, import style, response-shape drift vs. `execution_trajectory`, test correctness, file-scope compliance) found nothing — clean diff.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```
Verification after restart:
```bash
curl -fsS http://localhost:8115/projections/chat_session | python3 -m json.tool
curl -fsS http://localhost:8115/projections/route_arbitration | python3 -m json.tool
```
Both should return `{"ok": true, "projection": {...}}` given both reducers are already live and have processed real traffic.

## Risks / concerns

None identified. Purely additive read-only endpoints over data that's already being written; no new write path, no new dependency, no config surface.

## PR link

Push and open via: `git push -u origin feat/substrate-projection-debug-endpoints`, then open the compare URL GitHub prints (no `gh` auth in this environment).

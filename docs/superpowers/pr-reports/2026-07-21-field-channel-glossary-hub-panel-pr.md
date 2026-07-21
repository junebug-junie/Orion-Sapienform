# Field Channel Glossary Hub panel -- PR report

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1220
Branch: `feat/field-channel-glossary-hub-panel`

## Summary

- New Hub tab ("Field Channels") lists field-digester's 29 raw `FieldStateV1` channels (23 node + 8 capability, 2 overlapping) with a **live-computed** clean/dead verdict, not the field-digester README's frozen prose verdicts -- those had already gone stale once as fixes landed (agent-board note flagged this before this branch started).
- Structured channel metadata now lives at `config/field/field_channel_glossary.v1.yaml`, the machine-readable companion to the README's prose "Field channel glossary" section.
- Liveness classifier (`orion/self_state/field_channel_glossary.py`) generalizes `scripts/analysis/measure_capability_channel_health.py`'s already-validated heuristic (1e-100 subnormal cutoff, "live" if max-median > 0.05) from that script's 8 capability channels to all 29.
- New read-only API `services/orion-hub/scripts/field_channel_glossary_routes.py` (`GET /channels`, `GET /health?hours=1|6|24`), built on the existing `collect_field_channel_pressures()` merge so polarity/merge logic isn't re-derived.
- Tab wired into `index.html`/`app.js`/`api_routes.py` following the existing Substrate Lattice tab pattern exactly.

## Outcome moved

Operators can now see, per channel: is it alive, quiet-but-wired, dead, never-produced, or a suspected one-way ratchet -- computed fresh from `substrate_field_state` on every load, closing the specific staleness gap the field-digester README's hand-written verdicts already hit once.

## Current architecture

`services/orion-field-digester`'s README documented all 29 channels in prose only, with a verdict column that's a point-in-time snapshot. `services/orion-hub/scripts/substrate_field_routes.py` already exposed a single-tick `FieldStateV1` snapshot API but nothing that classified liveness over time or listed the channel glossary. Hub had three prior tabs (Drives Analytics, Pressure Analytics, Substrate Lattice) using an identical static-HTML-iframe pattern this branch follows.

## Architecture touched

- `config/field/` (new glossary config)
- `orion/self_state/` (new pure classifier module)
- `services/orion-hub/scripts/api_routes.py` (router registration)
- `services/orion-hub/templates/index.html`, `static/js/app.js` (tab wiring)
- `services/orion-hub/static/` (new self-contained panel page)
- `services/orion-field-digester/README.md` (cross-reference note)

## Files changed

- `config/field/field_channel_glossary.v1.yaml`: new structured 29-channel index
- `orion/self_state/field_channel_glossary.py`: new liveness classifier + glossary loader
- `services/orion-hub/scripts/field_channel_glossary_routes.py`: new FastAPI router
- `services/orion-hub/static/field-channel-glossary.html`: new self-contained panel page
- `services/orion-hub/scripts/api_routes.py`: registers the new router
- `services/orion-hub/templates/index.html`, `services/orion-hub/static/js/app.js`: new tab nav/panel/hash-routing, copied from the Substrate Lattice pattern
- `services/orion-field-digester/README.md`: cross-reference note pointing at the new live panel
- `tests/test_field_channel_glossary.py`, `services/orion-hub/tests/test_field_channel_glossary_routes.py`, `services/orion-hub/tests/test_field_channel_glossary_hub_tab.py`: new tests

## Schema / bus / API changes

- Added: `GET /api/field-channel-glossary/channels`, `GET /api/field-channel-glossary/health`
- Removed: none
- Renamed: none
- Behavior changed: none (new read-only surface only)
- Compatibility notes: no bus/schema contract changes; reads existing `substrate_field_state` table and existing `FieldStateV1`/`collect_field_channel_pressures()`

## Env/config changes

- Added keys: none (reuses existing `POSTGRES_URI`)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: not applicable, no new keys
- local `.env` synced: not applicable, no new keys
- skipped keys requiring operator action: none

## Tests run

```text
# Pure-logic + glossary tests (repo-root venv)
/tmp/orion-test-venv/bin/python -m pytest tests/test_field_channel_glossary.py tests/test_self_state_scoring.py -q
26 passed

# Hub route + tab-wiring tests (hub venv)
/tmp/orion-hub-test-venv/bin/python -m pytest tests/test_field_channel_glossary_routes.py tests/test_field_channel_glossary_hub_tab.py tests/test_pressure_analytics_hub_tab.py tests/test_substrate_lattice_hub_tab.py tests/test_substrate_field_debug_api.py -q
25 passed
```

Full `services/orion-hub/tests` suite was also run for regression-check purposes: 85-115 pre-existing failures (identical root cause -- missing `rdflib` in the test venv -- confirmed reproduced on unmodified `main` too, not introduced by this branch).

## Evals run

No eval harness exists for this observability surface; none added. This is a read-only debug/observability panel, not a cognition-affecting change, so no eval gap is being carried silently -- flagging explicitly per repo convention.

## Docker/build/smoke checks

Not run -- no Docker/runtime config changed (no new env keys, no compose changes, no dependency changes: PyYAML already in `services/orion-hub/requirements.txt`). Route dispatch was verified via a live `TestClient` request through the full `api_routes.router` aggregation (`GET /api/field-channel-glossary/channels` -> 200, 29 channels), since a newer FastAPI/Starlette version wraps included sub-routers as `_IncludedRouter` objects that don't show `.path` under naive `.routes` iteration -- confirmed this is a `.routes` inspection quirk, not a registration bug.

## Review findings fixed

- Finding: `/health` used `ORDER BY generated_at ASC LIMIT :row_cap`, so windows large enough to hit the row cap (hours=6/24 at ~1.8k rows/hour) silently classified from the *oldest* slice of the window instead of the newest -- the exact staleness failure mode this feature exists to replace.
  - Fix: `ORDER BY generated_at DESC LIMIT :row_cap`, then reverse in Python before building series.
  - Evidence: new test `test_health_endpoint_reverses_desc_rows_back_to_chronological_order`.
- Finding: 5 channels (`expected_offline_suppression`, `transport_pressure`, `contract_pressure`, `catalog_drift_pressure`, `observer_failure_pressure`) that are genuinely wired but read exactly `0.0` this tick get dropped by `collect_field_channel_pressures()`'s merge gate, and were misclassified `never_produced` (implying broken wiring) instead of `dead` (implying no current signal).
  - Fix: `build_channel_series()` now falls back to checking raw `node_vectors`/`capability_vectors` keys directly when a channel is missing from the merge, recording `0.0` if the key is structurally present; only a channel absent from every row's raw vectors all window is `never_produced`.
  - Evidence: new tests `test_build_channel_series_quiet_zero_channel_is_dead_not_never_produced`, `test_build_channel_series_genuinely_never_produced_when_key_absent_everywhere`.
- Finding: `ratchet_suspect` had no minimum-sample guard -- a 2-point up-step (a coin flip for any noisy-but-healthy channel) could false-positive as a suspected one-way ratchet.
  - Fix: added `RATCHET_MIN_SAMPLES = 4` guard.
  - Evidence: new tests `test_classify_two_point_up_step_is_not_ratchet_suspect`, `test_classify_ratchet_suspect_requires_minimum_sample_count`.
- Finding: malformed/unparsable historical rows were silently discarded with no diagnostic, so "every channel dead" and "100% of rows failed to parse" looked identical in the response.
  - Fix: added `unparsable_count` to the `/health` response and the static page's window-meta line.
  - Evidence: new test `test_build_channel_series_counts_unparsable_rows_separately_from_dead`.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build orion-hub
```

## Risks / concerns

- Severity: low
- Concern: `services/orion-hub/tests` has a pre-existing, wide `rdflib`-missing gap in the sandbox test venv (`/tmp/orion-hub-test-venv`) unrelated to this change; confirmed identical on unmodified `main`.
- Mitigation: none needed for this PR; separate venv/dependency fix, out of scope here.

- Severity: low
- Concern: `graphify update .` hit the known pre-existing destructive-update bug (2026-07-14 incident) during this session; the `safe_graphify_update.sh` wrapper caught it and auto-restored `graph.json`/`manifest.json`, nothing was committed. A stray regenerated `GRAPH_REPORT.md`/`graph.html` left over from the refused run were reverted/removed before committing so this branch's diff stays scoped to the feature.
- Mitigation: none needed; underlying graphify bug is a separate, already-tracked issue.

## Status

DONE

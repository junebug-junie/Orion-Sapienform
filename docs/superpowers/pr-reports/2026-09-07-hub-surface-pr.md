# PR report: Hub Surface — the decision-point dashboard for the attention/goal-bridge/durable-runs arc

## Summary

- Everything the attention-schema-surface / goal-bridge / durable-runs arc shipped was real, live, and
  invisible unless someone ran SQL by hand. This adds a Hub-hosted page (`/hub-surface`) that makes it legible.
- Not an activity monitor: two **decision points** the arc introduced, each showing what's possible, what's
  live, what's structurally off the table, and what actually happened — "does the goal producer's pick
  actually win Orion's attention?" and "does a long investigation survive to actually finish?"
- A small, honestly-labeled "over time" section (both mechanisms are under a week old — said plainly, not
  dressed up as a mature trend) plus the existing attention-schema-surface activity log, narrated in plain
  sentences.
- Two mockup rounds with Juniper (https://claude.ai/code/artifact/8e8e2dbc-1b66-4bbf-ba90-c68475e1b6d7)
  converged on this shape after the first draft (an activity feed + stats) missed the actual ask: "how are
  these new combined schemas affecting Orion's ability to change their world."

## Outcome moved

Before this: confirming whether the goal-bridge or durable-runs were actually doing anything meant a live
`psql` session against production — which is how every finding in this arc's own PR reports was produced.
After: `/hub-surface` answers it at a glance, live, for anyone with Hub access. Smoke-tested against the
real live database while building this (not a fixture): correctly surfaced that the *live* goal-bridge
`goal_matched_no_loop` share has drifted from 19.4% (an earlier read in this arc) to 28.1% since, and that
`HUB_CURIOSITY_KICKOFF_VIA_CORTEX` is currently `false` on the running Hub container — both real, current
facts nobody would otherwise have noticed without this page.

## Current architecture

Hub had no dashboard touching either `substrate_attention_schema` or `substrate_attention_self_model` or
`substrate_durable_run_state`. Standalone Hub pages follow one established pattern (`substrate.html`,
`causal_geometry.html`, `sentience_program.html`): Tailwind CDN, dark theme, a FastAPI route reading the
template file and doing a `{{HUB_UI_ASSET_VERSION}}` text-replace (no Jinja2 anywhere in this codebase),
paired with a dedicated vanilla-JS file that fetches its own small `/api/...` routes.

## Architecture touched

- New standalone page + its own API surface, following the existing pattern exactly — no new service, no new
  database table, no new bus channel.
- `voluntary_override_absent_reason`/`attention_reason` live inside `substrate_attention_self_model
  .self_model_json` (a jsonb column), not a plain SQL column — confirmed by investigation before writing any
  query, since the design doc's own language ("self-model rows") doesn't say this.

## Files changed

- `services/orion-hub/scripts/hub_surface_routes.py` (new): two routers — `/api/hub-surface/*` (5 read-only
  endpoints) and a bare `/hub-surface` page route. Pure aggregation functions (`summarize_bridge`,
  `summarize_durable_lifecycle`, `pick_example_run`) separated from I/O, matching this repo's own established
  convention (`attention_organ_routes.summarize_history`).
- `services/orion-hub/templates/hub_surface.html` (new): the page, matching `causal_geometry.html`'s shape.
- `services/orion-hub/static/js/hub-surface.js` (new): fetch + render, all free-text DB fields escaped.
- `services/orion-hub/tests/test_hub_surface_routes.py` (new): 12 tests — pure-function tests plus route-level
  tests against a fake SQLAlchemy engine (this repo's established convention from
  `test_attention_loops_reader.py`), including assertions on the actual SQL text sent, not just the
  happy-path result.
- `services/orion-hub/scripts/main.py`: two `include_router` lines.

## Schema / bus / API changes

- Added: 5 new read-only GET endpoints under `/api/hub-surface/`, one new page route `/hub-surface`.
- Removed / Renamed: none.
- Behavior changed: none to any existing path — purely additive reads.
- Compatibility notes: none needed; no existing contract touched.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- `.env_example` updated: no (nothing new to add — reuses `POSTGRES_URI` and the already-existing
  `HUB_CURIOSITY_KICKOFF_VIA_CORTEX`).
- local `.env` synced: n/a, no changes.
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-hub/tests/test_hub_surface_routes.py   12 passed
services/orion-hub/tests (full suite, this branch)    2269 passed, 49 failed, 4 skipped, 1 deselected
services/orion-hub/tests (full suite, clean main)     2260 passed, 50 failed, 4 skipped

Diffed the two failure lists directly: identical, except the one test this
branch deselected as a known-pre-existing failure (test_agent_trace_debug_panel
.py::test_memory_and_autonomy_modals_coordinate_scroll_lock_and_visibility,
confirmed failing on clean main too, unrelated app.js content assertion).
Zero regressions from this branch.
```

## Evals run

```text
No eval harness for orion-hub covers dashboard pages. In lieu of one: smoke-tested every new endpoint plus
the page itself against the REAL live Postgres (FastAPI TestClient, POSTGRES_URI pointed at the live
container's exposed port, no fixtures) -- confirmed real percentages, real run states, and correct
auto-selection of the actual most-retried run (ff8a379217d8) as the "concrete instance."
```

## Docker/build/smoke checks

```text
python scripts/check_env_template_parity.py orion-hub  ->  PASS (1 service(s) compared)
git diff --check                                        ->  clean
No container rebuild performed for this PR -- purely additive backend routes + a new page; verified live
against production Postgres via TestClient instead (see Evals run above). Restart command below for when
this is ready to go live.
```

## Review findings fixed

- Finding: The dashboard could silently go blind to a defect signal it exists to catch — the frontend only
  rendered 5 hardcoded `voluntary_override_absent_reason` values, but the real enum has 12, including
  `combiner_error` ("a defect signal, not a normal outcome" per the schema's own comment).
  - Fix: render every key the backend actually returns; defect-class values get a warning color regardless
    of magnitude.
  - Evidence: `hub-surface.js`'s `renderBridge` now derives its list from `Object.keys(branches)` union
    `Object.keys(baseline)`, not a fixed array.
- Finding: The "concrete instance" example query has no time-window filter while the stats above it do,
  so a short window could show "0 finished" next to an example run from days ago with nothing explaining
  the mismatch.
  - Fix: made the scope explicit (`example_scope: "all_time"` in the API response, matching UI copy) rather
    than forcing a window filter that would make the feature useless at short windows.
  - Evidence: `hub_surface_routes.py` comment + response field; `hub-surface.js` label now reads "all-time
    (not scoped to the window above)".
- Finding: The test fake recorded executed SQL text but no test ever asserted on it — deleting a `WHERE`
  clause or typoing a bind key would still pass every test.
  - Fix: added an assertion on the actual SQL text/params for the example-run lookup; added a real test for
    `bridge_trend()`, which had zero coverage.
  - Evidence: `test_hub_surface_routes.py`, now 12 tests (was 11), all passing.
- Finding: `/api/hub-surface/activity`'s `limit` param was the only one in this route-file family missing a
  lower bound (`ge=1`), so `limit=-1` or `0` would reach Postgres as a literal instead of a clean 422.
  - Fix: added `ge=1`.
  - Evidence: verified live — `limit=0` now returns `422` with a clear Pydantic validation message.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

If no restart is required:

```text
Restart required to make /hub-surface reachable on the live container -- everything above was verified
against live data via a local TestClient process, not the deployed container.
```

## Risks / concerns

- Severity: low
  Concern: no auth/rate-limiting on the new `/api/hub-surface/*` endpoints.
  Mitigation: confirmed consistent with every sibling read-only debug route in this file family
  (`substrate_attention_routes.py`, `attention_organ_routes.py`, ~35 routers total) — none of them have auth
  either, and Hub has no global auth middleware. Not a new gap introduced here.
- Severity: low
  Concern: the goal-bridge's kill switch (`ORION_GOAL_PROVENANCE_READS_COMPETITION`) lives on
  orion-attention-runtime, not Hub, so this page shows it as a static label, not a live-polled badge.
  Mitigation: deliberate — faking a "live" status Hub cannot actually confirm would be worse than an honest
  static label. Live cross-service kill-switch polling is a real but separate follow-up if wanted.
- Severity: informational, not a code risk
  Concern: while smoke-testing, found that `HUB_CURIOSITY_KICKOFF_VIA_CORTEX` is currently `false` on the
  live Hub container — durable runs are not being kicked off from the normal tick right now.
  Mitigation: none taken; this is an operational fact for Juniper to decide on, not something this PR should
  change unilaterally.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2137

# PR report: recent-attention cue on Hub Surface (real fix, replaces #2144)

**Date:** 2026-09-07
**Branch:** `feat/hub-surface-recent-attention`
**Program:** Sentience Striving Program (self-modeling / continuity)

## Summary

- **Reverts PR #2144's inert Cockpit wiring.** #2144 added `recent_attention`
  fields to the Cockpit HUD's `stance_inputs` hop on the wrong assumption
  that hop reflects cortex-exec's real stance-synthesis inputs. It doesn't:
  the Cockpit's `stance_inputs`/`stance_decision` hops are fed by a
  completely separate, Hub-side request to the `orion-thought` service
  (`turn_orchestrator.py`'s `stance_req = StanceReactRequestV1(...,
  stance_inputs={"user_message": user_message})` -- hardcoded, nothing else
  ever goes in it). Juniper checked the Cockpit for it and it wasn't there;
  confirmed by tracing the real data flow that it could never be there.
  Cleanly reverted (`_inject_recent_attention_to_inputs` in `chat_stance.py`,
  the `hop_from_stance_inputs` summary fields in `orion/cockpit/builders.py`,
  and their tests) rather than left as misleading dead code.
- **Real fix: a new "Live now" panel on Hub Surface** (the dashboard from
  earlier this arc, PR #2137) shows the exact ambient attention cue a real
  chat turn's internal stance prompt sees, live, refreshed on page load --
  no need to catch the Cockpit's live-turn-only window.
- **Hardening from this patch's own review**: the query and pure builder are
  now genuinely shared (one `RECENT_ATTENTION_QUERY_SQL` constant in
  `orion/substrate/recent_attention_cue.py`, imported by both cortex-exec's
  reader and this new route) instead of two hand-copied strings that could
  silently drift; the new route fails open instead of raising, since Hub
  Surface's frontend awaits every panel in one `Promise.all` and one
  unhandled 500 would have blanked the whole page.

## Outcome moved

Juniper can now see, on the same dashboard built for this arc, whether
Orion's chat stance synthesis has a fresh or stale sense of its own recent
attention -- verified against real live Postgres data during this patch
(see "Docker/build/smoke checks" below), not just asserted from tests.

## Current architecture

- `orion.substrate.recent_attention_cue.build_recent_attention_cue()` (PR
  #2141): pure function, no I/O, already shared-importable from any service.
- `services/orion-cortex-exec/app/recent_attention_reader.py` (PR #2141):
  the real consumer -- queries `substrate_attention_schema`, feeds the
  builder, lands in `chat_stance_brief.j2` via `ctx["recent_attention"]`
  (confirmed live in production logs this session: a real turn's
  `MetacogContextService` step listed `recent_attention` in its context
  keys, no fetch-failure warnings).
- `services/orion-hub/scripts/hub_surface_routes.py` (PR #2137): the
  existing Hub Surface dashboard backend -- one `_engine()`, several
  `@router.get` read routes, a `page_router` serving the page itself.
- **What #2144 got wrong**: `orion/hub/turn_orchestrator.py`'s Cockpit
  `stance_inputs` hop is built from `StanceReactRequestV1`, a request Hub
  sends to a *different* service (`orion-thought`, `ThoughtClient(bus)
  .react(stance_req)`) for a disposition judgment -- structurally unrelated
  to cortex-exec's `chat_stance.py::build_chat_stance_inputs()`, despite
  both using the phrase "stance inputs". No code path connects the two.

## Architecture touched

```
substrate_attention_schema (Postgres)
  --RECENT_ATTENTION_QUERY_SQL (shared constant, orion/substrate/recent_attention_cue.py)-->
  [cortex-exec's recent_attention_reader.py]     [Hub's hub_surface_routes.recent_attention()]
  --> build_recent_attention_cue() (same function, both callers)
  cortex-exec path --> ctx["recent_attention"] --> chat_stance_brief.j2   (real, live, unaffected by this patch)
  Hub path         --> GET /api/hub-surface/recent-attention --> hub_surface.js --> new "Live now" panel
```

## Files changed

**Revert of PR #2144 (dead code removed):**
- `services/orion-cortex-exec/app/chat_stance.py`: removed
  `_inject_recent_attention_to_inputs()` and its call site.
- `orion/cockpit/builders.py`: `hop_from_stance_inputs()` restored to its
  pre-#2144 shape.
- `orion/cockpit/tests/test_builders.py`,
  `services/orion-cortex-exec/tests/test_chat_relational_stance.py`:
  matching test removals.

**Shared-query hardening (this patch's own review finding):**
- `orion/substrate/recent_attention_cue.py`: new `RECENT_ATTENTION_QUERY_SQL`
  constant -- the query text itself is now shared, not just the row-shaping
  function, closing the "two hand-copied SQL strings" drift risk a reviewer
  flagged.
- `services/orion-cortex-exec/app/recent_attention_reader.py`: `_fetch_sync()`
  now imports and executes the shared constant instead of its own inline SQL.

**The real fix:**
- `services/orion-hub/scripts/hub_surface_routes.py`: new `GET
  /api/hub-surface/recent-attention` route. Fails open (returns a safe
  degraded cue + logs a warning) rather than raising, and discloses
  `mirrors_cortex_exec_defaults: true` in its response so a future
  fresh/stale disagreement with a real chat turn is traceable to a config
  change, not mistaken for a query bug.
- `services/orion-hub/templates/hub_surface.html`: new "Live now" card.
- `services/orion-hub/static/js/hub-surface.js`: `renderRecentAttention()`,
  wired into the existing `loadAll()` `Promise.all`.
- `services/orion-hub/tests/test_hub_surface_routes.py`: 6 new tests (fresh
  rows, stale/empty, fail-open on a DB error, the mirrored-defaults
  disclosure, shared-function identity, shared-query-text identity).

## Schema / bus / API changes

- Added: `GET /api/hub-surface/recent-attention` (Hub).
- Removed: the two `summary` fields #2144 added to the `stance_inputs`
  Cockpit hop (`recent_attention_items`, `recent_attention_stale`) -- never
  populated in production, so nothing observable regresses.
- Renamed: none.
- Behavior changed: `recent_attention_reader.py`'s executed SQL text is now
  sourced from a shared constant -- byte-identical to before, confirmed by
  the existing SQL-shape test still passing unmodified.
- Compatibility notes: fully additive on the Hub Surface side; the revert is
  a clean removal of code that never executed anything observable.

## Env/config changes

None. No new keys.

## Tests run

```text
python -m pytest orion/cockpit/tests/test_builders.py \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  orion/substrate/tests/test_recent_attention_cue.py \
  services/orion-cortex-exec/tests/test_recent_attention_reader.py \
  services/orion-cortex-exec/tests/test_recent_attention_prompt_contract.py \
  -q
=> 75 passed

# Hub's tests need cwd=services/orion-hub (scripts.* import resolution)
cd services/orion-hub && python -m pytest tests/test_hub_surface_routes.py -q
=> 18 passed (12 pre-existing + 6 new)

python scripts/check_env_template_parity.py
=> PASS (85 services compared)
```

## Evals run

No eval harness for Hub Surface or Cockpit builders (pure/unit-tested only,
matching every prior patch to these files).

## Docker/build/smoke checks

**Juniper has a live run going on Hub -- no restart was performed or is
requested by this report.** Instead, verified the new route directly against
the real, running Postgres (no Hub restart needed, since this is a plain
read against the already-live database, not a call into the Hub process):

```text
$ POSTGRES_URI=postgresql://postgres:postgres@127.0.0.1:55432/conjourney python -c \
  "from scripts.hub_surface_routes import recent_attention; print(recent_attention())"

{
  "items": [
    {"process": "substrate_attention", "narrative": "Pure bottom-up dispatch: ...the goal got what it wanted.", "age_label": "moments ago", ...},
    {"process": "substrate_attention", "narrative": "Pure bottom-up dispatch: 'None' selected...", "age_label": "moments ago", ...},
    {"process": "cortex_turn", "narrative": "already-known target should not be asked about again", "age_label": "about 1 minute ago", ...}
  ],
  "stale": false,
  "as_of": "2026-09-07T22:48:37+00:00"
}
```

Real, current, correctly-aged data -- not a mock.

## Review findings fixed

- Finding: the new route had no try/except around its DB call; Hub Surface's
  frontend awaits all six panels in one `Promise.all`, so an unhandled
  exception here would have blanked bridge/durable-runs/activity too, not
  just this panel.
  - Fix: wrapped in try/except, fails open to `{"items": [], "stale": True}`
    with a warning log, matching the underlying builder's own "never be the
    reason a chat turn fails" contract one layer up.
  - Evidence: `test_recent_attention_route_fails_open_on_db_error`.
- Finding: `RECENT_ATTENTION_LIMIT`/`RECENT_ATTENTION_STALE_AFTER_SEC` are
  hand-copied literals mirroring cortex-exec's live env-configurable
  defaults, with nothing enforcing they stay in sync.
  - Fix: not solved structurally (Hub genuinely cannot read another
    service's env without new cross-service plumbing, out of scope) --
    disclosed loudly instead: `mirrors_cortex_exec_defaults: true` is now in
    the API response itself, not just a source comment, so a future
    disagreement is traceable rather than mysterious.
  - Evidence: `test_recent_attention_route_discloses_mirrored_defaults`.
- Finding: the exact SQL query text was duplicated verbatim between this
  route and cortex-exec's reader, so a future fix to one (like the
  empty-narrative-before-`LIMIT` fix already made once) could silently apply
  to only one copy.
  - Fix: factored into `orion.substrate.recent_attention_cue
    .RECENT_ATTENTION_QUERY_SQL`, a single shared constant both readers
    execute. Not a new DB dependency in the pure module -- it's a plain
    string, still no I/O in that file.
  - Evidence: `test_recent_attention_query_sql_is_the_shared_constant`;
    cortex-exec's existing `test_query_filters_empty_narrative_before_limit`
    still passes unmodified against the refactored reader.

## Restart required

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build   # picks up the chat_stance.py / builders.py revert
scripts/safe_docker_build.sh orion-hub up -d --build           # picks up the new Hub Surface panel
```

**Do not run either now -- Juniper has a live run in progress on Hub.**
Both are safe, low-risk restarts (additive route, clean revert) whenever
that run is done; neither is urgent.

## Risks / concerns

- Severity: low
- Concern: the fail-open fix (try/except) is new only for this route; the
  same gap exists, unfixed, in this file's other four routes
  (`bridge`/`bridge_trend`/`durable_runs`/`durable_runs_trend`/`activity`).
- Mitigation: disclosed here rather than silently expanded scope to "fix" a
  pattern across a file this patch didn't otherwise touch. Worth a
  follow-up if Juniper wants Hub Surface hardened uniformly.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2146

# FCC harness route split — PR report

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1761
Branch: `feat/harness-route-split`
Date: 2026-08-20

## Summary

- Split the FCC/Claude Code CLI harness onto its own gateway route (`harness`), off `chat` — same reason `agent` was split off `chat` on 2026-08-14.
- Added `SYSTEM_LLM_ROUTES` — a new route-priority concept distinct from `background`: dispatches immediately (no slot-slack wait) but is never a human's Compute choice.
- Fixed a KeyError landmine in `scripts/smoke_llm_gateway_routes.py` that adding `harness` to the shared display order would have triggered.
- As shipped, `harness` is an interim alias of `chat`'s own worker — a labeling/observability seam, not physical isolation yet.

## Outcome moved

The gateway/admission system can now tell FCC harness turns and live Hub chat traffic apart at the route level. Today that's visibility only (same physical worker), but it's the seam any future isolation (dedicated worker, admission policy) needs to act on. Also closes a real UX bug caught in review before merge: `harness` would otherwise have shown up as an ordinary choosable lane in Hub's Compute picker.

## Current architecture

`~/.fcc/.env` sets `MODEL=llamacpp/chat`, so every FCC/Claude Code CLI harness turn (up to `HARNESS_FCC_TIMEOUT_SEC=900s`, full Bash/Docker/git/MCP access) resolves through `anthropic_passthrough.py::resolve_anthropic_route()` to the exact same route as live Hub chat traffic: `chat` → `circe-worker-1`, the `qwen36-35b-a3b-udq5km-2xv100-32gb-deep-cognition` profile, `n_parallel: 1`, zero admission/concurrency throttling (`priority_admission.py` only gates routes tagged `"background"`). `37f4fab9c` (2026-08-16) already fixed this exact class of problem for one much lighter call (the "5b reflection" background LLM call) by moving it off `chat` onto `agent`; FCC itself was never moved.

## Architecture touched

- `orion/llm/routes.py` — the shared route-name registry every route-aware service imports.
- `services/orion-llm-gateway/app/route_catalog.py` — `GET /routes` catalog, definitional-priority fallback.
- `services/orion-hub/scripts/llm_gateway_client.py` — Hub's server-side priority derivation.
- `services/orion-hub/static/js/app.js` — Hub's Compute picker filter (browser-side, mirrors the Python fail-safe).
- `scripts/smoke_llm_gateway_routes.py` — route dispatch smoke.

## Files changed

- `orion/llm/routes.py`: added `harness` to the accepted/display-order sets; added `SYSTEM_LLM_ROUTES` (mutually exclusive with `BACKGROUND_LLM_ROUTES`, validated at import time).
- `orion/llm/tests/__init__.py`, `orion/llm/tests/test_routes.py` (new): first direct test coverage for this shared module.
- `services/orion-llm-gateway/app/route_catalog.py`: `_definitional_priority()` returns `"system"` for `SYSTEM_LLM_ROUTES`.
- `services/orion-hub/scripts/llm_gateway_client.py`: `_priority_for()` mirrors the same fail-safe.
- `services/orion-hub/static/js/app.js`: new `isSystemRouteEntry()`, wired into `pickableComputeRouteIds()`.
- `scripts/smoke_llm_gateway_routes.py`: added `"harness"` to `DEFAULT_ROUTE_SERVERS`.
- `services/orion-llm-gateway/README.md`, `services/orion-fcc/README.md`: documented the split, the interim-alias caveat, and `priority: "system"`.
- `services/orion-llm-gateway/tests/test_anthropic_passthrough.py`, `test_route_catalog.py`: extended with `harness`-specific cases.

## Schema / bus / API changes

- Added: `harness` route key, `SYSTEM_LLM_ROUTES` set, `"priority": "system"` route-table value.
- Removed: none.
- Renamed: none.
- Behavior changed: `GET /routes` now reports a `harness` row; Hub's Compute picker now also excludes `priority: "system"` routes.
- Compatibility notes: `"system"` is a new, additive value for the existing `priority` field — no existing consumer breaks (only `== "background"` triggers admission-wait behavior anywhere in the dispatch path; confirmed via repo-wide grep).

## Env/config changes

- Added keys: none new, but `LLM_GATEWAY_ROUTE_TABLE_JSON` needs a `"harness"` entry, and `~/.fcc/.env`'s `MODEL`/`MODEL_SONNET`/`MODEL_OPUS` need to move from `llamacpp/chat` to `llamacpp/harness`.
- `.env_example` updated: **NOT in this PR** — see below.
- local `.env` synced: **NOT in this PR** — see below.

**Blocked by design, not an oversight:** the Edit tool's permission classifier refuses writes to `.env*`-pattern files/content in this environment — confirmed to be a real secrets guard (`~/.fcc/.env` has live plaintext API keys). Four files still need this change:

1. `services/orion-llm-gateway/.env_example` (tracked) — add `"harness":{"url":"http://100.112.254.99:8011","served_by":"circe-worker-1","backend":"llamacpp","priority":"system"}` to `LLM_GATEWAY_ROUTE_TABLE_JSON`, plus a doc comment mirroring this PR's README section.
2. `services/orion-llm-gateway/.env` (live, gitignored) — same JSON edit; needs a gateway restart.
3. `config/fcc.env_example` (tracked) — `MODEL`/`MODEL_SONNET`/`MODEL_HAIKU`/`MODEL_OPUS` default `chat` → `harness`.
4. `~/.fcc/.env` (live, host-local, real secrets) — `MODEL`/`MODEL_OPUS`/`MODEL_SONNET`: `llamacpp/chat` → `llamacpp/harness` (leave `MODEL_HAIKU` — already overridden to an external NIM model on this host).

Until applied, `harness` exists in the code/route-registry but is `not_configured` in the live route table — this PR is safe to merge/deploy with zero live behavior change until an operator applies the four diffs (staged at `/tmp/claude-1000/-mnt-scripts-Orion-Sapienform/d6049e80-71c9-4fdf-b4e5-263b79c45b71/scratchpad/harness-route-env-diffs.md` on the host this PR was authored from).

## Tests run

```text
uv run --no-project --with pytest python -m pytest orion/llm/tests/test_routes.py -q
  5 passed

uv run --no-project --with pytest --with pytest-asyncio --with-requirements requirements.txt \
  python -m pytest services/orion-llm-gateway/tests/ -q
  272 passed (full suite, not just touched files)

node --check services/orion-hub/static/js/app.js   → OK
python3 -m py_compile <every touched .py file>      → OK
```

`services/orion-hub/tests/test_llm_gateway_client_routes.py` did not run: Hub's `Settings()` requires a fully populated `.env` this fresh worktree doesn't have — pre-existing test-infra friction unrelated to this patch. The touched function (`_priority_for`) was reviewed by hand against the existing test file's patterns; the change mirrors `route_catalog.py`'s (tested) logic exactly.

The Hub-side JS picker fix (`isSystemRouteEntry`) has no automated test: `app.js` is a 13k-line monolithic browser script with no `module.exports`, and its sibling function `isBackgroundRouteEntry` (which predates this PR) has never had test coverage either — extracting either into a testable module is a real but separate, out-of-scope refactor.

## Evals run

No eval harness exists for `orion-llm-gateway` or the Hub route-picker surface. Not adding one here — out of scope for a route-registry patch.

## Docker/build/smoke checks

Not run — no runtime env available in this session, and this patch makes no live-behavior change until the `.env*` diffs above are applied. `scripts/smoke_llm_gateway_routes.py` was reviewed and patched by hand but not executed against a live gateway.

## Review findings fixed

- Finding: `harness` was invisible to Hub's picker-exclusion logic — `isBackgroundRouteEntry()` only checks `priority === 'background'`, and a route with no configured priority reports `null`, so `harness` would have appeared as an ordinary choosable lane in the Compute selector despite an in-code comment claiming it was "not human-interactive."
  - Fix: added `SYSTEM_LLM_ROUTES` (`orion/llm/routes.py`) as a distinct, mutually-exclusive-with-`background` priority concept; wired through `route_catalog.py`, `llm_gateway_client.py`, and a new `isSystemRouteEntry()` check in `app.js`.
  - Evidence: `orion/llm/tests/test_routes.py::test_harness_is_a_system_route_hidden_from_the_human_picker`, `test_route_catalog.py`'s new `by_id["harness"]["priority"] == "system"` assertion, both passing.
- Finding (self-caught during a follow-up repo sweep, same session): `scripts/smoke_llm_gateway_routes.py`'s `DEFAULT_ROUTE_SERVERS` dict had no `"harness"` entry — `routes_to_test` derives from `LLM_ROUTE_DISPLAY_ORDER` (now including `harness`), so the smoke would raise a bare `KeyError` in `_expected_served_by()` the first time it ran against a route table configuring `harness`.
  - Fix: added `"harness": "atlas-worker-1"` matching the existing `chat`/`agent`/`quick_background` pattern.
  - Evidence: `python3 -m py_compile scripts/smoke_llm_gateway_routes.py` passes; logic traced by hand against `_expected_served_by()`.

**Automated review infrastructure was unavailable this session** — three consecutive `code-review` subagent launches failed (one hit a session usage-limit mid-run, two returned "stopped, no completion record" consistent with session-level instability, not this patch). The findings above came from one partial review pass that returned one real finding before dying, plus a manual line-by-line self-review of the full diff and a repo-wide grep sweep for every other consumer of LLM route priority/display-order (confirmed no other hardcoded copies were missed).

## Restart required

```bash
# Only needed AFTER the four .env* diffs above are applied by an operator:
docker compose --env-file .env --env-file services/orion-llm-gateway/.env \
  -f services/orion-llm-gateway/docker-compose.yml up -d --build llm-gateway
```

No restart is required for this PR's code alone — `harness` stays `not_configured` until the route table is updated.

## Risks / concerns

- Severity: low
- Concern: this PR ships the `harness` route/picker mechanism but not the config that activates it — a second manual step (the four `.env*` diffs) is required before FCC actually moves off `chat`. If skipped, this PR is inert (safe) but the original chat/FCC contention problem remains live.
- Mitigation: exact diffs are prepared and handed off; `GET /routes` will visibly report `harness` as `not_configured` until applied, so the gap is observable, not silent.

- Severity: low
- Concern: even once fully configured, `harness` is an *interim alias* of `chat`'s own worker — this buys observability/admission-labeling, not physical isolation. An FCC turn can still occupy `circe-worker-1`'s one slot for up to 900s and block live chat, or vice versa.
- Mitigation: documented explicitly in-code and in both READMEs; a follow-up (dedicated worker or a real admission policy for `system`-priority routes) is the next real step, out of scope here.

- Severity: low
- Concern: no automated code review completed this session (infra instability, not patch-related) and the Hub-side JS fix has no automated test coverage (pre-existing gap, not introduced here).
- Mitigation: thorough manual self-review performed and documented above; full existing test suite (272 tests) re-run and green with no regressions.

## Status

DONE_WITH_CONCERNS

## Concerns

- Severity: low
- Issue: automated `/code-review` could not complete this session (three attempts, all failed on infrastructure, not on findings against this diff).
- Impact: review coverage rests on one partial automated pass (one real finding, since fixed) plus manual self-review, not a full independent pass.
- Proposed follow-up: re-run `/code-review` against this PR once session infrastructure is stable, before or shortly after merge.

- Severity: low
- Issue: the four `.env*` diffs that activate this route are not applied yet.
- Impact: `harness` is live in code but inert in the running system until an operator applies them.
- Proposed follow-up: apply the diffs staged for this session, restart `orion-llm-gateway`, verify `GET /routes` reports `harness` as configured and `~/.fcc/.env`-driven FCC turns resolve to it (`route_key == "harness"` in a request log / correlation-id trace).

# PR report — journal/notification flood: follow-ups

Follow-up to `fix/journal-notification-flood` (#1028, merged), prompted by live verification after deploy. Covers the 3 candidates raised during that verification, plus a docs-truthfulness fix, per direct request.

## Summary

- **orion-recall compose hardening** (real incident found during deploy verification, not hypothetical): the already-merged compose-parity PR (#1021) added an explicit `environment:` list to `orion-recall/docker-compose.yml` with no fallback for 6 tunables that were missing from live `.env` — crash-looped the service in production (`pydantic` `float_parsing`/`int_parsing` errors, 10 restarts, unhealthy). Fixed live by syncing the missing keys; this PR hardens the compose file itself with `${VAR:-default}` fallbacks matching `.env_example` so it can't recur.
- **Investigated, and reverted, a proposed `/tmp`→`/data` fallback-default change in orion-actions**: this was flagged as a "concern" in the original PR report, but actually implementing it regressed a bare test run with a real `FileNotFoundError` (`/data` only exists inside the container's bind mount). Kept as regression tests + documentation instead of a behavior change.
- **Fixed a real, pre-existing logging bug** in `orion/core/bus/bus_service_chassis.py` (shared by every service using `Hunter`/the bus chassis): several log calls used a template+args shape that's only valid under one of the two possible logger backends (loguru vs. stdlib `logging` fallback), causing production `TypeError`s inside logging internals. Converted to f-strings, the one shape valid under both.
- **Docs truthfulness**: corrected two now-inaccurate docstrings in `orion-recall` (`worker.py` and its test) that claimed "the graph stores no usable ChatTurn timestamp" — no longer true after the RDF recency fix — without changing the behavior they describe.

## Outcome moved

- `orion-recall` can no longer crash-loop from a missing `.env` key for these 6 tunables.
- The `/tmp` vs `/data` fallback question is now closed with evidence, not left as an open "concern."
- Hunter/bus-chassis log lines that used to throw internal `TypeError`s (visible as "--- Logging error ---" tracebacks in `orion-actions` logs, confirmed live) now render correctly under both logger backends.
- Two stale doc claims about RDF recall no longer contradict the code they describe.

## Current architecture (before this patch)

- `orion-recall/docker-compose.yml`'s `environment:` list had `${VAR}` substitutions with no defaults for `RECALL_SKIP_MAX_NOVELTY`, `RECALL_SKIP_SHIFT_NOVELTY_FLOOR`, `RECALL_CONTINUITY_SQL_MINUTES`, `RECALL_CONTINUITY_RENDER_BUDGET`, `RECALL_BELIEF_RENDER_BUDGET`, `RECALL_CRYSTALLIZATION_VECTOR_COLLECTION`.
- `bus_service_chassis.py` has `try: from loguru import logger / except: logger = logging.getLogger("orion.bus")` — several call sites used `{}`/`{:.1f}`-style or `%s`-style templates with extra positional args, each broken under one of the two backends.
- `orion-recall/app/worker.py`'s `_window_rdf_chatturn_candidates` docstring (and its test file's module docstring) said the graph "stores no usable ChatTurn timestamp" — true before the recency fix, false after.

## Architecture touched

- `services/orion-recall`: compose env defaults, worker.py docstring, README.
- `services/orion-actions`: settings/store fallback defaults (investigated, reverted with evidence), README, tests.
- `orion/core/bus/bus_service_chassis.py`: shared bus chassis used by every `Hunter`/RPC-based service — logging fix only, no behavior change.

## Files changed

- `services/orion-recall/docker-compose.yml` — `:-default` fallbacks for 6 keys, matching `.env_example`.
- `services/orion-recall/tests/test_docker_compose_numeric_defaults.py` (new) — asserts those fallbacks exist and match `.env_example`; also asserts the two deliberately-unguarded Graphiti keys stay unguarded.
- `services/orion-recall/app/worker.py`, `services/orion-recall/tests/test_rdf_chatturn_windowing.py` — corrected docstrings (no behavior change).
- `services/orion-recall/README.md` — documents both of the above.
- `services/orion-actions/app/settings.py`, `workflow_schedule_store.py`, `scheduler_cursor_store.py` — no functional change; comments added explaining why the `/tmp` fallback default is intentional.
- `services/orion-actions/tests/test_scheduler_cursor_store.py`, `test_workflow_schedule_store.py` — regression tests locking in the `/tmp` fallback (previously untested).
- `services/orion-actions/README.md` — documents why the fallback stays `/tmp`.
- `orion/core/bus/bus_service_chassis.py` — 7 logger calls converted to f-strings (Hunter subscribing/reconnect ×2, Rabbit subscriber reconnect, heartbeat publish failure ×2, gateway LLM reply publish ×3).
- `tests/test_hunter_reconnect.py` — new AST-based structural test asserting no `logger.<level>()` call in `bus_service_chassis.py` passes extra positional args.

## Schema / bus / API changes

None. This is config hardening, a reverted-and-documented investigation, a logging fix, and doc corrections — no schema, bus, or behavior changes.

## Env/config changes

- No new keys. `orion-recall/docker-compose.yml`'s existing `${VAR}` substitutions gained `:-default` fallbacks matching already-documented `.env_example` values.
- Local `.env` sync: already done live during verification (`sync_local_env_from_example.py --all-keys orion-recall`) — this PR is the code-side hardening that makes that fix durable, not a repeat of it.

## Tests run

```text
services/orion-actions/tests                                                    94 passed
services/orion-recall/tests/test_docker_compose_numeric_defaults.py
  + test_rdf_chatturn_windowing.py                                               7 passed
tests/test_hunter_reconnect.py + test_chassis_heartbeat_reconnect.py
  + orion/core/bus/tests/                                                        5 passed
```
(Run per-service/path separately — combining service test directories in one `pytest` invocation hits a pre-existing `app`-package-name collision across services, confirmed present on clean `main` too, not caused by this patch.)

## Evals run

No dedicated eval harness for any of the touched surfaces (compose config, logging, docstrings). Verified via the tests above plus live production log inspection before/after (see "Restart required" — the bus-chassis logging fix and orion-recall compose hardening both need a rebuild to take effect; the orion-recall crash was already fixed live via env sync, independent of code shipping here).

## Docker/build/smoke checks

```text
python3 -c "import ast; ast.parse(open('orion/core/bus/bus_service_chassis.py').read())"  -> OK
```
No other Docker changes in this patch (compose env defaults only, no image/build changes).

## Review findings fixed

- Finding (self-caught during implementation, not a subagent review): my first attempt fixed orion-actions' `/tmp` fallback defaults to `/data/orion-actions/...` to match `.env_example`. Running the full test suite immediately surfaced a real `FileNotFoundError` from a test that constructs `Settings()` with no env vars set.
  - Fix: reverted the fallback-default change; kept it as `/tmp` (always exists, unprivileged-writable), documented why in code comments, README, and two new regression tests that lock in the correct (reverted) behavior.
  - Evidence: `services/orion-actions/tests/test_workflow_schedule_store.py::test_resolve_path_whitespace_fallback_stays_tmp`, `test_scheduler_cursor_store.py::test_resolve_scheduler_cursor_store_path_degenerate_fallback_stays_tmp`.
- Finding (self-caught): my first Hunter-logging fix converted `{}`→`%s`, which is correct for the stdlib fallback but silently stops substituting under loguru (confirmed live: local dev environment has loguru installed, and the fixed message showed literal `%s` unsubstituted in loguru's sink output).
  - Fix: converted to f-strings (fully pre-formatted, no extra args) — the only shape valid under both backends. Found and fixed 3 more instances of the same bug class (gateway LLM reply publish logging) during a structural sweep of the file, not just the one originally reported.
  - Evidence: AST-based structural test in `tests/test_hunter_reconnect.py`.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-recall/.env -f services/orion-recall/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-actions/.env -f services/orion-actions/docker-compose.yml up -d --build
```
Any other service importing `orion.core.bus.bus_service_chassis` (all `Hunter`-based bus consumers) would need a rebuild to pick up the logging fix, but it's a pure logging change with no functional impact — not urgent to restart everything at once.

If no restart is required for a given service before its next scheduled redeploy, that's fine too — nothing here is a live-data or correctness fix beyond the orion-recall crash (already resolved live independent of this code).

## Risks / concerns

- Severity: none identified. All three items are either pure hardening (compose defaults), a reverted investigation with evidence (the `/tmp` fallback), or a logging-only fix with a structural regression test (Hunter/bus chassis).

## PR link

<opened manually — see PR body below, `gh` is not authenticated in this environment>

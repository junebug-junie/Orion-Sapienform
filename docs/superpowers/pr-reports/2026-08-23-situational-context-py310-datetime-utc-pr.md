# PR #1848: datetime.UTC import breaks every orion-mode chat turn on Python 3.10

- Branch: `fix/situational-context-py310-datetime-utc`
- PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1848

## Summary

- **Production was down**: every "orion" mode chat turn was crashing with an unhandled `ImportError`, reported by Juniper as the chat "hung."
- Root cause: `orion/situational/context.py` (shipped today, commit `397022326`) used `from datetime import UTC, ...` — `datetime.UTC` only exists in Python 3.11+, but `orion-athena-hub`'s actual container runtime is Python 3.10.12.
- Why it looked like a hang, not a crash: the failing import sat outside any try/except in `websocket_handler.py`, so the exception propagated all the way to the outer generic handler, which ends the WebSocket connection with no error frame sent to the browser at all.
- Already deployed live and verified on the real running `orion-athena-hub` container before this PR — this formalizes the fix on `main`. A follow-up commit addresses real code-review findings on the new error-handling path (see below), also already redeployed and re-verified live.

## Outcome moved

Orion-mode chat turns work again. Confirmed via direct import inside the real container (previously `ImportError`, now succeeds).

## Current architecture

`orion/situational/context.py` builds a "situation brief" (time/weather/presence/etc.) injected into the unified-turn harness prompt, imported by `orion/hub/turn_orchestrator.py`, imported by `services/orion-hub/scripts/websocket_handler.py` on every "orion" mode WebSocket chat turn (gated by `ORION_UNIFIED_TURN_ENABLED`/`ORION_HARNESS_GOVERNOR_ENABLED`, both live).

## Architecture touched

`orion/situational/context.py`, `scripts/smoke_situation_grounding.py`, `services/orion-hub/scripts/websocket_handler.py`, plus new/extended tests. No schema/bus/API changes.

## Files changed

- `orion/situational/context.py`: `from datetime import UTC, datetime, timedelta` → `from datetime import datetime, timedelta, timezone`; all 7 `datetime.now(UTC)` call sites → `datetime.now(timezone.utc)`. `timezone.utc` is Python 3.2+, behaves identically to `datetime.UTC` (which is literally an alias for `timezone.utc`, added in 3.11 purely for import convenience) — no behavior change, just compatibility.
- `scripts/smoke_situation_grounding.py`: same bug, same fix — this is the designated smoke test for the exact feature that broke; it shared the identical Python-version bug, so it couldn't even have caught this if run against a real 3.10 target.
- `services/orion-hub/scripts/websocket_handler.py`: the previously bare `from orion.hub.turn_orchestrator import run_unified_turn` import is now wrapped in `try/except ImportError`, sending a real `{"type": "turn_error", "phase": "import", ...}` frame to the client on failure via the file's own disconnect-safe `_safe_ws_send_json` helper, logging server-side first, and popping the just-appended-but-never-answered user turn out of `history`. Applied the same history-pop + safe-send fix to the pre-existing sibling `harness_governor_disabled` block, which had the identical latent weakness.
- `orion/situational/tests/test_py310_compat.py` (new): static regression guard against `from datetime import UTC` reappearing in `context.py` or its smoke script, plus a real execution test building a full situation brief.
- `services/orion-hub/tests/test_websocket_agent_claude_routing.py`: extended with a static-shape check for the import guard (guarded, `ImportError`-scoped, logged, sent via the safe helper, pops history).

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
python3 -c "import ast; ast.parse(...)"  # every touched file, AST OK

cd orion/situational && PYTHONPATH=... venv/bin/python -m pytest tests -q
  7 passed  (4 pre-existing + 3 new)

cd services/orion-hub && PYTHONPATH=... venv/bin/python -m pytest \
  tests/test_situation_request_builder.py tests/test_situation_settings_env.py \
  tests/test_websocket_agent_claude_routing.py -q
  7 passed  (was 6, +1 new)

cd services/orion-cortex-exec && PYTHONPATH=... venv/bin/python -m pytest \
  tests/test_situation_conversation_phase.py tests/test_situation_perception_context.py \
  tests/test_situation_provider.py tests/test_situation_settings_env.py -q
  62 passed
```

`test_situation_prompt_integration.py`'s 2 failures are pre-existing on `main` (confirmed by running it unmodified on the primary checkout — identical `jinja2.exceptions.UndefinedError: 'metadata' is undefined`), unrelated to this fix.

**Root cause of why local tests never caught this**: this session's shared dev venv runs Python 3.12.3 (matching the host), so `datetime.UTC` imported fine locally every time — the exact same class of bug as the loguru/Hub incident earlier this session (local environment newer/more permissive than the actual container runtime it ships to).

## Evals run

No dedicated eval harness for this module; live production verification (below) is the real evidence here.

## Docker/build/smoke checks

**This is the actual fix verification — production was broken, not a formality. Done twice: once for the core fix, once after the review-fix commit.**

```text
$ bash scripts/safe_docker_build.sh orion-hub up -d --build
  Container orion-athena-hub Started
$ curl -s http://localhost:8080/health
  200
$ docker exec orion-athena-hub python3 -c \
    "from orion.hub.turn_orchestrator import run_unified_turn; print('IMPORT_CHAIN_OK')"
  IMPORT_CHAIN_OK   # previously: ImportError: cannot import name 'UTC' from 'datetime'
```

Re-ran the exact same rebuild/redeploy/import-check cycle after the review-fix commit — still clean, no errors in `docker logs` since redeploy.

Also swept the rest of the live import chain (`orion/hub/`, `orion/schemas/situation.py`, `orion/situational/`, `services/orion-hub/`) for the same `from datetime import ... UTC` pattern — clean, this was the only occurrence in the actual crash path. `scripts/agent_board_lib.py` has the same import pattern but runs on the host (Python 3.12.3, confirmed fine), not inside any Python-3.10 container — left alone, genuinely out of scope.

The running production `orion-athena-hub` container is live on this branch's final commit right now.

## Review findings fixed

Real code-review skill run, 7 findings, 5 fixed:

- Finding: the new `except` block sent `turn_error` via raw `websocket.send_json` instead of this file's own disconnect-safe `_safe_ws_send_json` helper — a client disconnecting at the same moment the import failed would raise an unhandled `RuntimeError`, reproducing the exact silent-crash failure this hotfix was written to eliminate.
  - Fix: switched to `_safe_ws_send_json`. Also applied to the pre-existing sibling `harness_governor_disabled` block two lines above, which had the identical weakness.
  - Evidence: `test_turn_orchestrator_import_is_guarded_and_reports_a_client_facing_error` asserts `_safe_ws_send_json` appears in the guard's body.
- Finding: `history.append({"role": "user", ...})` runs unconditionally before both early-`continue` paths, and neither popped it back off — a client retrying on the same socket after a `turn_error` would accumulate two consecutive `{role: user}` entries with no assistant reply between them, violating alternation some downstream LLM call paths expect.
  - Fix: pop the just-appended entry on both early-exit paths before `continue`.
  - Evidence: same test, asserts `history.pop()` in the guard's body.
- Finding: the except block never logged the import failure server-side before sending the client-facing frame, unlike sibling error paths in the same function.
  - Fix: added `logger.error` with `correlation_id` before the send.
- Finding: `except Exception` was broader than the confirmed failure mode — any unrelated exception from the same import statement would be silently absorbed into an indistinguishable generic message on every future message on a long-lived connection instead of failing loud.
  - Fix: narrowed to `except ImportError`.
- Finding: no regression test existed for either half of the original fix — a future re-introduction of `datetime.UTC`, or a future refactor removing the import guard, would go uncaught.
  - Fix: `orion/situational/tests/test_py310_compat.py` (new, 3 tests: static guard on both files that had the bug, plus a real execution test) + extended `test_websocket_agent_claude_routing.py` (1 new test, static-shape check matching this file's own established convention — no WebSocket TestClient harness exists for this handler).
- Declined: extracting a shared helper for the `turn_error`-frame-dict-send-continue shape now used in two places — real DRY observation, but two occurrences isn't yet the N-copies problem the finding describes; revisit if a third `phase` appears.
- Declined: adding a Python-3.10 CI/lint target to catch the next 3.11+-only stdlib usage before it ships. This is a real, valuable systemic fix (root `pyproject.toml` declares `>=3.12,<3.13` and CI runs 3.12, but `orion-athena-hub`'s actual runtime is 3.10 — nothing catches this class of bug pre-merge) but a separate, larger change than this hotfix's scope. **Recommended follow-up, flagging explicitly rather than silently dropping it.**

## Restart required

Already done twice as part of live diagnosis/fix — `orion-athena-hub` is live on this branch's final commit right now. No further action needed once this merges to `main`, unless `main` diverges further before another deploy.

## Risks / concerns

- Severity: low. Concern: no Python-3.10-targeted CI/lint check exists to catch the next 3.11+-only stdlib usage before it reaches this exact container. Mitigation: flagged as a recommended follow-up above; out of scope for this hotfix.
- Everything else: none outstanding. Mechanical, behavior-preserving datetime fix; the error-handling hardening is narrowly scoped and live-verified twice.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1848

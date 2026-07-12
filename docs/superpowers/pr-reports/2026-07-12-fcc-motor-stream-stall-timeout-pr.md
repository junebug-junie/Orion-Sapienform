## Summary

- `orion/harness/fcc_motor.py`'s `run_fcc_turn()` bounded a single `stream-json` line's `readline()` wait by the *whole-turn* timeout (`HARNESS_FCC_TIMEOUT_SEC`, 900s default) instead of a much shorter per-line bound. A model that never completes one message (no stop token reached) hung the entire turn for up to 15 minutes with zero output. New `HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC` (default 180s) fails that case fast with `error_code=fcc_stream_stalled`, while the whole-turn timeout (`fcc_timeout`) still applies on top for turns whose many individually-fast steps add up past the turn budget.
- `orion/fcc/self_index_brief.py`'s GitNexus brief now tells the model to call GitNexus *before* falling back to raw source search on topology/ownership/impact/symbol-discovery questions, and to always check GitNexus status before trusting a result rather than skip that check whenever a query result happens to look fine.
- Rebuilt the live GitNexus index (`~/.gitnexus`, forced full rebuild) against current `main` HEAD — it was previously stamped against an abandoned branch/commit, so `gitnexus status` reported stale on every turn, and the model's own brief instructed it to disclose staleness and fall back when that happened.
- Documented a real GitNexus operability bug found live: incremental `analyze --index-only` can fail outright (`Failed calling LOWER: Invalid UTF-8`) and leaves an `incrementalInProgress` flag / stale index behind; documented the `--force` full-rebuild recovery in the README.

## Outcome moved

Caught a real, live hang in progress while investigating this (not a hypothetical): turn `corr=59915360-316c-4f3f-ad9a-2135a7a27825`, a two-word "hey" greeting, was stuck 7+ minutes with the `claude -p` subprocess and its `gitnexus mcp` child both asleep at ~0% CPU. `orion-athena-fcc`'s own log showed the actual mechanism: `api.response.stream_interrupted ... stream_chunks=4333 stream_bytes=540970 outcome=cancelled` — the upstream model streamed 540KB with no stop condition, and because Claude Code CLI only flushes a `stream-json` line once a message fully completes, the harness governor had zero visibility and zero fail-safe until the full 900s turn budget elapsed. After this patch, that same failure mode fails in ~180s with a diagnosable `fcc_stream_stalled` error code and a `steps_seen` count instead of a silent multi-minute hang.

## Current architecture

`orion-harness-governor` spawns `claude -p ...` per turn (`orion/harness/fcc_motor.py::run_fcc_turn`) and reads its `stream-json` stdout line by line. Before this patch, each `readline()` call was wrapped in a single `asyncio.wait_for(..., timeout=timeout_sec)` using the *entire* turn's timeout budget — so one stuck line looked identical to a healthy long-running turn until the whole-turn clock ran out. Separately, `orion/fcc/self_index_brief.py` renders GitNexus/Context Mode usage instructions into the harness prompt only when their respective env flags are on; the previous wording made GitNexus optional-first ("Use query to locate...") and let the model disclose staleness and bail without first checking the actual GitNexus status resource.

A prior, separate session had already prototyped and fully reverted a related fix (branch `fix/fcc-motor-idle-watchdog`: commits `a8e8221c`/`7df5e31b`/`f4c3d5dc` reverted by `e995278a`/`c863481b`/`7bd8be83`/`f43e1e78`/`a8671358`, no rationale recorded, net diff against base is empty). This PR does not resurrect that work; it's an independent fix derived from live evidence gathered in this session, deliberately narrower in scope (see Risks/concerns).

## Architecture touched

- `orion/harness/fcc_motor.py` — new `_stream_stall_timeout_sec()` helper and reworked `readline()` loop in `run_fcc_turn()`: tracks a wall-clock deadline, waits `min(stall_timeout, remaining)` per line, classifies a `TimeoutError` as `fcc_stream_stalled` or `fcc_timeout` depending on which bound fired, and includes `steps_seen`/`llm_response` on both outcomes.
- `orion/fcc/self_index_brief.py` — `gitnexus_brief_lines()` wording strengthened (call-before-fallback + mandatory status check).
- `services/orion-harness-governor/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md` — new `HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC` wired through the full operator-contract surface, plus a new README section documenting the GitNexus incremental-indexer failure mode and recovery.
- Live infra action (not a code change): rebuilt `~/.gitnexus`'s index for `/mnt/scripts/Orion-Sapienform` via `docker run --entrypoint gitnexus ... analyze --index-only --force --name orion` — now stamped at `main`@`26e0770d`.

## Files changed

- `orion/harness/fcc_motor.py`: stall-timeout guard, `steps_seen` counter, `_stream_stall_timeout_sec()` helper
- `orion/fcc/self_index_brief.py`: GitNexus brief text strengthened
- `orion/harness/tests/test_fcc_motor_mcp.py`: 2 new regression tests (`test_run_fcc_turn_fails_fast_on_stalled_stream`, `test_run_fcc_turn_whole_turn_timeout_wins_near_deadline`) + 2 unit tests for `_stream_stall_timeout_sec`
- `orion/harness/tests/test_harness_prefix.py`: updated assertions for the strengthened GitNexus brief wording
- `services/orion-harness-governor/.env_example`, `app/settings.py`, `docker-compose.yml`, `README.md`: new env key wired through, GitNexus incremental-failure recovery documented

## Schema / bus / API changes

- Added: new `error_code="fcc_stream_stalled"` value on the existing `run_fcc_turn()` error event shape (same dict shape as `fcc_timeout`, now also carrying `steps_seen` and `llm_response`); no schema/registry changes, this is an internal generator event, not a registered bus/schema contract.
- Removed: none
- Renamed: none
- Behavior changed: `fcc_timeout` and the new `fcc_stream_stalled` events now both include `steps_seen` and `llm_response` (previously `fcc_timeout` had neither). Verified `orion/harness/runner.py` and `services/orion-harness-governor/app/bus_listener.py` consume `error_code`/`grounding_status` generically (no hardcoded `== "fcc_timeout"` branch), so the new code flows through safely.
- Compatibility notes: none required; purely additive event fields.

## Env/config changes

- Added keys: `HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC` (default `180`)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes (`services/orion-harness-governor/.env_example`)
- local `.env` synced: yes, manually verified present in `services/orion-harness-governor/.env` (this repo's `scripts/sync_local_env_from_example.py` only reports services with a local `.env`; this worktree has none since git worktrees don't copy gitignored files — confirmed against the actual non-worktree checkout instead)
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=. pytest orion/harness/tests/ orion/fcc/tests/ -q
153 passed, 3 failed (pre-existing on main, unrelated: 2x test_grounding_capsule_consumers.py
identity-rendering assertions, 1x test_harness_runner.py::test_harness_runner_surfaces_fcc_error_code
— a pre-existing precedence bug in runner.py's error_msg-vs-error_code fallback order, confirmed
failing identically on main before this branch touched anything)

PYTHONPATH=services/orion-harness-governor:. pytest services/orion-harness-governor/tests/ -q
13 passed
```

## Evals run

No dedicated eval harness exists for `orion-harness-governor`'s FCC motor path (`orion/harness/evals/` covers the unified-turn introspection experiment, not this stream-reading seam). Flagging per repo contract rather than claiming eval coverage.

## Docker/build/smoke checks

```text
docker run --rm -v /mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform \
  -v $HOME/.gitnexus:/root/.gitnexus -w /mnt/scripts/Orion-Sapienform \
  --entrypoint gitnexus orion-harness-governor-harness-governor analyze --index-only --force --name orion
Repository indexed successfully (102.9s) — 72,108 nodes | 120,789 edges | 1766 clusters | 300 flows
registry.json now shows lastCommit=26e0770d (main), branch=main
```

Did not rebuild/restart `orion-athena-harness-governor` as part of this PR — the code change only takes effect once merged to `main` and the container is rebuilt (see Restart required).

## Review findings fixed

8-angle parallel code review ran against the working-tree diff.

- Finding: rewritten GitNexus brief text weakened a documented safety instruction — old wording unconditionally required checking the GitNexus status resource before any graph-grounded claim; my first draft changed the fallback trigger to "if a query genuinely returns thin results," which could let the model skip the authoritative status check whenever a result merely looked fine
  - Fix: reworded to make the status check unconditional and mandatory every turn, independent of how a query result looks
  - Evidence: `orion/fcc/self_index_brief.py::gitnexus_brief_lines`, `orion/harness/tests/test_harness_prefix.py`
- Finding: two near-duplicate `fcc_timeout` yield sites (a pre-loop deadline guard and the except-block's else branch), which would drift out of sync on the next edit, and neither included the new `steps_seen`/`llm_response` fields present on the sibling `fcc_stream_stalled` path
  - Fix: collapsed into a single classification (`stalled = stall_timeout_sec < remaining`) and a single yield site; both outcomes now carry `steps_seen` and `llm_response`
  - Evidence: `orion/harness/fcc_motor.py::run_fcc_turn`
- Finding: `fcc_stream_stalled` omitted `llm_response` entirely (unlike the sibling `fcc_stream_line_limit` branch), so `orion/harness/runner.py`'s partial-credit logic (`compliance_verdict="partial"` when `llm_response` is present) unconditionally scored a stalled turn as a hard failure even when real assistant text had already streamed through in earlier completed steps
  - Fix: both new yield sites include `"llm_response": accumulated or None`
  - Evidence: `orion/harness/fcc_motor.py::run_fcc_turn`, cross-referenced against `orion/harness/runner.py:279-289`
- Finding: `_stream_stall_timeout_sec`'s except-block reassigned `configured = DEFAULT_STREAM_STALL_TIMEOUT_SEC`, a no-op dead line since that's already `configured`'s value whenever `float(raw)` raises
  - Fix: removed the redundant reassignment
  - Evidence: `orion/harness/fcc_motor.py::_stream_stall_timeout_sec`
- Finding: the 1.0s floor on the effective stall timeout clamped silently, with no log, unlike the sibling invalid-parse branch which does warn — empirically verified (new test took 1.00s wall-clock against a configured 0.01s value)
  - Fix: added a warning log when the clamp actually changes the configured value
  - Evidence: `orion/harness/fcc_motor.py::_stream_stall_timeout_sec`
- Finding (declined): reuse `orion.fcc.context_budget._env_float` instead of hand-rolling env-float parsing in the new helper (that private helper's module is already imported into `fcc_motor.py`)
  - Fix: not applied — `_env_float` swallows invalid values silently with no warning log, and importing a leading-underscore (module-private) symbol across modules trades a real operator-facing diagnostic for a ~4-line reduction; judged not worth it. Noting the finding rather than silently dropping it.
  - Evidence: `orion/fcc/context_budget.py:21` (`_env_float`)

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml \
  up -d --build harness-governor
```

## Risks / concerns

- Severity: medium
  - Concern: `services/orion-hub/scripts/fcc_claude_bridge.py` (Hub's separate "agent-claude" WebSocket chat path, `run_turn_from_settings`) has the exact same unbounded per-line `readline()` timeout bug this PR fixes in the harness governor's motor. It is a parallel, independent implementation, untouched by this diff.
  - Mitigation: not fixed here to keep this patch service-bounded (harness-governor vs. hub, per this repo's own contract). Recommend a fast, near-identical follow-up patch to that file.
- Severity: medium
  - Concern: `services/orion-llm-gateway/app/anthropic_passthrough.py` forwards the Anthropic Messages request to the llama.cpp backend as a byte-for-byte passthrough — no `max_tokens` clamp, no stop-sequence injection, no generation-length governor — unlike `llm_backend.py`'s other routes, which do set explicit `n_predict`/`max_tokens`. This PR bounds how long the *harness* waits on a stuck generation; it does not bound the generation itself at the source.
  - Mitigation: not fixed here — cross-service, larger blast radius (affects every anthropic-passthrough consumer, not just the harness). Recommend as a follow-up: a request-shaping governor on that route specifically for local llama.cpp backends.
- Severity: low-medium (unverified, disclosed rather than assumed per this repo's "runtime truth beats config truth" rule)
  - Concern: `proc.kill()` terminates the local `claude -p` subprocess, but I did not verify whether that propagates a cancel through `orion-llm-gateway`'s SSE passthrough all the way to the llama.cpp backend. If it doesn't, the backend may keep generating (burning GPU/CPU) after the harness has already reported `fcc_stream_stalled` — and because turns now fail in ~180s instead of 900s, a busy period could produce more concurrent orphaned generations per unit time than before this patch, not fewer.
  - Mitigation: none in this PR. Follow-up: confirm via the llama.cpp backend's own request logs/metrics whether a client disconnect actually aborts generation.
- Severity: low
  - Concern: `HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC=180` default is chosen from this one incident plus a safety margin, not a measured distribution of legitimately-slow-but-healthy single steps (e.g. a large GitNexus query or heavy Bash/file-read tool call under host load). Could theoretically false-positive on a real but slow step.
  - Mitigation: configurable per-operator; set to `0` to fall back to the old whole-turn-only behavior if this proves too aggressive in practice.
- Severity: low
  - Concern: GitNexus's incremental indexer (`analyze --index-only`, no `--force`) crashed outright on this repo (`Failed calling LOWER: Invalid UTF-8`) during this session, leaving a corrupt `incrementalInProgress` state. Root cause not identified (which of ~4700 tracked files triggers it) — worked around via `--force` full rebuild, which succeeded cleanly.
  - Mitigation: documented the `--force` recovery path in the README. If this recurs on every future incremental update, it may need a real fix or a permanent switch to always-force in this repo's specific operator runbook.

## PR link

Branch pushed: `fix/fcc-motor-stream-stall-timeout` → https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/fcc-motor-stream-stall-timeout (PR not yet opened — `gh` CLI is not authenticated in this environment; open the compare link above manually, using this file as the description).

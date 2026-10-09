# FCC chat reply writer: always-warm Claude Code pool (spec L5, option a)

## Design (written before coding)

Juniper chose "keep all MCP servers, build always-warm". The orion-aitown MCP config is untouched.

**What it does.** Chat replies stop paying the ~3.4 s Claude Code start-up (MCP servers starting) on every turn. The governor keeps a small number of `claude` processes alive in streaming-input mode. Each chat turn borrows one, wipes its conversation with `/clear`, sends the prompt as a stream-json user message, and reads events until the CLI's own `result` event. Then the process goes back to the pool.

**Scope.** Only Juniper's chat replies (`utterance_origin="juniper"`, i.e. `chat_reply=True` in the motor), and only when the turn has no reading binding and is not reading-only. Investigations, curiosity, outreach and reading turns keep today's per-turn spawn.

### Checked against the real CLI before writing code (2.1.291, local stub, no model call)

- No `init` is printed until the first stdin message arrives. A `/clear` sent as the first message returns `conversation_reset` → `system/init` (new `session_id`) → `result` (empty, `num_turns:0`) with no `/v1/messages` call. So `/clear` doubles as the warm-up handshake and the per-turn reset.
- Each prompt after `/clear` starts with its own `system/init`, then `assistant`/`user` events, then `result`. Same shape a spawned turn yields.
- The turn after `/clear` sent no trace of the previous turn's prompt to the stub (checked by marker string in the request body).
- `CLAUDE_ENV_FILE` is re-read on every Bash call: a live process printed `DL=333 BUD=444` after the file was rewritten from `111/222` between turns.

### Every per-turn value, decided

| Value (today, per spawn) | Warm process |
|---|---|
| GPU lease header in `ANTHROPIC_CUSTOM_HEADERS` | Not in the process env. The process talks to a governor-local relay (`127.0.0.1`, one URL per slot); the relay adds the **current** turn's `X-Orion-Gpu-Lease` to each request and strips any copy the client sent. |
| `ANTHROPIC_BASE_URL` (llm-gateway when leased, FCC server otherwise) | Relay picks the upstream per turn from the same rule. |
| `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY` | Process gets a random per-slot secret. The relay refuses any request without it (401), then substitutes the turn's real credential (`orion-resource-lease` when leased, the FCC token otherwise). |
| Correlation id | Relay adds `X-Orion-Correlation-Id` per request and logs `fcc_warm_relay_request corr=... slot=... status=...`. (`x-request-id` left alone: the gateway keys pool calls on it.) |
| `ORION_TURN_BUDGET_SEC`, `ORION_TURN_DEADLINE_EPOCH`, `ORION_TURN_STEP_STALL_SEC` | Removed from the process env; written per turn to the slot's `CLAUDE_ENV_FILE` before `/clear`. Absent values are left out, matching today's "cleared, not inherited" rule. |
| Whole-turn deadline and per-step stall | Enforced by the same motor loop as today. Overrun → kill the process group → slot respawns. The relay unbinds the slot at turn end, so a late call from a killed turn gets 409, never the lease. |
| `--model` | Part of the slot's signature. Mismatch → this turn spawns as today; the idle slot is respawned with the new model in the background. |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` / `AUTO_COMPACT_WINDOW` (per-lane `n_ctx`) | Part of the signature, same rule. |
| MCP config (rendered per turn; only the file name carries the correlation id for chat turns) | Rendered once per process. Turns with a reading binding are not eligible. |
| `--settings` repeat-failure breaker | Static JSON. The hook keys on `transcript_path`, and `/clear` starts a new session (new transcript), so its state resets per turn. |
| `CLAUDE_CODE_DISABLE_AUTO_MEMORY` | Same env builder with `chat_reply=True`, so on by default (same flag as today). |
| Curiosity graph credentials from `~/.fcc/.env` | Static. The file's contents are part of the signature, so an edit respawns. |
| Permission argv, `--setting-sources`, all other env | Static, part of the signature. |
| Turn lock, write stamper, cancel registry | Still per turn in the motor. Cancel kills the warm process; it respawns. |

### Pool

- Size `HARNESS_FCC_CHAT_WARM_POOL_SIZE`, default 1. Observed chat concurrency: 220 chat turns in the last 30 days (`harness_turn_trace` joined to `chat_history_log`), maximum overlap 1.
- One turn per process at a time. A health task respawns dead slots. A failed `/clear` handshake kills the slot.
- Recycle after `HARNESS_FCC_CHAT_WARM_POOL_MAX_TURNS` turns (50) or `..._MAX_AGE_SEC` (3600). Graceful first (close stdin), then kill the process group.
- Warm-up on governor start with `HARNESS_FCC_CHAT_WARM_POOL_MODEL_LABEL` (`MODEL_SONNET`) and the env context ceiling. The first turn with a different signature retargets the slot.

### Fallback

Any failure before the prompt is sent (pool not started, no matching idle slot, process dead, `/clear` timeout, relay down) → today's per-turn spawn, logged as `fcc_warm_pool_fallback corr=... reason=...`. If a warm process dies after the prompt is sent but before any event, the turn also falls back. A death after events were yielded is reported as an error, the same as a spawned process dying mid-turn, because tools may already have run.

`HARNESS_FCC_CHAT_WARM_POOL_ENABLED=true` ships ON.

### Telemetry

Every turn, both modes: `fcc_turn_start_timing corr=... mode=warm|spawn spawn_or_acquire_ms=... first_event_ms=...`.
- `spawn_or_acquire_ms`: turn start → the turn's first `system/init` event. This is the same point the L5 probe measured.
- `first_event_ms`: turn start → the first `assistant`/`user` event, which includes model latency.

Both values are also added to the final frame's metadata as `fcc_spawn_mode`, `fcc_spawn_or_acquire_ms` and `fcc_first_event_ms`.

### Non-goals

- Carrying conversation between turns. Context is wiped every turn.
- Warm processes for investigation turns.
- Interrupting via the stream-json control protocol. Kill and respawn is simpler and already proven.
- Trimming MCP servers. This change does not touch the aitown config.

---

## Summary

- Juniper's chat replies borrow an already-running Claude Code process instead of starting a new one. Each turn wipes the conversation with `/clear` first.
- A relay inside the governor gives every model request the current turn's GPU lease header, upstream, credential and correlation id. The warm process's frozen environment carries none of them.
- The turn's clock (budget, deadline, stall cap) reaches the agent's Bash calls through a per-slot `CLAUDE_ENV_FILE`, rewritten before every turn.
- Any pool failure before the prompt is sent falls back to today's per-turn spawn. Investigation turns always spawn.
- New telemetry for both modes: `fcc_turn_start_timing corr=... mode=... spawn_or_acquire_ms=... first_event_ms=...`, plus the same values in the final frame metadata.

## Outcome moved

Time from turn start to Claude Code being ready, measured on the real CLI 2.1.291 against a local stub (`scripts/probe_fcc_warm_pool.py`, host, no MCP servers):
- spawn: median 3432 ms (3388-3459, n=5)
- warm: median 39 ms (30-57, n=5)

Two earlier host runs measured spawn at 419 ms and 3428 ms, while warm stayed at 35-36 ms. The host spawn cost varies between runs for reasons not investigated; the warm cost does not.

In the governor container with all MCP servers, PR #2509 measured spawn at 3.40 s. The live warm number in the container is **UNVERIFIED** until deploy. Read it from the `fcc_turn_start_timing` log lines.

## Current architecture

Every FCC turn ran `claude -p <prompt>` as a fresh process (`fcc_motor.run_fcc_turn`). Per-turn values were set through process env and argv. Start-up was about 3.4 s, almost all of it MCP servers.

## Architecture touched

- `orion/harness`: motor refactor, plus the new pool and relay.
- orion-harness-governor:
  - lifespan starts and stops the pool;
  - `/health` gains an `fcc_warm_pool` field;
  - settings, env template and compose gain the new keys.
- The relay listens on a container-local port (`127.0.0.1:7157`). It is not published, and no bus or schema contract changes.

## Files changed

- `orion/harness/fcc_motor.py`:
  - the stream loop moved into `_drive_fcc_turn`, shared by `_SpawnTurnIO` and `_WarmTurnIO`;
  - new `build_claude_argv`, `relay_binding_for_turn`, `turn_clock_env`;
  - the warm acquire and fallback path;
  - timing telemetry.
- `orion/harness/fcc_warm_pool.py` (new): slots, `/clear` handshake, recycle, health loop, respawn, fallback reasons.
- `orion/harness/fcc_warm_relay.py` (new): a Starlette/uvicorn relay on its own thread, with a per-slot secret and per-turn bindings.
- `orion/harness/tests/test_fcc_warm_pool.py`, `fake_claude_stream.py` (new): 21 tests that run real subprocesses against a stub upstream.
- `services/orion-harness-governor/app/{main,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`, `tests/test_fcc_warm_pool_wiring.py`.
- `scripts/probe_fcc_warm_pool.py` (new): live before/after probe on the real CLI.
- `.github/workflows/orion-fcc-warm-pool-tests.yml` (new): CI for the above.

## Schema / bus / API changes

- Added:
  - `/health` field `fcc_warm_pool`;
  - final-frame metadata `fcc_spawn_mode`, `fcc_spawn_or_acquire_ms`, `fcc_first_event_ms`;
  - relay request header `X-Orion-Correlation-Id`.
- Removed, renamed: none.
- Behavior changed: chat-reply turns run in a pooled process when one is warm and matches.
- Compatibility: additive. Metadata is a free-form dict.

## Env/config changes

- Added keys (orion-harness-governor):
  - `HARNESS_FCC_CHAT_WARM_POOL_ENABLED=true` (ships ON)
  - `HARNESS_FCC_CHAT_WARM_POOL_SIZE=1`
  - `HARNESS_FCC_CHAT_WARM_POOL_MAX_TURNS=50`
  - `HARNESS_FCC_CHAT_WARM_POOL_MAX_AGE_SEC=3600`
  - `HARNESS_FCC_CHAT_WARM_POOL_MODEL_LABEL=MODEL_SONNET`
  - `HARNESS_FCC_CHAT_WARM_POOL_RELAY_PORT=7157`
  - `HARNESS_FCC_CHAT_WARM_POOL_SPAWN_TIMEOUT_SEC=90`
  - `HARNESS_FCC_CHAT_WARM_POOL_CLEAR_TIMEOUT_SEC=5`
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py orion-harness-governor`: yes, all 8 keys added.
- Skipped keys requiring operator action: none. One divergence reported and left alone: `HARNESS_FCC_INTROSPECT_ENABLED` is locally `true` versus `false` in the example.

## Tests run

```text
pytest orion/harness/tests/test_fcc_warm_pool.py                    21 passed (x2, stable)
pytest orion/harness/tests orion/fcc/tests                         561 passed, 1 failed
  failure: test_claude_spawn.py::test_no_undiscovered_root_callers_of_claude_permission_argv
  (already failing on main; caller is services/orion-room-companion/app/claude_session.py, not touched here)
pytest services/orion-harness-governor/tests                        77 passed
fresh venv with only the CI install list                            89 + 4 passed
scripts/check_env_template_parity.py                                PASS
Mutation check: dropping /clear or the relay's lease injection fails 3 tests
```

## Evals run

```text
scripts/probe_fcc_warm_pool.py --turns 5 (real claude 2.1.291, local stub)
spawn median 3432 ms | warm median 39 ms | per-turn lease+corr OK (5/5 checked) | prior-turn leak: no
```

No eval harness exists for orion-harness-governor. The probe above is the before/after measurement. The live check is the `fcc_turn_start_timing` log lines after deploy.

## Docker/build/smoke checks

```text
Not run: no deploy or restart for this task. The governor image already has fastapi, uvicorn and httpx.
```

## Review findings fixed

- Finding 1 (material): a failure between acquire and the drive loop's `try` left the slot busy forever, with its lease still bound on the relay.
  - Fix: `run_fcc_turn` releases the turn in a `finally`. The health loop also reclaims a slot busy for more than 3 h.
  - Evidence: `test_failure_before_drive_try_still_releases_the_slot`.
- Finding 2 (material): the process dying between `/clear` and the prompt raised a broken-pipe error and failed the turn.
  - Fix: `send_prompt` treats the broken pipe as EOF, so the turn retries as a spawn.
  - Evidence: `test_death_between_clear_and_prompt_falls_back`.
- Finding 4: upstream stream closing relied on a Starlette `BackgroundTask`, which newer versions skip on client disconnect.
  - Fix: the body generator closes the stream in a `finally`.
- Finding 5: a background shell started in one turn could survive into the next turn while holding the slot secret.
  - Fix: a turn whose stream shows `run_in_background` recycles its process.
  - Evidence: `test_background_shell_turn_recycles_the_process`.
- Finding 7: MCP children were orphaned when the leader exited on its own.
  - Fix: the process group is swept right after reaping.
- Finding 8: an unbound model call got 503 `overloaded_error`, which the CLI may retry.
  - Fix: it now gets 409 `invalid_request_error`.
- Finding 9: a failed pool start leaked the relay and state dir, and `/health` showed it as "disabled".
  - Fix: cleanup on failure, and `/health` reports `start_error`.

## Restart required

```bash
python3 scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Minor (review #3): the first chat turn after each governor start usually misses on signature.
  - Cause: warm-up uses the env context window, while real chat turns pass the lane's live window (e.g. 131072). That turn spawns as today, and the slot retargets for every later turn.
  - If the pool-state probe flips between unknown and known, the slot respawns on each flip.
  - Mitigation: watch the `miss:signature_mismatch` counter in `/health`.
- Minor (review #6): a warm process keeps its start-up view of the sandbox for up to an hour.
  - Mitigation: `--setting-sources user,local` already excludes project settings, and the 1 h recycle bounds the rest.
- UNVERIFIED: start-up time inside the container with all MCP servers; memory use of one extra resident `claude` plus its MCP servers; whether the real CLI kills background shells on `/clear` (recycle covers this either way).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2514

🤖 Generated with [Claude Code](https://claude.com/claude-code)

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
| Whole-turn deadline and per-step stall | Enforced by the same motor loop as today. Overrun → kill the process group → slot respawns. The relay unbinds the slot at turn end, so a late call from a killed turn gets 503, never the lease. |
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

# FCC reply-writer spawn timing (spec L5 measurement)

Date: 2026-10-06. Spec: `docs/superpowers/specs/2026-10-06-unified-turn-latency-design.md`, item L5.
Status: measurement only. No production code, config, deploy or restart changed.

## Plain answer

The Claude Code start-up cost is almost all MCP tool servers (the external tools Claude Code is given), not the SessionStart hook and barely the context-mode plugin. Claude Code will not say "ready" until every MCP server has finished starting, and the slowest one, `gitnexus`, takes about 2.1 s on its own. Removing the hook saves nothing measurable. Removing the plugin saves about 0.3-0.5 s. Removing both still leaves about 2.9 s, which is over the spec's 1.5 s line.

So L5 option (b) as written (drop the hook and plugin) is not enough on its own. Two ways to get under 1.5 s:

1. **Trim the MCP server list for chat replies.** With no MCP servers, start-up is 0.45 s. With `github` only, 0.82 s. This only works if chat replies do not need `gitnexus` and `firecrawl`. That is a capability decision for Juniper, not a measurement.
2. **Keep one process alive (option a).** Verified with no model call: in a long-lived streaming-input process, sending `/clear` starts a fresh conversation in about 0.13 s. MCP servers stay connected, and the SessionStart hooks re-run. The catch is that some per-turn settings are fixed when the process starts (below).

**Recommendation:** do (1) if chat can drop `gitnexus`/`firecrawl`. Otherwise do (a), but only after the per-turn GPU lease header problem below is solved. Do not ship (b) alone: it recovers about 0.5 s of roughly 3.4 s.

## Method

- Ran inside `orion-athena-harness-governor` (Claude Code 2.1.291), cwd `/mnt/orion-fcc/repo`.
- argv and env come from the deployed motor's own functions (`/app/orion/harness/fcc_motor.py`):
  - `_build_subprocess_env(chat_reply=True)`, which sets `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`;
  - `setting_sources_argv("HARNESS_FCC_SETTING_SOURCES")`, giving `user,local`;
  - `repeat_failure_breaker_argv`, `_maybe_render_mcp_config` plus `extend_mcp_argv` (with the context-mode pre-approval), and `claude_permission_argv`.
- **No model call.** `ANTHROPIC_BASE_URL` points at a local stub inside the container, which records every request. It answers `/v1/models` with an empty list and anything else with 503. The process is killed when the first `POST /v1/messages` arrives.
- Two timings per run:
  - spawn → first stream-json `system/init` event;
  - spawn → first `POST /v1/messages`, the moment a real turn would start talking to the model.
- Before `init`, the CLI only makes `HEAD /api/hello` and `GET /v1/models`, at about 0.2 s and 0.45 s. `init` arrives after every MCP server reports connected or failed. The model request follows about 40 ms later, so `init` is a faithful no-API proxy.
- Variants (b)-(d) use a temporary `CLAUDE_CONFIG_DIR` under `/tmp` inside the container. Plugins and hooks are symlinked or copied in, with an edited `settings.json`. `/root/.claude` was never modified.
- `a2` is a control showing the temporary config dir itself does not change timing.
- Variants were interleaved run by run, so host-load noise spreads evenly.
- Script: `scripts/probe_fcc_spawn_timing.py`.

## Results (seconds, n=7 per variant for a-e, n=6 for f-h)

| Variant | init median | init p90 | first model request median | p90 |
|---|---|---|---|---|
| (a) exactly as the motor runs today | 3.40 | 3.59 | 3.44 | 3.64 |
| (a2) control: copied config dir, all on | 3.21 | 3.27 | 3.25 | 3.31 |
| (b) SessionStart hook removed | 3.33 | 3.43 | 3.39 | 3.47 |
| (c) context-mode plugin disabled | 2.87 | 3.12 | 2.92 | 3.15 |
| (d) minimal config dir (no hook, no plugin) | 2.86 | 3.00 | 2.93 | 3.05 |
| (e) minimal config, no MCP, no breaker hook | 0.45 | 0.53 | 0.49 | 0.57 |

The first run of the session (variant a, real config) took **8.93 s**. Nothing had spawned for a while before it. It is excluded above because it is a single sample, but live turns spaced minutes apart may hit this cold path. That is UNVERIFIED for real turns.

Per-MCP-server split (minimal config, breaker hook on):

| MCP servers loaded | init median | p90 |
|---|---|---|
| none (breaker hook only) | 0.58 | 0.65 |
| `github` only (stdio proxy → `docker run`) | 0.82 | 0.90 |
| `orion-aitown` only (**fails to start** every run) | 0.55 | 0.66 |
| `firecrawl` only (`npx -y firecrawl-mcp`) | 1.73 | 1.81 |
| `gitnexus` only (`gitnexus mcp`) | 2.65 | 2.84 |
| motor config minus `orion-aitown` | 3.20 | 3.96 |
| minimal config minus `orion-aitown` | 2.65 | 3.16 |

Reading:
- MCP servers start in parallel, and `init` waits for the slowest. `gitnexus` sets the floor at about +2.1 s over bare start-up, and `firecrawl` is next at about +1.2 s.
- The SessionStart hooks (`context-mode-cache-heal.mjs` plus the plugin's own) finish within about 0.4 s of spawn, while MCP is still starting, so they are off the critical path.
- The plugin's cost (~0.35-0.5 s) is its own MCP server, `plugin:context-mode:context-mode`.

Raw per-run values: `a` init `[3.413, 3.667, 3.037, 3.219, 3.31, 3.404, 3.588]`, `d` `[2.859, 2.95, 3.633, 3.002, 2.654, 2.503, 2.577]`, `e` `[0.568, 0.425, 0.409, 0.426, 0.469, 0.531, 0.447]`.

## Real-turn data

There is no persisted spawn → first-event timing for real turns.
- `harness_turn_trace.run_artifact` stores only `fcc_elapsed_sec`, the whole turn.
- The governor's logs for the last 48 h hold only `/health` lines; there are no motor log lines.
- Hub logs show no chat turns since its last restart.

The 2.4-4.4 s figure in the handoff has no recorded source. The probe's motor-exact 3.0-3.7 s (plus one 8.9 s cold start) is consistent with it, but the live-turn figure stays **UNVERIFIED**. If L5 ships, the motor should log spawn → `init` per turn, so the before/after can be measured on real turns.

## Streaming-input check (L5 option a)

- From `claude --help`: `--input-format stream-json` ("realtime streaming input", `--print` only) and `--replay-user-messages` exist. There is no flag for resetting history per message.
- Tested with no model call, against the same stub: sent a user message `/clear` on stdin to a live process.
  - The CLI emitted `{"type":"conversation_reset","trigger":"clear",...}`, re-ran `SessionStart:clear` hooks, and emitted a fresh `system/init` with a **new session_id** and `num_turns:0`.
  - With the motor config, all MCP servers stayed `connected` without respawning. Reset to `init` took about 0.13 s.
  - No request reached the stub. **So option (a), a warm process with cold context, is supported by the CLI.**
- Constraints this measurement surfaces for (a). These are fixed at spawn and cannot change per turn:
  - `ANTHROPIC_CUSTOM_HEADERS`, which carries the per-turn GPU lease header, and the base URL / auth (lease turns go to llm-gateway, others to FCC);
  - `ORION_TURN_DEADLINE_EPOCH`, `ORION_TURN_BUDGET_SEC` and `CLAUDE_CODE_AUTO_COMPACT_WINDOW` (per-lane `n_ctx`);
  - `--model`, the MCP config (rendered per turn with a correlation id and reading binding), and the `--settings` breaker keyed per run.
  - A persistent process needs these made turn-invariant, or a pool keyed by them. The GPU lease header is the hard one.

## Side finding

`orion-aitown` MCP server reports `failed` on every start (5/5 inits inspected). It is rendered because `HARNESS_AITOWN_ENABLED` is truthy, but it is not usable. It is not on the critical path, since it fails fast. Not investigated further.

## Recommendation

Per the spec's rule ("ship (b) if it recovers most of the cost; (a) only if (b) leaves more than about 1.5 s"):

- (b) as specced (hook plus plugin) leaves **2.86 s**. That is over 1.5 s, so on its own terms the spec points to (a).
- But the real lever is the MCP server set. A chat-specific config with no MCP servers is 0.45 s, and with `github` only it is 0.82 s, both under 1.5 s with no lifecycle work.
- **Next decision for Juniper:** do chat-reply turns need `gitnexus` and `firecrawl`?
  - If no: ship a chat config dir with no plugin and a chat-only MCP set. That also covers L7. Expected about 2.6-2.9 s off each turn.
  - If yes: (a) is worth it, but design the per-turn lease header first.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

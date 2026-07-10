# FCC motor ToolSearch — design spec

**Date:** 2026-07-09  
**Status:** Approved for implementation  
**Parent:** FCC motor / hub agent-claude MCP path (`orion/harness/fcc_motor.py`, `services/orion-hub/scripts/fcc_claude_bridge.py`)  
**Related:** `docs/superpowers/specs/2026-07-05-orion-imperative-first-motor-design.md` (tools available; motor decides depth — no pre-motor classifier)

**One line:** Keep MCP servers attached; stop dumping their schemas into every FCC turn — enable Claude Code ToolSearch so the motor loads a tool library only when it needs it.

---

## Arsonist summary

Casual chat should not pay for GitHub/Firecrawl/AI Town tool schemas.

Orion already attaches MCP on FCC turns when MCP is enabled. That is fine. The bug is **eager schema injection**: Claude Code, when `ANTHROPIC_BASE_URL` points at FCC (a non-first-party host), falls back to loading all MCP tool definitions into context at spawn. Live symptom on a ~137k context window: turns report on the order of **~71 tools** being chewed before useful work starts. Bigger context made the drag more obvious, not less.

Claude Code already has the world-class answer: **ToolSearch** — catalog stays light; the model pulls schemas into the working set when it needs a capability. Orion never sets `ENABLE_TOOL_SEARCH`, so the custom-base-URL fallback wins and dumps everything.

Do not build stance gates, intent classifiers, or mid-turn MCP respawn routers. Flip the library switch. Confirm GitHub toolsets still reach the container so we are not also advertising an oversized GitHub surface.

---

## Current architecture

| Seam | Behavior today |
|------|----------------|
| `HARNESS_FCC_MCP_ENABLED` / `HUB_AGENT_CLAUDE_MCP_ENABLED` | When true, render ephemeral MCP config and pass `--mcp-config` + `--allowedTools mcp__<server>` |
| `orion/fcc/mcp_config.py` | Defaults `GITHUB_TOOLSETS` to `repos,pull_requests` if unset in `~/.fcc/.env` |
| `orion/fcc/context_budget.extend_fcc_subprocess_env` | Shared subprocess env for motor + hub (PYTHONPATH, MCP result caps, context ceiling) |
| `ANTHROPIC_BASE_URL` | Set to FCC server URL on every spawn |
| `ENABLE_TOOL_SEARCH` | **Unset** in Orion — Claude Code therefore falls back to upfront MCP schema load on custom base URL |
| Operator brief | Prompt discipline only (relational vs instrumental); does not change which schemas enter context |

**Evidence (operator-reported, UNVERIFIED in this design session):** ~137k ctx turns show ~71 tools in play; latency/drag is severe on ordinary chat.

**Operator FCC env note:** `~/.fcc/.env` currently has Firecrawl secrets but no explicit `GITHUB_TOOLSETS` line. Code default should still apply at render time; implementation must verify the rendered MCP JSON actually carries the restricted toolsets into the github-mcp container env.

---

## Problem

1. MCP schemas are loaded into context as soon as a turn starts, including rando chat that never calls them.
2. Custom `ANTHROPIC_BASE_URL` disables Claude's default ToolSearch deferral unless `ENABLE_TOOL_SEARCH` is set explicitly.
3. A large advertised tool surface (~71) burns tokens and time before the motor can hop.

This is not a missing Orion abstraction. It is a missing env contract on the existing spawn path.

---

## Design

### 1. Enable ToolSearch on every FCC claude spawn

In `orion/fcc/context_budget.extend_fcc_subprocess_env`, set:

```text
ENABLE_TOOL_SEARCH=true
```

Use `setdefault` so an operator can override for debugging (`false` / `auto` / `auto:N`) without a code change.

Both consumers already call this helper:

- harness motor: `orion/harness/fcc_motor.py` → `_build_subprocess_env`
- hub agent-claude: `services/orion-hub/scripts/fcc_claude_bridge.py` → `_build_subprocess_env`

No duplicate wiring in each bridge beyond tests.

### 2. Keep MCP attachment as-is

- Do **not** remove GitHub/Firecrawl/AI Town from the MCP template.
- Do **not** gate MCP on stance, imperative keywords, or turn mode.
- Do **not** add mid-session MCP attach/respawn logic.
- Motor still decides whether to *use* tools while hopping; ToolSearch decides when schemas enter context.

### 3. Confirm GitHub toolset restriction still applies

Same patch, thin verification only:

- Assert rendered MCP config (or motor render path) still injects `GITHUB_TOOLSETS` (default `repos,pull_requests`) into the github server env.
- Document that operators may set `GITHUB_TOOLSETS` in `~/.fcc/.env` to tighten further; empty/missing continues to use the code default.
- Do **not** invent a new Orion tool taxonomy or allowlist cathedral.

### 4. Config / docs surface

- Comment in hub + harness-governor `.env_example` near MCP flags: FCC spawns set `ENABLE_TOOL_SEARCH=true` via shared helper; override only for debug.
- No new required operator env key for the happy path (code sets it).
- If a service `.env_example` documents optional override `ENABLE_TOOL_SEARCH`, sync local `.env` via `python scripts/sync_local_env_from_example.py`.

### 5. Failure / compatibility note

ToolSearch requires the model/proxy path to accept `tool_reference` blocks. FCC is a custom `ANTHROPIC_BASE_URL`.

- **Happy path:** `ENABLE_TOOL_SEARCH=true` forces the beta/deferral path through the proxy; casual turns stop dumping full MCP schemas; tool-needing turns ToolSearch then call MCP.
- **If live smoke fails** (proxy rejects `tool_reference`, Haiku-class model, or schemas still dump): mark live path `UNVERIFIED` / `DONE_WITH_CONCERNS` and open a follow-up on FCC/gateway support — do **not** replace this with a stance router in the same patch.

---

## Non-goals

- Stance-based or imperative-classifier MCP attach/detach
- Progressive respawn (“bare hop then re-spawn with MCP”)
- Custom MCP multiplexer / lazy server proxy
- Changing which MCP servers exist in `fcc_claude_mcp.template.json`
- Changing Claude built-in tools (Bash/Read/…)
- Claiming sentience or “smart tool routing” beyond ToolSearch

---

## Files likely to touch

| Path | Change |
|------|--------|
| `orion/fcc/context_budget.py` | `ENABLE_TOOL_SEARCH=true` via `setdefault` |
| `orion/fcc/tests/test_context_budget.py` | Assert env key |
| `orion/harness/tests/test_fcc_motor_mcp.py` (or sibling) | Spawn env includes ToolSearch |
| `services/orion-hub/tests/test_fcc_claude_bridge_mcp.py` (or sibling) | Same for hub bridge |
| `services/orion-hub/.env_example` | Comment / optional override note |
| `services/orion-harness-governor/.env_example` | Same |
| `services/orion-hub/README.md` (short) | Document ToolSearch on FCC MCP path |

---

## Acceptance checks

1. Unit: `extend_fcc_subprocess_env` sets `ENABLE_TOOL_SEARCH=true` when unset; preserves explicit override.
2. Unit: motor and hub FCC spawn env (via shared helper or captured subprocess env) include `ENABLE_TOOL_SEARCH=true`.
3. Unit: MCP render still defaults `GITHUB_TOOLSETS` to `repos,pull_requests` when FCC env omits it.
4. Live smoke (when runnable): ordinary chat turn does not advertise/load the full MCP schema set into context; a turn that needs GitHub/Firecrawl still reaches those tools via ToolSearch. If smoke cannot run, report `UNVERIFIED` with exact command for Juniper.
5. No new keyword/stance MCP gate.

---

## Recommended next patch

Implementation plan via writing-plans: one thin PR — shared env setdefault + tests + env_example comments + short README note. Optional live smoke after restart of hub / harness-governor FCC consumers.

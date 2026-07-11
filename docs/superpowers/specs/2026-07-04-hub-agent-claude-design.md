# Hub Agent Claude — FCC harness in chat (v1)

**Date:** 2026-07-04  
**Status:** Draft — pending operator review  
**Scope:** `orion-hub` only (no FCC package changes, no new Orion service)  
**Goal:** Add Hub mode `agent-claude` that runs one Claude Code harness turn per chat message, streams harness events into the existing Hub chat WebSocket, and routes inference through the existing `fcc-server` → `orion-llm-gateway` stack.

---

## 1. Problem

Terminal workflow is verified: `fcc-server` + `fcc-claude` against `orion-llm-gateway` with `~/.fcc/.env` model mappings. Hub has no equivalent mode. Operator wants the same harness behavior in the Hub chat window without forking or patching the `free-claude-code` package (upgrade-safe).

---

## 2. Operator contract (confirmed)

### v1

| Requirement | Behavior |
|-------------|----------|
| Mode | New Hub mode **`agent-claude`** (Mode dropdown) |
| Turns | **Single-turn** per message; no cross-turn Claude session resume |
| Streaming | Harness `stream-json` events relayed live to chat UI |
| Model pick | Dropdown shows **FCC env key labels** (`MODEL`, `MODEL_OPUS`, …) for keys that are set in `~/.fcc/.env`; Hub sends label, bridge maps to Claude tier model name |
| Workspace | Fixed repo root (`HUB_AGENT_CLAUDE_WORKSPACE`), default `/mnt/scripts/Orion-Sapienform` |
| Gateway / Compute | Existing Hub **Compute** dropdown unchanged for other modes; **not** used for `agent-claude` |
| FCC package | **Never modified** — invoke installed `claude` binary, read `~/.fcc/.env` for labels only |
| Slash commands | **Deferred** — architecture must not block v2 |

### v2 (planned, not v1)

| Tier | Examples | Priority |
|------|----------|----------|
| Hub operator | `/stop`, `/stats`, `/clear` | **High** |
| Claude harness pass-through | `/compact`, `/help`, other Claude-internal verbs in prompt text | **High** |
| FCC messaging parity | reply-scoped stop, thread trees, `/fork` | **v3 at earliest** |

### v2 session resume (optional later slice)

Persist Claude `session_id` from stream-json; pass `--resume` on follow-up turns. Independent of slash tiers above.

---

## 3. Approach

**Hub subprocess bridge (Approach 1).** All on Athena node; no extra service hop.

```text
Browser Hub chat (WebSocket /ws)
    → websocket_handler (mode=agent-claude)
    → prepare_agent_claude_input()   [v1: pass-through; v2: slash dispatch]
    → fcc_claude_bridge.run_turn()
        → spawn: claude -p <prompt> --output-format stream-json …
        → cwd: HUB_AGENT_CLAUDE_WORKSPACE
        → env: ANTHROPIC_BASE_URL → fcc-server (HUB_FCC_SERVER_URL)
        → --model <tier-model-id> from fcc_model_label
    → parse stdout (stream-json lines) → WS frames
    → fcc-server POST /v1/messages
    → orion-llm-gateway route table
    → llama.cpp lane (per ~/.fcc/.env values)
```

**Rejected alternatives:**

- **Patch FCC** with `/v1/hub/run` — breaks upgrade path.
- **`orion-fcc-bridge` service** — unnecessary latency/complexity on single node (negligible ms, but extra failure surface).
- **`fcc-server` /v1/messages only** — no Claude Code tool loop; not the verified harness.

---

## 4. Architecture

### 4.1 Choke points

| Layer | File | Responsibility |
|-------|------|----------------|
| Mode routing | `services/orion-hub/scripts/websocket_handler.py` | Branch `mode=agent-claude` before cortex/context-exec |
| HTTP parity | `services/orion-hub/scripts/api_routes.py` | Same branch for HTTP chat if used |
| Bridge | `services/orion-hub/scripts/fcc_claude_bridge.py` | Subprocess spawn, parse, cancel registry |
| Env catalog | `services/orion-hub/scripts/fcc_env_catalog.py` | Read `~/.fcc/.env`; expose set model labels |
| Input seam | `services/orion-hub/scripts/agent_claude_input.py` | `prepare_agent_claude_input()` — v2 slash hook |
| UI mode | `services/orion-hub/templates/index.html` + `static/js/app.js` | Mode option + FCC model dropdown |
| Catalog API | `GET /api/fcc-model-labels` | Labels for dropdown (hub reads catalog module) |

### 4.2 Model label → Claude tier

FCC `fcc-server` resolves tiers by **substring** in the Claude model name (`opus`, `sonnet`, `haiku`) to `MODEL_*` env values. Hub bridge maps dropdown label to a stable tier model id:

| Label (from `.env` key) | `--model` passed to `claude` |
|-------------------------|------------------------------|
| `MODEL` | `claude-sonnet-4-20250514` (default tier) |
| `MODEL_OPUS` | `claude-opus-4-20250514` |
| `MODEL_SONNET` | `claude-sonnet-4-20250514` |
| `MODEL_HAIKU` | `claude-haiku-4-20250514` |

If multiple keys share the same resolved value (e.g. `MODEL` and `MODEL_OPUS` both → `llamacpp/chat`), show **each label** in the dropdown anyway; operator picks semantic tier, FCC routes identically.

Catalog endpoint returns only keys with non-empty values in the env file.

### 4.3 Subprocess invocation

Mirror FCC managed mode (no FCC import):

```bash
claude -p "<prompt>" \
  --output-format stream-json \
  --dangerously-skip-permissions \
  --verbose \
  --model <tier-model-id>
```

Optional v2 fields (ignored in v1): `--resume <session_id>`, `--fork-session`.

Environment (subset):

```text
ANTHROPIC_BASE_URL=<HUB_FCC_SERVER_URL>   # e.g. http://127.0.0.1:8082
ANTHROPIC_AUTH_TOKEN=<from HUB_FCC_AUTH_TOKEN or read ~/.fcc/.env>
CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
TERM=dumb
```

Binary resolution: `HUB_AGENT_CLAUDE_BIN` (default `claude`), must be on PATH where Hub process runs.

---

## 5. WebSocket contract

### 5.1 Client → Hub (v1)

Existing chat payload plus:

```json
{
  "mode": "agent-claude",
  "text": "explain the hub websocket handler",
  "session_id": "hub-…",
  "fcc_model_label": "MODEL_HAIKU",
  "claude_session_id": null,
  "resume": false
}
```

`claude_session_id` and `resume` accepted but **ignored** in v1.

### 5.2 Hub → Client (streaming)

Preserve stream-json shape; do not flatten to plain text before send.

**Step frame** (repeat):

```json
{
  "kind": "claude_step",
  "correlation_id": "…",
  "mode": "agent-claude",
  "step": {
    "type": "assistant",
    "raw": { }
  }
}
```

`step.type` and `step.raw` mirror the parsed JSON line from Claude stdout (same events as terminal `stream-json`).

**Final frame:**

```json
{
  "llm_response": "<extracted final user-visible text>",
  "mode": "agent-claude",
  "correlation_id": "…",
  "metadata": {
    "fcc_model_label": "MODEL_HAIKU",
    "claude_session_id": "… or null",
    "duration_ms": 12345,
    "exit_code": 0
  }
}
```

**Error frame:**

```json
{
  "error": "human-readable",
  "error_code": "fcc_claude_spawn_failed | fcc_claude_timeout | fcc_claude_nonzero_exit",
  "mode": "agent-claude",
  "correlation_id": "…"
}
```

### 5.3 UI rendering (v1)

- Reuse live-step pattern from `agent_step` (append incremental blocks in conversation pane).
- New renderer `appendLiveClaudeStep()` in `app.js` (or extend agent-trace panel with `claude_step` kind).
- Final answer appended as normal assistant bubble from `llm_response`.

---

## 6. Bridge module interface (v2-ready)

```python
# fcc_claude_bridge.py — public surface

async def run_turn(
    *,
    prompt: str,
    fcc_model_label: str,
    correlation_id: str,
    workspace: str,
) -> AsyncIterator[dict]: ...

async def cancel_turn(correlation_id: str) -> bool: ...

def active_turns() -> list[dict]: ...

# agent_claude_input.py

def prepare_agent_claude_input(text: str) -> TurnRequest | HubCommandResult:
    """v1: always TurnRequest. v2: intercept /stop, /stats, /clear."""
```

`cancel_turn` sends SIGTERM to registered subprocess; enables v2 `/stop` without FCC messaging.

---

## 7. Configuration

| Key | Default | Meaning |
|-----|---------|---------|
| `HUB_AGENT_CLAUDE_ENABLED` | `false` | Gate mode in UI and handlers |
| `HUB_FCC_ENV_PATH` | `~/.fcc/.env` | Model label catalog source |
| `HUB_FCC_SERVER_URL` | `http://127.0.0.1:8082` | `ANTHROPIC_BASE_URL` for subprocess |
| `HUB_FCC_AUTH_TOKEN` | empty | Override; else read from env file `ANTHROPIC_AUTH_TOKEN` |
| `HUB_AGENT_CLAUDE_BIN` | `claude` | CLI binary name/path |
| `HUB_AGENT_CLAUDE_WORKSPACE` | `/mnt/scripts/Orion-Sapienform` | subprocess `cwd` |
| `HUB_AGENT_CLAUDE_TIMEOUT_SEC` | `900` | Max turn duration |
| `HUB_AGENT_CLAUDE_MAX_CONCURRENT` | `1` | Parallel harness turns per hub instance |

Update `services/orion-hub/.env_example`, `settings.py`, `docker-compose.yml` (env pass-through only; **do not** require CLI inside container unless operator opts in — see risks).

---

## 8. Error handling

| Failure | Behavior |
|---------|----------|
| `HUB_AGENT_CLAUDE_ENABLED=false` | Mode hidden; WS returns `error_code=agent_claude_disabled` |
| `fcc-server` unreachable at turn start | Fail before spawn; message cites `HUB_FCC_SERVER_URL` |
| `claude` binary missing | `fcc_claude_spawn_failed` with PATH hint |
| Timeout | Kill subprocess; `fcc_claude_timeout` |
| Nonzero exit | Emit partial steps received; error frame with exit code |
| Malformed JSON line | Skip or wrap as `{type: "raw", content: line}`; do not abort turn |
| Second turn while one active | Reject when `MAX_CONCURRENT=1`; v2 `/stop` clears |

Preflight: `GET {HUB_FCC_SERVER_URL}/health` optional in `GET /api/fcc-model-labels` response as `fcc_server_ok`.

---

## 9. Testing

### Gate tests (deterministic)

- `test_fcc_env_catalog.py` — parse fixture `.env`; labels only for set keys.
- `test_fcc_model_label_mapping.py` — label → tier model id.
- `test_fcc_claude_bridge_parse.py` — fixture stream-json lines → step frames + final text extraction.
- `test_agent_claude_input.py` — v1 pass-through; stub v2 `/stop` dispatch contract.
- `test_websocket_agent_claude_routing.py` — mode branches to bridge mock, not cortex.

### Live smoke (operator)

```bash
# fcc-server running; claude on PATH; HUB_AGENT_CLAUDE_ENABLED=true
python services/orion-hub/scripts/verify_agent_claude_stream_live.py \
  --ws ws://127.0.0.1:8080/ws \
  --text "list files in services/orion-hub/scripts" \
  --fcc-model-label MODEL_HAIKU
# Exit 0 iff: >=1 claude_step frames AND final llm_response non-empty
```

---

## 10. Non-goals

- Modifying `free-claude-code` package or forking `fcc-server`
- Hub Compute dropdown changes for non-`agent-claude` modes
- FCC messaging tree / reply-scoped commands (v3)
- Running Claude harness inside Hub Docker image by default
- Slash command UI autocomplete (v2 polish)
- Orion bus / cortex / context-exec integration for this mode

---

## 11. v2 extension map

| Feature | Seam used | Effort |
|---------|-----------|--------|
| `/stop` | `cancel_turn()` + `prepare_agent_claude_input` | Small |
| `/stats` | `active_turns()` + bridge counters | Small |
| `/clear` | Hub-local; no subprocess | Small |
| Claude `/compact`, etc. | Pass prompt unchanged to `claude -p` | Trivial |
| Session resume | Capture `session_id` from stream-json; `--resume` argv | Medium |
| FCC messaging parity | New design; likely v3 | Large |

---

## 12. Risks

| Severity | Risk | Mitigation |
|----------|------|------------|
| High | Hub runs in Docker without `claude` on PATH | Document host-network dev; mount repo; run Hub on host for v1; `HUB_AGENT_CLAUDE_ENABLED=false` in container default |
| Medium | Long-running tool turns block WS | Timeout + cancel registry; v2 `/stop` |
| Medium | `~/.fcc/.env` not visible in container | `HUB_FCC_ENV_PATH` mount or copy labels into hub `.env` |
| Low | Stream-json schema drift from Claude Code upgrades | Pass through `raw`; tolerant parser |

---

## 13. Acceptance checks (v1)

1. Mode **Agent Claude** appears when `HUB_AGENT_CLAUDE_ENABLED=true`.
2. Model dropdown lists labels from `~/.fcc/.env` (not values).
3. Chat message streams `claude_step` frames then final `llm_response`.
4. Inference hits `fcc-server` logs (correlation visible) and gateway route matching selected tier.
5. No changes under `free-claude-code` install.
6. Gate tests pass; live smoke script documented.

---

## 14. Files likely to touch (implementation)

- `services/orion-hub/scripts/fcc_claude_bridge.py` (new)
- `services/orion-hub/scripts/fcc_env_catalog.py` (new)
- `services/orion-hub/scripts/agent_claude_input.py` (new)
- `services/orion-hub/scripts/websocket_handler.py`
- `services/orion-hub/scripts/api_routes.py`
- `services/orion-hub/app/settings.py`
- `services/orion-hub/.env_example`
- `services/orion-hub/templates/index.html`
- `services/orion-hub/static/js/app.js`
- `services/orion-hub/tests/test_fcc_*.py`
- `services/orion-hub/scripts/verify_agent_claude_stream_live.py` (new)
- `services/orion-hub/README.md`

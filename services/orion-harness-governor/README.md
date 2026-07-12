# orion-harness-governor

Bus worker for unified Hub turns. Listens on `orion:harness:run:request`, runs fcc motor + three-beat finalize (5a/5b/5c), replies with `HarnessRunV1`, and publishes audit artifacts.

## Channels

| Env key | Default | Role |
|---------|---------|------|
| `CHANNEL_HARNESS_RUN_REQUEST` | `orion:harness:run:request` | RPC intake from Hub |
| `CHANNEL_HARNESS_RESULT_PREFIX` | `orion:harness:run:result:` | Reply channel prefix |
| `CHANNEL_HARNESS_RUN_ARTIFACT` | `orion:harness:run:artifact` | Audit publish after each run |
| `CHANNEL_FINALIZE_APPRAISAL_REQUEST` | `orion:substrate:finalize_appraisal:request` | 5a draft molecule RPC |
| `CHANNEL_POST_TURN_CLOSURE` | `orion:substrate:post_turn_closure` | Step 7 learning closure |

## Flow

```text
LISTEN orion:harness:run:request
  → validate HarnessRunRequestV1 + thought disposition
  → HarnessRunner.run() — fcc motor + grammar receipts + draft_text
  → run_harness_finalize_chain() — 5a substrate / 5b reflect / 5c voice / 6b outcome
  → REPLY HarnessRunV1
  → PUBLISH orion:harness:run:artifact
  → emit_post_turn_closure (step 7)
```

## Local checks

```bash
PYTHONPATH=services/orion-harness-governor:. ./orion_dev/bin/python -m pytest services/orion-harness-governor/tests/ -v
PYTHONPATH=. ./orion_dev/bin/python -m pytest orion/harness/tests/ -v

docker compose \
  --env-file .env \
  --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml config
```

## Health

`GET http://localhost:7156/health`

## FCC MCP (Orion mode)

When `HARNESS_FCC_MCP_ENABLED=true`, harness turns spawn ephemeral MCP config (GitHub + Firecrawl; optional AI Town when `HARNESS_AITOWN_ENABLED=true`; optional GitNexus/Context Mode, below). The container image includes `docker`, Node 22, `npx`, the orion-aitown MCP package, and pinned `gitnexus@1.6.9` + `context-mode@1.0.169`.

### Semantic self-indexing (GitNexus + Context Mode)

Both are default-off, fail-open, and need no secrets:

- `HARNESS_FCC_GITNEXUS_ENABLED=true` adds the GitNexus code-graph MCP (`gitnexus mcp`). Prerequisite: build the index against the host checkout. The reliable path is the governor image itself (it bakes the LadybugDB FTS extension; without it search silently degrades to "FTS indexes missing"):

  ```bash
  mkdir -p ~/.gitnexus   # BEFORE first compose up, or docker root-owns it
  docker run --rm \
    -v /mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform \
    -v $HOME/.gitnexus:/root/.gitnexus \
    -w /mnt/scripts/Orion-Sapienform \
    --entrypoint gitnexus orion-harness-governor-harness-governor \
    analyze --index-only --name orion
  ```

  The generated `.gitnexus/` is gitignored; compose mounts `~/.gitnexus` read-only for registry discovery. Re-run after merges so `gitnexus status` reports up-to-date (the MCP discloses staleness but stale structure is never authority).
- `HARNESS_FCC_CONTEXT_MODE_ENABLED=true` adds the Context Mode MCP (MCP-only stage, no Claude hooks). Working data lives in the `harness-context-mode` volume at `HARNESS_FCC_CONTEXT_MODE_DIR` — operational data, not an Orion memory store; never expose it through Hub APIs.

#### Hook mode (Stage B)

`HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED=true` runs Context Mode as a Claude Code plugin (PreToolUse/PostToolUse/PreCompact/SessionStart/Stop hooks) instead of the standalone MCP server, adding session continuity across compaction. This is the ordinary Orion-mode default; disable it only for an explicit hook isolation smoke. The plugin is installed once by the operator into the persistent `harness-claude-config` volume (mounted at `/root/.claude`), not baked at image build:

```bash
docker exec -it <container> claude plugin marketplace add mksglu/context-mode
docker exec -it <container> claude plugin install context-mode@context-mode
```

The smoke script `scripts/context_mode_hooks_smoke.py` must pass before changing this wiring. No duplicate registration: when both `HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED` and `HARNESS_FCC_CONTEXT_MODE_ENABLED` are true, hook mode wins and the standalone server is skipped.

Turning the env flag off stops Orion-side hook wiring, but an already-installed
Claude plugin can still load from the persistent volume. To fully disable hook
mode after an experiment:

```bash
docker exec -it <container> claude plugin disable context-mode
```

The unified-turn introspection experiment for these flags lives at `scripts/run_unified_turn_introspection_eval.py` with its fixture in `orion/harness/evals/fixtures/`.

`HARNESS_FCC_SKIP_PERMISSIONS=true` (default in compose) passes `--dangerously-skip-permissions` to `claude -p` even when the governor runs as root — otherwise Bash/MCP steps stall on approval prompts with no operator in Orion mode.

`HARNESS_FCC_STREAM_IDLE_TIMEOUT_SEC=180` is the maximum silence between
Claude `stream-json` events before the governor kills only that FCC subprocess
and returns `fcc_idle_timeout`. The full turn budget remains
`HARNESS_FCC_TIMEOUT_SEC`; set the idle value to `0` only for debugging if you
want the old whole-turn timeout behavior.

`HARNESS_FCC_INCLUDE_PARTIAL_MESSAGES=true` passes
`--include-partial-messages` to `claude -p` so the governor can observe local
FCC token streams before Claude Code emits a complete assistant/tool event.
The motor folds text deltas into runtime evidence, throttles progress frames
with `HARNESS_FCC_PARTIAL_PROGRESS_INTERVAL_SEC=15`, and fails as
`fcc_partial_stream_timeout` after
`HARNESS_FCC_PARTIAL_STREAM_TIMEOUT_SEC=90` seconds of partial-only output. The
partial text is marked unsafe, so finalize will not turn a local-model loop
into a user-visible answer.

`HARNESS_FCC_FORCE_NO_THINKING_MODEL=true` rewrites local `llamacpp/...` FCC
model ids to FCC's `claude-3-freecc-no-thinking/...` gateway ids before
spawning Claude Code. This keeps Claude Code from requesting unsupported
extended-thinking stream blocks from the local llama.cpp rail.

### Required secrets

Mount host `~/.fcc` (already wired in compose). In `~/.fcc/.env` (or path from `HARNESS_FCC_ENV_PATH`):

| Key | Used by |
|-----|---------|
| `GITHUB_PAT` | GitHub MCP (`docker run ghcr.io/github/github-mcp-server`) |
| `FIRECRAWL_API_KEY` | Firecrawl MCP (`npx firecrawl-mcp`) |

When `HARNESS_AITOWN_ENABLED=true`, also set `AITOWN_CONVEX_URL`, `AITOWN_ADMIN_KEY`, and `AITOWN_WORLD_ID` (optional: `AITOWN_ORION_AGENT_ID`, `AITOWN_ORION_PLAYER_ID`).

### Docker socket

GitHub MCP runs sibling containers via the host Docker daemon. Compose mounts `/var/run/docker.sock:/var/run/docker.sock` (same pattern as orion-hub).

### Enable and restart

```bash
# services/orion-harness-governor/.env
HARNESS_FCC_MCP_ENABLED=true
HARNESS_AITOWN_ENABLED=false   # optional

docker compose \
  --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml \
  up -d --build
```

Rebuild/restart after toggling MCP flags or changing `~/.fcc/.env` secrets.

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

## Tool-provenance audit

`orion/harness/tool_provenance_audit.py::detect_tool_provenance_mismatch()` runs once per turn in `HarnessRunner.run()`, after the fcc stream completes: flags when `draft_text` uses live-immediacy language ("this turn", "right now", "happening now", "in the background") while that same turn's `grammar_receipts` show a fetch-shaped tool call (`get_file_contents`, `read_file`, a web fetch). It's a post-hoc audit, not prevention — the fcc subprocess is single-shot with no mid-run injection point, so nothing here can stop a confabulated claim before it's generated (that's `orion/harness/prefix.py`'s `CONTEXT PROVENANCE` block's job, in the compiled motor prompt).

On a mismatch: `HarnessMotorResult.tool_provenance_audit` / `HarnessDraftMoleculeV1.tool_provenance_audit` are set (both `None` otherwise), a `GrammarAtomV1(atom_type="uncertainty_marker", semantic_role="exec_tool_provenance_mismatch")` is published on `CHANNEL_HARNESS_RESULT_PREFIX`'s underlying grammar channel alongside the rest of the turn's grammar receipts, and a `harness_tool_provenance_mismatch` warning is logged with the correlation ID. Deliberately kept separate from `grounding_status` (an overloaded error/overflow code that downstream consumers surface as a user-visible error) — this is a soft grounding signal on the claim, not a motor failure.

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

  The generated `.gitnexus/` is gitignored; compose mounts `~/.gitnexus` read-only for registry discovery. Re-run after merges so `gitnexus status` reports up-to-date (the MCP discloses staleness but stale structure is never authority, and a stale/unindexed state pushes the model toward source search instead — see the harness motor prefix in `orion/fcc/self_index_brief.py`).

  The incremental update (bare `analyze --index-only`, no `--force`) can fail outright on some repo states (observed 2026-07-12: `Failed calling LOWER: Invalid UTF-8` mid-run), which leaves an `incrementalInProgress` flag and a stale index behind. If a re-run reports that or `gitnexus status` won't clear, force a full rebuild instead of retrying incremental:

  ```bash
  docker run --rm \
    -v /mnt/scripts/Orion-Sapienform:/mnt/scripts/Orion-Sapienform \
    -v $HOME/.gitnexus:/root/.gitnexus \
    -w /mnt/scripts/Orion-Sapienform \
    --entrypoint gitnexus orion-harness-governor-harness-governor \
    analyze --index-only --force --name orion
  ```
- `HARNESS_FCC_CONTEXT_MODE_ENABLED=true` adds the Context Mode MCP (MCP-only stage, no Claude hooks). Working data lives in the `harness-context-mode` volume at `HARNESS_FCC_CONTEXT_MODE_DIR` — operational data, not an Orion memory store; never expose it through Hub APIs.

#### Hook mode (Stage B)

`HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED=true` runs Context Mode as a Claude Code plugin (PreToolUse/PostToolUse/PreCompact/SessionStart/Stop hooks) instead of the standalone MCP server, adding session continuity across compaction. The plugin is installed once by the operator into the persistent `harness-claude-config` volume (mounted at `/root/.claude`), not baked at image build:

```bash
docker exec -it <container> claude plugin marketplace add mksglu/context-mode
docker exec -it <container> claude plugin install context-mode@context-mode
```

The smoke script `scripts/context_mode_hooks_smoke.py` must pass before enabling this on ordinary turns. No duplicate registration: when both `HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED` and `HARNESS_FCC_CONTEXT_MODE_ENABLED` are true, hook mode wins and the standalone server is skipped.

The unified-turn introspection experiment for these flags lives at `scripts/run_unified_turn_introspection_eval.py` with its fixture in `orion/harness/evals/fixtures/`.

`HARNESS_FCC_SKIP_PERMISSIONS=true` (default in compose) passes `--dangerously-skip-permissions` to `claude -p` even when the governor runs as root — otherwise Bash/MCP steps stall on approval prompts with no operator in Orion mode.

### Stream stall detection

Claude Code only writes a `stream-json` line once a step fully completes — with no `--include-partial-messages`, a single assistant message that never reaches a stop condition produces zero output. Before `HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC` existed, the governor's only defense was `HARNESS_FCC_TIMEOUT_SEC` (900s default) applied to *each* `readline()` call, so one stuck message could hang a turn for the full 15 minutes with the Hub UI showing nothing.

`HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC=180` (default) bounds a single line separately from the whole-turn budget; a turn that goes this long without completing a step fails fast with `error_code=fcc_stream_stalled` instead of running out the whole-turn clock. The whole-turn timeout (`fcc_timeout`) still fires if the aggregate turn — many steps, each individually under the stall cap — exceeds `HARNESS_FCC_TIMEOUT_SEC`. Set the stall value to `0` to fall back to the old whole-turn-only behavior.

This does not fix a runaway upstream generation (e.g. a local model that never emits a stop token) — that failure mode lives in the model-serving stack outside this repo. It bounds how long a turn can be stuck waiting on one before the operator gets a diagnosable, fast failure instead of a silent hang.

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

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

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), independent of the request/cancel bus workers above.

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

## Conversation history

Each turn's `claude -p` subprocess is still single-shot: it starts, runs, and exits with no mid-run injection point (see below). What changed (2026-08-20) is that the *prompt it starts with* now carries real recent-turn history, not just the current message. `HarnessRunRequestV1.recent_turns` (`orion/schemas/harness_finalize.py`) holds up to `HARNESS_RECENT_TURNS_MAX` (8) bounded prior user/assistant messages, built by `orion.hub.turn_orchestrator.execute_unified_turn` from the caller's `continuity_messages` via the same `build_turn_window` normalizer the pre-turn-appraisal client already used. `orion/harness/prefix.py::compile_harness_prefix()` renders it into a `RECENT CONVERSATION` section of the compiled prompt before the subprocess ever starts. Before this, every unified-turn (`mode="orion"`) prompt was built from only the single current `user_message` — no prior-turn content at all — which produced generic, isolated-per-turn responses and repeated opening greetings on a live multi-turn session.

## Tool-provenance audit

`orion/harness/tool_provenance_audit.py::detect_tool_provenance_mismatch()` runs once per turn in `HarnessRunner.run()`, after the fcc stream completes: flags when `draft_text` uses live-immediacy language ("this turn", "right now", "happening now", "in the background") while that same turn's `grammar_receipts` show a fetch-shaped tool call (`get_file_contents`, `read_file`, a web fetch). It's a post-hoc audit, not prevention — the fcc subprocess is single-shot with no mid-run injection point (the prompt, including the `RECENT CONVERSATION` section above, is fully assembled before the subprocess starts and cannot be changed once it is running), so nothing here can stop a confabulated claim before it's generated (that's `orion/harness/prefix.py`'s `CONTEXT PROVENANCE` block's job, in the compiled motor prompt).

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

  The extension is baked in offline, from a host-local cache (`HARNESS_LBDB_EXT_CACHE_DIR`, default `/mnt/telemetry/duckdb-extensions/staging`) rather than downloaded from `extension.ladybugdb.com` at build time — that origin has had real sustained outages (Cloudflare 522) that block the build outright. Seed the cache once per host (idempotent; auto-refetches when `LBDB_EXT_VERSION`/`LBDB_EXT_PLATFORM` change):

  ```bash
  scripts/seed_lbdb_fts_extension_cache.sh
  ```

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

`HARNESS_FCC_SKIP_PERMISSIONS=true` (default in compose) makes `orion/fcc/claude_spawn.py::claude_permission_argv()` pass full-auto-approve permissions to `claude -p` — `--dangerously-skip-permissions` on the host, `--permission-mode bypassPermissions` when running as root (this container always does; no `USER` directive), requiring the Dockerfile's `ENV IS_SANDBOX=1`. See that function's docstring for why (Claude Code's own root-sandbox gate, and why the previous `dontAsk` mode was silently deny-by-default rather than auto-approve — confirmed live 2026-08-13). Otherwise Bash/MCP steps stall or get silently denied with no operator in Orion mode.

**This is genuinely full, unprompted Bash/tool access, not a narrowed grant** — know what the container can reach before relying on it. This container mounts `/var/run/docker.sock` (host Docker daemon) and `${HOME}/.ssh:/root/.ssh:ro` (the operator's real SSH key, for `git push`) — both real capabilities, not repo-write-only. The two things standing between a bad turn and real damage are (1) `HARNESS_FCC_WORKSPACE`'s disposable sandbox checkout, whose only path back to this repo is `git push` to a non-main branch gated by GitHub branch protection (`orion/fcc/sandbox_sync.py`), and (2) `--setting-sources user,local`, which drops this repo's own project-level hooks (including `destructive_git_guard`) for FCC turns — deliberately, since the read-only repo mount already covers what that hook protects, but it means no repo-committed hook gates a root FCC Bash call; whatever gates it must live in the operator-managed `harness-claude-config` volume instead (not checked by this repo or its tests).

### Stream stall detection

Claude Code only writes a `stream-json` line once a step fully completes — with no `--include-partial-messages`, a single assistant message that never reaches a stop condition produces zero output. Before `HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC` existed, the governor's only defense was `HARNESS_FCC_TIMEOUT_SEC` (900s default) applied to *each* `readline()` call, so one stuck message could hang a turn for the full 15 minutes with the Hub UI showing nothing.

`HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC=180` (default) bounds a single line separately from the whole-turn budget; a turn that goes this long without completing a step fails fast with `error_code=fcc_stream_stalled` instead of running out the whole-turn clock. The whole-turn timeout (`fcc_timeout`) still fires if the aggregate turn — many steps, each individually under the stall cap — exceeds `HARNESS_FCC_TIMEOUT_SEC`. Set the stall value to `0` to fall back to the old whole-turn-only behavior.

This does not fix a runaway upstream generation (e.g. a local model that never emits a stop token) — that failure mode lives in the model-serving stack outside this repo. It bounds how long a turn can be stuck waiting on one before the operator gets a diagnosable, fast failure instead of a silent hang.

### Served-model self-context

`HARNESS_LLM_GATEWAY_URL` (default `http://llm-gateway:8210`, the same `app-net` bridge-network hostname `orion-cortex-exec`/`orion-context-exec` already use) points `orion.harness.fcc_motor.probe_current_served_model` at orion-llm-gateway's `GET /routes`. Before every turn's prompt is compiled, this resolves the requested `fcc_model_label` (e.g. `MODEL_SONNET`) through `~/.fcc/.env` to a route key, reads that route's live-probed real model off the (already-existing, 15s-TTL-cached) `/routes` response, and injects it into the harness system prompt as `Backend model currently serving this turn: <model>` — the same fact that lands in `chat_history_log.response_identity` after the turn, but available to Orion *before* it answers, not just to operators after the fact.

Fails open to no line at all (never a placeholder) on: no label, a non-llamacpp backend (`MODEL_HAIKU`'s `nvidia_nim` route isn't in this route table), an unreachable gateway, or a route with no cached model yet (worker down). A self-context probe must never block or fail a turn over a missing fact about itself.

### Draft length ceiling

`orion/harness/fcc_motor.py::run_fcc_turn` kills the fcc subprocess with `error_code=fcc_draft_length_ceiling_exceeded` if the accumulated draft size reaches the model's context ceiling (`max_context_chars()` in `orion/fcc/context_budget.py` — `HARNESS_FCC_MAX_CONTEXT_TOKENS` tokens times `ORION_FCC_CHARS_PER_TOKEN`, 65536 × 4 chars by default). The ceiling is deliberately generous: it never fires on normal turns, and it explicitly skips the terminal `"result"` stream event (the CLI's own signal that a turn already finished), so a legitimately long-but-completed answer can't get its own already-generated payload double-counted into a false-positive kill. It only fires on true runaway generation.

### Reflection fail-closed fallback

If the 5b reflect LLM call itself fails (`run_finalize_reflection` in `orion/harness/finalize.py`) and the deterministic quick-lane gate is also blocked, the degraded fallback verdict is `alignment_verdict="misaligned"` (`reflection_source="degraded_llm_failure_fallback"`) — not `"aligned"`. Reflection failing is not evidence the draft is fine, so the fallback fails closed instead of open. The 5c voice-finalize pass always runs regardless of verdict, but a `"misaligned"` verdict is the documented signal (in `orion_voice_finalize.j2`) to materially revise the draft rather than pass it through unreviewed. This does not fix why 5b failed — check `alignment_notes` on the verdict artifact (`reflect_llm_failed: <exception excerpt>`) for that.

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

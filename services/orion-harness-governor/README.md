# orion-harness-governor

Bus worker for unified Hub turns. Listens on `orion:harness:run:request` (chat compute lane)
**and** `orion:harness:run:request:agent` (agent compute lane) via two independent dispatch
loops — same handler code either way, just two separate queues so a long agent-lane turn
(curiosity, Mode=Agent+Compute=Agent) can never make a chat-lane turn wait behind it. Runs fcc
motor + three-beat finalize (5a/5b/5c), replies with `HarnessRunV1`, and publishes audit
artifacts.

**Lane split (2026-09-07).** Before this, both lanes shared one dispatch loop that processed
turns strictly one at a time regardless of source — confirmed live that a single 40-minute
agent-lane run left a real chat turn waiting the entire time with no visible error until Hub's
RPC timeout. Which lane a turn's request goes out on is decided once, Hub-side, from the SAME
resolved model label that already picks the turn's model (`orion.hub.turn_orchestrator
._is_agent_compute_lane`) — not from "who called it" — so a manual Mode=Agent+Compute=Agent chat
turn shares the agent lane with curiosity (they already share one GPU), while ordinary chat never
waits behind either. See `orion/fcc/turn_lock.py` for why concurrent turns sharing the FCC
sandbox checkout is already a supported, pre-existing case, not a new risk this introduces.

## Channels

| Env key | Default | Role |
|---------|---------|------|
| `CHANNEL_HARNESS_RUN_REQUEST` | `orion:harness:run:request` | RPC intake from Hub — chat compute lane |
| `CHANNEL_HARNESS_RUN_REQUEST_AGENT` | `orion:harness:run:request:agent` | RPC intake from Hub — agent compute lane (independent dispatch loop) |
| `CHANNEL_HARNESS_RESULT_PREFIX` | `orion:harness:run:result:` | Reply channel prefix (shared — keyed by correlation ID, not lane) |
| `CHANNEL_HARNESS_RUN_ARTIFACT` | `orion:harness:run:artifact` | Audit publish after each run |
| `CHANNEL_FINALIZE_APPRAISAL_REQUEST` | `orion:substrate:finalize_appraisal:request` | 5a draft molecule RPC |
| `CHANNEL_POST_TURN_CLOSURE` | `orion:substrate:post_turn_closure` | Step 7 learning closure |

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), independent of the request/cancel bus workers above.
`GET /health` reports `lane_chat_alive` / `lane_agent_alive` so a dispatch loop that dies
silently is visible immediately rather than inferred later from turns going unanswered.

## RPC-health publish (on by default)

Every `RPC_HEALTH_PUBLISH_INTERVAL_SEC` (30s) the governor drains the shared dispatch
bus's RPC-health window and publishes `RpcHealthSnapshotV1` to `orion:rpc_health:snapshot`
(`instance="main"`; `orion-signal-gateway` passes it through as `rpc_health_harness_governor`,
`orion-equilibrium-service` folds `channel_latency` into its log-only EWMA baseline).
With `RPC_HEALTH_CHANNEL_LATENCY_ENABLED=true` the snapshot carries per-hop stats:

| Hop key | Meaning |
|---------|---------|
| `orion:cortex:exec:request:background` | finalize reflect / response repair RPC to cortex-exec (`rpc_request`) |
| `orion:substrate:finalize_appraisal:request` | 5a draft-molecule appraisal RPC |
| `fcc:<role>` / `fcc:route:<route>` | FCC motor leg wall time (`HarnessRunV1.fcc_elapsed_sec`: placement probe + `claude -p` subprocess + lifecycle publish). Success on exit code >= 0, timeout on `fcc_timeout`/`fcc_stream_stalled`; Hub cancels (negative exit), pre-spawn refusals and output-limit kills are skipped. `fcc:<role>` = the GPU pool role the turn's hold was granted (every call under a hold runs there); `fcc:route:<route>` = no hold, so only the requested gateway route is known (each call is placed separately); `fcc:unknown` when neither is known. Replaced `fcc:<served_model>` on 2026-09-29, which split one lane into several keys under pool spill |

`fcc:*` outcome mapping: `fcc_timeout` / `fcc_stream_stalled` (the motor's own timeout-kill)
-> timeout; any other run that spawned the subprocess -> success with its wall time; a
pre-spawn refusal (bad model label, lane context too small, spawn/MCP preflight failure) or
an externally killed run (negative exit code, i.e. Hub cancel) -> not recorded.
Consumer-first rollout: `channel_latency` is a new field on an `extra="forbid"` model --
rebuild `orion-signal-gateway` and `orion-equilibrium-service` on PR #2312's build first.
Hop key conventions: `orion/core/bus/rpc_health.py` module docstring.

## Flow

```text
LISTEN orion:harness:run:request          (chat lane, one loop)
LISTEN orion:harness:run:request:agent    (agent lane, another loop — same code)
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

`HARNESS_FCC_INTROSPECT_ENABLED=true` adds `orion-introspect`, a read-only MCP that lets Orion look up their own recorded activity. See [orion-introspect](#orion-introspect-orion-reading-back-their-own-records) below.

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

#### Repeat-failing-call breaker

Every FCC turn (including reading-only turns) gets a harness-owned PreToolUse hook, passed as `claude --settings` so it applies regardless of `--setting-sources` and independently of the Context Mode plugin. Before each tool call it re-reads the turn's own session transcript; once the same tool has failed `HARNESS_FCC_REPEAT_FAILURE_THRESHOLD` times (default 3, `0` disables) with the same normalized input, further identical calls are blocked for the rest of the turn and the model is told the call will not succeed and to change approach or answer with what it has. A call that ever succeeded in the turn is never blocked; different inputs are different calls. State is the transcript, so it is per turn by construction. Trace: governor log line `fcc_repeat_failure_breaker_fired corr=<id>`; the blocked tool_result (marker `[orion-repeat-failure-breaker]`) also rides the normal step frames. Code: `orion/fcc/repeat_failure_breaker.py`. Incident: urgent run a153451fe423 (2026-10-01), 19 identical failing `firecrawl_scrape` calls.

The unified-turn introspection experiment for these flags lives at `scripts/run_unified_turn_introspection_eval.py` with its fixture in `orion/harness/evals/fixtures/`.

`HARNESS_FCC_SKIP_PERMISSIONS=true` (default in compose) makes `orion/fcc/claude_spawn.py::claude_permission_argv()` pass full-auto-approve permissions to `claude -p` — `--dangerously-skip-permissions` on the host, `--permission-mode bypassPermissions` when running as root (this container always does; no `USER` directive), requiring the Dockerfile's `ENV IS_SANDBOX=1`. See that function's docstring for why (Claude Code's own root-sandbox gate, and why the previous `dontAsk` mode was silently deny-by-default rather than auto-approve — confirmed live 2026-08-13). Otherwise Bash/MCP steps stall or get silently denied with no operator in Orion mode.

**This is genuinely full, unprompted Bash/tool access, not a narrowed grant** — know what the container can reach before relying on it. This container mounts `/var/run/docker.sock` (host Docker daemon) and `${HOME}/.ssh:/root/.ssh:ro` (the operator's real SSH key, for `git push`) — both real capabilities, not repo-write-only. The two things standing between a bad turn and real damage are (1) `HARNESS_FCC_WORKSPACE`'s disposable sandbox checkout, whose only path back to this repo is `git push` to a non-main branch gated by GitHub branch protection (`orion/fcc/sandbox_sync.py`), and (2) `--setting-sources user,local`, which drops this repo's own project-level hooks (including `destructive_git_guard`) for FCC turns — deliberately, since the read-only repo mount already covers what that hook protects, but it means no repo-committed hook gates a root FCC Bash call; whatever gates it must live in the operator-managed `harness-claude-config` volume instead (not checked by this repo or its tests).

### Stream stall detection

Claude Code only writes a `stream-json` line once a step fully completes — with no `--include-partial-messages`, a single assistant message that never reaches a stop condition produces zero output. Before `HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC` existed, the governor's only defense was `HARNESS_FCC_TIMEOUT_SEC` (900s default) applied to *each* `readline()` call, so one stuck message could hang a turn for the full 15 minutes with the Hub UI showing nothing.

`HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC=420` (default, raised from 180 on 2026-09-08) bounds a single line separately from the whole-turn budget; a turn that goes this long without completing a step fails fast with `error_code=fcc_stream_stalled` instead of running out the whole-turn clock. The whole-turn timeout (`fcc_timeout`) still fires if the aggregate turn — many steps, each individually under the stall cap — exceeds `HARNESS_FCC_TIMEOUT_SEC`. Set the stall value to `0` to fall back to the old whole-turn-only behavior.

This does not fix a runaway upstream generation (e.g. a local model that never emits a stop token) — that failure mode lives in the model-serving stack outside this repo. It bounds how long a turn can be stuck waiting on one before the operator gets a diagnosable, fast failure instead of a silent hang.

**Raised 180 -> 420 (2026-09-08):** every confirmed world-pulse-read Stage 2 `empty_generation` failure investigated (`services/orion-hub/scripts/world_pulse_read_stage2.py`) matched the identical signature — `grounding_status = "fcc stream stalled for 180.0s without completing a step ..."`, two of them with `fcc_served_model=null`, meaning the model server hadn't even confirmed it started serving before the 180s ran out. That is a contended-shared-GPU symptom, not a genuinely dead process. The whole-turn budget (`HARNESS_FCC_TIMEOUT_SEC`) had already been raised three times for the same contended-GPU reason (900 -> 1600 -> 2400 across the curiosity-turn-budget and fcc-deadline-chain PRs) but this stall cap was never touched in any of those raises. 420s leaves ample room under the current 7200s whole-turn ceiling to still catch a genuinely dead process fast.

### Served-model self-context

`HARNESS_LLM_GATEWAY_URL` (default `http://llm-gateway:8210`, the same `app-net` bridge-network hostname `orion-cortex-exec`/`orion-context-exec` already use) is where a turn holding a GPU pool lease sends its Anthropic-compatible calls (below). It is no longer used for self-context: GPU pool stage 6.3 moved the pre-turn "what model am I about to run on" read off the gateway's retiring `GET /routes`. The runner now reads pool state once per turn over `orion:gpu_pool:state:request` (with the pool's config, `HarnessRunner._read_pool_state`) and shares it with the prompt line and the motor's context window (`orion/gpu_pool/route_view.py`). Before every turn's prompt is compiled, it resolves the requested `fcc_model_label` (e.g. `MODEL_SONNET`) through `~/.fcc/.env` to a route key, reads the model the pool discovered on the role that route lands on right now, and injects it into the harness system prompt as that route's *default* model (`Default backend model for route <route>: <model> ...`). Under the GPU pool that route read is only a default: a turn that holds a pool lease (`HarnessRunRequestV1.gpu_lease`, a durable run's hold) instead reads the pool's live state over `orion:gpu_pool:state:request` and states the model discovered on its granted role (`Backend model serving this turn: <model> (GPU pool role <role>, profile <profile>; ...)`), because every call under a hold runs on that role while an unheld call may be spilled elsewhere (`orion/gpu_pool/placement.py`; spec `docs/superpowers/specs/2026-09-24-gpu-pool-design.md`, reader impacts item 5).

Fails open to no line at all (never a placeholder) on: no label, a non-llamacpp backend (`MODEL_HAIKU`'s `nvidia_nim` route isn't in this route table), an unreachable gateway, or a route with no cached model yet (worker down). A self-context probe must never block or fail a turn over a missing fact about itself.

### Draft length ceiling

Claude's `system/thinking_tokens` progress counters are excluded from draft
accounting and grammar steps. Their UUID/session metadata is not model context;
counting each counter prematurely exhausted the ceiling and bloated finalization.
Actual assistant `thinking` text counts against the same context ceiling as
other content. Whole-turn and stalled-stream deadlines remain in force.

`orion/harness/fcc_motor.py::run_fcc_turn` kills the fcc subprocess with `error_code=fcc_context_ceiling_exceeded` (named `fcc_draft_length_ceiling_exceeded` before 2026-10-02 -- it was never about the draft) when its estimate of the turn's LIVE context reaches the lane's ceiling (`max_context_chars()` in `orion/fcc/context_budget.py` -- the lane's `n_ctx`, else `HARNESS_FCC_MAX_CONTEXT_TOKENS`, times `ORION_FCC_CHARS_PER_TOKEN`). The estimate starts at the prompt size, grows by every tool result, tool input, thinking and text block, and is **rebased** on every claude CLI `compact_boundary` stream event to the CLI's own `post_tokens` (or the prompt size when it reports none). Before that rebase it was a lifetime total, so long turns that had already compacted were still killed (live 2026-09-22..10-01). With the CLI autocompacting at `HARNESS_FCC_AUTOCOMPACT_PCT_OVERRIDE` (70%), this guard now only fires when compaction failed to keep up. It skips the terminal `"result"` event so a finished answer is never double-counted into a kill.

**Cut-short turns** (`orion/harness/cut_short.py`): when the motor stops a still-working turn (`fcc_context_ceiling_exceeded`, `fcc_timeout`, `fcc_stream_stalled`, `fcc_stream_line_limit`), the draft is no longer the last text fragment (usually a lead-in like "Let me check X:"). It is built from the turn's own recorded findings -- interim notes and every tool result paired with its call -- under a `[Cut short - not a finished answer.]` marker, with `compliance_verdict=partial`, `grounding_status=<code>`. A cut-short turn with no findings fails instead of shipping a marker with nothing behind it, and the governor re-attaches the marker if response repair rewrote it away.

### Reflection fail-closed fallback

If the 5b reflect LLM call itself fails (`run_finalize_reflection` in `orion/harness/finalize.py`) and the deterministic quick-lane gate is also blocked, the degraded fallback verdict is `alignment_verdict="misaligned"` (`reflection_source="degraded_llm_failure_fallback"`) — not `"aligned"`. Reflection failing is not evidence the draft is fine, so the fallback fails closed instead of open.

Mandatory `orion_voice_finalize` is gone. After 5a/5b, an ordinary turn either returns the exact motor draft (aligned + strain resolved) or runs conditional `orion_response_repair` only when 5b marks the draft `misaligned`, `uncertain`, or `strain_unresolved`. Fail-closed: if repair is required and fails, the known-bad draft is not published. `finalize_ran=true` means 5a/5b completed — it is not “a rewrite LLM ran.” Use `response_repair_ran` / `response_repair_reason` for the repair pass. Reading-only machine turns still validate and canonicalize motor JSON and skip prose repair. This does not fix why 5b failed — check `alignment_notes` on the verdict artifact (`reflect_llm_failed: <exception excerpt>`) for that.

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

## General reading tools

Unified Chat and curiosity may carry a server-authored `reading_binding` in `HarnessRunRequestV1`. The FCC motor renders an `orion-reading` stdio MCP entry with only `recommend_reading(url, why_now)` and `reading_status(request_id)`. Binding provenance is outside tool arguments; the transport uses the existing internal bus to Hub's Postgres queue owner. No writer database credentials are passed to this tool. Existing `HARNESS_FCC_MCP_ENABLED` controls normal turn MCP exposure.

Reading acceptance is transcript-grounded, not inferred from the draft. The motor matches each reading `tool_use` to its `tool_result`, requires an explicit `ok=true` envelope with the bound turn's deterministic request ID, and retains the validated status in `HarnessRunV1.reading_receipts`. Unknown acceptance replaces reading persistence/future-action prose before draft materialization; the same gate runs after voice finalization so a later model pass cannot restore it. Successful responses always expose the durable ID and current status.

Stage 1/2 reading turns instead set the trusted `reading_only` flag. Their actual process receives `--tools WebFetch,WebSearch --strict-mcp-config --setting-sources ''` and an explicit empty MCP config. This prevents source content from reaching shell, mutable graph tools or plugin execution while the server-owned queue/journal/Concept Atlas path retains responsibility for persistence. During finalization, 5b reflection still runs on the `agent` lane, but the structured motor response is deterministically parsed and canonicalized rather than passed through prose-oriented 5c. Invalid JSON fails the turn instead of masquerading as a successful empty-shell result. Ordinary turns retain the existing 5c voice pass. The general chat/curiosity tool configuration is unchanged apart from the new narrow entry.

Rebuild this service and Hub after applying the additive queue migration. See the [reading implementation report](../../docs/superpowers/pr-reports/2026-09-10-general-reading-pr.md) for exact tests, restart commands and unverified production behavior.

## orion-introspect: Orion reading back their own records

**What it is.** A read-only tool server Orion gets during a harness turn. With
it, Orion can look up what actually happened to them instead of reconstructing
it from impression.
- Today it covers what they learned from reading.
- As later slices land it will cover their dreams, reveries, curiosity runs
  and memories.

Every answer comes from the service that owns the data, over the bus, under a
correlation ID, so any claim Orion makes from it can be traced back to a stored
record.

Design: [`2026-09-28-orion-introspect-mcp-design.md`](../../docs/superpowers/specs/2026-09-28-orion-introspect-mcp-design.md).
Search by meaning: [`2026-09-28-orion-introspect-slice1b-semantic-search.md`](../../docs/superpowers/specs/2026-09-28-orion-introspect-slice1b-semantic-search.md).

### Tools

| Tool | What Orion can ask | Answered by | Request channel | Status |
|---|---|---|---|---|
| `reading_results` | What a reading actually taught them: recent finished reads, one read by `url`/`request_id`, or `query=` by meaning | orion-hub ([responder](../orion-hub/README.md#introspect-responder-reading_results)) | `orion:reading:tool:request`, operation `reading_result` | Live (slices 1 + 1b) |
| `dreams` | Narrative dreams and the sleep-cycle hypotheses they have already been offered, recent / one / by meaning | orion-dream | `orion:introspect:dream:request` | Designed (slice 2) |
| `reveries` | Their spontaneous-thought chains | orion-thought | `orion:introspect:reverie:request` | Planned |
| `curiosity` | What their curiosity runs set out to do and what came of it | orion-substrate-runtime | `orion:introspect:curiosity:request` | Planned |
| `memories` | Memory cards by meaning, with sensitivity labels | orion-recall | `orion:introspect:memory:request` | Planned; never listed when an outward-facing tool is attached |

`orion/introspect/tests/test_readme_coverage.py` fails when a tool the server
lists is missing from this table, or when a live request channel is not
documented in the README of the service that answers it. A new tool cannot
ship undocumented.

### Which turns get it

- **Flags.** `HARNESS_FCC_MCP_ENABLED` and `HARNESS_FCC_INTROSPECT_ENABLED`
  must both be true.
- **Turn type.** The turn must carry a reading binding: Unified Chat and
  curiosity turns do. Reading stages (`reading_only`) never get it, because
  their job is to read the source, not to recall.
- **Binding.** Built by the server from runtime facts
  (`orion/introspect/binding.py`). It is never part of tool arguments, so
  the model cannot set it. It holds no secrets: it sits in the MCP
  subprocess environment and the per-turn MCP config file.
  - It carries the parent run/trace IDs and `memory_allowed`, which is
    `not HARNESS_AITOWN_ENABLED`.
  - `orion/fcc/mcp_config.py` refuses to render a config with both an
    outward-facing tool and memory access (`fcc_introspect_outward_memory`),
    or without `ORION_BUS_URL` (`fcc_introspect_bus_missing`).
- **Launch.** Per turn, over stdio:
  `python3 -P -m orion.introspect.mcp_server`, with `ORION_BUS_URL` and
  `ORION_INTROSPECT_BINDING` in its environment.
- **Brief.** When attached, `orion/introspect/brief.py` adds usage lines to
  the harness prefix. The lines say when to call it, and that an error means
  unknown.
- **Motor accounting.** Its calls count as context-gathering in the motor's
  `context_gathering_ratio` (`orion/harness/fcc_motor.py`).

### Request path

```text
claude -p (FCC motor)
  -> orion-introspect (stdio, one per turn; validates arguments, rejects extra fields)
  -> bus RPC, 15 s timeout, reply on a channel derived from a fresh correlation ID
  -> owning service: Postgres read (+ Chroma for query=), re-gated
  -> IntrospectResultV1 -> tool result JSON
```

### Truth rules (every tool)

- **Read-only.** SELECTs only. Nothing is queued, retried, charged or written.
- **Bounded.** At most 5 items, 900-char text per item, `truncated` set when
  cut. Five full items stay under the 12,000-char MCP result budget
  (`ORION_FCC_MCP_TOOL_RESULT_MAX_CHARS`); a test pins the worst case. The
  proxy that enforces that budget does not wrap this server today; the
  bound keeps wrapping it later safe.
- **Scaled.** Every success carries `as_of` and `total_available`, so "5 of
  40" is distinguishable from "all 5".
- **Empty is not unknown.**
  - `items=[]` with `total_available=0` means nothing matched.
  - A timeout, malformed or mismatched reply, owner error, or search outage
    becomes an MCP tool error saying the answer is unknown. It never becomes
    an empty list.
  - The brief tells Orion to say "unknown", never "nothing happened".
- **Labeled.** Each item's `epistemic_status` is `record` (it happened, e.g.
  a memory card or a run outcome) or `unsettled` (something Orion had or read,
  not a settled fact: readings, dreams, reveries).
- **Trusted reply path.** Responders answer only when `reply_to` is exactly
  the channel derived from the request's correlation ID and the message kind
  matches. They never reply to a model-supplied subject.

### Search by meaning

Same pattern for every domain that supports `query=`:

- **Indexing.**
  - Each record is embedded once via vector-host `/embedding` (bge-large,
    1024-dim).
  - It is upserted through orion-vector-writer
    (`orion:vector:semantic:upsert`) into that domain's own Chroma
    collection, keyed by record ID with a `content_hash`.
- **Index loop.** A hash-aware loop in the owning service re-embeds changed
  text and doubles as the backfill.
- **Querying.**
  - A query embeds only the question and keeps hits at or above the
    domain's similarity floor.
  - Each hit is re-read from Postgres through the same gate as the recent
    view. The index is never the record.
- **Failure.** An embedder or Chroma failure, or an unbuilt or empty index,
  means unknown.
- **Floor.** Set per domain by a calibration eval on real data. Readings:
  `services/orion-hub/evals/run_reading_search_calibration.py`.
- **Code.** Reading search today: `orion/world_pulse_read/search.py`.

### Verify it live

- **Responder log.** `introspect op=<op> corr=<id> items=<n> total=<n> mode=<...>`
  on success. Readings: `introspect op=reading_result`, failures
  `reading_tool_failed ... category=...`.
- **Governor log.** `harness_grammar_step_published corr=<turn> ... tool=mcp__orion-introspect__<tool>`.
- **Smoke (read-only).**

  ```bash
  ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --limit 3
  ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --query "graphics cards"
  ```

  Exit 0 means a coherent answer, 1 a degenerate one, 2 an unknown one.
- **Tests.** `orion/introspect/tests`, run in CI by
  `.github/workflows/orion-reading-tests.yml`.

### Turn it off

Set `HARNESS_FCC_INTROSPECT_ENABLED=false` in
`services/orion-harness-governor/.env`, then recreate the governor from a
worktree that has the `.env` files:

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --no-build
```

Responders can keep running; nothing calls them.

### Adding a tool (one domain per PR)

1. `orion/schemas/introspect.py`: the operation plus an arguments model
   (`extra="forbid"`); register new models in `orion/schemas/registry.py`.
2. `orion/bus/channels.yaml`: request channel and result channel entries.
3. The owning service's responder, and an **"Introspect responder:
   `<tool>`"** section in that service's README. It must cover what it
   answers, which tables, the gate, the epistemic label, log lines and
   failure modes.
4. `orion/introspect/tools.py` (spec + description) and
   `orion/introspect/brief.py`.
5. The row in the table above; the coverage test enforces this.
6. A smoke mode in `scripts/smoke_introspect.py`; a calibration eval if it
   has `query=`.
7. `.github/workflows/orion-reading-tests.yml`: the owning service's path in
   the trigger list, and its responder tests in the `pytest` command.

Steps 1–5 land in the same PR. The coverage test fails if a request channel
exists without a Live row, a responder section and a listed tool, so a
contract-only PR ahead of the responder is refused on purpose: a channel with
no responder would read as "unknown" on every call.

## Admitted turns (GPU pool hold)

`HarnessRunRequestV1.gpu_lease` (a durable run's GPU pool hold ref, `GpuLeaseRefV1`) and
`inference_timeout_sec` are optional. Admitted requests can execute concurrently even when they
share an intake channel; legacy requests retain one executing turn per channel. Intake owns and
cancels its tasks on shutdown, and duplicate in-flight requests for the same correlation and hold
generation do not start a second motor. The FCC subprocess receives only its own encoded hold ref
(`X-Orion-Gpu-Lease`) in `ANTHROPIC_CUSTOM_HEADERS`; an inherited one is removed while unrelated
custom headers survive. Finalization carries the same hold through its Cortex requests
(`options.gpu_lease`), so every call attaches to the hold. The older durable token
(`resource_lease` / `X-Orion-Resource-Lease`) was deleted in GPU pool stage 4.6.

For held turns only, `ANTHROPIC_BASE_URL` targets the existing
`HARNESS_LLM_GATEWAY_URL` directly (default `http://llm-gateway:8210`). The external
FCC proxy has no repository-controlled guarantee that it forwards lease headers.
Direct Gateway delivery makes fencing inspectable and leaves the legacy FCC proxy
path unchanged. A leased subprocess uses a nonsecret CLI placeholder token and
removes the inherited Anthropic API key; the FCC proxy credential is not sent to
Gateway. The pool (via the gateway's `attach`) is the held request's fencing authority.

## Durable admission owner

Admitted harness turns carry the same GPU pool hold through the FCC motor,
reflection, optional re-reflection, and conditional response repair. These LLM calls attach to
the hold, so a continuation does not wait behind its own reservation; they name route `agent`
(the hold's work class, never the hold's role). Unheld finalization uses the turn owner lane:
chat for non-agent FCC labels (default Hub chat / `MODEL_SONNET`), agent for the agent FCC
model label.

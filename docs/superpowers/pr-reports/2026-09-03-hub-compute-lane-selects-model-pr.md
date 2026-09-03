# PR report: Hub's COMPUTE lane actually picks the model, and the lane's real window is the budget

## Summary

- Mode=Agent + Compute=Agent has **never once** run the 27B on circe:8015.
  Live: `harness_turn_trace` shows **132 of 132** harness turns over 7 days
  served by `Qwen3.6-35B-A3B-UD-Q5_K_M` (the chat/harness worker on :8011).
- Cause: two knobs for one concept. The COMPUTE dropdown writes `llm_route`;
  the unified turn picks its model from `fcc_model_label`, an unrelated
  `~/.fcc/.env` key. `llm_route` appears **nowhere** in `orion/hub/` or
  `orion/fcc/`. Mode=Agent sends no explicit label, so every turn fell through
  to `MODEL_SONNET` -> `llamacpp/harness` -> the 35B. COMPUTE was inert.
- Fixed on one axis, with **no second chat mechanism**: the lane now resolves
  to the FCC model, through `orion/llm/routes.py` — the module that already
  owns the route vocabulary and exists precisely because a copy of it drifted.
- The window then has to follow the lane. `agent` serves **32768** while
  `chat`/`harness` serve **131072**, against one env-wide
  `HARNESS_FCC_MAX_CONTEXT_TOKENS=131072`. llm-gateway now publishes each
  route's live `n_ctx`, and the motor budgets the turn against it.
- **No JS change was needed.** The payload already carried
  `{mode: "agent", llm_route: "agent"}`; the backend simply ignored it. That
  keeps UI risk at zero.
- **Live-verified end to end:** an Agent+Agent turn now reaches the 27B. It
  then fails on context — see "Outcome moved" and "Risks". That failure is the
  real remaining blocker and it is now legible instead of silent.

## Outcome moved

`Mode=Agent + Compute=Agent` goes from *silently running a different model than
the operator selected* to *actually dispatching to the selected lane*, with the
lane's true context window enforced end to end.

It does **not** yet complete a turn. Live 2026-09-03 the routed turn reached
circe:8015 and came back `Prompt is too long` / `claude exited with code 1` /
`verdict=partial`. Measured floor: a bare `claude -p "Say OK."` with no MCP
config and no harness prefix already costs **11,515 input tokens**; real
production harness turns measured p90 19,151 / max 25,003 on the 131072 lane.
The 27B's 32768 window cannot hold Claude Code's baseline plus the harness
prefix plus MCP tool schemas. **This is a worker-side ctx_size decision on
circe, not a code fix** — see "Risks / concerns".

Two things that were previously invisible are now not:

- Before, an overrun returned **HTTP 200**, `stop_reason: "end_turn"`, exit 0,
  with llama.cpp's 400 as the *assistant's own text* — Orion would have spoken
  the provider error as its answer and it would have persisted as a success.
- Before, the pressure nudge on a 32768 lane was computed off 131072, firing at
  ~91750 tokens, i.e. never.

## Current architecture (before this patch)

```
COMPUTE dropdown --> payload.llm_route --> cortex_request_builder --> context-exec / cortex
                                             (never read by the unified turn)

Mode dropdown ----> payload.mode ------> turn_orchestrator --> HarnessRunRequestV1
                                             fcc_model_label = payload.get(...)
                                                or "MODEL_SONNET"   <-- always this
                                             --> ~/.fcc/.env MODEL_SONNET=llamacpp/harness
                                             --> gateway route `harness` --> circe:8011 (35B)

context ceiling: HARNESS_FCC_MAX_CONTEXT_TOKENS (governor 131072 / hub 65536),
                 a process-wide constant, identical for every lane.
```

## Architecture touched

- `orion/llm/routes.py` — the route vocabulary's single owner; gains the
  route -> FCC-model translation.
- `orion/hub/turn_orchestrator.py` — unified-turn model selection.
- `orion/harness/fcc_motor.py` — label resolution, lane probe, per-lane budget,
  lane-capacity pre-flight, laundered-provider-error detection.
- `orion/fcc/context_budget.py` — the budget helpers take a live window.
- `services/orion-llm-gateway/app/route_catalog.py` — publishes `n_ctx`.

Deliberately **not** touched: `context_exec_agent_bridge.py` /
`context_exec_client.py` / anything context-exec (dead prototype, explicitly
out of scope), and `services/orion-hub/scripts/fcc_claude_bridge.py` (see
"Risks").

## Files changed

- `orion/llm/routes.py`: `fcc_model_for_route()` + `FCC_LLAMACPP_MODEL_PREFIX`.
  Delegates to `normalize_llm_route`, inheriting its policy rather than
  restating it — aliases resolve, unknown is `None` ("no override", never a
  guess), and `SYSTEM_LLM_ROUTES` (`harness`) is refused.
- `orion/hub/turn_orchestrator.py`: `_resolve_fcc_model_label()`. Explicit
  label wins outright; only Mode=Agent derives from COMPUTE; anything else
  keeps today's default.
- `orion/harness/fcc_motor.py`:
  - `label_to_claude_model_id` accepts an already-resolved
    `"<backend>/<route>"` spec, checked **before** the env lookup.
  - `probe_current_served_model` split into `probe_route_runtime` (model +
    `n_ctx`) with the old name kept as a thin wrapper for its existing callers.
  - lane-capacity pre-flight; per-lane `ceiling_chars` / `pressure_chars` /
    `CLAUDE_CODE_AUTO_COMPACT_WINDOW`; provider-error envelope check.
- `orion/fcc/context_budget.py`: `max_context_tokens(n_ctx)` and friends;
  `is_provider_error_envelope()`; `context_risk_level` warn-line fix.
- `services/orion-llm-gateway/app/route_catalog.py`: `n_ctx` on
  `RouteHealthEntry`, `_probe_model`, `_probe_backend`, `_entry_from_probe`,
  `_entry_to_dict`.
- `orion/harness/tests/test_compute_lane_model_and_ceiling.py` (new, 30 tests)
- `services/orion-llm-gateway/tests/test_route_catalog_n_ctx.py` (new, 11 tests)

## Schema / bus / API changes

- Added: `n_ctx` on each row of `GET /routes` (llm-gateway HTTP API).
  Additive and nullable; `null` means "not known" (older worker, worker down,
  unexpected shape) and every consumer falls back rather than treating it as
  unlimited.
- Removed / renamed: none.
- Behavior changed: `fcc_model_label` may now carry a
  `"<backend>/<route>"` spec in addition to a `~/.fcc/.env` key. Existing
  callers passing `MODEL_SONNET`/`MODEL_HAIKU` are unaffected.
- No bus channel or schema-registry change. `HarnessRunRequestV1` is
  **unchanged** — the governor resolves the window itself from the label it
  already receives, so there is no rolling-deploy compatibility concern.

## Env/config changes

- Added / removed / renamed keys: **none**.
- `.env_example` updated: not required (no key changes).
- local `.env` synced: n/a — nothing to sync.
- Note, not changed by this PR: `services/orion-llamacpp-host/.env` carries
  `ATLAS_AGENT_HOST_PORT=8014` while `.env_example`, the compose default and
  the gateway route table all say **8015**, and 8014 collides with
  orion-circe-diffusion-host. Athena's copy is not what deploys circe, so this
  is a parity break rather than the cause of anything here.

## Tests run

```text
PYTHONPATH=. pytest orion/harness/tests/ orion/fcc/tests/test_context_budget.py -q
-> 289 passed, 3 failed

The 3 failures are PRE-EXISTING, verified by running the same files on a clean
`main` with `git status` empty -- identical 3 failures, same names:
  test_grounding_capsule_consumers.py::test_stance_react_prompt_renders_identity_when_present
  test_grounding_capsule_consumers.py::test_stance_react_prompt_renders_without_identity
  test_harness_runner.py::test_harness_runner_surfaces_fcc_error_code

cd services/orion-llm-gateway && pytest tests/ -q
-> 286 passed

PYTHONPATH=services/orion-hub:. pytest \
  services/orion-hub/tests/test_turn_orchestrator_ws_frames.py \
  services/orion-hub/tests/test_hub_agent_mode_fcc_routing.py \
  services/orion-hub/tests/test_llm_route_selector.py -q
-> 66 passed, 3 failed

Those 3 are also PRE-EXISTING and all context-exec (baselined identically on
clean main); PR #2048's report names the same 3.
```

### Mutation testing

A green suite proves nothing until the code is broken and the suite goes red.
Every gate added here was mutated against the real file (not a synthetic
fixture) and the mutation reverted after each run:

```text
CAUGHT route-spec passthrough removed (falls back to MODEL = wrong model, silently)
CAUGHT explicit label no longer wins (COMPUTE would steer Orion mode)
CAUGHT mode gate dropped entirely (lane steers every mode)
CAUGHT live window ignored, env ceiling always wins
CAUGHT warn line back on the process env instead of the caller's ceiling
CAUGHT envelope check widened to any error substring (eats real prose)
CAUGHT n_ctx type guard removed (a bool/str would be accepted)
CAUGHT reads n_ctx_train (the trained max) instead of n_ctx (the started window)
CAUGHT bool guard removed (True accepted as a window)
CAUGHT stale window survives a down worker
CAUGHT n_ctx dropped from the serialized contract
CAUGHT guard removed entirely
CAUGHT guard fires on any known lane
CAUGHT unknown window treated as zero-capacity
CAUGHT compares chars against tokens
CAUGHT boundary flipped
```

Two of the lane-guard tests were **vacuous on the first pass** and mutation
testing is the only reason that was found: both survived a guard comparing
characters against tokens, and one survived a guard that refused on an unknown
window. Both were rewritten to pin the distinguishing case — a prompt sized
between `n_ctx` and `n_ctx * chars-per-token`, and a prompt over the env
fallback with the window unknown.

## Evals run

None — no eval harness exists for Hub chat-mode/lane dispatch. The live
production smoke below is the closest equivalent, and is what actually found
the remaining blocker.

## Docker/build/smoke checks

All three affected services rebuilt and redeployed on athena via
`scripts/safe_docker_build.sh` from the worktree. Safe here specifically
because the governor's repo mount defaults to an **absolute** path
(`${HARNESS_REPO_MOUNT:-/mnt/scripts/Orion-Sapienform}`) and its Dockerfile
`COPY orion ./orion` from a `context: ../..` build context — so no disposable
worktree path is pinned into the running deploy.

```text
safe_docker_build.sh orion-llm-gateway     up -d --build   -> started
safe_docker_build.sh orion-harness-governor up -d --build   -> started
safe_docker_build.sh orion-hub             up -d --build   -> "Startup complete — Hub is ready."
```

Live `GET /routes` after the gateway deploy — the runtime proof that the window
is read, not assumed:

```text
route                  status    n_ctx  model
chat                   up       131072  Qwen3.6-35B-A3B-UD-Q5_K_M.gguf
quick                  up         4096  Qwen_Qwen3-8B-Q4_K_M.gguf
quick_background       up         4096  Qwen_Qwen3-8B-Q4_K_M.gguf
metacog                up         4096  Qwen_Qwen3-8B-Q5_K_M.gguf
metacog_background     up         4096  Qwen_Qwen3-8B-Q5_K_M.gguf
agent                  up        32768  Qwen3.8-27B-UD-Q4_K_XL.gguf
harness                up       131072  Qwen3.6-35B-A3B-UD-Q5_K_M.gguf
```

Cross-checked directly against each worker's own `/v1/models`
(`meta.n_ctx`): 8011=131072, 8012=4096, 8013=4096, 8015=32768. This is what
surfaced the 4096 lanes and prompted the pre-flight guard.

Live Agent+Agent turn, `POST /api/chat {"mode":"agent","llm_route":"agent"}`:

```text
type         final
mode         agent
chat_route   unified_turn_harness
correlation  19947bef-cc23-455a-976e-871cfb19014a
```

The reply text was NOT taken as evidence — a model's claim about its own
identity proves nothing. The ground truth came from the governor log and the
persisted artifact:

```text
harness_motor_complete steps=9 verdict=partial grounding=fcc_nonzero_exit
fcc motor error code=fcc_nonzero_exit err=claude exited with code 1
harness_turn_trace.run_artifact->>'draft_text' = "Prompt is too long"
```

i.e. the turn **did** route to the 27B and **did** overrun its window. Routing
verified; capacity is the open item.

Baseline measurement behind that conclusion (`claude -p` inside the governor,
no MCP config, no harness prefix):

```text
input_tokens 11515   <- Claude Code's floor for "Say OK."
```

A discarded hypothesis, recorded because it looked right: that `max_tokens`
counted against `n_ctx`. Probed directly — a ~5k-token prompt with
`max_tokens` of 512 / 8192 / 16384 / 30000 all returned OK, so it does not.
The prompt itself is genuinely over 32768.

## Review findings fixed

- Finding (self-review during live verification): the lane-capacity guard was
  placed after `_maybe_render_mcp_config`, so returning early would leak the
  rendered MCP config file — the cleanup lives in a `finally` that only starts
  at the subprocess.
  - Fix: moved the guard to immediately after the lane probe, before the FCC
    preflight, the MCP render, the turn lock and the subprocess.
  - Evidence: `test_a_lane_too_small_fails_before_spending_anything` booby-traps
    both `_preflight_fcc_server` and `_maybe_render_mcp_config` to raise, so a
    guard that fires too late fails loudly.
- Finding: `probe_current_served_model` would have silently stopped resolving
  for lane-selected turns (`env.get("llamacpp/agent")` misses), costing the
  harness prompt its "what model am I running on" self-context on exactly the
  lane where it differs from the default.
  - Fix: same two label shapes, same order, as `label_to_claude_model_id`.
- Finding: `context_risk_level` took the ceiling from its caller but the warn
  line from process env — on a 32768 lane under a 131072 default, `warn_at`
  (~91750) sat **above** critical, so `warn` was unreachable and a turn's first
  signal was `critical`, after the overrun.
  - Fix: derive the warn line from the same ceiling the caller passed.
  - Evidence: `test_warn_is_reachable_below_critical_on_a_small_lane`.
- Finding (found in the live failure): the overflow operator hint quoted the
  container default, not the lane — the failed turn told the operator
  "context window full (~131072 tokens)" when the real window was 32768.
  - Fix: pass `n_ctx=lane_n_ctx` on the nonzero-exit path too.
  - Evidence: `harness_turn_trace` draft_text for
    `19947bef-cc23-455a-976e-871cfb19014a`.

## Restart required

Already done on athena during this session (gateway, governor, hub — all
rebuilt, recreated, healthy). No further restart required for this PR.

**Separately, and NOT done here** — raising the agent lane's window is a
circe-side change, see below.

## Risks / concerns

- Severity: **high (open blocker, not introduced by this PR)**
  - Concern: the 27B's 32768-token window cannot host a full FCC harness turn.
    Claude Code's floor alone is 11,515 tokens before any MCP schema or harness
    prefix; production turns measured 19,151 (p90) / 25,003 (max) on the
    131072 lane. Mode=Agent + Compute=Agent therefore routes correctly and then
    fails on context.
  - Mitigation: raise the agent worker's `ctx_size` on circe. The model's
    `n_ctx_train` is 262144, so 65536 or 131072 is architecturally fine; the
    cost is KV-cache VRAM on that card. **I could not author this change
    safely**: no profile in `config/llm_profiles.yaml` matches
    `Qwen3.8-27B-UD-Q4_K_XL.gguf`, so circe's agent worker is launched from a
    profile this repo does not contain, and Tailscale SSH to circe is denied
    from this host (`tailnet policy does not permit ... as user "athena"`), so
    I can neither read its config nor check GPU 1's free VRAM. Needs to be done
    where `nvidia-smi` is visible.
- Severity: medium
  - Concern: COMPUTE defaults to `quick`, which serves **4096** tokens. Mode=Agent
    on the already-selected default now routes there and is refused by the
    pre-flight guard. Before this PR it "worked" by silently running the 35B —
    i.e. by ignoring the operator's selection, which is the bug being fixed.
  - Mitigation: the refusal names the lane, the prompt size and the window, so
    it is self-explanatory. A UI-side filter that greys out lanes too small for
    Agent mode is the natural follow-up now that `n_ctx` is on `GET /routes`.
- Severity: medium
  - Concern: there really are **two FCC motors**. `orion/harness/fcc_motor.py`
    line 1 says "patterns adapted from
    services/orion-hub/scripts/fcc_claude_bridge.py", and the hub-local copy
    (532 lines) is still imported and called by `websocket_handler.py:894` and
    `api_routes.py:2986` — only for `mode == "agent-claude"`, a mode **no**
    `HUB_MODE_SPECS` entry produces, so it is UI-unreachable but still
    HTTP-reachable and still `HUB_AGENT_CLAUDE_ENABLED=true`. It carries its own
    ceiling (`HUB_AGENT_CLAUDE_MAX_CONTEXT_TOKENS=65536`) which disagrees with
    the governor's 131072. This PR adds nothing to it and does not make the
    duplication worse.
  - Mitigation: deliberate non-goal here — deleting it touches `turn_cancel.py`,
    `api_routes.py`, `websocket_handler.py`, a JS function
    (`applyAgentClaudePayloadFields`, itself dead: it gates on a mode no spec
    declares) and an env block. It deserves its own PR with its own
    "any other live callers?" check, not a rider on this one.
- Severity: low
  - Concern: `is_provider_error_envelope` could in principle reclassify a real
    answer as a failure.
  - Mitigation: matched on the envelope's opening framing, not on any error
    word, precisely because Orion writes genuine prose about its own
    infrastructure; a test pins that such prose survives, and a mutation
    widening it to a substring check is caught.

## PR link

<filled in after push>

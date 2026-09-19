# Stance capacity refusal surfaces by name, and reading/curiosity get a bounded agent-lane-then-chat-lane fallback

## Summary

- Autonomous reading (world-pulse Stage 1/2) and curiosity turns were failing
  ~70% of the time with the unhelpful `stance_react_failed: stance_react exec
  result missing thought payload`. The real cause: both GPU lanes are
  single-slot, and these turns ask for the busy "agent" lane and wait their
  entire step budget behind it, get shed by the gateway, and the gateway's
  shed reply (empty text + `raw.error`) was carried forward by cortex-exec as
  a normal, successful, empty answer instead of a failure.
- `services/orion-cortex-exec/app/executor.py`: new `gateway_error_step_failure()`
  recognizes that empty-content-with-`raw.error` reply shape and fails the LLM
  step by name (e.g. `gateway_capacity_rejected:capacity_wait_budget_exhausted`)
  instead of letting it pass as success.
- `services/orion-thought/app/bus_listener.py`: new
  `execute_stance_react_with_lane_fallback()` — for a turn that prefers the
  agent lane without a durable admission lease, try a short, bounded agent-lane
  attempt (new `STANCE_REACT_AGENT_LANE_BUDGET_SEC`, default 60s), then fall
  back once to the idle chat lane with a fresh correlation id and whatever
  budget is left.
- New `StanceReactRequestV1.caller_handles_lane_fallback` field lets
  endogenous outreach (which already runs its own agent-then-chat retry
  around the whole turn, PR #2163) opt out of this second fallback layer, so
  one outreach tick still gets exactly 2 stance attempts, not 4.
- Named failures propagate through `orion-thought`'s
  `extract_stance_react_payload`/`exec_failure_reason`, so a deferred turn now
  reads `stance_react_failed: gateway_capacity_rejected:...` instead of the
  generic message.

## Outcome moved

A reading/curiosity turn that hits a busy agent lane now spends at most ~60s
queueing there before falling back to the idle chat lane, instead of spending
its entire ~240s step budget queued and generating nothing. When both lanes
genuinely fail, the deferred-turn reason names which gateway refusal happened
and where, instead of the generic "missing thought payload" that gave no clue
the real cause was a GPU-lane queue.

## Current architecture

Before this patch: `orion.hub.turn_orchestrator` set `StanceReactRequestV1.llm_route="agent"`
for reading/curiosity turns (no durable lease), but `orion-thought`'s
`run_stance_react` made exactly one RPC to cortex-exec with the verb's full
timeout (`stance_react.yaml`, `timeout_ms: 240000`) and no lane-aware retry.
Meanwhile `orion-llm-gateway` sheds an overloaded/capacity-refused request as
a normal `llm.chat.result` with empty `content`/`text` and `raw.error` set
(`_overloaded_result`, the `CapacityRejected`/`ResourceLeaseRejected` catch in
`_dispatch_chat`) — its own docstring already flagged that cortex-exec's
chat-turn step did not check `raw.error` or empty content, so that empty
answer was carried forward as a successful step. `extract_stance_react_payload`
then found nothing usable anywhere in the plan result and raised the generic
`ValueError("stance_react exec result missing thought payload")`.
Endogenous outreach already had its own two-attempt agent-then-chat fallback
around the whole unified turn (PR #2163), but reading and curiosity had no
equivalent.

## Architecture touched

- `services/orion-cortex-exec/app/executor.py` — LLM step result handling
  (`call_step_services`).
- `services/orion-thought/app/bus_listener.py` — stance_react plan-request
  construction and RPC orchestration.
- `services/orion-thought/app/settings.py`, `.env_example`,
  `docker-compose.yml` — new `STANCE_REACT_AGENT_LANE_BUDGET_SEC` key.
- `orion/schemas/thought.py` — additive `StanceReactRequestV1.caller_handles_lane_fallback`
  field.
- `orion/hub/turn_orchestrator.py` — stamps that field for outreach's turns.

## Files changed

- `services/orion-cortex-exec/app/executor.py`: added `gateway_error_step_failure()`
  and wired it into the LLM step's result handling so a shed/refused gateway
  reply fails the step by name.
- `services/orion-cortex-exec/tests/test_llm_gateway_overloaded_reply.py`
  (new): regression tests using the literal gateway reply shapes.
- `services/orion-thought/app/bus_listener.py`: added `exec_failure_reason()`,
  `lane_fallback_applies()`, `_run_stance_react_attempt()`,
  `execute_stance_react_with_lane_fallback()`; `build_stance_react_plan_request()`
  gained `llm_route_override`/`step_timeout_cap_sec`/`request_id` params;
  `extract_stance_react_payload()` now raises the named `exec_failure_reason`
  when available instead of only the generic message; `run_stance_react()`
  calls the new fallback orchestrator.
- `services/orion-thought/tests/test_stance_react_lane_fallback.py` (new):
  covers `exec_failure_reason`, `lane_fallback_applies`, the happy path, named
  agent-lane failure falling back to chat, both lanes failing, the
  below-gateway-floor budget clamp, the "not enough budget left to try chat"
  skip, the chat-lane step-cap floor, a durable-lease request (no fallback),
  and a `caller_handles_lane_fallback=True` request (no fallback).
- `services/orion-thought/app/settings.py`: added `stance_react_agent_lane_budget_sec`.
- `services/orion-thought/.env_example`, `services/orion-thought/docker-compose.yml`:
  added `STANCE_REACT_AGENT_LANE_BUDGET_SEC`.
- `orion/schemas/thought.py`: added `StanceReactRequestV1.caller_handles_lane_fallback`
  (additive, default `False`).
- `orion/hub/turn_orchestrator.py`: `stance_req` now sets
  `caller_handles_lane_fallback=payload.get("source") == "endogenous_outreach"`.

## Schema / bus / API changes

- Added: `StanceReactRequestV1.caller_handles_lane_fallback: bool = False`
  (`orion/schemas/thought.py`). Additive field on a model with no
  `extra="forbid"` — an older producer that omits it, or an older consumer
  reading a payload that carries it, both continue to work unchanged. No
  registry update needed (the registry references the class, not a field
  list).
- Removed: none.
- Renamed: none.
- Behavior changed: a stance_react plan step that previously "succeeded" with
  empty content and a gateway `raw.error` now fails with a named reason.
  Anything reading `PlanExecutionResult.status`/`.error` for the stance_react
  verb sees more `fail` results than before in the capacity-refusal case — this
  is the intended fix, not a regression, since the prior "success" carried no
  usable content anyway.
- Compatibility notes: `StanceReactRequestV1` producers other than
  `turn_orchestrator.py` (there are none live today besides it and any future
  ones) get the safe default (`False`, i.e. get the new fallback) with no
  changes needed on their part.

## Env/config changes

- Added keys: `STANCE_REACT_AGENT_LANE_BUDGET_SEC` (default `60`,
  `services/orion-thought/.env_example`, `services/orion-thought/app/settings.py`,
  `services/orion-thought/docker-compose.yml`).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (`services/orion-thought/.env_example`).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes
  — `orion-thought: +STANCE_REACT_AGENT_LANE_BUDGET_SEC='60'` (written to the
  primary checkout's `.env`, since the sync script always targets there per
  this repo's convention).
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-cortex-exec/tests/test_llm_gateway_overloaded_reply.py
services/orion-cortex-exec/tests/test_resource_lease_forwarding.py
  22 passed

services/orion-thought/tests/test_stance_react_lane_fallback.py
services/orion-thought/tests/test_stance_context_llm_route.py
services/orion-thought/tests/test_bus_listener_errors.py
  23 passed (18 + 5)

services/orion-thought/tests (full)
  428 passed, 3 failed, 12 skipped
  -- the 3 failures (test_settings_mind_enrichment, test_settings_salience_flags,
     test_reverie_spontaneous_thought) are pre-existing, host-local-.env-value
     mismatches against .env_example defaults (ORION_MIND_BASE_URL,
     ORION_ATTENTION_SALIENCE_V2_ENABLED, a DB hostname), unrelated to any file
     this patch touches. Independently confirmed by the review subagent.

services/orion-cortex-exec/tests (full, excluding 14 files with a pre-existing
verb-registry double-registration collection error when the whole ~1050-test
directory is collected together -- confirmed via git-stash A/B: 14 identical
collection errors exist on the unpatched tree too)
  946 passed, 105 failed
  -- A/B against the unpatched tree (same exclusion list): 935 passed, 102
     failed. The only diff is this patch's own 3 new tests joining the same
     pre-existing full-suite-only instability (they pass cleanly in every
     targeted/isolated run above) -- confirmed identical failure list via diff,
     modulo those 3. No regression introduced.

services/orion-hub/tests/test_turn_orchestrator_utterance_origin.py
services/orion-hub/tests/test_turn_orchestrator_ws_frames.py
services/orion-hub/tests/test_thought_client.py
services/orion-hub/tests/test_curiosity_investigation.py
services/orion-hub/tests/test_endogenous_outreach.py
services/orion-hub/tests/test_endogenous_outreach_decisions.py
services/orion-hub/tests/test_endogenous_outreach_self_prior_filter.py
  394 passed
```

## Evals run

No eval harness exists for `orion-thought` or `orion-cortex-exec` for this
seam. Not added here — this is a bug fix to an existing failure-classification
and retry path, covered by unit/regression tests against the literal gateway
reply shapes and the timeout-budget boundary math; a live post-deploy check is
listed below as the real behavioral proof.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-thought build       -> Built
scripts/safe_docker_build.sh orion-cortex-exec build   -> Built (4 images:
  cortex-exec, cortex-exec-chat, cortex-exec-spark, cortex-exec-background)
```

## Review findings fixed

- Finding: `stance_react_agent_lane_budget_sec` had no floor -- a
  misconfigured value under cortex-exec's own 45s gateway-read-timeout floor
  would make this service give up on its RPC before the gateway's shed/serve
  decision could land.
  - Fix: `execute_stance_react_with_lane_fallback` clamps a sub-floor
    configured budget up to `STANCE_REACT_MIN_STEP_TIMEOUT_SEC` (45s) and
    logs a warning naming the misconfiguration.
  - Evidence: `test_agent_lane_budget_below_gateway_floor_is_clamped_up`.
- Finding: no test covered the two timeout-budget boundary branches (skip the
  chat attempt entirely when remaining budget is already at/under the gateway
  floor; floor the chat attempt's own step cap at 45s when remaining budget is
  tight) -- exactly where an off-by-one or sign error would first show up.
  - Fix: added both boundary tests.
  - Evidence: `test_insufficient_remaining_budget_skips_chat_attempt_entirely`,
    `test_chat_attempt_step_cap_floors_at_45s_when_remaining_is_tight`.
- Finding (recorded, not a defect): `_run_stance_react_attempt` catches bare
  `Exception` and treats every failure shape as a fallback trigger, including
  a hypothetical future bug in `build_stance_react_plan_request` itself. This
  is a deliberate, documented tradeoff (bounded to 2 total attempts) — no
  change made.
- Finding (recorded, not a defect): the step-timeout cap in
  `build_stance_react_plan_request` loops over every step in the plan, not
  just the LLM step. `stance_react.yaml` has exactly one step today, so this
  is a no-op concern; flagged as a tripwire for whoever next edits that verb's
  plan, no change made.
- Independently confirmed clean (no fix needed): timeout math never goes
  negative or double-counts; `gateway_error_step_failure` matches every real
  reply shape `orion-llm-gateway` can produce with no plausible false
  positive; the plan-step-capping wrapper is the sole caller of
  `build_plan_for_verb("stance_react", ...)` in the repo; the full
  `endogenous_outreach.py` -> `turn_orchestrator.py` ->
  `ThoughtClient.react` -> `orion-thought` bus listener call path was traced
  end-to-end and outreach gets exactly 2 total stance attempts, not 4; the
  new schema field is additive-safe.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: `endogenous_outreach.py`'s own caller-side fallback (PR #2163)
    still exists unchanged; this patch relies on `caller_handles_lane_fallback`
    to prevent it from stacking with orion-thought's new fallback. If a future
    caller sets `llm_route="agent"` with no lease and also implements its own
    retry without setting this flag, it would get 4 stance attempts instead of
    2.
  - Mitigation: the field's docstring on `StanceReactRequestV1` and the
    comment at its one producer (`turn_orchestrator.py`) both name this
    explicitly; `test_caller_handled_request_makes_exactly_one_call` pins the
    contract for outreach specifically.
- Severity: low
  - Concern: the live fix cannot be proven from a worktree — it needs the
    agent lane genuinely held busy by a real curiosity/harness turn to
    reproduce the queue-then-shed path.
  - Mitigation: see UNVERIFIED section below for the exact post-deploy checks.

## UNVERIFIED (live)

The fallback's actual effect on production world-pulse reads is **UNVERIFIED**
— it was not exercised against a live busy agent lane in this session. Exact
checks to run after deploy:

```bash
psql "postgresql://postgres:postgres@127.0.0.1:55432/conjourney" -c \
  "select status,last_error,completed_at from world_pulse_read_seed order by completed_at desc nulls last limit 10"
```
Expect: reads completing, or a `last_error` naming
`gateway_overloaded:*`/`gateway_capacity_rejected:*` instead of "missing
thought payload".

```bash
docker logs orion-athena-thought | grep stance_react
```
Expect: `stance_react_attempt ... lane=agent ... outcome=failed` followed by
`stance_react_lane_fallback ... from=agent to=chat` followed by
`stance_react_attempt ... lane=chat ... outcome=ok` lines for turns that hit a
busy agent lane.

## PR link

<opened after push — see final response>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

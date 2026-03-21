# Agent no-recall timeout manifest — pass 2

## A. Claim status

### Proven by tests

- Disabled-recall Hub Agent requests preserve the same supervised routing intent as recall-enabled Hub Agent requests.
- Orch emits dedicated per-request verb-runtime reply channels.
- Exec publishes verb-runtime results to the dedicated reply channel and mirrors to the legacy shared channel.
- Gateway replies to the per-request Hub gateway result channel.
- Disabled recall by request shape alone does not suppress the PlannerReact -> AgentChain supervised route.
- The live proof harness now records same-corr request config, ordered hops, exact reply channels, and required log keys.

### Proven by live trace

- **Not yet proven in this container.**
- This environment cannot reach the configured Redis bus, so a successful same-corr live stack trace was not captured here.
- The code now contains the same-corr proof path and artifact generation needed to capture that trace on a live stack with one command.

### Still inferred

- The timeout diagnosis is now strongly supported by code-level tests and by the dedicated-reply implementation, but the specific historical failing run remains **inferred** until the new proof command is executed against a reachable live stack and its artifact shows the full same-corr success path.

## B. Exact root cause

Supported root-cause statement:

- The prior Orch <-> Exec verb-runtime RPC/result handling used the shared `orion:verb:result` fan-in stream.
- That design made the user request lifecycle difficult to prove by one corr_id and left the Orch waiter exposed to unrelated background verb traffic on the same stream.
- The fix moves Orch waiting to a dedicated per-request reply channel while preserving the existing Cortex path and keeping a legacy mirror for compatibility.

What is **not** supported strongly enough to claim yet:

- We cannot claim from this container alone that every historical timeout was definitively caused by a shared-channel collision, because we could not replay the live failing system state.

## C. Same-corr live proof

### Status

- A successful same-corr disabled-recall live trace was **not captured here** due unreachable live infrastructure.
- The proof command below is now the authoritative way to capture it on a live stack.

### One-command live proof

```bash
python scripts/run_answer_depth_live_proof.py --scenario discord --disable-recall
```

### What the command now records

Artifact path:

- `docs/postflight/proof/live/discord_deploy_live_evidence.json`
- `docs/postflight/proof/live/discord_deploy_live_evidence.md`
- `docs/postflight/proof/live/live_proof_summary.json`

For one corr_id it records:

- request mode / supervised / force_agent_chain / recall config
- ordered same-corr bus hops
- exact `reply_to` / reply channels observed
- whether PlannerReact ran
- whether AgentChain ran
- whether a dedicated `verb.result` channel was observed
- whether a `cortex.gateway.chat.result` returned to the Hub reply channel
- pass/fail verdict
- same-corr log keys to grep in service logs

### Required same-corr success sequence on a live stack

For the live claim to become proven, one artifact must show a single corr_id with the following ordered path:

1. `cortex.gateway.chat.request` on `orion:cortex:gateway:request`
2. `cortex.orch.request` on `orion:cortex:request`
3. `verb.request` on `orion:verb:request`
4. `agent.planner.request` on `orion:exec:request:PlannerReactService`
5. optional `agent.chain.request` on `orion:exec:request:AgentChainService`
6. `verb.result` on `orion:verb:result:<corr_id>:<request_id>`
7. `cortex.gateway.chat.result` on `orion:cortex:gateway:result:<corr_id>`

### Exact same-corr log keys to grep

The proof command writes these expected same-corr log keys into the artifact:

- `hub_route_egress`
- `hub_ingress_result`
- `gateway_intake`
- `gateway_publish_orch`
- `gateway_wait_orch`
- `orch_intake`
- `orch_plan_wiring`
- `orch_publish_verb_runtime`
- `orch_wait_verb_runtime`
- `verb_runtime_intake`
- `plan_start`
- `verb_runtime_result`
- `orch_verb_runtime_result`
- `gateway_publish_hub_result`
- `[planner-react] intake`
- `[planner-react] replied`
- `[agent-chain] intake`
- `[agent-chain] replied`
- `agent_runtime_stop`

## D. Before / after channel behavior

### Before

- Orch published `verb.request` and then waited on the shared `orion:verb:result` channel.
- Exec published `verb.result` only to the shared channel.
- Same-corr proof was weak because unrelated background traffic could appear in the same result stream.

### After

- Orch publishes `verb.request` with `reply_to=orion:verb:result:<corr_id>:<request_id>`.
- Orch waits on that dedicated reply channel.
- Exec publishes `verb.result` to the dedicated `reply_to` channel.
- Exec also mirrors to `orion:verb:result` for compatibility.
- Proof artifacts now check for a dedicated same-corr `verb.result` hop explicitly.

## E. Files changed

- `services/orion-hub/scripts/cortex_request_builder.py`
  - normalized explicit recall payload shape.
- `services/orion-hub/scripts/websocket_handler.py`
  - added same-corr Hub result log.
- `services/orion-hub/scripts/api_routes.py`
  - added same-corr Hub result log for HTTP parity.
- `services/orion-cortex-gateway/app/bus_client.py`
  - added structured Gateway proof logs.
- `services/orion-cortex-orch/app/main.py`
  - added structured Orch intake proof log.
- `services/orion-cortex-orch/app/orchestrator.py`
  - dedicated per-request verb-runtime reply channel and Orch reply-wait logs.
- `services/orion-cortex-exec/app/main.py`
  - Exec dedicated reply publish and proof logs.
- `scripts/run_answer_depth_live_proof.py`
  - one-command same-corr proof path, stricter no-fallback default, ordered hops, reply channels, same-corr grep keys, artifact hardening.
- `services/orion-hub/tests/test_cortex_request_builder.py`
  - request-shape / routing-intent proof tests.
- `tests/test_cortex_gateway_error_reply.py`
  - Gateway forwarding and reply-channel proof tests.
- `services/orion-cortex-orch/tests/test_verb_runtime_rpc.py`
  - Orch dedicated reply-channel proof tests.
- `services/orion-cortex-exec/tests/test_depth_routing.py`
  - disabled-recall supervised routing proof.
- `services/orion-cortex-exec/tests/test_verb_runtime_reply_channel.py`
  - Exec dedicated reply publish proof.
- `tests/test_agent_no_recall_live_proof.py`
  - proof-artifact behavior tests.

## F. Tests

- `services/orion-hub/tests/test_cortex_request_builder.py`
  - proves recall-enabled and recall-disabled agent requests keep supervised intent and explicit recall structure.
- `tests/test_cortex_gateway_error_reply.py`
  - proves Gateway forwards disabled-recall agent requests correctly and replies on the Hub result channel.
- `services/orion-cortex-orch/tests/test_verb_runtime_rpc.py`
  - proves Orch uses dedicated per-request verb-runtime reply channels and disabled recall still builds planner + agent-chain supervised steps.
- `services/orion-cortex-exec/tests/test_depth_routing.py`
  - proves disabled-recall agent requests still hand off into the supervisor path.
- `services/orion-cortex-exec/tests/test_verb_runtime_reply_channel.py`
  - proves Exec publishes `verb.result` to the dedicated reply channel.
- `tests/test_agent_no_recall_live_proof.py`
  - proves the proof harness writes ordered same-corr hops, grep keys, and dedicated reply-channel evidence.

## G. Live run instructions

### Same-corr Hub Agent no-recall proof

```bash
python scripts/run_answer_depth_live_proof.py --scenario discord --disable-recall
```

### Optional supervisor-oriented proof

```bash
python scripts/run_answer_depth_live_proof.py --scenario supervisor --disable-recall
```

### If Gateway is intentionally unavailable and you want supplemental Orch-only evidence

```bash
python scripts/run_answer_depth_live_proof.py --scenario discord --disable-recall --allow-orch-fallback --allow-partial-pass
```

### After the run

Inspect:

- `docs/postflight/proof/live/discord_deploy_live_evidence.json`
- `docs/postflight/proof/live/discord_deploy_live_evidence.md`
- `docs/postflight/proof/live/live_proof_summary.json`

Then grep the service logs using the emitted corr_id and the artifact’s `same_corr_log_keys`.

## H. QA verdict

- **Is the no-recall timeout issue proven fixed?**
  - **Partially.** It is proven at code level that the shared Orch <-> Exec result handling has been replaced by dedicated per-request reply handling, and the proof harness now checks for that behavior. A successful same-corr live stack run is still required to elevate the verdict to fully proven in production-like conditions.

- **Is Hub Agent no-recall proven to reach PlannerReact / AgentChain?**
  - **Proven by tests, not yet by live trace from this container.**

- **Remaining caveats**
  - Live infrastructure was unreachable here, so the historical timeout is not closed by direct replay evidence in this environment.
  - The pass-2 proof path is now strict enough that the next live run should resolve the remaining uncertainty with one corr_id and one artifact set.

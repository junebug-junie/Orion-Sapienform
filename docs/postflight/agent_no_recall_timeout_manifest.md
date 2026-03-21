# Agent no-recall timeout postflight manifest

## Root cause

The timeout was not caused by recall being disabled by itself.

The actual defect was in the Orch -> Exec verb-runtime RPC leg: `call_verb_runtime()` published `verb.request` and then waited on the shared fan-in channel `orion:verb:result` instead of a per-request reply channel. Under concurrent background traffic (especially `spark-introspector` / brain-mode verb activity), the user request and unrelated verb results were multiplexed onto the same result stream, which made the request lifecycle ambiguous and could stall correlation-dependent tracing and timeout diagnosis.

A second issue made this harder to see: Hub emitted a sparse disabled-recall object (`{"enabled": false}`), while enabled-recall emitted a fuller shape with profile information. That did not define the architecture, but it made boundary comparisons noisier and made the no-recall branch look more suspicious than it really was.

## Why it only showed up, or appeared to show up, when recall was disabled

Disabling recall removed the `RecallService` step and its latency from the supervised agent request path. That made the user request reach the shared Orch -> Exec verb-result fan-in sooner, increasing overlap with unrelated background verb traffic. The correlation with recall was therefore timing/observability related, not an architectural requirement that agent mode depends on recall.

## Files changed

- `services/orion-hub/scripts/cortex_request_builder.py`
- `services/orion-hub/tests/test_cortex_request_builder.py`
- `services/orion-cortex-gateway/app/bus_client.py`
- `services/orion-cortex-orch/app/main.py`
- `services/orion-cortex-orch/app/orchestrator.py`
- `services/orion-cortex-orch/tests/test_verb_runtime_rpc.py`
- `services/orion-cortex-exec/app/main.py`
- `services/orion-cortex-exec/tests/test_depth_routing.py`
- `tests/test_cortex_gateway_error_reply.py`
- `scripts/run_answer_depth_live_proof.py`

## Before / after request shapes

### Hub egress before

Agent + recall enabled:

```json
{
  "mode": "agent",
  "options": {"supervised": true},
  "recall": {
    "enabled": true,
    "profile": "reflect.v1"
  }
}
```

Agent + recall disabled:

```json
{
  "mode": "agent",
  "options": {"supervised": true},
  "recall": {
    "enabled": false
  }
}
```

### Hub egress after

Both branches now emit the same explicit recall structure:

```json
{
  "mode": "agent",
  "options": {"supervised": true},
  "recall": {
    "enabled": false|true,
    "required": false,
    "mode": "hybrid",
    "profile": null|"reflect.v1"
  }
}
```

That keeps supervised routing intent stable while making disabled-recall requests just as explicit and grep-friendly as enabled-recall requests.

## Before / after result-channel behavior

### Before

- Orch published `verb.request`.
- Exec published `verb.result` onto the shared channel `orion:verb:result`.
- Orch waited on the shared channel and filtered by `request_id`.
- Concurrent background verb traffic could pollute the same stream and obscure the user request lifecycle.

### After

- Orch publishes `verb.request` with a dedicated `reply_to` channel:
  - `orion:verb:result:<corr_id>:<request_id>`
- Exec replies directly to that per-request channel.
- Exec also mirrors to legacy `orion:verb:result` for compatibility.
- Orch waits on the dedicated reply channel, so a single `corr_id` trace is no longer mixed with unrelated verb traffic.

## New logs added / improved

### Hub

- Existing `hub_route_egress` remains the concise route summary.
- Hub request shape is now explicit for both recall branches because the recall object is normalized.

### Gateway

- `gateway_intake`
- `gateway_publish_orch`
- `gateway_wait_orch`
- `gateway_orch_result`
- `gateway_publish_hub_result`
- `gateway_orch_timeout`
- `gateway_early_exit`

Each log includes, where applicable:
- `corr_id`
- source service
- reply channel
- mode
- verb
- supervised / force-agent-chain flags
- recall enabled/profile
- packs
- output mode / response profile when available
- explicit reason for early exit

### Orch

- `orch_intake`
- existing `orch_plan_wiring`
- `orch_publish_verb_runtime`
- `orch_wait_verb_runtime`
- `orch_verb_runtime_result`
- `orch_verb_runtime_skip`
- `orch_wait_verb_runtime_decode_failed`
- `orch_wait_verb_runtime_invalid_payload`

### Exec

- `verb_runtime_intake`
- `verb_runtime_result`
- `verb_runtime_result_legacy_mirror`
- `verb_runtime_validation_failed`

PlannerReact / AgentChain already had corr-aware intake/reply logs; those remain part of the proof chain.

## Tests added / updated

1. **Hub / Gateway request-shape coverage**
   - `services/orion-hub/tests/test_cortex_request_builder.py`
   - added explicit disabled-recall supervised-route shape assertion.

2. **Gateway forwarding / result reply coverage**
   - `tests/test_cortex_gateway_error_reply.py`
   - added a disabled-recall forwarding test that verifies supervised routing intent, explicit recall shape, and reply-to behavior.

3. **Orch / Exec routing coverage**
   - `services/orion-cortex-exec/tests/test_depth_routing.py`
   - strengthened the supervisor handoff test to assert the disabled-recall agent request still becomes `agent_runtime` with the correct recall config.

4. **End-to-end-ish internal RPC proof**
   - `services/orion-cortex-orch/tests/test_verb_runtime_rpc.py`
   - verifies per-request verb-runtime reply channels are used instead of the shared fan-in channel.

5. **Manual proof harness improvement**
   - `scripts/run_answer_depth_live_proof.py`
   - added `--disable-recall` while preserving the supervised agent path.

## Step-by-step live verification from Hub UI

1. Start the normal Orion stack with Hub, Cortex Gateway, Cortex Orch, Cortex Exec, PlannerReact, and AgentChain.
2. Open Hub UI.
3. Select **Agent** mode.
4. Run one request with recall enabled.
5. Run the same request with recall disabled.
6. Capture the user request corr_id from Hub logs (`hub_route_egress`).
7. Grep that corr_id across services and verify this chain:
   - Hub `hub_route_egress`
   - Gateway `gateway_intake`
   - Gateway `gateway_publish_orch`
   - Orch `orch_intake`
   - Orch `orch_plan_wiring`
   - Orch `orch_publish_verb_runtime`
   - Exec `verb_runtime_intake`
   - Exec `plan_start`
   - Supervisor `planner_decision` / `agent_runtime_continue` / `agent_runtime_stop`
   - PlannerReact `[planner-react] intake` / `replied`
   - AgentChain `[agent-chain] intake` / `replied`
   - Exec `verb_runtime_result`
   - Orch `orch_verb_runtime_result`
   - Gateway `gateway_publish_hub_result`
8. Confirm the Hub request returns on `orion:cortex:gateway:result:<corr_id>` in both recall-enabled and recall-disabled runs.
9. Optionally run:

```bash
python scripts/run_answer_depth_live_proof.py --scenario discord --disable-recall --allow-partial-pass
```

That now emits a proof artifact with explicit `recall_enabled=false` while keeping the supervised path intact.

## Known caveats

- This environment could not reach the live Redis bus, so live proof execution from the container could not be completed here.
- Exec still mirrors verb results to legacy `orion:verb:result` for compatibility; the authoritative Orch waiter now uses the dedicated per-request reply channel.

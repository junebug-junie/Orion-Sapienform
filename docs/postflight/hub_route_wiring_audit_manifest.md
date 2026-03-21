# Hub Route Wiring Audit Manifest

## Verdict

### Is Hub Agent fully wired correctly?
**Yes, after this pass.**

Hub Agent already selected the Orch/Exec supervised runtime path indirectly by emitting `mode="agent"`, which Orch turns into the `agent_runtime` plan and Exec routes through `Supervisor`. This pass makes that intent explicit from Hub by emitting `options.supervised=true`, adds Hub-side structured route logs, and mirrors `supervised` / `force_agent_chain` / `output_mode_decision` into Orch exec args so the downstream request shape is audit-friendly and resilient.

### Is Hub Auto fully wired correctly?
**Yes, with caveats documented below.**

Hub Auto emits `mode="auto"` plus `route_intent="auto"`, which allows Orch's `DecisionRouter` to promote delivery-oriented asks into depth-2 `agent_runtime` execution. Delivery-oriented asks continue to pick up `output_mode`, `response_profile`, and `delivery_pack` downstream.

### Should `force_agent_chain` be set in Hub Agent mode?
**No, not by default.**

`force_agent_chain` is useful for the proof harness when you need to force the fallback escalation regardless of planner outcome. For normal Hub Agent UX, it should stay unset so PlannerReact can still complete successfully when it already has a strong final answer or delegate normally. Hub Agent should express “deep supervised delivery path,” not “always skip planner success and force chain escalation.”

## Audit Findings

### End-to-end: Hub Agent
1. Hub UI sends `mode="agent"`.
2. Hub request builder now emits:
   - `mode="agent"`
   - `route_intent="none"`
   - `options.supervised=true`
   - user-selected `packs`
   - metadata containing the selected UI route
3. Gateway preserves mode/options/packs and forwards them to Orch.
4. Orch builds the two-step `agent_runtime` plan (`planner_react` -> `agent_chain`).
5. Orch now mirrors `supervised`, `force_agent_chain`, and `output_mode_decision` into `PlanExecutionArgs.extra` for clearer downstream preservation.
6. Exec routes `agent_runtime` through `Supervisor` even without `force_agent_chain`; `force_agent_chain` remains optional.
7. Delivery-oriented prompts still get classified into delivery-oriented output modes and receive `delivery_pack`.

### End-to-end: Hub Auto
1. Hub UI sends `mode="auto"`.
2. Hub request builder emits:
   - `mode="auto"`
   - `route_intent="auto"`
   - no forced `supervised`
   - user-selected `packs`
3. Orch `DecisionRouter` classifies the request.
4. Delivery-oriented prompts can be promoted to:
   - `mode="agent"`
   - `verb="agent_runtime"`
   - `options.execution_depth=2`
   - `options.output_mode` / `options.response_profile`
   - `options.output_mode_decision`
5. Orch `build_plan_request()` preserves those signals, injects `delivery_pack`, and sends them to Exec.
6. Exec receives a deterministic plan and supervises depth-2 flows.

## Exact Files Changed
- `services/orion-hub/scripts/cortex_request_builder.py`
- `services/orion-hub/scripts/api_routes.py`
- `services/orion-hub/scripts/websocket_handler.py`
- `services/orion-cortex-orch/app/orchestrator.py`
- `services/orion-hub/tests/test_cortex_request_builder.py`
- `services/orion-cortex-orch/tests/test_auto_router.py`
- `docs/postflight/hub_route_wiring_audit_manifest.md`

## Request Shape: Before vs After

### Hub Agent before
```json
{
  "mode": "agent",
  "route_intent": "none",
  "verb": null,
  "packs": ["...user packs..."],
  "options": {},
  "metadata": {"source": "hub_http|hub_ws"}
}
```

### Hub Agent after
```json
{
  "mode": "agent",
  "route_intent": "none",
  "verb": null,
  "packs": ["...user packs..."],
  "options": {
    "supervised": true
  },
  "metadata": {
    "source": "hub_http|hub_ws",
    "hub_route": {
      "selected_ui_route": "agent"
    }
  }
}
```

### Hub Auto before
```json
{
  "mode": "auto",
  "route_intent": "auto",
  "verb": null,
  "packs": ["...user packs..."],
  "options": {
    "route_intent": "auto"
  },
  "metadata": {"source": "hub_http|hub_ws"}
}
```

### Hub Auto after
```json
{
  "mode": "auto",
  "route_intent": "auto",
  "verb": null,
  "packs": ["...user packs..."],
  "options": {
    "route_intent": "auto"
  },
  "metadata": {
    "source": "hub_http|hub_ws",
    "hub_route": {
      "selected_ui_route": "auto"
    }
  }
}
```

### Orch -> Exec preservation after this pass
`PlanExecutionArgs.extra` now explicitly mirrors:
- `mode`
- `packs`
- `options`
- `diagnostic`
- `supervised`
- `force_agent_chain`
- `output_mode_decision`
- `trace_id`
- `session_id`
- `verb`

## Recommended Live Manual Test Procedure from Hub UI

### Test A: Agent mode deep delivery
1. Open Hub.
2. Select **Agent**.
3. Ask a delivery-oriented prompt such as:
   - “Write me a deployment guide for this service.”
   - “Generate a code scaffold for a FastAPI endpoint that validates this payload.”
4. Confirm in logs:
   - Hub `hub_route_egress` shows `selected_ui_route=agent`, `emitted_mode=agent`, `supervised=true`, `force_agent_chain=false`.
   - Orch `orch_plan_wiring` shows delivery-oriented `output_mode` and `packs` including `delivery_pack`.
   - Exec `supervisor_wiring` shows delivery pack-backed tool availability.
5. Expect this to be the preferred path for live manual delivery testing.

### Test B: Auto mode smart promotion
1. Open Hub.
2. Select **Auto**.
3. Ask a delivery-oriented prompt such as:
   - “Compare Docker Compose vs Kubernetes for this service and recommend one.”
   - “How do I deploy this service step by step?”
4. Confirm in logs:
   - Hub `hub_route_egress` shows `selected_ui_route=auto`, `emitted_mode=auto`, `supervised=false`.
   - Orch `auto_depth_result` promotes the request to depth 2 when appropriate.
   - Orch `orch_plan_wiring` shows the chosen `output_mode` and `delivery_pack`.
5. Expect Auto to remain useful for realistic UX testing of router promotion, but less deterministic than Agent.

## Recommended Answer to “For live manual testing in Hub, should I use Auto or Agent?”
**Use Agent for live manual delivery-quality testing.**

- Agent is now the explicit deep supervised path.
- Auto is correct for testing router intelligence and promotion behavior.
- Use Auto when you specifically want to verify routing.
- Use Agent when you want the most reliable delivery-oriented execution path from the current Hub UI.

## Remaining Caveats
- Agent mode still does **not** force `agent_chain` every time; it allows PlannerReact to succeed or delegate naturally.
- Auto remains heuristic/LLM-router dependent, so some borderline prompts can still route shallow by design.
- The Hub UI still does not expose every proof-harness knob directly; this pass makes the intended live path explicit without turning normal UX into proof-only behavior.

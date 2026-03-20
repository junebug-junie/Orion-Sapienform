# Discord Deploy Live Evidence

- timestamp: `2026-03-20T06:28:11.250679+00:00`
- correlation_id: `e0531ae9-8e36-4a1a-9a00-c70df75e9080`
- request_channel: `orion:cortex:request`
- reply_channel: `orion:cortex:result:e0531ae9-8e36-4a1a-9a00-c70df75e9080`
- result_kind: `None`
- overall_pass: `False`

## Runtime Signals
- output_mode: `None`
- response_profile: `None`
- packs: `None`
- resolved_tool_ids (excerpt): `[]`
- tool_sequence: `None`
- triage_blocked_post_step0: `None`
- repeated_plan_action_escalation: `None`
- finalize_response_invoked: `None`

## Path Observed
- {"gateway_request_kind": false, "orch_request_kind": true, "exec_request_kind": false, "verb_request_kind": true, "planner_request_kind": true, "agent_chain_request_kind": false, "llm_request_kind": true}

## Pass Checks
```json
{
  "real_orch_path_observed": true,
  "plannerreact_bus_observed": true,
  "llm_bus_observed": true,
  "output_mode_expected": null,
  "response_profile_expected": null,
  "delivery_pack_active": null,
  "delivery_verbs_visible": null,
  "triage_not_after_step0": null,
  "repeated_plan_action_not_shallow": null,
  "finalization_when_needed": null,
  "quality_evaluator_rewrite": null,
  "answer_quality_concrete": false
}
```

## Quality Checks
```json
{
  "positive_checks": {
    "discord": false,
    "(developer portal|application|app setup|bot setup|create .*bot)": false,
    "(token|env var|environment variable|DISCORD_BOT_TOKEN)": false,
    "(intent|permission|oauth|invite)": false,
    "(deploy|hosting|host|process|systemd|docker)": false,
    "(test|troubleshoot|debug|verify)": false
  },
  "negative_checks": {
    "gather requirements": false,
    "create a guide": false,
    "review and refine": false,
    "test deployment, then refine": false,
    "purely managerial": false
  },
  "positive_pass": false,
  "negative_pass": true,
  "overall_pass": false
}
```

## Final Answer Excerpt
```text

```

## Error
```text
RPC timeout waiting on orion:cortex:result:e0531ae9-8e36-4a1a-9a00-c70df75e9080
```

## Unavailable Fields
```json
{
  "runtime_debug": "metadata.answer_depth and AgentChain runtime_debug not present.",
  "tool_sequence": "No step_name sequence available from cortex_result.steps.",
  "gateway_path": "Gateway path timed out in this run; evidence captured via real Orch path fallback."
}
```

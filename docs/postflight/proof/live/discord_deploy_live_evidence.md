# Discord Deploy Live Evidence

- timestamp: `2026-03-20T04:54:50.617449+00:00`
- correlation_id: `9fa3aa5b-b07e-49c5-8851-b52f81b18edd`
- request_channel: `orion:cortex:request`
- reply_channel: `orion:cortex:result:9fa3aa5b-b07e-49c5-8851-b52f81b18edd`
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
- {"gateway_request_kind": false, "orch_request_kind": true, "exec_request_kind": false, "verb_request_kind": false, "planner_request_kind": false, "agent_chain_request_kind": false, "llm_request_kind": false}

## Pass Checks
```json
{
  "real_orch_path_observed": false,
  "plannerreact_bus_observed": false,
  "llm_bus_observed": false,
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
RPC timeout waiting on orion:cortex:result:9fa3aa5b-b07e-49c5-8851-b52f81b18edd
```

## Unavailable Fields
```json
{
  "runtime_debug": "metadata.answer_depth and AgentChain runtime_debug not present.",
  "tool_sequence": "No step_name sequence available from cortex_result.steps.",
  "gateway_path": "Gateway path timed out in this run; evidence captured via real Orch path fallback."
}
```

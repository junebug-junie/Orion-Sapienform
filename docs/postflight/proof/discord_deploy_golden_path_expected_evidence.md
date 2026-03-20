# Discord Golden-Path Expected Evidence (Pass 3)

Canonical prompt:
`Please provide instructions on how to deploy you onto Discord.`

## Expected runtime-debug fields (agent-chain)

- `output_mode`: `implementation_guide`
- `response_profile`: `technical_delivery`
- `packs` includes `delivery_pack`
- `resolved_tool_ids` includes delivery verbs:
  - `write_guide`
  - `finalize_response`
- guard / escalation flags:
  - `triage_blocked_post_step0` is `true`
  - `repeated_plan_action_escalation` is `true`
  - `finalize_response_invoked` is `true`
  - `quality_evaluator_rewrite` is `true`

## Expected representative log excerpt (agent-chain)

These lines should appear (correlation id may vary):

```text
[agent-chain] wiring corr=... output_mode=implementation_guide profile=technical_delivery packs=[..., delivery_pack] tools=[..., write_guide, finalize_response, ...]
[agent-chain] repeated_plan_action_escalation=1 -> tool_id=write_guide output_mode=implementation_guide
[agent-chain] triage_blocked_post_step0=1 step=2 prior_trace_len=1 -> finalize_response
[agent-chain] quality_evaluator_rewrite=1 output_mode=implementation_guide
```

## Expected final answer shape

The returned `final_text` should look like a concrete Discord deployment guide and include at least:
- Discord app/bot setup
- bot token handling via env vars (e.g. `DISCORD_BOT_TOKEN`)
- gateway intents and permissions
- invite / authorization steps
- hosting/process steps
- testing/troubleshooting section

It should **not** contain meta-plan scaffolding phrases like:
- `gather requirements`
- `create a guide`
- `review and refine`


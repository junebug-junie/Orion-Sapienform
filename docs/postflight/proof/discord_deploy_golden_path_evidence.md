# Discord Golden-Path Evidence (Pass 3)

This evidence is produced by a deterministic integration-style harness (LLM/planner are stubbed; tool resolution and guards are real).

## Runtime-debug excerpt
```json
{
  "output_mode": "implementation_guide",
  "response_profile": "technical_delivery",
  "packs": [
    "executive_pack",
    "memory_pack",
    "delivery_pack"
  ],
  "resolved_tool_ids_excerpt": [
    "analyze_text",
    "triage",
    "plan_action",
    "assess_risk",
    "goal_formulate",
    "evaluate",
    "extract_facts",
    "summarize_context",
    "answer_direct",
    "finalize_response",
    "write_guide",
    "write_tutorial",
    "write_runbook",
    "write_recommendation",
    "compare_options",
    "synthesize_patterns",
    "generate_code_scaffold"
  ],
  "flags": {
    "triage_blocked_post_step0": true,
    "repeated_plan_action_escalation": true,
    "finalize_response_invoked": true,
    "quality_evaluator_rewrite": true
  }
}
```

## Log excerpt
```
[agent-chain] wiring corr=corr-discord-deploy output_mode=implementation_guide profile=technical_delivery packs=['executive_pack', 'memory_pack', 'delivery_pack'] tools=['analyze_text', 'triage', 'plan_action', 'assess_risk', 'goal_formulate', 'evaluate', 'extract_facts', 'summarize_context', 'answer_direct', 'finalize_response', 'write_guide', 'write_tutorial', 'write_runbook', 'write_recommendation', 'compare_options', 'synthesize_patterns', 'generate_code_scaffold']
[agent-chain] repeated_plan_action_escalation=1 -> tool_id=write_guide output_mode=implementation_guide
[agent-chain] triage_blocked_post_step0=1 step=2 prior_trace_len=1 -> finalize_response
[agent-chain] quality_evaluator_rewrite=1 output_mode=implementation_guide
```

## Tool call sequence
```
plan_action -> write_guide -> finalize_response -> finalize_response
```

## Final answer excerpt
```text
Discord Deployment Guide (Orion Discord Bot Bridge)

1) Create the Discord application
- Go to the Discord Developer Portal.
- Create a new Application and add a Bot.
- Enable the Bot and copy the Bot token.

2) Configure environment variables
- Set `DISCORD_BOT_TOKEN` in your runtime environment (never hard-code it).
- Example:
  - export DISCORD_BOT_TOKEN="your-token-here"

3) Choose Gateway intents + permissions
- Enable only the intents you need.
- Common starting point for bridges:
  - `Guilds`
  - `GuildMessages`
  - `MessageContent` (if applicable in your server settings)

4) Host the bot process
- Start the bridge process with a persistent runner (systemd, Docker, or a supervised container).
- Ensure the process restarts on failure.

5) Invite the bot to your server
- Generate the OAuth2 invite URL for the application.
- Use the required scopes (typically bot) and permissions.
- 
```
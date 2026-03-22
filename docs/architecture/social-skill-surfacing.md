# Social Skill Surfacing for `social_room`

This phase adds a **tiny internal skill layer** for `social_room` so Oríon can surface a few bounded, conversation-native helpers when they materially improve a reply.

## Purpose

- improve conversational competence in shared rooms
- keep the live path socially grounded and inspectable
- avoid introducing a general tool or agent runtime

## Allowlisted social skills

- `social_summarize_thread`
- `social_safe_recall`
- `social_self_ground`
- `social_followup_question`
- `social_room_reflection`
- `social_exit_or_pause`

These are **internal, text-only helpers**. They do not call external tools, world actions, MCP, ADK, planner-react, or agent-chain.

## Where selection happens

Skill choice happens in the Hub `social_room` request assembly seam:

- `services/orion-hub/scripts/social_room.py`
- `services/orion-hub/scripts/cortex_request_builder.py`

The bridge does **not** choose skills. It only provides room-local policy/continuity context. Hub heuristics may then:

1. keep the default no-skill path
2. select one allowlisted social skill
3. inject the compact result into `chat_social_room` prompt metadata

## Safety / privacy boundary

- default behavior remains plain conversational generation
- tools/actions remain effectively `none`
- only one narrow social skill may be injected
- blocked/sealed/private memory text is suppressed from `social_safe_recall`
- no broad tool inventory is exposed to peers
- no unrestricted journals, raw mirrors, or internal traces are surfaced

## Non-goals

- not a general tool system
- not planner-driven behavior
- not world-action execution
- not broader recall access
- not arbitrary bridge-controlled tooling

## Summary

- Rewrote `orion-npc-answer-first.patch` so it **actually applies** after pair-turn continuity (the Aug 30 fix had been silently broken).
- For **every** NPC: plan is role background, not “goals for the conversation”; no forced previous-conversation callback; continue can break abstract circling.
- Rewrote all four NPC plans + softened Sofia’s coffee-menu identity bait.
- Added `resyncAgentDescriptions` engine input + script so live Circe picks up the new cards (baked agentDescriptions otherwise keep the old plans forever).

## Outcome moved

NPC↔NPC chats stop being structurally ordered to narrate job props / prior metaphor loops. Deploy + resync required for live effect.

## Current architecture

AI Town NPC speech: `conversation.ts` injects `agentDescription.identity` + `plan` every turn. Plans lived in `town_cards.yaml` → `orion-character.patch` → baked into Convex `agentDescriptions` at `createAgent`. Aug 30 answer-first targeted a pre-pair-turn `fetchTownContinuity(otherName)` shape and stopped applying.

## Architecture touched

- `services/orion-ai-town` patches, cards, tests, README, resync script

## Files changed

- `patches/orion-npc-answer-first.patch`: regenerated against post-continuity conversation.ts
- `patches/orion-resync-agent-descriptions.patch`: new
- `patches/orion-character.patch`: all four plans + Sofia identity
- `cards/town_cards.yaml`: plans + Sofia public_description
- `scripts/apply_upstream_patches.sh`, `scripts/resync_agent_descriptions.py`
- tests + README

## Schema / bus / API changes

- Added: engine input `resyncAgentDescriptions` (empty args)
- Removed: none
- Behavior changed: NPC speech prompt contracts; plan labeling
- Compatibility notes: must run resync after deploy or live agents keep old plans

## Env/config changes

- none

## Tests run

```text
pytest services/orion-ai-town/tests/ -q
# 131 passed, 1 skipped
# Plus clean full patch-chain apply on vanilla upstream archive (all 15 patches OK)
```

## Evals run

```text
No NPC dialogue eval harness; structural patch/card gates + apply-chain proof.
```

## Docker/build/smoke checks

```text
Not deployed this session. After Circe AI Town rebuild: run resync_agent_descriptions.py, then watch a Mara↔Sofia (or any NPC↔NPC) chat.
```

## Review findings fixed

- Finding: resync never set `descriptionsModified`, so plans wouldn't persist to Convex
  - Fix: set `game.descriptionsModified = true` in the input handler; gate test asserts it
  - Evidence: patch line + full apply chain shows the assignment
- Finding: circling guidance told NPCs to inject role-background props (coffee bait path)
  - Fix: keep anti-repetition; change subject or end — do not re-inject role props
  - Evidence: patch + test asserts old "role background" circle line is gone
- Finding: resync script could hit idle engine
  - Fix: call `heartbeat_world()` before sendInput

## Restart required

```bash
# On Circe (AI Town host), rebuild/redeploy orion-ai-town so patches apply, then:
export AITOWN_CONVEX_URL=... AITOWN_ADMIN_KEY=... AITOWN_WORLD_ID=...
PYTHONPATH=. python3 services/orion-ai-town/scripts/resync_agent_descriptions.py
```

## Risks / concerns

- Severity: medium
- Concern: without resync after deploy, live agents keep old baked plans
- Mitigation: script + README; verify via world:gameDescriptions

## PR link

(filled after open)

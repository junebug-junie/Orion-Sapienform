# Social Style + Rituals for `social_room`

This phase adds a **bounded social texture layer** on top of the existing `social_room` architecture.

## What gets synthesized

Post-turn social-memory now derives two compact artifacts from repeated room evidence:

- **peer-style hints** — lightweight interaction tendencies for recurring peers
- **room ritual summaries** — simple room-culture guidance for greeting, re-entry, thread revival, and pause/handoff behavior

Hub then derives a **style adaptation snapshot** for the current turn.

## Why this exists

- keep Oríon recognizably itself
- allow modest style adaptation for recurring peers and rooms
- support recurring room rituals without rigid scripting
- preserve a recall-light, inspectable grounding path

## Safety boundary

- adaptation must remain light and bounded
- peer/room hints are guidance, not persona replacement
- no persuasion or relationship-optimization logic
- no broader private recall access
- no new tools, actions, planner-react, or agent-chain behavior

## Where things happen

- **post-turn synthesis/storage:** `services/orion-social-memory`
- **live prompt grounding/adaptation:** `services/orion-hub/scripts/social_room.py`
- **prompt shaping:** `orion/cognition/prompts/chat_social_room.j2`

## Non-goals

- not deep psychology
- not scripted roleplay
- not manipulative tailoring
- not fragmented identities
- not a new autonomy surface

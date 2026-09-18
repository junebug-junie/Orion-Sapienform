# Curiosity outreach hop synthesis — design

**Date:** 2026-09-18  
**Status:** approved (Door A first)  
**Scope:** Curiosity investigation → compose → `offer_message` only. Endogenous soft-peek path (Door B) is out of scope.

## Problem

When curiosity investigation decides a finding is worth telling Juniper, the second turn (`build_outreach_composition_prompt`) dumps the journal finding and says “say something interesting.” It does **not** feed hop notes, and it does **not** require the message to be:

1. a synthesis of what Orion has been thinking through those hops, and  
2. a synthesis of why they want to share it with Juniper.

That produces free-floating prose instead of continuing the curiosity train of thought out loud.

## Goal

Door A compose messages must land both layers:

- **Thinking thread:** aggregate of the hop path (what I’ve been working through)  
- **Why share:** why this is for Juniper now (not only “I found something”)

Exact `PASS` remains if the thing does not survive being written down.

## Choke points

| Piece | File / symbol |
|---|---|
| Compose prompt | `orion/curiosity/outreach_prompt.py` → `build_outreach_composition_prompt` |
| Caller | `services/orion-hub/scripts/curiosity_investigation.py` → `CuriosityInvestigation._maybe_reach_out` |
| Hop source | `orion.curiosity.worldview.read_hop_notes(reader, run_id)` → `list[tuple[int, str]]` |

## Data contract

`build_outreach_composition_prompt` gains `hop_notes: Sequence[tuple[int, str]] = ()`.

- Numbered hop notes in order, capped (count + per-note chars) so context stays bounded.  
- Finding text and `reach_out_why` stay as today (finding truncated, why shown when non-empty).  
- Instructions require both layers explicitly.  
- Still requires exact `PASS` for decline.

`_maybe_reach_out`:

- Accept optional `hop_notes`.  
- If omitted and a worldview reader exists, load via `read_hop_notes` in a worker thread.  
- Live tick / self-inquiry paths that already hold hops pass them through (no double-fetch).  
- Durable completion path omits them and lets the fetch run.

## Non-goals

- Door B (`build_outreach_prompt` / endogenous soft peeks)  
- Provenance / Hub UI changes  
- New bus channels or services  
- Post-generation semantic grader that the message “really” synthesized (prompt contract + structural tests only this patch)  
- Changing when `reach_out` is decided (still first-turn outcome)

## Acceptance

1. Compose prompt with multi-hop notes includes the hops and requires thinking-thread + why-share.  
2. Empty hops still builds a prompt from finding + why; still exact `PASS`.  
3. `_maybe_reach_out` wires hops into the builder (unit/wire test).  
4. Existing curiosity investigation suite still passes.

## Rollback

Revert the prompt + `_maybe_reach_out` hop threading. Delivery path unchanged.

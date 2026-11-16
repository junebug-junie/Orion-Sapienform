## Summary

- This PR adds one design document. It builds nothing.
- The design answers Juniper's question: what should Orion do while someone plays a game on Orion?
- Orion records each sitting, looks at the screen about once a minute, speaks through the unified chat turn, and answers a chat box inside the game.
- Looks and lines bid on tier 1. Everything goes on the temporal record. The play becomes a third attention candidate. Prime sees it, regulates it, and is woken by it at most twice per sitting.
- Voice in and out is shaped for, and left for a later design.

Spec: `docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md`. Refs #3500.

## Outcome moved

None yet. The document names the outcome to move: about 27 of 31 hours of play left no trace in Orion.

## Current architecture

- The game engine is sealed and has no link to Orion's bus.
- Hub relays the picture and the inputs, and never decodes a frame.
- A play record is written only when Orion held the controller.
- Orion's only signal during a 249-minute sitting on 2026-11-15 was the office camera's guess.

## Architecture touched

None. Docs only.

## Files changed

- `docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md`: the design.
- `docs/superpowers/pr-reports/2026-11-16-game-company-watch-and-commentary-design-pr.md`: this report.

## Schema / bus / API changes

- Added: none in this PR. The spec proposes them under "Proposed schema / API changes".
- Removed: none.
- Renamed: none.
- Behavior changed: none.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, no env template changed.
- skipped keys requiring operator action: none.

## Tests run

```text
python3 scripts/check_personal_terms.py   -> no personal terms in tracked files
git diff --cached --check                 -> clean
anchor check: 47 cited file:line anchors read back from the code; 1 was off by 23 lines and was fixed
```

## Evals run

```text
None. Docs only.
```

## Docker/build/smoke checks

```text
None. Docs only. Live facts in the spec were read from the running system on 2026-11-16:
/props on each circe lane, gpu_pool_leases, grammar_acts, harness_turn_trace,
temporal_self_event, endogenous_outreach_decisions, and the local game journals.
```

## Review findings fixed

- Finding: the spec cited the wrong line for the sentence "Juniper attached".
  - Fix: `orion/harness/attachment_staging.py:253`.
  - Evidence: anchor check above.
- Finding: a line turn runs in Juniper's chat session, so the hand-back rule would record it as Juniper speaking.
  - Fix: the spec now decides the new `play` class by the turn's source, before the rule for Juniper's chat.
  - Evidence: `orion/prime/handback.py:198-210`.
- Finding: "Six of the seven Prime seams exist today" overstated it. Each needs wiring.
  - Fix: reworded in section 6.
  - Evidence: the spec text.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium
- Concern: two things cannot be known from code: whether the chat-lane model reads pixel art, and how long a line takes end to end. The spec estimates 60 to 90 seconds.
- Mitigation: step 0 of the patch order is a spike with no deploy that measures both before any build. It waits for Juniper's go.

- Severity: low
- Concern: the quick lane cannot see today, so the look starts on the chat lane, not on the lane Juniper named first.
- Mitigation: the design binds only to a route name. Moving the look later is one config line.

Issues opened: #3500 (design, needs-juniper). Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3517

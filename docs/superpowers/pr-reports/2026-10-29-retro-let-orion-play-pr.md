## Summary

- Juniper asked for a button that lets Orion play for real, not a 60 s experiment. **Let Orion play** (next to New game) hands the controller to Orion's goal loop in 30-minute turns, Retro's maximum, and renews each turn when it runs out.
- The player takes the game back by:
  - clicking the game;
  - pressing a game key (except on a focused button or menu);
  - pressing any controller button;
  - Take this controller or Stop.

  A hidden tab, a lost connection or a pause ends it.
- Only a turn that ran out its time is renewed: the page compares the session frame with the grant's `started_frame + duration`. If someone else took the controller, Orion stops and the page says "Someone else took the controller, so Orion stopped playing." It never takes the game back from a person.
- The button shows its state (`aria-pressed`, label "Orion is playing"). Try Orion goals and the scripted baseline are unchanged and clear Let Orion play.

## Outcome moved

Orion can play Zelda for as long as Juniper lets it, with no click every 60 s.

## Current architecture

A hand-over was a single 60 s grant (`duration_seconds:60`). The harness re-handed the controller by script.

## Architecture touched

Hub front end only (`retro-panel.js`, `retro.html`), plus the eval fixture and the runbook.

## Files changed

- `services/orion-hub/static/js/retro-panel.js`:
  - `ORION_PLAY_SEC`, `setOrion`, `orionTurn`, `renewOrion` and `takeBack`;
  - the pad poll, and the budget check in `showPolicy`;
  - take-back hooks on click, key, Stop, Take this controller and pause;
  - `handoff(cognition, seconds)`.
- `services/orion-hub/templates/retro.html`: the button and `?v=13`.
- `services/orion-hub/evals/retro_play_panel.py`: the fixture registers a started hand-over as a new grant, and `end_grant?expired=1` simulates a turn that ran out.
- `services/orion-hub/evals/retro_play_panel.mjs`: covers the 1800 s start, renewal of an expired turn, click take-back, no renewal after take-back, and no renewal when another controller took the port.
- `services/orion-hub/tests/retro/test_play_panel_static.py`: the version and markers.
- `docs/runbooks/retro-browser.md`: the behaviour and its limits.

## Schema / bus / API changes

None. Retro already accepts `duration_seconds` up to 1800.

## Env/config changes

None.

## Tests run

```text
cd services/orion-hub && pytest tests/retro -q -> 195 passed
node --check static/js/retro-panel.js          -> ok
```

## Evals run

```text
python services/orion-hub/evals/retro_play_panel.py (real page, fixture Retro) -> PASS (run locally; CI does not run this eval)
Mutations: no renewal -> "the ended turn was not renewed"; no click take-back -> fails;
           renew on any grant end -> "Someone else took the controller" never shown (fails)
```

## Docker/build/smoke checks

```text
No build. The post-pull rebuild covers orion-hub.
```

## Review findings fixed

- Finding (high): renewal took the controller back from a person in another window, again and again.
  - Fix: renew only when the grant ran out its time (frame budget). Otherwise Orion stops and says so.
  - Evidence: an eval step; the mutation fails.
- Finding (medium): after a pause from elsewhere, Orion's play mode stayed on. It swallowed the next click, and the scripted baseline could start a 30-minute turn.
  - Fix: a pause clears it; the scripted baseline clears it; `takeBack` falls through between turns.
- Finding (medium): Enter on a focused button became a take-back.
  - Fix: keys on a button, summary, link or select act normally.
- Finding (low): stale error text after a take-back during a hand-over.
  - Fix: the error is reported only if Orion was still meant to play.
- Accessibility: `aria-pressed` and the label show the state.
- Not fixed, for Juniper (medium): there is no heat or strain gate on these turns. Co-play planning is interactive priority (#2905), never heat-shed, so an all-evening session keeps calling the planner during a cabinet lockdown. This is documented in the runbook.

## Restart required

```bash
git pull   # rebuilds orion-hub; reload /retro (v=13)
```

## Risks / concerns

- Severity: medium
  - Concern: there is no heat gate (above).
  - Mitigation: the planner is one call per decision, behind a lock with a 429 on busy. Juniper decides whether long turns should stop on `cabinet_hot` / `strained`.
- Severity: low
  - Concern: a dropped connection ends Orion's play, and you press the button again.
  - Mitigation: Retro cancels a grant whose sponsor disconnects, so renewing over a dead socket cannot work anyway.
- Severity: low
  - Concern: Orion's turns count as "human recent" for the leisure gate, so solo play waits 10 minutes after the page closes.
  - Mitigation: harmless today.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2940

🤖 Generated with [Claude Code](https://claude.com/claude-code)

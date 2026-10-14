## Summary

- Reverie's "EARLIER TODAY" block now groups Orion's own daydreams into one counted line, followed by the two most recent other things Orion did (curiosity run, image attempt, conversation, attention, sleep, body-sensor stretch).
- Before: the block showed the last 3 closed arcs of any kind. Daydreams are the commonest arc (160 of 258 closed on 2026-10-13), so reverie was mostly shown its own previous thoughts.
- With no daydream closed yet today, the block shows 3 other arcs, as before.
- On a frame truncated past 64 arcs (daydreams dropped first), the line says "at least N" and gives no start time.

## Outcome moved

The note's job was to show reverie what Orion did before now; it was showing reverie's own last daydreams instead. Live frame, same moment, old vs new render:

```text
old: - 00:57-00:58 (under a minute) reverie: open-loop-4409448b6975 [arc:b0d12eec8f48b5d9]
new: - 00:04-00:58 (6 daydreams) your own daydreaming, the latest ended 00:58 [arc:b0d12eec8f48b5d9]
     - 00:50-00:56 (6 min) a curiosity run [arc:73a4426c0d2843b1]
     - 00:03-00:39 (35 min) interoception: node:substrate.biometrics [arc:02a3d4b38d951af2]
```

## Current architecture

`orion/temporal_self/recent_arcs.py` (pure template, no LLM) renders from the durable-runs frame; `services/orion-thought/app/reverie.py` fetches the frame and passes the text to `reverie_narrate.j2`.

## Architecture touched

Only the template function. No schema, bus, API or env key change.

## Files changed

- `orion/temporal_self/recent_arcs.py`: daydream grouping, "at least"/"up to" on truncated frames.
- `orion/temporal_self/tests/test_recent_arcs.py`: 4 new tests.
- `services/orion-thought/app/settings.py`, `services/orion-thought/.env_example`: comments only.

## Schema / bus / API changes

None.

## Env/config changes

- Keys: none added/removed. `.env_example` comment only; `python scripts/sync_local_env_from_example.py` run; parity PASS.

## Tests run

```text
pytest orion/temporal_self/tests/test_recent_arcs.py services/orion-thought/tests/test_reverie_recent_arcs.py -> 28 passed
```

## Evals run

None. Live check after deploy: thought logs `reverie_recent_arcs ... arcs=3`, and the rendered block carries one daydream line.

## Docker/build/smoke checks

Rendered against the live frame (above).

## Review findings fixed

- Finding: on a truncated frame the start time was the earliest *kept* daydream, stated as fact.
  - Fix: truncated frames render `up to HH:MM` with no start time.
  - Evidence: `test_one_daydream_is_singular_and_truncated_frame_says_at_least`.

## Restart required

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns

- Severity: low. Concern: reverie may still repeat itself for reasons unrelated to this block. Mitigation: compare repetition before and after the deploy.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

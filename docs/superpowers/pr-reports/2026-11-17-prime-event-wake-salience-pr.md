## Summary

- A funded event want may wake Prime on its own only when it is in the top tenth of its own source (`salience >= 0.9`, `EVENT_WAKE_SALIENCE`). Ordinary funded events stay in Prime's frame at its next thought when still open and in the top 8 by net.
- The hunger replay gains a planted event stream at the live rate; the live fixture carried no event wants, which is why the replay missed this.

## Outcome moved

Live 11-16 18:06Z..11-17 03:00Z (after patch 5 deployed): 11 of 12 Prime thoughts were woken by funded reading wants and all 11 ended "no move" (Prime: "low-value and not actionable"). 3-day funded reading wants: 114, mean salience 0.09; only 9 funded wants in 3 days, across all families, were >= 0.9. Replay with the live-rate stream: 4 thoughts in 2 days with the gate (one per top-tenth event), 16 without.

## Current architecture

`orion/prime/hunger.py` `event_horizon_needs`: every funded event want Prime does not hold, nearing expiry, got a horizon drive capped at 0.5 = the wake threshold, so each woke Prime once regardless of value.

## Architecture touched

orion-mind (Prime) only. No schema, bus, env or config change.

## Files changed

- `orion/prime/hunger.py`: `EVENT_WAKE_SALIENCE`, the gate, docstring states what an ordinary event loses.
- `orion/prime/evals/run_prime_hunger_replay.py`: planted `event_stream`, check `ordinary_events_do_not_wake` (exactly one wake per top-tenth event).
- `orion/prime/tests/test_prime_hunger.py`, `orion/prime/tests/test_prime_evals.py`: tests.

## Schema / bus / API changes

- Added / Removed / Renamed: none.
- Behavior changed: below-band funded events no longer wake Prime.
- Compatibility notes: none.

## Env/config changes

- None. `.env_example` updated: no. Local `.env` sync: not needed.

## Tests run

```text
pytest orion/prime -> 281 passed
pytest orion/prime services/orion-mind/tests -> 700 passed (before review fixes)
```

## Evals run

```text
run_prime_hunger_replay.py: all checks PASS; ordinary_events_do_not_wake: 4 thoughts in 2 days, top-tenth events 4
mutation: EVENT_WAKE_SALIENCE=0.0 -> FAIL (16); =1.01 (blocks all) -> FAIL (0)
```

## Docker/build/smoke checks

```text
Not run (pure logic). Live check after deploy: Prime thoughts' woke: lines; reading-woken thoughts should be rare.
```

## Review findings fixed

- Finding: docstring/commit claimed every funded event still reaches the frame.
  - Fix: docstring says only open events in the top 8 by net, names what is lost (3 h reads, 15 min cooling acts), and that salience not net decides the wake (deliberate).
  - Evidence: `orion/prime/hunger.py` docstring.
- Finding: replay check passed if the gate blocked everything.
  - Fix: asserts exactly one horizon wake per top-tenth event.
  - Evidence: mutation at 1.01 now fails.
- Finding: no test at exactly 0.9.
  - Fix: `test_a_top_tenth_funded_event_wakes_prime_at_exactly_the_band`.
- Found, not this PR: camera asks leave Prime's menu after 24 h though open 7 days (pre-existing) -> 3558.

## Restart required

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns

- Severity: low. Concern: Prime now rarely notices cooling acts (15 min windows) or ordinary reads. Mitigation: the acts run on their own; top-tenth events still wake Prime.
- Severity: low. Concern: a camera ask with high net but low salience never wakes Prime. Mitigation: asks never woke Prime under the old rule either (their drive started at 84 h, after they left the 24 h menu).
- Issues: Refs #3427 (hunger 14-day check). Opened 3558.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

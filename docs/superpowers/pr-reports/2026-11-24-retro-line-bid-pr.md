## Summary

- Orion's game comments (`hub.retro_commentary`) never ran live: 367 of 367 bids deferred, `capacity_reserved`, net +0.000 (want ledger, 2026-11-18..23). Orion only spoke when spoken to.
- Cause 1: each bid was a new want opened "now", and the family closes no act until a line is said, so hunger read 0 s forever and rank stayed 0.
- Cause 2: the chat lane read "full", and the line treated any non-`funded` answer as a refusal. The economy's own rule (`bind_verdict`) says a capacity deferral with net > 0 queues at the GPU pool.
- Fix: the sitting keeps one want clock across rebids until a line is delivered or passed; `line_admitted` runs a line on `funded` or `queued_at_pool`.

## Outcome moved

Lines can now be funded at all. Expected: hunger reaches ~0.63 after one 300 s tau, so a line wins about 5-7 min into a watched sitting. UNVERIFIED live.

## Current architecture

Patch 3 (PR #3745) bid per tick with a fresh `WantV1(opened_at=now)` and ran only on `outcome == "funded"`.

## Architecture touched

`services/orion-hub/scripts/retro_commentary.py` only (plus comments in binding.py, families.yaml, contract.yaml, settings.py).

## Files changed

- `services/orion-hub/scripts/retro_commentary.py`: `_wanted_since` clock; `line_admitted()` via `bind_verdict`; reset after delivered/passed and on sitting change.
- `services/orion-hub/tests/test_retro_commentary_service.py`: 5 tests (pool-queued runs; held cases hold; clock held across ticks; reset on delivery; reset on new sitting).
- `orion/physiology/binding.py`, `config/physiology/families.yaml`, `config/physiology/contract.yaml`, `services/orion-hub/app/settings.py`: comments now name `queued_at_pool`.

## Schema / bus / API changes

- None.

## Env/config changes

- None.

## Tests run

```text
services/orion-hub: tests/test_retro_commentary_service.py + tests/retro: 302 passed
orion/physiology: 379 passed
scripts/check_body_economy_contract.py: OK
Mutation check: disabling both clock resets fails 2 new tests.
```

## Evals run

```text
None new. Live check after deploy: #3741.
```

## Docker/build/smoke checks

```text
None: Hub code only; post-pull rebuild covers it.
```

## Review findings fixed

- Finding: tests called `_want()` directly; disabling the resets still passed.
  - Fix: tick-level tests. Evidence: mutation check fails 2 tests.
- Finding: clock reset on admission, before yield/insert checks.
  - Fix: reset only after `delivered` or `passed`.
- Finding: stale entry when one sitting replaces another.
  - Fix: clear on sitting change, with a test.
- Finding: four comments said "only funded runs a line".
  - Fix: updated.
- Finding: `datetime.now()` instead of the injected clock.
  - Fix: `line_admitted(decision, now)`.

## Restart required

```text
No manual restart: the post-pull rebuild rebuilds orion-hub.
```

## Risks / concerns

- Severity: low. Concern: the first line ever (no act close yet) can carry long hunger after a long chat. After the first close, hunger counts from the last close. Mitigation: by design of `appetite.hunger`.
- Severity: low. Concern: a `queued_at_pool` line has no lane reservation in the market's book. Mitigation: the economy's documented design; the pool queue is the real queue.
- Issues opened: none. Closed: none. Live proof: #3741.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3775

🤖 Generated with [Claude Code](https://claude.com/claude-code)

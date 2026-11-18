# Body economy Patch C3b (dream): shadow bid for the dream cycle

## Summary

- A dream check that would sleep now asks the body economy "would you fund this sleep?", and ignores the answer. The answer is only logged, next to what the cycle really did.
- The bid happens once a check would sleep (pressure due, the overdue backstop, or forced). Checks that are too soon or not due are not a want, so they do not bid.
- The bid's `want_id` is the sleep act's correlation id (`act_emit.cycle_correlation_id(cycle_id)`). The cycle id is now minted before the arousal read so a held check has one too.
- The bid is a background task. The cycle, its act events and what it persists are the same whatever the economy answers.
- Switch `DREAM_PHYSIOLOGY_SHADOW_BID_ENABLED`, shipped on.

## Outcome moved

The arousal hold is one of the strain vetoes Patch D replaces with a bid. Each would-be sleep now leaves a `physiology_shadow_bid ... shadow=<funded|deferred|none> real=<ran|blocked:reason>` line, and durable-runs writes the decision to `want_ledger`, so the two can be compared before anything is cut over.

## Current architecture

Patch C1 built the market and client. orion-dream's `run_cycle_once` gated a sleep on too-soon, pressure (or the backstop), then arousal, and opened a `make` act (actor `dream.cycle`) for a sleep with material to replay.

## Architecture touched

- orion-dream only, plus one contract registration and the env sync list. No new bus channel.

## Files changed

- `services/orion-dream/app/physiology_shadow.py`: new. Same `ShadowBid` helper as the hub patch.
- `services/orion-dream/app/cycle.py`:
  - The `if not force:` gate block is split. Too-soon and not-due still run first. Then the cycle id is minted and the bid is placed. Then the arousal hold (non-forced only).
  - Settles: `blocked:arousal_<level>`, `blocked:cancelled` (a cancel mid arousal read), `blocked:nothing_to_replay` (the `empty` cycle), `ran` (the act is about to open).
  - `_cycle_shadow` never raises (falls back to a no-op).
- `services/orion-dream/app/settings.py`, `.env_example`, `docker-compose.yml`: the switch (compose lists this service's env explicitly). The client's default URL resolves on app-net (checked from the running container).
- `scripts/sync_local_env_from_example.py`: `DREAM_PHYSIOLOGY_SHADOW_BID_ENABLED` in `SYNC_EXACT` (no `DREAM_` prefix is synced).
- `config/physiology/contract.yaml`: the helper registered as `display`.
- `services/orion-dream/tests/conftest.py`: autouse fixture so tests never POST to the live market.
- `services/orion-dream/tests/test_physiology_shadow_bid.py`: new (9 tests).
- `services/orion-dream/README.md`: section.

## Correlation join

| family | want_id | act correlation id | verified by |
|---|---|---|---|
| `dream.cycle` (Tier 1) | `act_emit.cycle_correlation_id(cycle_id)` | the `act_opened` event's `correlation_id` (`act_emit.opened_event`) | `test_a_sleep_is_identical_for_every_answer_and_joins_its_act` asserts want_id == the emitted act_opened correlation id |

## Real-gate field

`ran` (the gates passed and the act opens; a sleep that then fails still ran and its act closes at 0), `blocked:arousal_<level>`, `blocked:nothing_to_replay`, `blocked:cancelled`.

## Schema / bus / API changes

- Added / Removed / Renamed: none.
- Behavior changed: none (shadow only). The cycle id is minted a few lines earlier; nothing on the hold path reads it.
- Compatibility notes: none.

## Env/config changes

- Added keys: `DREAM_PHYSIOLOGY_SHADOW_BID_ENABLED=true` (orion-dream).
- `.env_example` updated: yes. `docker-compose.yml` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (primary checkout `services/orion-dream/.env`).
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-dream: pytest tests -> 299 passed, 7 skipped (incl. test_physiology_shadow_bid.py 9)
orion-static-gates.yml python steps (38 gates, incl. check_body_economy_contract, settings defaults) -> all OK
```

## Evals run

```text
None. The shadow-vs-real comparison needs live decisions (scripts/analysis/body_economy_shadow_vs_real.py, another agent).
```

## Docker/build/smoke checks

```text
docker exec orion-athena-dream: GET http://orion-athena-durable-runs:8121/health -> 200.
The live bid from a real sleep is UNVERIFIED until deploy.
```

## Review findings fixed

- Finding: shadow construction sat outside any try; an import error would stop a sleep.
  - Fix: `_cycle_shadow` builds and places inside a try and falls back to a no-op.
  - Evidence: `test_a_shadow_that_cannot_be_built_never_stops_a_sleep`.
- Finding: a cancel during the arousal read left a placed bid unsettled (no line, a funded reservation held to expiry).
  - Fix: settle `blocked:cancelled` and re-raise.
- Finding: the join test compared the helper with itself; act events were not checked across answers.
  - Fix: the test records the real act events and asserts want_id == the act_opened correlation id, and that the act events are identical across the four answers and flag-off.
- Finding: `ran` with the act ledger off, and hold volume, were undocumented.
  - Fix: README lines.
- Nits: raw `_enabled` assignment replaced with monkeypatch; a redundant format removed.

## Restart required

```bash
scripts/safe_docker_build.sh orion-dream up -d --build
```

Run it from the clean primary `main` checkout after merge.

## Risks / concerns

- Severity: low. Concern: a long arousal hold bids every check (about 6 rows an hour at the 600 s interval). Mitigation: documented; the report should count hold episodes.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2929

🤖 Generated with [Claude Code](https://claude.com/claude-code)

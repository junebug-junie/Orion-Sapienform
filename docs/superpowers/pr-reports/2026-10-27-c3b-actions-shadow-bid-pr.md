# Body economy Patch C3b (actions): shadow bid for the in-process journal

## Summary

- Each in-process journal compose now asks the body economy "would you fund this?", and ignores the answer. The answer is only logged, next to what the compose really did.
- `actions.journal` is Tier 1. Tier 1 still bids; only Tier 0 never bids.
- The bid happens once the journaling switch and the dedupe cooldown let a trigger through. A cooldown skip is not a want and does not bid.
- The bid's `want_id` is the compose correlation id, which is the journal act's correlation id and the id the compose's GPU leases carry.
- The bid is a background task. The compose, its act events and its cortex request are the same whatever the economy answers.
- Switch `ACTIONS_PHYSIOLOGY_SHADOW_BID_ENABLED`, shipped on.

## Outcome moved

Patch D needs evidence of where the economy would have deferred upkeep that really ran. Each journal compose now leaves a `physiology_shadow_bid ... shadow=<funded|deferred|none> real=<ran|blocked:reason>` line, and durable-runs writes the decision to `want_ledger`.

## Current architecture

Patch C1 built the market and client. `_dispatch_journal` gated on the journaling switch and the dedupe cooldown, waited for the semaphore, then opened an `actions.journal` / `make` act on a per-compose correlation id.

## Architecture touched

- orion-actions only, plus one contract registration and the env sync list. No new bus channel.

## Files changed

- `services/orion-actions/app/physiology_shadow.py`: new. Same `ShadowBid` helper as the hub patch.
- `services/orion-actions/app/main.py`: `_journal_shadow` builds and places the bid (never raises); `_settle_shadow` settles `ran` right after the semaphore is acquired, or `blocked:raised:<Exception>` on the error and cancel paths (once only).
- `services/orion-actions/app/settings.py`, `.env_example`: the switch (compose passes it through `env_file`). The client's default URL resolves on app-net (checked from the running container).
- `scripts/sync_local_env_from_example.py`: `ACTIONS_PHYSIOLOGY_SHADOW_BID_ENABLED` in `SYNC_EXACT`.
- `config/physiology/contract.yaml`: the helper registered as `display`.
- `services/orion-actions/tests/conftest.py`: tests never POST to the live market.
- `services/orion-actions/tests/test_physiology_shadow_bid.py`: new (7 tests).
- `services/orion-actions/README.md`: section.

## Correlation join

| family | want_id | act correlation id | verified by |
|---|---|---|---|
| `actions.journal` (Tier 1) | `compose_correlation_id(dedupe_key, trigger corr)` (the trigger's id when acts are off) | `ActTrace.correlation_id` = the same compose id | `test_compose_is_identical_for_every_answer_and_joins_the_act` asserts want_id == act event correlation id == the cortex compose request's correlation id |

## Real-gate field

`ran` once the semaphore is acquired and the act opens (a compose that then fails still ran; its act closes at 0). `blocked:raised:<Exception>` if it was cancelled or raised before that. A funded shadow decision on a blocked compose is released.

## Not bid here

- Durable `journal.compose` runs (world_pulse_digest): durable-runs admission covers them.
- `actions.daily_metacog`: off live (`ACTIONS_DAILY_METACOG_ENABLED=false`), not in scope.

## Schema / bus / API changes

- Added / Removed / Renamed: none. Behavior changed: none (shadow only).

## Env/config changes

- Added keys: `ACTIONS_PHYSIOLOGY_SHADOW_BID_ENABLED=true` (orion-actions).
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (primary checkout `services/orion-actions/.env`).
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-actions: pytest tests -> 281 passed (incl. test_physiology_shadow_bid.py 7, test_act_emit.py)
orion-static-gates.yml python steps (38 gates, incl. check_body_economy_contract) -> all OK
```

## Evals run

```text
None. The shadow-vs-real comparison needs live decisions (scripts/analysis/body_economy_shadow_vs_real.py, another agent).
```

## Docker/build/smoke checks

```text
docker exec orion-athena-actions: GET http://orion-athena-durable-runs:8121/health -> 200.
The live bid from a real journal is UNVERIFIED until deploy.
```

## Review findings fixed

- Finding: the blocked path and the release were untested.
  - Fix: `test_a_compose_that_never_got_its_slot_is_blocked_and_releases` (the semaphore raises; `real=blocked:raised:RuntimeError`; the funded reservation is released).
- Finding: with acts off, the want_id joined nothing.
  - Fix: want_id is `compose_corr or` the trigger's id, the id the compose request then carries.
- Finding: "ran" could be read as "produced".
  - Fix: README says `ran` means the real gates let it run.
- Not changed: bid tasks are not cancelled at shutdown (each is bounded by the 0.5 s client timeout; cosmetic).

## Restart required

```bash
scripts/safe_docker_build.sh orion-actions up -d --build
```

Run it from the clean primary `main` checkout after merge.

## Risks / concerns

- Severity: low. Concern: a funded shadow decision for a compose that ran but failed holds its reservation until durable-runs expires it (UNVERIFIED whether an act close releases it). Mitigation: shadow only; no real gate reads the book.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2930

🤖 Generated with [Claude Code](https://claude.com/claude-code)

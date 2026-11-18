## Summary

- Prime's thought (and every other resident loop on `orion/mindloop/runtime.py`) now goes ahead when the body economy says "the lane is full, but this is worth its price" (`capacity_reserved` with a net above 0). The GPU pool's own queue holds the call, and the seat wait still bounds it.
- This is the rule the day letter got in #3241, and the rule every binding family already uses (`bind_verdict` in `orion/physiology/binding.py`, `goes_ahead` in `orion/physiology/prime_charge.py`).
- Never fail open is kept: no answer from the market still waits. A consent block never goes ahead.
- A queued thought logs `<name>_bid_queued_at_pool outcome=... reason=... binding=... net=...`. A refusal still logs `<name>_bid_not_funded`.
- The Prime eval report now counts `went_ahead` next to `funded`.

## Outcome moved

Prime's thought and the curiosity replica no longer treat a lane-full, worth-it bid as a refusal.

Live evidence (`want_ledger`, read-only):

- Last 48 h: 0 thought bids were capacity deferrals with a positive net. Prime: 9 of 9 thought bids funded. Replica: 9 of 9 funded. Today this patch is a consistency fix, not a live unblock.
- Last 30 days: 910 Prime thought bids (2026-10-30 to 2026-11-03) were `deferred capacity_reserved lane:chat` with a positive net. Each one was a thought that waited. Under this rule each would have queued at the pool. They stopped after the cost-ceiling fix (#3004), not because of this rule.
- The 22 `make` bids in the last 48 h (all `capacity_reserved lane:agent`, net > 0) are the day letter, which #3241 and the later tier-0 change already handled.
- `docker logs orion-mind --since 48h` has 0 `_bid_not_funded` lines. The container restarted about 10 minutes before the check, so the logs only cover that window. The ledger above is the real evidence.

## Current architecture

`MindLoopRuntime.tick()` bid once per thought and thought only on `outcome == "funded"`. Every other answer called `_on_not_funded` and waited.

## Architecture touched

`orion/mindloop/runtime.py` (shared by `PrimeRuntime` and `ReplicaRuntime` in orion-mind). No schema, bus, or env change.

## Files changed

- `orion/mindloop/runtime.py`: gate on `goes_ahead(decision)` plus `consent_block`. Queued log line. Docstring.
- `services/orion-mind/app/prime_loop.py`, `services/orion-mind/app/replica_loop.py`: docstrings.
- `services/orion-mind/README.md`, `config/physiology/contract.yaml`, `docs/superpowers/specs/2026-11-07-attention-patch3-market-decides-design.md`: stale "only funded thinks" text.
- `orion/prime/evals/run_prime_report.py`: `went_ahead` count. The bids query reads `net`.
- `orion/mindloop/tests/echo_instance.py`, `orion/mindloop/tests/test_second_instance.py`: regression tests on the runtime.
- `services/orion-mind/tests/test_prime_loop.py`, `services/orion-mind/tests/test_replica_loop.py`, `orion/prime/tests/test_prime_evals.py`: instance-level tests.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: a `capacity_reserved` deferral with a finite net > 0 now runs the thought.
- Compatibility notes: the market made no reservation for a deferral, so the release after the thought is a no-op for it (`ReservationBook.release` returns 0).

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=services/orion-mind:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  services/orion-mind/tests orion/mindloop/tests orion/physiology orion/prime
850 passed
```

New tests:

- `test_a_capacity_deferral_worth_its_price_thinks_and_queues_at_the_pool` (runtime): one model call, act opened and closed, want released, queued log line. It fails on origin/main.
- `test_a_capacity_deferral_not_worth_its_price_waits` (runtime, net 0 / -0.1 / None / NaN) and Prime's equivalent.
- `test_another_deferral_with_a_positive_net_still_waits` (reason `price`).
- `test_no_answer_from_the_market_waits` (None decision).
- `test_a_consent_block_never_goes_ahead_even_when_worth_its_price`.
- `test_a_capacity_deferral_worth_its_price_thinks` (Prime) and `test_a_capacity_deferral_worth_its_price_plans_within_the_deadline` (replica: seat wait still cut to the deadline).
- The Prime replay golden (`test_prime_replay.py`) is unchanged and passes.

## Evals run

```text
orion/prime/tests/test_prime_evals.py (cadence now reports went_ahead) -- passed in the run above
```

## Docker/build/smoke checks

```text
Not run. Library and docstring change only; no dependency, config, or compose change.
After deploy: docker logs orion-mind 2>&1 | grep -E "_bid_(queued_at_pool|not_funded)"
```

## Review findings fixed

- Finding: `services/orion-mind/README.md` and `config/physiology/contract.yaml` still said "funded thinks, deferred or no answer waits".
  - Fix: both now name the worth-it capacity deferral.
  - Evidence: diff.
- Finding (nit): `goes_ahead` skips the consent check that `bind_verdict` runs first.
  - Fix: the runtime also refuses when `consent_block(decision)` is set.
  - Evidence: `test_a_consent_block_never_goes_ahead_even_when_worth_its_price`.
- Finding (nit): no replica test; the Prime test did not check the log line or NaN.
  - Fix: replica test added; Prime test checks the log line and NaN.
- Finding (nit): the Prime eval report counted only funded bids.
  - Fix: `cadence()` adds `went_ahead` and `went_ahead_share`.
- Finding (nit): the patch-3 design table row was stale.
  - Fix: dated note added.
- Review confirmed clean: the gate, release after a deferred thought, the replica deadline, Prime state, and imports.

## Restart required

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns

- Severity: low. Concern: a queued thought can wait the full `PRIME_SEAT_WAIT_SEC` and get no seat. The next try then goes on the starved route, which has higher pool priority, so the starved route may be used more often. Mitigation: this is existing behavior, and it is still behind Juniper. After deploy, watch for `prime_try_no_model kind=no_seat`.
- Severity: low. Concern: the queued thought holds no reservation, so the market can fund other bids on the same lane while it runs. Mitigation: durable runs already work this way for `queued_at_pool`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3315

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## Summary

- Prime's day-letter bid now goes ahead when the body economy says "lane full, but worth its price" (`capacity_reserved` with net > 0). The GPU pool's own queue holds the run. This is the same rule every other binding family already uses (`orion/physiology/binding.py` `bind_verdict`, `orion/physiology/prime_charge.py` `goes_ahead`).
- Before: Part 3 (#3232) accepted only `outcome == "funded"`.

## Outcome moved

Live 11-12: 20 of 20 letter bids for 2026-11-11 were deferred `capacity_reserved, lane:agent` with net +0.67, from 09:05 to 18:20 UTC. No letter was written and no email was sent. The bid window ends 2026-11-13 03:05 UTC.

## Current architecture

`PrimeLoop._bid_letter` (`services/orion-mind/app/prime_loop.py`) treated any answer except `funded` as not funded.

## Architecture touched

orion-mind only. No schema, bus, or env change.

## Files changed

- `services/orion-mind/app/prime_loop.py`: use `pc.goes_ahead(decision)`.
- `services/orion-mind/tests/test_prime_day_letter.py`: the live answer orders the letter; a capacity deferral with net <= 0 still holds.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=services/orion-mind:. pytest -q -p no:cacheprovider services/orion-mind/tests  ->  268 passed
```

## Evals run

None changed. The live check is below.

## Docker/build/smoke checks

After deploy: `docker logs orion-mind 2>&1 | grep prime_day_letter_funded` within 30 min, then `prime_day_letter_ordered`, then an `orion_day_letter` row for 2026-11-11 and its email.

## Review findings fixed

- Finding: the `prime_day_letter_funded` log could not tell funded from queued.
  - Fix: the line now logs `outcome` and `reason`.
- Finding: docstrings said only "funded" goes ahead.
  - Fix: prime_loop.py, day_letter.py, orion_day.py, orion_day_order.py now name the worth-it capacity deferral.
- Review verdict otherwise: no material findings (release is a no-op for an unreserved want; the letter run is not a binding family in durable-runs, so it is not re-bid and held).

## Restart required

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns

- Severity: low. Concern: the letter turn now waits in the agent pool queue behind other work. Mitigation: that is the designed queue; the run deadline (12 h) still bounds it.

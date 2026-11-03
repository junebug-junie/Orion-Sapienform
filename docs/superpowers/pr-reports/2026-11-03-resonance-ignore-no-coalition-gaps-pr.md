## Summary

- The resonance alarm ("reverie theme X resonance is worsening") now counts a quick re-fire as a cooldown breach only when the chain before it actually started a cooldown.
- Chains that could not produce a first thought (`no_coalition`) do not start a cooldown on purpose, so a fast retry after one is the gate working, not a breach.
- The worsening health check treats alert rows older than two cooldown periods as stale. Without this, the open page for a theme that stops resonating could never close.

## Outcome moved

Stops repeated false pages for `open-loop-7376a3da4050` (and any theme with the same pattern).

Live evidence (2026-11-03, last 10 days of this theme): 104 of 104 re-fires inside the 15-minute cooldown came right after a `no_coalition` chain. 0 of 99 came after a chain that armed the cooldown. Logs show `reverie_chain_skipped reason=refractory` holding normally. Replaying the live last-200-chain window: old detector = 18 violations (matches the live alert); new detector = no alert.

## Current architecture

`detect_resonance` (orion/reverie/resonance.py) counted every same-theme gap under `refractory_sec` as a violation. `run_reverie_chain` only arms the cooldown for discharged/settled/max_steps/low_salience chains, or after 3 `no_coalition` chains in a row. The two disagreed.

## Architecture touched

- `ThemeEvent.armed` (default `True`, so other callers keep old behaviour).
- `store.load_recent_chain_theme_events` also returns `terminal_reason`.
- `resonance_monitor._is_worsening` gets an optional staleness limit (2 x `reverie_refractory_sec`).

## Files changed

- `orion/reverie/resonance.py`: skip gaps after unarmed chains.
- `orion/reverie/tests/test_resonance.py`: unarmed / mixed regression tests.
- `services/orion-thought/app/chain.py`: mark `no_coalition` chains unarmed.
- `services/orion-thought/app/store.py`: select `terminal_reason`.
- `services/orion-thought/app/resonance_monitor.py`: stale samples read healthy.
- `services/orion-thought/tests/test_reverie_chain.py`, `tests/test_resonance_monitor.py`: regression tests.

## Schema / bus / API changes

- None. No bus, schema, or table change.

## Env/config changes

- None.

## Tests run

```text
orion-thought: tests/test_resonance_monitor.py tests/test_reverie_chain.py tests/test_store.py evals -> pass
orion/reverie/tests -> pass
(final run: 131 passed)
```

## Evals run

```text
services/orion-thought/evals (incl. test_reverie_efficacy_resonance_eval.py) -> pass
```

## Docker/build/smoke checks

```text
Live replay of detector over the last 200 substrate_reverie_chain rows: old=18 violations, new=None.
Container not rebuilt from this branch (deploys run from main after merge).
```

## Review findings fixed

- Finding: once no new alert rows are saved, the monitor re-reads the last 2 rows (17 -> 18) forever, so the page never recovers.
  - Fix: newest sample older than 2 x refractory_sec reads healthy.
  - Evidence: `test_monitor_recovers_once_trend_goes_stale`, `test_is_worsening_stale_samples_read_healthy`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns

- Severity: low
- Concern: a `no_coalition` chain that hit the 3-in-a-row forced cooldown is also treated as unarmed, so a breach right after a forced cooldown is not counted.
- Mitigation: a broken cooldown store would still show up after normal chains.
- Severity: follow-up (not fixed here)
- Concern: this theme fails to produce a first thought 72% of the time (260 of 359 chains). Its thoughts are dropped as `unanchored_evidence_outside_coalition`.
- Mitigation: separate thought-quality investigation.

## PR link

TBD

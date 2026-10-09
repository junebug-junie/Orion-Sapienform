## Summary

- **The grader now judges rest by the gate's own decisions.** `scripts/analysis/grade_transport_baseline.py` asks "would this hop raise a false alert while calm?" It no longer judges bands the gate ignores.
  - **A rest FAIL is:**
    - a spike or saturation the gate itself opened during quiet hours (`would_emit_by_condition`); or
    - a typical quiet-hour ratio at or above the gate's saturation line (2.0) with a material gap of 250 ms or more.
  - **Everything else is a NOTE.**
- **The gap is `floor × (ratio − 1)`,** which is the gate's own level-minus-floor. It is no longer `baseline_ms − floor_ms`, because `baseline_ms` is the guarded fast mean and does not absorb a step change.
- **Excluded hops are shown but never decide the overall verdict.** `--spike-z`, `--saturation-ratio` and `--min-excess-ms` mirror the live gate config.
- **`EQUILIBRIUM_TRANSPORT_BASELINE_EMIT` defaults to `true`** in settings, `.env_example`, compose and the README. It has been live since 2026-10-01.

## Outcome moved

- **First live night, old rules:** FAIL on 12 of 17 hops. Every failure was one the gate could never alert on:
  - ratios on tiny gaps (`orion:state:request`: 59 ms against a 23 ms best);
  - excluded hops;
  - quiet-hour z of 0.5–3 from the night-versus-day load difference.
- **Same data, new rules:** PASS, 11 of 11 hops that can alert. `orion-mind`'s LLM hop read z +4.02 in its one quiet hour, but the gate opened no spike: the gap wasn't material or didn't last. It is now a note.
- **The would-emit rows,** about 30 opens in 19 hours, each matched a real timeout in the same hour. Today's logic produces hundreds of rows a day. On that evidence, the gate is now the publisher.

## Scope, stated

- **What this grades:** whether a hop would raise a false alert at rest.
- **What it doesn't grade:** whether a hop is drifting. Slow creep below the saturation line is caught only by the floor-rise check, which is unchanged.

## Files changed

- `scripts/analysis/grade_transport_baseline.py`: gate-rule grading, the gap formula, notes, excluded-hop handling and CLI overrides.
- `scripts/analysis/tests/test_grade_transport_baseline.py`: 24 tests. The fixtures keep `baseline_ms` at the floor, so a test cannot lean on it.
- `services/orion-equilibrium-service/{app/settings.py,.env_example,docker-compose.yml,README.md}`: EMIT defaults to true. The README also fixes the stale `transport` row, which still listed bus_synaptic.

## Env/config changes

- **Default changed:** `EQUILIBRIUM_TRANSPORT_BASELINE_EMIT` false → true.
- **Local `.env`:** edited by hand to `true`, because the sync script only adds keys. Backup in the session scratchpad.
- **Live:** equilibrium was restarted on the same image (`safe_docker_build.sh ... up -d`). It confirmed `EMIT=true` and `transport_baseline resumed keys=73`, so the learned state was kept.

## Tests run

```text
scripts/analysis/tests/test_grade_transport_baseline.py + services/orion-equilibrium-service/tests -> 214 passed
orion/metacog/tests + equilibrium evals (earlier run on this branch) -> 424 passed
check_env_template_parity PASS; check_definition_drift --gate PASS
Mutation checks: gap via baseline_ms -> 3 tests fail; ignoring the gate's opened episodes -> 1 fails;
materiality removed -> 1 fails; excluded hops counted -> 2 fail.
Live: grade_transport_baseline.py --days 2 -> Overall PASS (11 pass, 0 fail, 6 excluded).
```

## Review findings fixed

- **HIGH: the gap read the guarded fast mean, which hid saturation on a step.**
  - **Fix:** the gap is `floor × (ratio − 1)`.
  - **Evidence:** `test_material_saturation_at_rest_fails_even_when_fast_stayed_low`, mutation-checked.
- **MEDIUM: `|z| ≥ spike_z` FAILed hops that can never fire** (negative z, tiny-ms hops).
  - **Fix:** a FAIL now requires a spike or saturation the gate opened at rest; z alone is a note.
  - **Evidence:** new tests, mutation-checked.
- **LOW:** a ratio below 0.8 printed as "-X ms above".
  - **Fix:** it now reads "faster than its recorded best".
- **LOW, stated rather than changed:** the narrower scope (false alerts at rest, not drift). See above.

## Restart required

```text
No restart required: EMIT is already live; this PR aligns the checked-in defaults with it.
```

## Risks / concerns

- **Severity: low.**
  - **Concern:** only 0.8 days of hourly readings so far, where the spec asks for a week.
  - **Mitigation:** the hourly table keeps recording. Re-run `grade_transport_baseline.py --days 7` after a week. Back out with `EQUILIBRIUM_TRANSPORT_BASELINE_EMIT=false` and an equilibrium restart.

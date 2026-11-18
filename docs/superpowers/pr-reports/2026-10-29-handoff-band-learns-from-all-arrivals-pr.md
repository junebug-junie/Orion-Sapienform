## Summary

- The cross-camera handoff loop now switches from its hand-set arrival-time guess (the prior band) to its own measured window after 10 arrivals of any kind: met, early or late. Before, only `met` arrivals counted.
- Why: off a wrong guess, most arrivals land early or late, so the switch could never happen. Live: approach_door's guess was 3-30 s, and real arrivals took 1-74 s (3 met, 6 late, 2 early in 5 days).
- `VISION_HANDOFF_MIN_MET` is renamed `VISION_HANDOFF_MIN_ARRIVALS`, because its meaning changed. Met counts are removed from `learned_gaps`.
- The replay eval counts arrivals, and judges "too thin" on the best-fed portal and kind (that is how the switch works), not the sum over all portals.

## Outcome moved

On the real history since the doorbell fix, the replay switched approach_door to the learned band at its 10th arrival. The prior band was used 52 times, the learned band 2 times. Right after deploy, approach_door has 11 arrivals on record, so its next prediction uses the measured window.

Spot check of whether those 11 arrivals were real: the doorbell saw a person within 40 s after 8 of them, including the 1.1, 2.4 and 7.5 s ones, so the fast arrivals are not wrong matches. The 3 with no person on the door camera were probably Ring motions set off by something else.

## Current architecture

`choose_band` (orion-sql-writer `app/vision_handoff.py`) received `g` for every met, early and late row, but gated on a separate count of `met` rows.

## Architecture touched

orion-sql-writer only: `vision_handoff.py`, `vision_handoff_loop.py`, settings, `.env_example`, README, the replay eval, tests.

## Files changed

- `services/orion-sql-writer/app/vision_handoff.py`: `choose_band` gates on `len(learned[kind])`. `min_met` is renamed `min_arrivals`.
- `services/orion-sql-writer/app/vision_handoff_loop.py`: `learned_gaps` returns (learned, missed). Callers updated.
- `services/orion-sql-writer/app/settings.py`, `.env_example`, `README.md`: `VISION_HANDOFF_MIN_ARRIVALS`.
- `services/orion-sql-writer/evals/run_vision_handoff_replay_eval.py`: arrival-based switch and verdict.
- `services/orion-sql-writer/tests/test_vision_handoff.py`: new signature, plus a regression test built from the live 10-day pattern.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the learned band takes over after 10 arrivals of any kind.
- Compatibility notes: none. The table is unchanged, and existing early and late rows count right away.

## Env/config changes

- Added keys: `VISION_HANDOFF_MIN_ARRIVALS=10`
- Removed keys: `VISION_HANDOFF_MIN_MET`
- Renamed keys: `VISION_HANDOFF_MIN_MET` -> `VISION_HANDOFF_MIN_ARRIVALS`
- `.env_example` updated: yes
- local `.env` synced: yes. Renamed in place in the primary checkout's `services/orion-sql-writer/.env` (the sync script reads the template from its own checkout, which did not have this branch yet).
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-sql-writer/tests services/orion-sql-writer/evals -k handoff -> 65 passed, 16 skipped (Postgres tests; no test DB here)
New: test_late_arrivals_count_toward_the_switch_off_a_wrong_prior (the live 3 met / 6 late / 1 early values -> learned at 10, prior at 9)
scripts/check_env_template_parity.py -> PASS
```

## Evals run

```text
run_vision_handoff_replay_eval.py --since 2026-10-24T00:20Z (read-only)
  approach_door: met 4, late 5, early 2, unscorable 16; band_source prior 52 / learned 2
  learned g approach_door/person: n=11, median 42.4 s, p10 2.4 s, p90 63.2 s
  verdict: TOO THIN (5 met p-values, need 20); learned spread above 50% of the median
```

## Docker/build/smoke checks

```text
None: no image or compose change.
```

## Review findings fixed

- Finding: the eval's "too thin" verdict summed arrivals over all portals and kinds, but the switch is per portal and kind.
  - Fix: the verdict judges the best-fed portal and kind.
  - Evidence: replay above.
- Finding (checked, no change): fast arrivals might be wrong matches on timing alone.
  - Evidence: the doorbell saw a person after 8 of 11 arrivals, including the fast ones.
- Finding (noted, not fixed): arrivals past the old 84 s matching cutoff during the prior era were graded missed and never trained. The cutoff grows with the learned band, so this fades as new data comes in.

## Restart required

```bash
git pull --ff-only && scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: real approach times look bimodal (straight to the door in seconds vs about a minute). One 10-90% band covers both, so the window is wide (about 1-74 s) and less specific.
  - Mitigation: grading stays honest. A per-route model is a possible follow-up once there is enough data.
- Severity: low
  - Concern: 3 of 11 training arrivals had no person on the door camera (probably Ring motion from something else).
  - Mitigation: a follow-up could train only on arrivals the door camera confirms.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2948

🤖 Generated with [Claude Code](https://claude.com/claude-code)

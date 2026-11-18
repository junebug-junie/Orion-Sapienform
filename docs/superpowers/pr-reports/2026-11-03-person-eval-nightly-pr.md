## Summary

- `make eval-person-recognition`: runs the person-learning eval with `--write`, so every run stores one `vision_person_model_eval` row.
- Nightly host cron (09:23 UTC) after merge; README documents it.
- First stored row written by hand today: `eval-58d03b6ab5994a04b20f5cf39ddee4ee`.

## Outcome moved

The spec's Phase 1 acceptance check ("the eval writes a `vision_person_model_eval` row") had never been met: 0 rows since 10-20, because nothing ran the eval. Now there is a nightly history, and the McNemar comparison against the previous model version has something to compare to.

First row (2026-11-03): 113 labelled clusters, 0 wrong accepts, 0 correct accepts, unsure rate 1.0, top unsure guess right 50 of 103; cross-day cluster AUC body 0.853, face 0.614.

## Current architecture

`services/orion-sql-writer/evals/run_person_recognition_eval.py` existed (read-only unless `--write`). No make target, no schedule.

## Architecture touched

Makefile target; sql-writer README. No service code.

## Files changed

- `Makefile`: `eval-person-recognition` target (refuses without `POSTGRES_URI`).
- `services/orion-sql-writer/README.md`: nightly run documented.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
POSTGRES_URI= make eval-person-recognition   -> "POSTGRES_URI is required", exit 2
POSTGRES_URI=<live> make eval-person-recognition -> exit 0, row eval-58d03b6ab5994a04b20f5cf39ddee4ee written
```

## Evals run

```text
see above (this PR is the eval's runner)
```

## Docker/build/smoke checks

Not applicable (no image change).

## Review findings fixed

- Finding: from the primary checkout (where cron runs), `METRIC_PYTHON` is the relative `.venv/bin/python`; the recipe's `cd services/orion-sql-writer` made it point at a file that does not exist, so every nightly run would exit 127 and write no row. The worktree test could not catch it.
  - Fix: no `cd`; the eval puts its own paths on sys.path. Each run also logs a UTC timestamp line.
  - Evidence: `make -n -C /mnt/scripts/Orion-Sapienform -f <worktree>/Makefile eval-person-recognition` expands to `.venv/bin/python services/orion-sql-writer/evals/...`, which exists in the main checkout.

## Restart required

```text
No restart required. After merge, add the host cron line:
23 9 * * * POSTGRES_URI=$(grep -m1 '^POSTGRES_URI=' /mnt/scripts/Orion-Sapienform/services/orion-hub/.env | cut -d= -f2-) make -C /mnt/scripts/Orion-Sapienform eval-person-recognition >> /mnt/scripts/Orion-Sapienform/logs/orion-person-recognition-eval.log 2>&1
```

## Risks / concerns

- Severity: low. Concern: ~15-20 min of CPU nightly on athena. Mitigation: 09:23 UTC is 03:23 local, a quiet hour.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3068

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## Summary

- Recalibrated `HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH` from 8 to 6.
- Pure config/constant change with documentation updates — no logic change, no schema/bus/API change.
- Follows directly from PR #1837 (durable decision log): once that log made per-tick decisions inspectable, live data showed the organic trigger had gone essentially silent, and root-causing why led here.

## Outcome moved

`MIN_RUN_LENGTH` was derived once (2026-08-16 replay) to gate roughly the top 1% of real field-tension persistence episodes. The live field has since gotten noisier — more nodes actively compete for the Borda win (including `node:rpc_timeout`), and 55.96% of ticks in a sampled trailing hour had no winner at all. Re-running the same replay technique against the trailing 24h (2026-08-22) found the old bar of 8 had silently drifted to gating roughly the top 0.2% of episodes, not 1% — 8 qualifying runs out of 3,625 in that 24h window, versus 71 at a bar of 6. This is exactly the risk the module's own docstring named ("if that tuning drifts later, the distribution this bar rests on drifts with it and nothing here would notice") — and nothing did notice, until asked directly a second time why outreach still wasn't firing after PR #1837 shipped.

## Current architecture

`services/orion-hub/scripts/tension_outreach_trigger.py::current_run()` walks backward from the latest `substrate_field_state` tick and requires `run_length >= MIN_RUN_LENGTH` consecutive same-winner ticks (no gap tolerance — a NULL or different-winner tick breaks the run) before returning a `TensionTriggerReason`. `MIN_RUN_LENGTH` was a module constant (8) with a `HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH` env override, explicitly designed to be operator-tunable from live firing-rate data without a code/deploy change.

## Architecture touched

- `services/orion-hub/app/settings.py` — `HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH` Field default
- `services/orion-hub/scripts/tension_outreach_trigger.py` — module constant + docstring
- `services/orion-hub/.env_example`, `services/orion-hub/README.md` — operator-facing docs

## Files changed

- `services/orion-hub/app/settings.py`: Field default 8 → 6; comment cross-references `tension_outreach_trigger.py`'s docstring as the single source of truth for the replay numbers (an earlier draft of this patch inline-restated them here too, which a code review caught as breaking the file's own established single-source-of-truth convention for this exact paragraph)
- `services/orion-hub/scripts/tension_outreach_trigger.py`: module constant `MIN_RUN_LENGTH` 8 → 6; "WHERE THE BAR CAME FROM" docstring section now documents both the 2026-08-16 original derivation and the 2026-08-22 recalibration side by side
- `services/orion-hub/.env_example`: `HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH` 8 → 6
- `services/orion-hub/README.md`: updated the stated default (was still saying 8 after the code change — code review finding)
- `services/orion-hub/.env` (gitignored, not committed): synced to 6 by hand — the sync script's diverged-key protection correctly refused to auto-overwrite a locally-differing value without `--force`, since it can't tell "stale" from "an intentional host override" on its own; this one just needed the manual review it was asking for

## Schema / bus / API changes

None. Pure constant recalibration.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- Changed default: `HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH` 8 → 6
- `.env_example` updated: yes
- local `.env` synced: yes, by hand (both `services/orion-hub/.env` and the primary checkout's copy) — `python3 scripts/sync_local_env_from_example.py` reported this key as "diverged" (correct behavior: a locally-differing value is treated as a possible intentional override, not silently clobbered) and needed the manual edit it flagged
- skipped keys requiring operator action: none

## Tests run

```
/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest services/orion-hub/tests/test_tension_outreach_trigger.py services/orion-hub/tests/test_endogenous_outreach.py -q
109 passed, 2 warnings
```
All `MIN_RUN_LENGTH`-dependent tests reference `tension_outreach_trigger.MIN_RUN_LENGTH` dynamically rather than hardcoding 8, so they exercise the new value automatically with no test changes needed.

## Evals run

None — no eval harness exists for this subsystem; not added here (see PR #1837's own report for the same note).

## Docker/build/smoke checks

```
scripts/safe_docker_build.sh orion-hub build   # clean build
```

## Review findings fixed

- Finding: commit message claimed the value was "applied live in .env (both root and services/orion-hub)" — root `.env`/`.env_example` carry zero `HUB_*` keys; this is a hub-service-scoped setting that has never lived there.
  - Fix: corrected in this PR report and the final commit message; the claim only ever applied to `services/orion-hub/.env`.
- Finding: `settings.py` and `tension_outreach_trigger.py` both cited this PR report file before it existed.
  - Fix: this file.
- Finding: `services/orion-hub/README.md` still stated the old default (8) after the code change.
  - Fix: updated to reflect the 2026-08-22 recalibration.
- Finding: `settings.py`'s new comment inline-restated the full replay numbers instead of cross-referencing `tension_outreach_trigger.py`'s docstring, breaking that file's own established single-source-of-truth convention (the untouched 2026-08-16 paragraph right above it already just says "see that module's docstring").
  - Fix: trimmed to a cross-reference.
- Finding (disclosed, not fixed — legitimate scope note per the reviewer, not a defect in this patch): this is the second manual recalibration of this constant, and no automated drift-detection exists — the only way to notice future drift is another one-off hand-run SQL replay.
  - Not fixed here: building a periodic/automated drift check is real, separate follow-up work (an eval or a scheduled check comparing the live run-length distribution against the configured bar), out of scope for what was asked (recalibrate the value). Flagging it explicitly rather than silently deferring it.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
- Concern: a future `FieldTensionCompetition` tuning change will silently invalidate this bar again, the same way it did between 2026-08-16 and 2026-08-22, since no automated drift detection exists.
- Mitigation: disclosed in both the module docstring and this report; the constant remains cheaply retunable from live data without a deploy. A real fix (automated periodic re-derivation, or the wall-clock-persistence redesign the module already names as the deeper fix for the poll-cadence gap) is separate follow-up work.

## PR link

<filled in after push>

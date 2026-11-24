## Summary

- Causal geometry Patch 2 (#3290): a read-only replay that measures Orion's whole-body structure on live data and gives each structure feature a keep or drop verdict.
- New pure module `orion/substrate/whole_body_structure.py`. It turns levels into surprise (each channel's departure from its own hour-of-day average, earlier days only). It computes coupling, effective modes, the leading mode, redundancy vs synergy, body-mind coupling and drift. It includes the shifted null and the day-to-day mode match.
- New `merge_copies`. One bus message often lights several channels at once, and counted apart they read as coupling even at rest. Near-exact copies are now merged into one channel.
- New `scripts/whole_body_structure_replay.py`, with its evidence in `docs/superpowers/evidence/2026-11-24-whole-body-structure-replay/`.
- Fixes a field demon bug found on live data. Its "clock" baseline ignored today's level, so a level offset read as about 6% foresight all the way to 2 h.

## Outcome moved

Every Layer 2 feature now has a verdict from real data:
- **Keep:** body coupling (30 days); Layer 1 (46 of the top 50 links are not body hardware); whole-body body-mind coupling, provisional (#3794).
- **Hold:** whole-body coupling and redundancy. They beat chance but never come back to chance on quiet nights (#3795).
- **Drop:** the rest.

No mode is stable from day to day, so the integration rung (#3296) has no organs to cut between yet. The field demon's first preview read: 20% better than simple guesses at 5 min, gone by about 45 min. Splitting the modes costs 22% of the 5-minute foresight, beyond every shifted copy.

## Current architecture

Patch 1 (#3282) stores 5-minute bus traffic. No code computed Layer 2 features. The field demon core (#3728) existed, but had never run on live data.

## Architecture touched

- A pure module and a read-only replay script.
- A one-line baseline change in `orion/substrate/demon.py`, which no live service imports.
- The CI workflow.
- No services, contracts, schema, bus or config.

## Files changed

- `orion/substrate/whole_body_structure.py`: Layer 2 math (new).
- `orion/substrate/tests/test_whole_body_structure.py`: 19 known-answer tests (new).
- `scripts/whole_body_structure_replay.py`: the live replay and its gate (new).
- `orion/substrate/demon.py`: the clock baseline is now the training-window mean.
- `orion/substrate/tests/test_demon.py`: a level-offset regression test, and one check moved to where signal remains.
- `.github/workflows/field-demon-tests.yml`: runs the new tests.
- `docs/superpowers/evidence/2026-11-24-whole-body-structure-replay/{report.md,results.json}`: the verdicts (aggregates only).
- `docs/superpowers/pr-reports/2026-11-24-whole-body-structure-replay-pr.md`: this report.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: the `demon.rolling_errors` clock baseline. It has no live consumer.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/substrate/tests/test_demon.py orion/substrate/tests/test_whole_body_structure.py
51 passed in 8.32s

The new level-offset test fails on the old demon.py, as it should.
scripts/check_definition_drift.py --gate   PASS
scripts/check_metric_lineage.py --gate     PASS
scripts/check_integration_seam.py          OK
tests/test_no_private_eval_artifacts.py    3 passed
```

## Evals run

```text
scripts/whole_body_structure_replay.py --out <scratch> (live Postgres, read-only, seed 3290)
whole body: 8 days, 168 windows, 55 channels; body only: 30 days, 624 windows, 16 channels
Verdicts: docs/superpowers/evidence/2026-11-24-whole-body-structure-replay/report.md
```

## Docker/build/smoke checks

```text
None needed. No service or image changes.
```

## Review findings fixed

The code review subagent ran. Every finding is fixed, and the replay was rerun:

- **Finding:** the quiet-night filter never ran. Juniper's messages and Hub attention fail the 90% activity rule, so they were never among the channels, and every night counted as quiet (4 of the 8 "quiet" nights had Juniper messages).
  - **Fix:** the quiet test reads the raw columns, passed in as a parameter (this replaces a function-attribute hack). It refuses to run if either column is missing.
  - **Evidence:** results.json `rest.nights` against `rest.quiet_nights`: 8 against 3, and 27 against 15. The verdicts are restated.
- **Finding:** at rest, drift compared nights 24 h apart against a null 1 h apart.
  - **Fix:** drift is not scored at rest.
  - **Evidence:** the report's rest column.
- **Finding:** the report said the full cut costs 0.061 (30%). That number was the excess over the null; the actual cost is 0.044 (22%).
  - **Fix:** the report is corrected.
- **Finding:** the null's minimum gap (95 min) was not disclosed.
  - **Fix:** it is stated in the report.
- **Finding:** the bus's own body-telemetry channels were grouped as mind.
  - **Fix:** `orion:biometrics*` is now body. This moved body-mind coupling from 49% to 98% beating its null. Its link to the spark signal is flagged as possibly one source read twice (#3794).
- **Finding:** an anti-copy (r ≤ -0.95) would cancel itself out in a merge.
  - **Fix:** members are sign-aligned before averaging.
  - **Evidence:** `test_merged_anti_copy_does_not_cancel`.
- **Finding:** the report claimed world channels that did not survive selection, and quoted first-run numbers that are not in results.json.
  - **Fix:** the report now lists only what is in results.json.

## Restart required

```text
No restart required.
```

## Risks / concerns

- **Severity:** medium. **Concern:** body-mind coupling may be one source read twice. **Mitigation:** it is display-only until #3794 traces the spark-signal producer.
- **Severity:** medium. **Concern:** the whole-body rest test rests on only 3 quiet nights. **Mitigation:** re-run when more history exists; #3795 tracks the coupling floor.
- **Severity:** low. **Concern:** the field demon's horizon ("first step where skill reaches 0") saturates on slow modes even though the useful foresight is gone by about 45 min. **Mitigation:** raised on #3296 as a spec question. The cost curve and skill half-life are reported instead.
- **Issues opened:** #3794 (trace the spark signal), #3795 (the coupling floor at rest).
- **Issues closed:** Fixes #3290 (the replay and results are committed, and every feature has a verdict).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3796

🤖 Generated with [Claude Code](https://claude.com/claude-code)

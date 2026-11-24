## Summary

- Adds the slow test that Juniper asked for: are Orion's modes stable when blocks are several days long instead of one?
- New `block_stability` in `orion/substrate/whole_body_structure.py`. It matches each block's top modes to the next block's and compares them with a chance floor from shifted copies of the data.
- New `--stability-only` mode in `scripts/whole_body_structure_replay.py`. It runs 1, 2 and 4-day blocks on the whole body and 1, 2, 3, 5 and 7-day blocks on the body alone.
- Evidence: `docs/superpowers/evidence/2026-11-24-whole-body-structure-replay/slow/`.

## Outcome moved

Juniper asked: why not loosen the stability bar from 0.7 to 0.65? The data now answers it.
- **The modes are real.** Day to day, every top mode beats every chance draw. Whole body: 0.56-0.63 against at most 0.33. Body alone: 0.56-0.59 against at most 0.46.
- **They drift slowly.** Whole-body rank 1 rises 0.56, then 0.69, then 0.87 as blocks grow from 1 to 4 days. The 4-day value is one pair.
- **The 0.7 bar is now the deciding choice.** At 0.65 a few ranks would pass at 2-day blocks.
- **Still no organs for Φ (#3296).** No block length gives two ranks that clear 0.7. Re-run around 12-08, with 24+ days of bus history.

## Current architecture

The replay (#3796) tested mode stability only day to day, against a fixed 0.7 bar with no chance floor.

## Architecture touched

A pure function, a replay flag, and evidence. No services, contracts, schema, bus or config.

## Files changed

- `orion/substrate/whole_body_structure.py`: `STABLE_SIMILARITY`, `BlockStability`, `block_stability`.
- `orion/substrate/tests/test_whole_body_structure.py`: 5 tests. Fixed modes pass. Modes that change every block fail. 2-channel noise clears the bar but not the chance floor. A block too long for a null has no verdict. Fewer than 2 blocks returns None.
- `scripts/whole_body_structure_replay.py`: `--stability-only`, `stability_by_timescale` and a shared `_body_block` helper. The bar now comes from the module.
- `docs/superpowers/evidence/2026-11-24-whole-body-structure-replay/slow/{report.md,stability.json}`: results.
- `docs/superpowers/pr-reports/2026-11-24-slow-mode-stability-pr.md`: this report.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: none. The full replay is unchanged without the new flag.
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
pytest orion/substrate/tests/test_whole_body_structure.py orion/substrate/tests/test_demon.py
56 passed
Mutation: with the chance gate removed, the 2-channel noise test fails.
orion/substrate/tests (all): 1 unrelated failure, test_felt_state_lived_answers, which also fails on main.
```

## Evals run

```text
scripts/whole_body_structure_replay.py --stability-only (live Postgres, read-only, seed 3290)
whole body: 55 channels, 8.2 days; body only: 16 channels, 27 days
Results: docs/superpowers/evidence/2026-11-24-whole-body-structure-replay/slow/report.md
```

## Docker/build/smoke checks

```text
None needed. No service or image changes.
```

## Review findings fixed

The code review subagent ran. Fixed:

- **Finding:** the chance floor compared the real median over up to 26 pairs against the maximum of 20 single draws. That was far stricter than the real value at many pairs and looser at one pair. It made body modes read as near chance.
  - **Fix:** each chance draw is now the median over the same number of pairs as the real value.
  - **Evidence:** body day-to-day chance max dropped from 0.64 to 0.46, and the verdict changed: body modes beat chance.
- **Finding:** the null's channel gap was thrown away, and the report said the copies hold no real mode even at long blocks.
  - **Fix:** the gap is in `stability.json`, and the report says the floor is conservative at long blocks.
- **Finding:** the tests passed with the chance gate deleted.
  - **Fix:** added a 2-channel noise test, which clears 0.7 by construction, and a no-room test.
  - **Evidence:** the mutation check above.
- **Finding:** the report made claims that `stability.json` did not support: the 08:00 cut, "3 weeks" for 5 pairs, and no mention of multiple comparisons.
  - **Fix:** `first_block_start` is recorded, the wording is 24+ days, and multiple comparisons are stated.
- **Finding:** `except ValueError` also caught `LinAlgError`.
  - **Fix:** an explicit room check replaces it.
- **Finding:** `--stability-only` needed the quiet-night columns, and the body block code was copied.
  - **Fix:** the check moved after the branch, and a `_body_block` helper replaces the copy.

## Restart required

```text
No restart required.
```

## Risks / concerns

- **Severity:** low. **Concern:** long blocks give few pairs (whole body at 4 days: 1 pair), so a high match there cannot yet beat chance. **Mitigation:** the report says so; re-run around 12-08.
- **Issues opened:** none. **Issues closed:** none. The organ question stays on #3296.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)

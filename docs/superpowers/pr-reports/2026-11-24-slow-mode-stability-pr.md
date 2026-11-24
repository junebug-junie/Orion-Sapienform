## Summary

- Adds the slow test that Juniper asked for: are Orion's modes stable when blocks are several days long instead of one?
- New `block_stability` in `orion/substrate/whole_body_structure.py`. It matches each block's top modes to the next block's and compares them with a chance floor from shifted copies of the data.
- New `--stability-only` mode in `scripts/whole_body_structure_replay.py`. It runs 1, 2 and 4-day blocks on the whole body and 1, 2, 3, 5 and 7-day blocks on the body alone.
- Evidence: `docs/superpowers/evidence/2026-11-24-whole-body-structure-replay/slow/`.

## Outcome moved

The question "why not loosen the stability bar to 0.65?" now has an answer from data:
- Whole-body modes are real (day-to-day match 0.56-0.63 against chance 0.25) but drift. Rank 1 settles as blocks get longer (0.56, 0.69, 0.87). The 4-day value is one pair, so it is not proof yet.
- Body-only modes sit near chance (0.56-0.59 against a chance median of about 0.45 and a chance max of about 0.64). The earlier 0.66-0.67 came from where the day boundary was cut; moving it 4 hours drops them to 0.56-0.59. A 0.65 bar would have passed noise.
- No block length gives two stable modes, so there are still no organs for Φ (#3296). Re-run around 12-08, when bus history covers about 3 weeks.

## Current architecture

The replay (#3796) tested mode stability only day to day, against a fixed 0.7 bar with no chance floor.

## Architecture touched

A pure function, a replay flag, and evidence. No services, contracts, schema, bus or config.

## Files changed

- `orion/substrate/whole_body_structure.py`: `STABLE_SIMILARITY`, `BlockStability`, `block_stability`.
- `orion/substrate/tests/test_whole_body_structure.py`: 3 tests (fixed modes pass; modes that change every block fail; fewer than 2 blocks returns None).
- `scripts/whole_body_structure_replay.py`: `--stability-only` and `stability_by_timescale`; the bar now comes from the module.
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
54 passed
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

REVIEW_PLACEHOLDER

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

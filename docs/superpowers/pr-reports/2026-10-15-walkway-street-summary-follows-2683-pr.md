## Summary

- Follow-up to #2682 (merged) for sibling PR #2683 (patio fingerprinting on, stationary fixtures).
- The chat street summary no longer counts stationary fixtures (the patio skeleton) as people "around".
- It finds the patio head-count row by the new `head_count_zone` flag, falling back to the older `no_embed_zone`.
- Comments that said patio people can never be named or be individuals are corrected.

## Outcome moved

With #2683 merged, a prop that has not moved for 30+ minutes would otherwise read as "an unfamiliar person" in every chat turn's street line. Now it is left out. Works before and after #2683's migration: if the `stationary` column is missing, the read falls back to the old statement (live today: column absent, fallback ran, summary returned normally: `['On the walkway just now: ...', 'Someone is on the patio.']`).

## Current architecture

`orion/situational/perception_reader.py::fetch_street_summary` read sightings without a stationary filter and found the patio count row only by `no_embed_zone = true`; its privacy comment said the patio is never fingerprinted and yields only a count.

## Architecture touched

Only `orion/situational/perception_reader.py` (read side of the street summary). No schema, bus, or env changes.

## Files changed

- `orion/situational/perception_reader.py`: `AND NOT s.stationary` with a pre-migration fallback statement; patio row matched by `head_count_zone OR no_embed_zone` (SQL and Python); privacy/stationary comment rewritten.
- `orion/schemas/situation.py`: `street_summary` field comment only.
- `orion/situational/tests/test_street_material.py`, `test_street_summary.py`: stationary + fallback test, `head_count_zone` test; missing-tables test now expects 3 rollbacks.
- `docs/superpowers/pr-reports/2026-10-15-walkway-street-into-mind-pr.md`: #2683 follow-up section (this change).

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: street summary excludes stationary sightings once the column exists; reads either patio flag.
- Compatibility notes: safe before and after `manual_migration_walkway_privacy_v2.sql`.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
orion/situational/tests                                   179 passed
services/orion-hub (curiosity/situation/street files)     458 passed, 3 skipped
services/orion-cortex-exec tests/test_situation_perception_context.py  56 passed
```

## Evals run

```text
No eval harness for the street summary; live read-only smoke against the real database (fallback path) passed.
```

## Docker/build/smoke checks

```text
No docker build. Live read-only smoke: fetch_street_summary('walkway') -> read_ok=True, 2 lines, via the pre-stationary fallback.
```

## Review findings fixed

- Covered by the #2682 review (same file, same patterns); this change adds one guarded statement and one OR'd flag with tests for both paths. No separate review run.

## Restart required

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

(Both build the situation brief. From the primary checkout on main after merge.)

## Risks / concerns

- Severity: low. Concern: any failure of the stationary statement (not only a missing column) triggers one fallback read. Mitigation: bounded to one extra read with the same 1.5 s statement timeout; both logged.

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

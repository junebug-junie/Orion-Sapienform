## Summary

- `orion-sql-writer-tests` fails on every branch since 2026-10-24 (2 tests in `test_vision_osnet_shadow.py`).
- Cause: the test writes fake camera crops at a fixed date (2026-10-16). The eval's `--live` mode only reads the last 8 days of the real clock, so the crops aged out of that window.
- Fix: write the crops relative to the real clock.

## Outcome moved

CI on every PR that touches orion-sql-writer goes green again (it was failing #2908 and at least 4 other branches).

## Files changed

- `services/orion-sql-writer/tests/test_vision_osnet_shadow.py`: `_street` stamps crops at real now minus 2 h.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
Throwaway postgres:16-alpine, migrations applied exactly as the CI workflow does:
this branch: 16 passed
main:        2 failed, 14 passed (the same 2 as CI)
```

## Evals run

N/A (test-only fix).

## Docker/build/smoke checks

N/A.

## Review findings fixed

Not run: one-line, test-only change proven against real Postgres on both sides.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: info. The test file's docstring migration list is stale (it omits migrations that add `frame_width`). Use the CI workflow's list.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

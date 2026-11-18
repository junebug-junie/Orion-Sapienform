## Summary

- The redaction test now uses an obviously fake sample credential in its sample database address.
- The three assertions that check the password is stripped now check for the fake value.

## Outcome moved

The test fixture can no longer be mistaken for a real credential. Test behavior is unchanged.

## Current architecture

`orion/harness/tests/test_trace_secret_redaction.py` builds a sample Postgres DSN and checks that the password part is removed from trace steps and final replies.

## Architecture touched

Test file only. No runtime code.

## Files changed

- `orion/harness/tests/test_trace_secret_redaction.py`: sample DSN password is now `fake-test-password-not-real`; assertions updated.
- `docs/superpowers/pr-reports/2026-11-17-fake-sample-dsn-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
python -m pytest orion/harness/tests/test_trace_secret_redaction.py -q
13 passed in 0.30s
```

## Evals run

```text
None. Test-fixture-only change; no behavior to evaluate.
```

## Docker/build/smoke checks

```text
Not applicable. No runtime change.
```

## Review findings fixed

- None. One-line fixture change; diff reviewed by hand.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: none
- Concern: none
- Mitigation: n/a
- Issues opened/closed: none

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3630

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## Summary

This is a follow-up to #3677 (merged), which made Orion's memory-correction step classify, vote, record validity times, and gate on an eval. It does two small things:

- The one-time pass now retries a transient gateway fault up to 3 times (a dropped connection, or a 5xx other than the existing 503 wait) before it counts the vote as an error. A dry run lost 61 votes this way (55 dropped connections, 6 5xx), and each failed vote held its row. A timeout is still not retried.
- The #3677 PR report gains its live-store dry-run section: 735 pairs, 0 errors, 2 proposals, and all six watched rows still active.

## Outcome moved

A gateway blip no longer turns votes into errors that hold whole rows. The live proof for #3659 and #3660 is now recorded in the repo, ids-free.

## Current architecture

`gateway_ask` in `scripts/memory_supersede_pass.py` already waited out 503 (seat busy). Any other 5xx, and any dropped connection, raised at once.

## Architecture touched

`scripts/memory_supersede_pass.py` only, plus a doc.

## Files changed

- `scripts/memory_supersede_pass.py`: retry a dropped connection or a 5xx up to 3 times (5 s, 10 s, 15 s); never a timeout.
- `orion/memory/episode/tests/test_supersede_pass_script.py`: retry a 5xx, retry a dropped connection, the 4-call cap, no retry on a 4xx or a timeout.
- `docs/superpowers/pr-reports/2026-11-18-memory-temporal-corrections-pr.md`: the dry-run results and the #3677 link.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. Behavior changed: the pass only. Compatibility notes: none.

## Env/config changes

- None. `.env_example` not changed; no local `.env` sync needed.

## Tests run

```text
PYTHONPATH=. pytest -q orion/memory/episode/tests/test_supersede_pass_script.py   8 passed
```

## Evals run

```text
None new. The live dry run that this report records is the evidence (read-only, private files in /tmp/memory-supersede-pass-v3/).
```

## Docker/build/smoke checks

```text
Not applicable: a host-side script and docs only.
```

## Review findings fixed

- Finding: the retry covered only 5xx, but 55 of the 61 lost calls were dropped connections (ReadError).
  - Fix: retry `httpx.TransportError` with the same cap and backoff. Re-raise timeouts.
  - Evidence: `test_gateway_ask_retries_a_dropped_connection`.
- Finding: the report implied the retry covered the whole outage.
  - Fix: the report now gives the split (55 dropped, 6 5xx) and says what is retried.
- Finding: the tests did not pin the retry cap, a 4xx, or a timeout.
  - Fix: `test_gateway_ask_caps_retries_and_never_retries_a_4xx_or_timeout`.
- Finding: the report said "mode 600" for a directory whose two manifest files are 644 (ids only).
  - Fix: it now says "directory mode 700". The pet detail was trimmed to "the pet correction".

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low.
- Concern: a read timeout is still not retried. It is recorded as an error vote, which holds that row.
- Mitigation: re-running the dry run replays cached answers and asks only what failed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3694

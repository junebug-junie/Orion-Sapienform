## Summary

Follow-up to #3866 (#3498), which was merged before its code review finished. This PR fixes what that review
found. Two readers still had a path where the model answered but the read stayed open. That path made the
coverage report call it "never closed". It also offered the same experiences to the reader again.

- Play (retro lines): a line that was claimed but whose delivery then failed now closes the read (no cites).
  Before, the branch chain had no `else`.
- Play: a turn that returned text in a race with Juniper's yield now closes the read (no cites). A yield
  before any answer still leaves it open.
- Dream: a run whose persist gives up after its retries now closes the read (no cites), best effort, as
  reverie already does on a persist failure.
- Docs and comments only: the dream docstring says retries reuse the run's one read, and only the next run reads
  again. The Prime docstring states the one default read after a restart. Coverage `NOW_READS_SQL` says "not
  closed" includes turns still running at report time and the reads stored before `closed_at` existed.
- CI: the durable-runs workflow now runs when the two experience migrations or `orion/integration/coverage.py`
  change. The Postgres lifecycle test checks `NOW_READS_SQL`.

## Outcome moved

A model call that ran is never counted as "never closed", and it never leaves its experiences offered again.

## Current architecture

As of #3866: a read moves the organ's cursor only when `POST /temporal-self/now/use` closes it. Readers close
once their model ran. Prime keeps its own `since=` cursor and also posts the close, even with no cites. It
never depends on the server cursor. For the Prime decision and the re-offer bound, see
`docs/superpowers/pr-reports/2026-11-24-now-cursor-moves-on-close-pr.md`.

## Architecture touched

orion-hub (`scripts/retro_commentary.py`), orion-durable-runs (`app/dream_day_graph.py`), orion-mind (docstring
only), `orion/integration/coverage.py` (comment only), `.github/workflows/orion-durable-runs-tests.yml`.

## Files changed

- `services/orion-hub/scripts/retro_commentary.py`: close on claimed-then-not-sent and on a yield race with text.
- `services/orion-durable-runs/app/dream_day_graph.py`: close on final persist failure; docstring.
- `services/orion-mind/app/prime_loop.py`: docstring.
- `orion/integration/coverage.py`: comment.
- `.github/workflows/orion-durable-runs-tests.yml`: path filters (pull_request and push).
- Tests: `services/orion-hub/tests/test_retro_commentary_service.py` (3 new),
  `services/orion-durable-runs/tests/test_dream_day.py` (1 new).

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: the read closes on 3 more paths where the model ran (listed above).
- Compatibility notes: none.

## Env/config changes

- None. `.env_example` not touched; no sync needed.

## Tests run

```text
services/orion-hub: pytest tests -k "curiosity_now or retro"     450 passed, 8 skipped
services/orion-hub: test_retro_commentary_service.py             42 passed
services/orion-durable-runs: pytest tests                         see CI (full suite also run locally)
services/orion-durable-runs: test_dream_day.py test_experience_now.py  53 passed
services/orion-mind: test_prime_now.py                            18 passed
orion/dig/tests/test_organs.py tests/test_report_integration_coverage.py  28 passed
```

## Evals run

```text
None new. The real-Postgres lifecycle test from #3866 covers the cursor SQL in CI.
```

## Docker/build/smoke checks

```text
No image change. Not deployed (by instruction). Live path UNVERIFIED for the new close paths.
```

## Review findings fixed

These findings come from the review of #3866. This PR fixes them.

- Finding: retro `sent`/`yielded`/`not run.claimed` chain had no `else`, so a claimed-then-failed delivery never closed.
  - Fix: final `else` closes the read empty (and still marks failure when unclaimed).
  - Evidence: `test_a_claimed_line_whose_delivery_then_fails_still_closes_the_now_read`.
- Finding: dream persist exhausting retries left the read open (reverie closes on persist failure).
  - Fix: best-effort empty close before returning `failed`.
  - Evidence: `test_a_run_whose_persist_gives_up_still_closes_the_now_read`.
- Finding: retro `run.yielded` check ran before the text check, so a race-finished answer never closed.
  - Fix: `answered` computed first; yielded with text closes empty, yielded without text stays open.
  - Evidence: `test_a_turn_that_answered_in_a_race_with_the_yield_closes_the_now_read`,
    `test_a_yield_before_any_answer_leaves_the_now_read_open`.
- Finding: dream docstring claimed a retry takes a fresh read; Prime docstring said its close never moves a
  server cursor; coverage comment omitted in-flight and pre-migration rows; CI path filter missed the
  migrations and `coverage.py`.
  - Fix: wording corrected; path filters added.
  - Evidence: diff.
- Not changed: the merged migration file's comments (editing an applied migration for a comment is not worth
  the drift risk; the pre-migration note lives in `coverage.py`).

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

orion-mind: docstring only, no restart needed.

## Risks / concerns

- Severity: low. Concern: the dream close on persist failure is best effort. If it also fails, the read
  stays open. That is the old behavior. Mitigation: a log line `dream_day_now_close_failed`.
- Issues opened: none. Issues closed: none (#3498 closed by #3866).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3876

🤖 Generated with [Claude Code](https://claude.com/claude-code)

# PR: surface the artifacts Orion was already producing

Branch: `feat/rateable-artifact-surface`
Follows: #1806 (artifact ratings), which shipped a rating path with no subject.

## Summary

- #1806 added a way to rate an artifact Orion produced. Nothing produced an
  artifact — `build_artifact_ref` appeared in the schema, a test and the CLI
  and nowhere else. It was a contract with a producer, a consumer, and no
  subject.
- **The artifacts already existed.** `substrate_dispatch_results` has been
  storing the real prose from `substrate.summarize` / `inspect` / `observe`
  the whole time. 630 successful results in 3 hours, ~161 characters each, in
  readable English. No human has ever read one.
- So this builds no producer. It surfaces what was already there, and fixes
  two things found while doing it.

Sample, verbatim from live:

> *"The transport capability is currently operating with a moderate priority
> and minimal risk, indicating it is functioning within expected parameters
> without urgent demands."*

## Outcome moved

`rate_artifact.py --list` shows unrated artifacts with their text; `--show`
reads one in full. `score_artifact_ratings.py` can now resolve **two thirds
more** of them, because the fast path (`substrate_action_outcomes`) only
covers actions that declared a signal — and the ones that didn't are exactly
the ones a human rating is the *only* available grade for, having no pressure
claim to be scored against.

## Files changed

- `scripts/rate_artifact.py`: `--list` reads the real corpus and excludes
  already-rated items; `--show <dispatch-id>` prints one in full with Orion's
  own confidence.
- `scripts/score_artifact_ratings.py`: frame-based resolver fallback; rater
  quarantine; unattested raters visible in the readout.
- `orion/autonomy/rating.py`: `rated_by` / `rating_source` on `ScoredRating`.
- `services/orion-sql-db/manual_migration_artifact_rating_attestation.sql`:
  new, **applied**.
- `tests/test_artifact_rating.py`: 36 tests (2 new classes).

## Two defects found and fixed

**Attestation died at the scoring boundary.** #1806 made an artifact rating
*require* a `user_id`, reasoning that an unattributed rating cannot be
distinguished from Orion rating itself — and then dropped it.
`substrate_action_ratings` recorded the verdict, the categories and the free
text, and not who gave it. Proven on the resolver's very first run: the first
row through it was a deploy smoke test tagged `deploy-smoke`, and nothing
downstream could have known. `rated_by` and `rating_source` now ride onto the
scored row; a NULL stays NULL rather than being backfilled to a plausible name.

**A smoke rating was one command from becoming a belief.** Once the resolver
worked, that `deploy-smoke` row became scoreable — and the only thing between
it and a permanent posterior was a paragraph in a commit message, which a cron
entry or this script's own "RUN IT PERIODICALLY" docstring both defeat.
`QUARANTINED_RATERS` now filters it in the query, with `--include-rater` as an
explicit opt-in.

## Review findings NOT fixed in this PR

Recorded rather than quietly carried. Reviewed adversarially; 14 findings.

- **Double-rating (MEDIUM).** `up` then `down` on the same artifact produces
  two different fingerprints, so both store and both fold into the belief —
  `posterior_n` reaches 2 and the variance shrinks as if two artifacts were
  graded. The scorer's docstring claims "an observation counted twice corrupts
  the belief permanently"; that guarantee does not currently hold. Needs a
  unique index on `substrate_action_ratings (dispatch_id)`.
- **`--list`'s `NOT EXISTS ... LIKE` is not sargable** and cannot be: `LIKE`
  against a non-constant pattern forces a nested loop. Measured at a simulated
  100k feedback rows: **431ms**. Fine today (3 rows) because the 2026-07-23
  disk loss emptied that table. Fix is an exact-suffix `regexp_replace`
  comparison.
- **The posterior is still unattested.** `rated_by` survives to the ledger and
  not to `substrate_action_rating_posterior`, which is the only thing a
  consumer reads. Same disease, one table down.
- **`unrated_count` has no writer** and prints `0` beside a corpus of ~142,000
  unrated artifacts — a false readout, not merely dead weight.
- **One bad row rolls back the whole scoring run** (single transaction, only
  `parse_artifact_ref` guarded).
- **Two comment numbers are wrong**: "32.3% declares a signal" is 15.9% over
  the population that sentence is actually about; "~200 an hour, every hour"
  is 0–1,291 with four zero hours in the last twelve.
- `--list` will surface status tokens as artifacts (998 rows whose entire text
  is `success`).

## Tests run

```text
PYTHONPATH=. .venv/bin/python -m pytest tests/test_artifact_rating.py -q
36 passed
```

## Docker/build/smoke checks

Not deployed. Both scripts are operator tools, not services; the migration is
applied. `--list` and `--show` verified against live Postgres.

## Restart required

```text
No restart required — scripts only. Migration already applied.
```

## Risks / concerns

- Severity: MEDIUM — double-rating (above) is reachable today by any operator
  who changes their mind about an artifact.
- Severity: LOW — the LIKE anti-join degrades with `chat_response_feedback`
  growth, which will happen as chat ratings resume.

## PR link

<filled in on push>

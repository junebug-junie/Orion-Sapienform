# Visual Reverie analytics

## Status

`DONE_WITH_CONCERNS`: database models, access boundaries, semantic compilation,
content schemas, tests, evals, and review pass. Live Lightdash deployment and
rendered dashboard verification still require the replacement CLI token to be
installed locally after the exposed token was revoked.

## Summary

- Added separate visual-chain and visual-artifact facts without unioning them
  with text Reverie or creating a cross-fact Lightdash join.
- Reused the UTC date and terminal-reason dimensions through many-to-one joins.
- Added content-free operational measures for persisted chains/images,
  artifacts per chain, missing artifacts, caption coverage, artifact bytes,
  continuity use, recency, late gaps, and terminal reasons.
- Added five visual charts and a unified `Reverie Overview` dashboard with
  visibly separate text and visual sections.
- Expanded the transformer role to exactly two additional operational sources;
  the Lightdash reader remains analytics-only and read-only.

## Metric-quality result

Counts conserve declared primary-key/foreign-key grains. Coverage and rate
metrics are labeled as derived operational diagnostics, not independent
cognitive signals. Late intervals reuse the live 45-minute watchdog contract.

The requested generation-to-artifact persistence delay was rejected: the
producer records neither generation-start time nor artifact-persistence time.
The dashboard reports observed late gaps rather than inventing missing-run
counts from a sequential run-then-sleep worker.

## Live evidence

```text
visual chains=1442
distinct visual chain IDs=1442
persisted artifacts=1433
distinct artifact IDs=1433
chains without artifact=9
captioned artifacts=1398
rows with continuity marker=1257
rows proving continuity used=940
late gaps over 45 minutes=9
all source/fact, duplicate, rollup, and join-fanout deltas=0
```

The latest visual chain was roughly 60 hours old during validation. The metric
reports that recency without asserting why the producer is stale.

## Tests and validation

- Analytics Python contract suite: 12 passed.
- dbt 1.9.0 parse: passed.
- dbt compile: 8 models, passed.
- dbt run: 8 views, passed.
- dbt test: 95 passed.
- dbt docs generation: passed.
- Lightdash 2.184.6 schema lint: passed.
- Lightdash semantic compile with warehouse-column validation: 5 Explores,
  0 errors.
- Reader probe: both visual facts readable; both operational sources denied;
  analytics schema creation denied.
- `git diff --check`: passed.

## Review findings fixed

- Finding: `prior_description IS NOT NULL` means continuity is available for a
  future run, not that the current run used it.
  - Fix: derive a nullable flag from the producer's per-run numeric
    `chain_json.continuity_streak`; legacy or malformed markers remain unknown
    and are excluded from the rate.
  - Evidence: live values changed from the misleading 1,407 nonblank outputs to
    940 proven uses among 1,257 marked rows; a source-to-fact dbt regression
    test covers positive, non-positive, absent, and non-numeric markers.
- Finding: docs said the source artifact hash was removed although it remains
  the fact primary key.
  - Fix: docs now state that the key is retained in the analytics view but
    hidden at the Lightdash presentation layer.
  - Evidence: static contract coverage and direct diff review.

Final adversarial re-review found no remaining actionable grain, privacy,
role-security, metric-provenance, test, or content-as-code issues.

## Runtime deployment

Operational rows and Orion application services were not changed or restarted.
The analytics views and grants are live. The first Lightdash deploy attempt was
rejected because the prior CLI token had correctly been revoked; no replacement
token was printed or requested in chat.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2197

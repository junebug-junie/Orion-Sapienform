# Self-monitor engine reuse

## Summary

Fixes the Hub full-suite failure on current `main`:

```text
test_no_new_per_call_engines_in_shared_orion_code
create_engine() called per call in [('self_monitor/report.py', 'collect')]
```

## Change

- `orion.self_monitor.report` now keeps one read-only SQLAlchemy engine per configured
  database URL instead of constructing and disposing a pool on every `collect()` call.
- First use is serialized, so concurrent callers cannot create duplicate losing pools.
- The cache has a hard four-URL capacity and refuses unexpected URL churn rather than
  evicting an engine another thread might still be using.
- Failed engine construction removes its placeholder and can be retried.
- `_dispose_cached_engines()` deterministically closes every cached pool in tests and is
  registered for process shutdown.

## Tests

- concurrent first use creates one engine;
- capacity refusal does not evict/dispose live engines;
- explicit cleanup disposes every cached engine;
- factory failure does not consume a slot and retries successfully;
- the read-only timeout/pool settings remain intact;
- Hub's static engine-reuse gate passes.

## Env/config/schema

No env, configuration, bus, schema, or deployment changes.

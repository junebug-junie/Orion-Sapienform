## Summary

- Crystallization duplicate detection never fired: every intake row has a unique `memory_window:<id>` scope, and `detect_duplicates` required scopes to overlap. Live: "Run github compactor." x8, "Compact the last 24 hours..." x6, "hi" x4 among non-rejected rows.
- `scopes_overlap()` now ignores `memory_window:` tags; real topic scopes still gate.
- New `find_exact_duplicates` SQL lookup so an old low-salience copy outside the top-200 list is found; an exact copy counts as a duplicate outright (short text like "ok" has zero token overlap).
- Reinforce-on-duplicate moved after the intimate and identity-scope gates, so a sensitive window can never write evidence into an active row unreviewed.

## Outcome moved

Identical chat turns reinforce the existing memory instead of inserting a new row. Existing rows are untouched (no cleanup).

## Files changed

- `orion/memory/crystallization/detection.py`: `scopes_overlap`, used by duplicate and contradiction detection.
- `orion/memory/crystallization/repository.py`: `find_exact_duplicates`.
- `orion/memory/crystallization/intake_pipeline.py`: exact-copy lookup feeds duplicate_id.
- `orion/memory/crystallization/formation_policy.py`: duplicate check ordered after intimate/identity gates.
- `tests/test_crystallization_window_scope_dedup.py`: regression tests.

## Schema / bus / API changes

None. Behavior change: hub `validate` (`crystallization_routes.py:213-225`) also uses `detect_duplicates`, so window-scoped proposals with a >=0.72 near-copy now fail validation. Intended; untested.

## Env/config changes

None.

## Tests run

```text
pytest tests/test_crystallization_window_scope_dedup.py + 6 related crystallization files + services/orion-memory-consolidation/tests: 330 passed, 8 skipped
```

## Evals run

None; the service has no dedup eval. Follow-up: replay eval over stored windows.

## Docker/build/smoke checks

New SQL run read-only against live `memory_crystallizations`: returns 8 rows for "Run github compactor.". Live reinforce path UNVERIFIED until deployed.

## Review findings fixed

- Finding: reinforce ran before intimate/identity gates (privacy leak, now live).
  - Fix: reordered in `formation_policy.py`; two tests.
- Finding: SQL `btrim` vs Python strip mismatch on newlines/tabs.
  - Fix: `btrim(regexp_replace(...))`.
- Finding: Jaccard misses short exact copies; copy order could pick a proposed row over an active one.
  - Fix: exact match is a duplicate outright, active preferred; test.
- Not fixed: SQL lookup itself has no DB-backed test; reinforcement trail is only merged evidence.

## Restart required

```bash
scripts/safe_docker_build.sh orion-memory-consolidation up -d --build
```

## Risks / concerns

- Severity: low. Near-duplicates with different wording still get through.
- Mitigation: follow-up, embedding or fuzzy match.

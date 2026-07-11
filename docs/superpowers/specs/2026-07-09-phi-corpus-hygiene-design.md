# φ corpus hygiene — projection cap + ingestion gate

**Mode:** Design. Deterministic "no garbage in" — no cognition change, no proposal mode required. Independent of the other two specs; can land in parallel.

## Arsonist summary

Two deterministic gates that stop garbage at the source: (1) cap/prune the unbounded `execution_trajectory` projection (25 MB / 29,165 runs since May 25), and (2) reject degenerate rows at φ-corpus **write** time instead of discovering them at fit time. Both are the CLAUDE.md mandate: turn a repeated failure into a failing gate, not a reminder.

## Part A — cap/prune the execution_trajectory projection

### Current
`services/orion-substrate-runtime/app/store.py` `save`/`load` for `EXECUTION_TRAJECTORY_PROJECTION_ID="active_execution_trajectory"` (`execution_loop/constants.py`). The reducer keeps a `runs` dict keyed by trace_id and **never evicts** — 29,165 runs, 25 MB, growing since May 25. The introspector fetches the whole blob every tick and filters to a 120s window, so 99.98% of the payload is dead weight.

### Proposed
- Bound `runs` to the **most-recent N by `last_updated_at`** (LRU) and/or a max age (e.g. keep ≤ 24h or ≤ 2,000 runs — tune to comfortably exceed the 120s consume window plus headroom). Evict on write in the reducer/store.
- Same O(N)-growth pattern as [[feedback_execution_merge_cap]] / [[feedback_substrate_performance]] — reuse that cap discipline.

### Files
- `services/orion-substrate-runtime/app/store.py` (or the reducer that materializes runs) — eviction on save.
- `services/orion-substrate-runtime/app/settings.py` + `.env_example` — `EXECUTION_TRAJECTORY_MAX_RUNS` / `EXECUTION_TRAJECTORY_MAX_AGE_SEC`.
- Tests: reducer evicts oldest past cap; endpoint payload bounded.

### Acceptance
- After N+k runs, projection holds ≤ N; payload size bounded; freshest runs always retained (introspector still sees the 120s window).
- One-time: existing 29k-run projection compacted (backfill/prune script or first-write eviction).

## Part B — ingestion-time corpus-health gate

### Current
`scripts/backfill_phi_corpus.py` and the live spark-introspector emit path append `InnerStateFeaturesV1` rows to the corpus JSONL with **no health gate**. Degenerate rows (cognitive source `execution_trajectory.none`, per-dim frozen, `phi_health != ok`, `grammar_truth_degraded`) enter freely and are only excluded later inside `fit_phi_encoder.py::_filter_training_rows`. Garbage lands on disk; we discover it at fit time.

### Proposed
- A shared, deterministic `is_corpus_row_healthy(inner) -> (bool, reasons)` used by both the backfill writer and the live emit:
  - reject `phi_health != "ok"` or `grammar_truth_degraded` (mirrors the fit filter — apply at write).
  - reject rows whose cognitive features all carry the `.none` source (no active execution signal).
  - optional: tag (not reject) rows contributing to a frozen column, for observability.
- Rejected rows are counted + logged (not silently dropped); a `--allow-degraded` escape hatch for debugging.
- Keep it a pure function so it is unit-testable and reused everywhere the corpus is written.

### Files
- `orion/schemas/telemetry/inner_state.py` or a new `orion/telemetry/corpus_gate.py` — the pure predicate.
- `scripts/backfill_phi_corpus.py` — gate on write, report counts.
- `services/orion-spark-introspector/app/worker.py` — gate the live corpus append.
- Tests: synthetic degenerate row rejected; healthy row accepted; counts reported.

### Acceptance
- A synthetic `.none`/frozen/`phi_health!=ok` row is rejected at write with a reason; a healthy row passes.
- Backfill reports `kept / rejected(reason)` totals.
- Live emit path drops degraded rows and logs the count (no silent loss).

## Non-goals

- Not changing feature semantics (that's seed-v4).
- Not deleting historical corpus rows (fit already filters them); this stops *new* garbage.

## Recommended next patch

Part B first (pure predicate + writer gate — smallest, highest leverage: stops new garbage immediately). Part A second (projection cap + one-time compaction).

# doc-semantic-drift: live producer

## Summary

- Built the live `doc_semantic_drift` producer in `orion-cocreation-signals`, implementing `docs/superpowers/specs/2026-07-30-doc-semantic-drift-design.md` now that diff-scoped embedding was calibrated and confirmed real (PRs #1560, #1563).
- Split pure logic (`orion/structural_mass/doc_semantic_drift.py`: `changed_doc_files`, `diff_hunks`, `doc_semantic_drift_changes`) from scheduling/publishing (`services/orion-cocreation-signals/app/producers/doc_semantic_drift.py`), following the codebase's established convention.
- New `DocSemanticDriftV1` wire schema, published on a new `orion:substrate:doc_semantic_drift` channel — shadow-write (`consumer_services: []`), default `COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_ENABLED=false`.
- Resolved a real architecture fork mid-implementation: orion-vector-host's embedding-request bus contract unconditionally persists every embedded text to the shared vector store. Per Juniper's explicit call (AskUserQuestion, 2026-08-11), this producer uses that real contract as-is, scoped to its own `doc_semantic_drift` vector collection rather than commingling with chat/social memory or building a bypass.
- 19 new producer-level tests + 7 pure-logic tests, all passing. Code review found and fixed one real leak (unclosed forked RPC bus client).

## Outcome moved

Codebase-mass instrumentation now has a fifth live producer (alongside git_delta, pr_lifecycle, graph_delta, affective_state) — real diff-scoped embedding-diff signal per real doc commit, publishing to the bus. Currently shadow-write only (no consumer), pending a live-stream sanity pass before flip-on.

## Current architecture

Before this patch: `doc_semantic_drift`'s only real code was the offline calibration script (`scripts/analysis/measure_doc_semantic_drift.py`) and its design doc. No live producer, no schema, no channel.

## Architecture touched

- `orion-cocreation-signals` service: new producer loop, settings, main.py wiring, docker-compose/.env_example.
- `orion/bus/channels.yaml` + `orion/schemas/registry.py`: new contract.
- `orion/structural_mass/` and `orion/schemas/`: new pure-logic module + schema.

## Files changed

- `orion/structural_mass/doc_semantic_drift.py`: pure git-diff logic — `conventional_commit_prefix`, `changed_doc_files` (git diff --name-only, `*.md` only), `diff_hunks` (git diff --unified=0, +/- lines only, explicit `before_rev`/`after_rev` params so a multi-commit poll window is covered, not just the last commit's own parent), `doc_semantic_drift_changes` (orchestrates the above into `DocHunkChange` records).
- `orion/structural_mass/tests/test_doc_semantic_drift.py`: 7 tests against real throwaway git repos, including a regression guard for the range bug caught before it ever shipped.
- `orion/schemas/doc_semantic_drift.py`: `DocSemanticDriftV1` — aggregate scalar only (`diff_scoped_embedding_diff: float | None`, `possibly_truncated: bool`, char-length fields), never persists hunk text.
- `orion/bus/channels.yaml`: new `orion:substrate:doc_semantic_drift` entry, `producer_services: [orion-cocreation-signals]`, `consumer_services: []`, `stability: experimental`.
- `orion/schemas/registry.py`: registered `DocSemanticDriftV1`.
- `services/orion-cocreation-signals/app/producers/doc_semantic_drift.py` (new): cold-start-sha polling loop (same pattern as `git_delta_loop`), requests two real embeddings per changed doc file (hunk_removed, hunk_added) via a forked RPC bus client against `orion:embedding:generate`, computes `diff_scoped_embedding_diff = 1 - cosine_similarity`, publishes `DocSemanticDriftV1`.
- `services/orion-cocreation-signals/app/settings.py`: new `COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_*` fields + `CHANNEL_DOC_SEMANTIC_DRIFT` / `CHANNEL_EMBEDDING_GENERATE`.
- `services/orion-cocreation-signals/app/main.py`: wired `doc_semantic_drift_loop` into `run_producers()`.
- `services/orion-cocreation-signals/docker-compose.yml`, `.env_example`: new env var passthrough.
- `services/orion-cocreation-signals/tests/conftest.py`: added `FakeBus.close()` so producers that fork a dedicated RPC client can exercise real teardown in tests.
- `services/orion-cocreation-signals/tests/test_doc_semantic_drift_producer.py` (new): 12 tests — cold start, real-change publish, no-op, zero-doc-changes-still-advances-sha, failed-publish non-advancement, disabled-bus, tick-exception-survives, RPC-client-close-on-exit, cosine similarity edge cases.

## Schema / bus / API changes

- Added: `orion:substrate:doc_semantic_drift` channel, `DocSemanticDriftV1` schema.
- Removed: none.
- Renamed: none.
- Behavior changed: none (new signal only).
- Compatibility notes: no new channel needed for the embedding request — `orion:embedding:generate`'s `producer_services` already includes `"*"`.

## Env/config changes

- Added keys: `COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_ENABLED` (default `false`), `COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_POLL_INTERVAL_SEC` (300.0), `CHANNEL_DOC_SEMANTIC_DRIFT`, `CHANNEL_EMBEDDING_GENERATE`, `COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_EMBED_COLLECTION` (`doc_semantic_drift`), `COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_EMBED_TIMEOUT_SEC` (30.0), `COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_TRUNCATION_CHAR_THRESHOLD` (2048).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced: yes — manually appended to both the primary checkout's and this worktree's `services/orion-cocreation-signals/.env` (the sync script only flags diverged *existing* keys, doesn't add brand-new ones, so this was done by hand per CLAUDE.md's env-parity mandate).
- skipped keys requiring operator action: none.

## Tests run

```text
.venv/bin/python -m pytest services/orion-cocreation-signals/tests/ orion/structural_mass/tests/test_doc_semantic_drift.py -q
49 passed, 14 warnings in 4.66s
```

## Evals run

No dedicated eval harness for this producer yet — the real calibration evidence is the offline replay in `docs/superpowers/pr-reports/2026-08-11-doc-semantic-drift-diff-scoped-embedding.md` (max trivial=0.3841 < min real=0.4150 on real historical commits). Follow-up: once live and flipped on, a periodic eval replaying real published events against that same calibration set would close this gap.

## Docker/build/smoke checks

Not deployed live this cycle — `COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_ENABLED` ships `false` by default (same "shadow write, flip on deliberately" convention as `affective_state`'s own rollout), so no live Docker smoke was required before merge. Import-checked directly:

```text
PYTHONPATH=. .venv/bin/python -c "from app.producers.doc_semantic_drift import doc_semantic_drift_loop; from app import main" -> ok
```

## Review findings fixed

- Finding: Forked RPC bus client (`fork_rpc_client`) was created once at loop start and never closed — a real Redis connection leak on every task cancellation/redeploy, contradicting the docstring's own claim to mirror `orion-chat-memory`'s pattern (which does close its equivalent handle).
  - Fix: wrapped the loop body in `try/finally`, closing `rpc_bus` in the `finally` with its own `except Exception` guard so a close failure never masks the real shutdown path.
  - Evidence: new regression test `test_forked_rpc_bus_client_is_closed_on_loop_exit`; `FakeBus.close()` added to the shared test fixture so the real teardown path is exercised, not just implied.

## Restart required

```text
No restart required — COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_ENABLED ships false; no live container is running this code path yet.
```

## Risks / concerns

- Severity: low
- Concern: `possibly_truncated` is a char-length heuristic, not a real token count — the live bus embedding-request contract doesn't expose one the way the offline calibration script's direct container access did.
- Mitigation: disclosed explicitly in both the schema docstring and the settings field comment; may under/over-flag relative to the real tokenizer. Revisit if this producer's real published data shows the heuristic diverging badly from the real embedding host's own truncation behavior.

- Severity: low
- Concern: only 3 real non-truncated samples in the offline calibration replay so far — not enough to set a real alert/consumer threshold yet.
- Mitigation: shipped default-off (shadow write only); flip on deliberately once the live stream has accumulated enough real samples for its own sanity pass, per the same convention already used for `affective_state`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/doc-semantic-drift-producer

# feat(recall): bounded retrieval — query intake, capped expansion, run-once feeds, deadline

## Summary

Recall was slow because it searched for the whole prompt. A 30,663-character self-inquiry prompt turned into 268 separate searches, and every search re-ran every backend one after another, so that recall took 72 seconds. This PR is Phases 1 and 2 of the approved design (`docs/superpowers/specs/2026-09-29-recall-retrieval-query-architecture-design.md`, PR #2405). Recall now defends itself against any caller:

- **It picks a short search text first.** It uses the caller's `retrieval_query` when one is sent. Otherwise, a fragment over 600 characters is condensed to at most 600, question sentences first. Anything shorter is searched as-is.
- **It caps the fan-out.** At most 4 extracted names become extra searches. Sentence-starters like "The", "It", "No" and "You" are dropped, and the most specific names (ids, file names, multi-word names) come first. The order is the same on every run.
- **"What's going on" feeds run once.** Recent chat, recent timeline and bus anomalies ignore the search text, so they now run once per recall instead of once per search. The actual retrievers still run per search.
- **Everything runs at once, under one deadline.** The deadline is 80% of the caller's `deadline_ms`, or 60s by default. When it fires, recall answers with what arrived and flags `deadline_hit=true`, instead of timing out and looking like "no memory".
- **Contract (Phase 2, consumer-first).** `RecallQueryV1` gains optional `retrieval_query`, `deadline_ms` and `mode` (`context_only` = feeds only). The decision and `recall_telemetry` gain `query_chars`, `retrieval_query_source`, `sub_query_count`, `candidates_fetched`, `candidates_kept`, `deadline_hit` and `timings_ms`.
- **Smaller fixes from the design's trace:**
  - Falkor chat now filters by time inside the query instead of fetching and discarding.
  - Two regexes that could never match are fixed: the anchor rail and the "show recent memories" shortcut.
  - `latency_ms` is now measured end to end.
  - Fusion tokenizes the query once instead of once per candidate.

## Outcome moved

On the real stored queries (stubbed backends, so this is work done, not live latency):

- The 30,663-char prompt goes from 268 searches and a projected 1,876 backend calls to 3 searches and 9 calls.
- Every long real query now does at most 13 backend calls.
- For short questions, the top 8 results match the uncapped pipeline 97.7% of the time on average. Every difference comes from no longer searching for the bare word "What".

Live effect (design acceptance check 6: stance_react max latency under 10s, no RecallService RPC timeouts) is **UNVERIFIED** until deploy.

## Current architecture

`process_recall` took the fragment, ran chat_general-only trimming, then `_expand_query` produced fragment + verb + intent + every capitalized word, with no cap. For each signal, one at a time, it awaited `_query_backends`. That ran every backend, including the query-independent ones, and `related_by_entities` re-extracted all entities from the full query each time. There was no deadline. `latency_ms` stopped before the entity boost and fusion.

## Architecture touched

- `orion-recall` only, plus the shared `RecallQueryV1`/`RecallDecisionV1` contract.
- `_query_backends` is now an ordered list of feed/retriever units, with `include_feeds`/`include_retrievers` switches and a partial-results sink. It kept its signature and its default single-signal behavior (same gates, same order), so existing callers and test stubs still work.
- No new channels. Nothing in the repo re-validates `RecallDecisionV1` off `orion:recall:telemetry`: sql-writer is declared as a consumer but does not subscribe (`KNOWN_UNSUBSCRIBED`).

## Files changed

- `orion/core/contracts/recall.py`: new optional query and decision fields.
- `tests/test_recall_bounded_retrieval_contract.py`: round-trip, old-payload, bad-value and `resolve()` tests.
- `services/orion-recall/app/worker.py`:
  - intake and condensation (`_intake_query`, `_condense_query`);
  - bounded expansion (`_ranked_entities`, `_expand_query`, `_max_sub_queries`);
  - unit-based `_query_backends`;
  - concurrent fetch with a deadline in `process_recall`;
  - regex fixes, deterministic `_extract_entities`, the boost using the ranked list, end-to-end timings;
  - telemetry DDL and insert.
- `services/orion-recall/app/storage/falkor_chat_adapter.py`: `WHERE t.ts >= $cutoff`, plus `allow_empty_query` for context-only recall.
- `services/orion-recall/app/fusion.py`: memoized query tokenization.
- `services/orion-recall/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: three new knobs.
- `services/orion-recall/sql/recall_telemetry.sql`: new nullable columns.
- `services/orion-recall/tests/test_recall_bounded_retrieval.py`: 27 tests covering the acceptance checks.
- `services/orion-recall/tests/test_recall_telemetry_persist.py`: column DDL and insert tests.
- `services/orion-recall/tests/test_docker_compose_numeric_defaults.py`: new keys hardened.
- `services/orion-recall/tests/test_process_recall_active_turn_exclusion.py`, `test_recall_policy_harness.py`: the `_query_backends` stubs now accept `**kwargs`. They already failed on main because they didn't accept `lane`.
- `services/orion-recall/evals/`:
  - `run_recall_bounded_retrieval_eval.py`, `test_recall_bounded_retrieval_eval.py`, `conftest.py`;
  - `fixtures/recall_telemetry_queries_2026-09-29.json`: 80 distinct live queries, scanned for secrets, IPs and emails (none).

## Schema / bus / API changes

- Added: `RecallQueryV1.retrieval_query` (≤1000 chars), `.deadline_ms` (>0) and `.mode` (`retrieve`|`context_only`, default `retrieve`). `RecallDecisionV1.query_chars`, `.retrieval_query_source`, `.sub_query_count`, `.candidates_fetched`, `.candidates_kept`, `.deadline_hit` and `.timings_ms`.
- Removed / renamed: none.
- Behavior changed: `RecallDecisionV1.latency_ms` is now end to end (it used to exclude the boost and fusion), so expect it to read higher for the same work.
- Compatibility: every field is optional. Consumer first: deploy orion-recall before any caller sends the new fields, because `RecallQueryV1` is `extra="forbid"`. No caller sends them yet (that is Phase 3).
- `recall_telemetry` gets 7 nullable columns via `ADD COLUMN IF NOT EXISTS` on the first write after boot.

## Env/config changes

- Added keys: `RECALL_MAX_QUERY_CHARS=600`, `RECALL_MAX_SUB_QUERIES=4` (0 = old uncapped fan-out, the rollback lever), `RECALL_DEADLINE_MS_DEFAULT=60000`.
- `.env_example` updated: yes. `docker-compose.yml` gets `:-default` fallbacks.
- Local `.env` synced: yes, with `python scripts/sync_local_env_from_example.py --all-keys orion-recall`. It wrote the three keys to the primary checkout's live `services/orion-recall/.env`. The default (no `--all-keys`) mode skips these keys because they're outside its prefix list.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-recall: pytest tests evals -q -p no:cacheprovider
  main baseline:  3 failed, 284 passed
  this branch:    2 failed, 320 passed
  Still failing, both pre-existing:
    test_recall_policy_harness::test_process_recall_diagnostic_contains_gating_suppression_and_selection
      (its old TypeError is fixed; it now reaches a stale assert that vector is "enabled",
       but vector was removed from recall earlier)
    test_recall_vector_amputation::test_worker_and_recall_v2_import_without_vector_adapter
      (the subprocess prints pydantic warnings to stderr; unchanged)
  Now passing: test_process_recall_active_turn_exclusion
Repo: tests/test_recall_bounded_retrieval_contract.py + recall contract/registry tests: 20 passed
  orion/cognition/tests/test_reflect_recall_integration.py::test_process_recall_includes_sql fails on main too
Mutation check: making feeds run per sub-query again fails 4 of the new tests.
Static gates (local equivalents of orion-static-gates.yml): all PASS, including
  check_definition_drift --gate (0 changed, no re-lock) and check_metric_lineage --gate.
  check_schema_registry.py / check_bus_channels.py don't exist in the repo.
make check-env-compose-parity SERVICE=orion-recall: same 3 pre-existing gaps as main, new keys present.
```

## Evals run

```text
python services/orion-recall/evals/run_recall_bounded_retrieval_eval.py
rows=80 (distinct query/verb/profile from recall_telemetry 2026-09-29)
max sub-queries (K=4): 6  [bound K+2=6]
max chars searched: 600  [bound 600]
stopword sub-queries: none
context feeds called at most once per recall: True
short queries (<=500 chars): n=64 top-8 overlap bounded-vs-uncapped mean=0.977 min=0.750
long queries (>600 chars): n=16 max wall=33ms (stub I/O 5ms/call) backend calls new=13 vs old sequential projected=1876
  stance_react 30663 chars -> condensed 600 chars, 3 sub-queries (uncapped 268), calls 9 (old projected 1876)
  stance_react  7577 chars -> condensed 600 chars, 6 sub-queries (uncapped 75),  calls 13 (old projected 525)
pytest services/orion-recall/evals: 5 passed
```

What the eval can measure: that intake and expansion stay bounded on real inputs, that feeds run once, that the work no longer grows with prompt length, and how much the cap moves the top 8 when retrieval is lexical. What it cannot measure: real backend latency, real recall quality (the stub retriever is not Falkor or Postgres), or anything about the live deploy.

## Docker/build/smoke checks

```text
Not run: no deploy or restart, per instructions.
Read-only live check: the pushed-down Cypher (WHERE t.ts >= $cutoff) ran against orion-athena-falkordb
and returned the 6 newest chat.history turns inside a 2-day window. All 1,916 stored ChatTurn.ts values
are 32-char ISO strings ending +00:00, the same shape as the cutoff, so the string comparison is like-for-like.
```

## Review findings fixed

- Code review is being run by the orchestrator (not in this session).
- Self-found before review:
  - Finding: once the anchor regex was fixed, it treated UUID segments and trace ids ("be03", "cb4dd9417c8d4020") as anchors on 12 of 80 real queries.
    - Fix: strip UUIDs before matching, and drop pure-hex tokens of 8+ characters.
    - Evidence: 0 of 80 real queries now trigger it. There's a new test.
  - Finding: once fixed, the memory-browse shortcut would have sent any long prompt that mentions "recall" and "context" down the recent-only path.
    - Fix: it only applies to text up to 160 characters.
    - Evidence: a test on the 30k prompt. 0 of 80 real queries trigger it.
  - Finding: pure clause-score condensation turned the 30k self-inquiry prompt into a random refuted prior, not its question.
    - Fix: question sentences rank first.
    - Evidence: the condensed text now starts with the standing question (pinned by a test).

## Restart required

```bash
# orion-recall only (contract is additive; deploy recall before any Phase 3 caller change)
scripts/safe_docker_build.sh orion-recall up -d --build
```

## Risks / concerns

- **Severity: medium.** Two dead paths come back to life:
  - the anchor rail, which runs exact-match SQL for tokens like `gpu1`;
  - the memory-browse shortcut, for short "show recent memories"-style requests.
  - These change results for queries that contain them. The v2 shadow compare also fires on anchor queries now, bounded by the remaining deadline.
  - Mitigation: the id-shaped false positives are guarded, and browse is limited to short text. Watch `recall_telemetry` `backend_counts.sql_timeline_anchor` after deploy.
- **Severity: medium.** Condensation is a backstop, not the fix. For a long prompt it searches the question plus the densest sentences, which can still miss what the turn is about. Phase 3 (callers send `retrieval_query`) is the real fix.
- **Severity: low.**
  - The deadline cancels waiting on thread-offloaded calls (the RDF adapters, graph compression), but the thread itself runs to completion in the default executor.
  - SQL chat windowing (`fetch_chat_turn_timestamps`) runs after the fetch deadline and is not covered by it.
- **Severity: low.** `latency_ms` now includes the boost, fusion and shadow compare. Dashboards comparing before and after will see a step up that is measurement, not regression.
- **Severity: low.** Entity specificity ranking uses shape and length, not Falkor entity degree: that would be an extra round trip before retrieval. The boost still applies its own degree discount.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## Summary

Chat-turn memory recall sometimes ran out of time while the reranker was scoring, threw away all of that work, and fell back to exact-word search. And when a memory search could not run properly, nothing told Orion that this is not the same as having no memory.

- The reranker now gets the candidates in small chunks (8 at a time), best-retrieved first. When the deadline or an error stops it, the chunks already scored are kept and served. Only a rerank that scored nothing falls back to exact words.
- The first chunk holds the message's top meaning hits and top exact-word hits, taking turns. A name-only hit is no longer last in line.
- A part-scored search that found nothing above the bar is reported as degraded (reason `rerank_partial`) and serves the unscored exact-word hits. It is never logged as a silent "below the bar".
- `orion.dig know` lists what a part-scored search left unscored as weak exact-word matches.
- The dig brief now carries one sentence: if `know` could not search properly (skipped, degraded, timed out, error), say you could not look, never that you have no memory. The `know` output for too-few-words now says it is NOT the same as having no memory.
- The replay eval gained a budget mode (`--budget-sec`, `--noise-limit`) that reports the degraded rate, and `--rewrite-cache` so a before/after pair sees the same rewritten searches.

Fixes #3662
Fixes #3663

## Outcome moved

Why it happened (live `recall_telemetry`, chat-turn calls, 2026-11-17):

- 5 of the 7 degraded calls overlapped orion-whisper-tts speech synthesis on the same Tesla P4 GPU as the reranker. All 5 were long messages. Across the 72 h window, 7 calls overlapped synthesis and 5 of them degraded. When the GPU was free, 2 of 58 degraded.
- Rerank time while synthesis ran: p50 4.2 s. While the GPU was free: p50 1.8 s.
- The other 2 degraded calls lost 1.8-2.6 s to the query rewrite before the rerank started.
- Not the cause: concurrent recalls (none overlapped), a larger candidate pool alone (synthetic 32-candidate calls took 0.8-2.1 s on a free GPU).
- fp16 was not tried: the P4 (Pascal GP104) runs fp16 at a fraction of its fp32 speed.

Replay of the last 100 real chat messages, live settings (5.5 s budget, live rewrite on the fast lane, links k=2). 85 of them were long enough to search:

| Condition | Before: degraded | After: degraded | After: part-scored | Total p50 / p90 before | after |
|---|---|---|---|---|---|
| GPU free | 1/85 (1.2%) | 0/85 (0%) | 0 | 2.64 / 4.33 s | 2.38 / 3.54 s |
| GPU fully loaded the whole time (synthetic, worse than live) | 45/85 (52.9%) | 1/85 (1.2%) | 20 | 5.13 / 5.26 s | 3.75 / 5.26 s |

- The one degraded call after the fix was an HTTP error on the first chunk (likely the vector host's "busy" refusal), not a timeout.
- Rerank p90 on a free GPU dropped from 2.9 s to 2.0 s: each chunk is padded only to its own longest pair.

Hit rate on the lived-memory recall bank (22 referring messages, same rewritten searches for both runs via `--rewrite-cache`): before 17/22 (explicit 9/9, implicit 8/13), after 17/22 (explicit 9/9, implicit 8/13). Small talk injecting: 9.1% both. Without the shared cache, two runs of the unchanged code scored 15/22 and 17/22: that spread is the rewrite model's run-to-run variance, not this change.

## Current architecture

`services/orion-recall/app/lived_memory.py` `LivedMemoryRecall.recall` ran rewrite, then meaning and exact-word search, then one `/rerank` call for the whole pool (up to 32 candidates) on orion-vector-host (bge-reranker-v2-m3 on the athena Tesla P4, shared with orion-whisper-tts). If that one call missed the deadline, every score was lost and the turn used the exact-word fallback (#3573).

## Architecture touched

- orion-recall lived-memory retrieval (rerank loop, `know()`, trace).
- `orion/dig` brief line and `know` rendering.
- No bus, schema, API, env or Docker change.

## Files changed

- `services/orion-recall/app/lived_memory.py`: chunked rerank that keeps scored chunks; interleaved chunk order; part-scored-and-empty degrades with reason `rerank_partial`; `know()` shows unscored exact-word hits; `rerank_partial` in the persisted trace.
- `services/orion-recall/tests/test_lived_memory_step2.py`: 9 tests (chunking, deadline keeps chunks, error after a chunk, no chunk still degrades, part-scored-and-empty degrades, empty note, `know` weak hits, exact-word-only hit in chunk 1, HTTP chunk sizes and timeouts).
- `services/orion-recall/evals/run_lived_memory_replay_eval.py`: `--budget-sec`, `--noise-limit`, `--rewrite-cache`; reports `degraded_rate`, `degraded_reasons`, `rerank_partial`, `rerank_errors`.
- `services/orion-recall/README.md`: chunked rerank and the measured cause.
- `orion/dig/brief.py`: one sentence; `MAX_BRIEF_CHARS` 640 -> 740 (line is 739).
- `orion/dig/know.py`: the too-few-words line says nothing was searched and it is NOT the same as having no memory.
- `orion/dig/tests/test_know_cmd.py`: the brief sentence and length; every non-search status never says "No stored memory".

## Schema / bus / API changes

- Added: `lived_trace.rerank_partial` (`{"scored", "of", "reason"}`) in `recall_telemetry` (JSONB, no migration). New degraded reason `rerank_partial`.
- Removed: none.
- Renamed: none.
- Behavior changed: a deadline mid-rerank keeps the scored chunks (status `ok` when it served). A part-scored search that served nothing above the bar is `degraded`, not `below_bar`.
- Compatibility notes: `/lived/know` response shape unchanged.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
services/orion-recall: pytest tests evals -q -> 479 passed, 2 failed
  (test_recall_policy_harness::test_process_recall_diagnostic_contains_gating_suppression_and_selection and
   test_recall_vector_amputation::test_worker_and_recall_v2_import_without_vector_adapter fail the same way on main)
services/orion-recall lived tests + evals: 48 passed (step2 + evals)
orion/dig/tests: 309 passed
services/orion-hub/tests/test_orion_abilities_catalog.py: 10 passed
orion/physiology/evals/physiology_state_skill_eval.py: passed (brief within MAX_BRIEF_CHARS)
scripts/check_metric_lineage.py --gate: PASS; --prompt-semantics: PASS
scripts/check_definition_drift.py --gate: PASS
scripts/check_integration_seam.py: OK
scripts/check_async_routes_not_blocking.py: OK
```

## Evals run

```text
run_lived_memory_replay_eval.py --variants step2 --budget-sec 5.5 --noise-limit 100 --rewrite-timeout 2.5 --link-top-k 2
  GPU free:   before degraded 1/85 (1.2%), total p90 4327 ms | after 0/85 (0%), total p90 3542 ms
  GPU loaded: before degraded 45/85 (52.9%)                  | after 1/85 (1.2%), 20 part-scored
run_lived_memory_replay_eval.py --variants step2 --min-fit 0.004 --rewrite-timeout 2.5 --link-top-k 2 --rewrite-cache <host file>
  before 17/22 (explicit 9/9, implicit 8/13) | after 17/22 (explicit 9/9, implicit 8/13); small talk injecting 9.1% both
```

GPU load for the loaded run: a throwaway container from the vector-host image ran fp32 matmuls on the P4 for the length of each replay, then stopped. No live container was changed.

## Docker/build/smoke checks

```text
Not deployed (per task). No Dockerfile, compose or dependency change.
The replays ran this branch's code on the host against the live vector host, Chroma, Postgres and the bus.
```

## Review findings fixed

- Finding: a part-scored `know` lookup could end in "No stored memory" while most candidates were never scored.
  - Fix: a part-scored search that served nothing is degraded (`rerank_partial`); `know()` lists unscored exact-word hits as weak.
  - Evidence: `test_partial_rerank_with_nothing_above_the_bar_degrades_and_says_so`, `test_know_shows_what_a_partial_rerank_left_unscored_as_weak`.
- Finding: chunk order was meaning-score first, so an exact-word-only hit sat in the last chunk.
  - Fix: meaning and exact-word lists take turns for the chunk order (link seeds unchanged).
  - Evidence: `test_exact_word_only_hit_is_in_the_first_chunk`.
- Finding: a part-scored chat turn with nothing above the bar was logged `below_bar`.
  - Fix: same as the first finding; status is `degraded`.
  - Evidence: `status_of` asserts in the tests above.
- Finding: budget mode reused cached rewrites, so later variants timed a 0 ms rewrite.
  - Fix: budget mode uses a fresh rewrite cache per variant.
  - Evidence: eval code; the idle replay rewrite p90 stays 2.5 s.
- Finding: an idle-GPU budget replay is only a lower bound.
  - Fix: help text says so; the loaded replay above; verify-live issue #3668.
  - Evidence: `--budget-sec` help.
- Finding: no test of the HTTP path with chunking.
  - Fix: `httpx.MockTransport` test asserts chunk sizes and a live per-chunk timeout.
  - Evidence: `test_http_rerank_goes_in_chunks_each_with_a_live_timeout`.
- Nits fixed: unused counter removed, per-chunk HTTP timeout keeps the finish reserve, budget mode skips the referring fetch, walrus removed, README updated, brief headroom noted.

## Restart required

```bash
scripts/rebuild_services_from_git_diff.sh
```

orion-recall needs a rebuild to pick up the change (the post-pull rebuild does this from the clean primary main checkout after merge). The dig brief and `know` change reach Orion when the FCC sandbox syncs after that deploy.

## Risks / concerns

- Severity: low. Concern: under GPU load about 1 in 4 searches is part-scored, so a candidate deep in the pool can go unscored. Mitigation: the first chunks hold the best meaning and exact-word hits; `rerank_partial` is in the trace; #3668 tracks the live share.
- Severity: low. Concern: the reranker still shares the P4 with whisper-tts, which slows it. Mitigation: this patch stops the loss, not the slowdown. If #3668 shows a high part-scored share, moving one of them is the next step.
- Severity: low. Concern: the dig brief is 739 of 740 characters. Mitigation: noted in `brief.py`.
- Issues opened: #3668 (verify-live: degraded under 2% over 7 days, p90 not above 3.5 s). Commented on #3536 with the scoreboard part of #3663 (count a "no memory" answer after a non-search as a false denial, by cause), since the scoreboard is not built yet.
- Issues closed: #3662, #3663.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3676

🤖 Generated with [Claude Code](https://claude.com/claude-code)

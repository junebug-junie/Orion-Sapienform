# doc-semantic-drift: chunk below the model's real token ceiling, score by max chunk-pair drift

## Summary

- The embedding model silently clips at 512 tokens. Every real `doc_semantic_drift` event this repo ever published carried `possibly_truncated=True` — a 20KB PR-report diff and its first 2KB scored identically, and the tail was never measured.
- Chunk both hunk sides below that ceiling, embed every window, and score by the **most-changed chunk pair** rather than a pooled centroid.
- Replace `possibly_truncated` (a constant, therefore not a signal) with `chunk_count_removed`/`chunk_count_added`.
- Add `change_kind` from git's own `--name-status`, so a `None` score is interpretable instead of ambiguous.
- The first commit in this branch did **not** actually fix the truncation it was written to fix. Code review caught it; the second commit does. Both are kept in history rather than squashed, because the failure mode is the instructive part.

## Outcome moved

`diff_scoped_embedding_diff` now measures the whole diff instead of its first ~2KB, and does so on a scale that does not collapse with document length. Measured on a real 5/16-chunk doc modification from this repo's history: **0.2401** under the new scoring vs **0.0464** under the mean-pooled version rejected here — 5.2x, on real content.

## Current architecture

`doc_semantic_drift_loop` (`services/orion-cocreation-signals/app/producers/doc_semantic_drift.py`) polls this repo's real `*.md` changes, extracts diff hunks, requests two embeddings over `orion:embedding:generate` (orion-vector-host's live contract), and publishes `1 - cos(removed, added)` on `orion:substrate:doc_semantic_drift`.

Before this patch, each hunk side was sent as a single embedding request with a best-effort `possibly_truncated` character-length flag attached. The channel has `consumer_services: []` — still shadow-write, no consumer.

## Architecture touched

`orion-cocreation-signals`'s `doc_semantic_drift` producer, its schema, and the shared hunk-extraction module. No new channels, no new services, no consumer wiring.

## Files changed

- `services/orion-cocreation-signals/app/producers/doc_semantic_drift.py`: added `_chunk_text()` (line-boundary windows, hard-splits an oversized single line), `_max_pair_drift()` (symmetric max chunk-pair). `_score_change()` embeds every chunk and nulls a whole side if any chunk fails. `_request_embedding()` now rejects the embedding host's error-shaped replies. Removed `_mean_pool()`.
- `orion/structural_mass/doc_semantic_drift.py`: added `changed_doc_files_with_status()` reading git's `--name-status`; `DocHunkChange` gained `change_kind`. Removed `changed_doc_files()` (no production caller).
- `orion/schemas/doc_semantic_drift.py`: `+change_kind`, `+chunk_count_removed`, `+chunk_count_added`, `-possibly_truncated`.
- `services/orion-cocreation-signals/app/settings.py`, `.env_example`, `docker-compose.yml`: `..._TRUNCATION_CHAR_THRESHOLD` → `..._CHUNK_CHAR_SIZE`, default 2048 → 1200, `gt=0`.
- `services/orion-cocreation-signals/tests/test_doc_semantic_drift_chunking.py` (new, 24 tests), `test_doc_semantic_drift_producer.py`, `orion/structural_mass/tests/test_doc_semantic_drift.py`.

## The measurements this patch rests on

All taken against the **live** `orion-athena-vector-host` container, not estimated.

**The model's real ceiling** — from its own files in the running container:

```text
tokenizer_config.json  model_max_length:        512
config.json            max_position_embeddings: 512
services/orion-vector-host/app/embedder.py:56 passes truncation=True
```

**Why 2048 chars was wrong** — 400 real chunks from this repo's `*.md` history, tokenized with the container's own tokenizer:

```text
median 541 tokens | p90 608 | max 722
EXCEEDING 512: 266/400 = 66.5%
chars/token on near-full chunks: median 3.59, min 2.81   (the code assumed 4.0)
```

**Why 1200 is right** — same procedure, same corpus, at the new size:

```text
median 312 tokens | p99 426 | max 461
EXCEEDING 512: 0/500 = 0.00%
```

**Why mean-pooling was rejected** — real model, one section rewritten to unrelated real content, document length varied:

```text
real single-section drift:  0.2392
N=1 chunk  -> 0.2392
N=2 chunks -> 0.0655
N=4 chunks -> 0.0168
N=8 chunks -> 0.0044     (54x dilution for the same real edit)
```

Decay is ~1/N², worse than the naive 1/N: near cosine 1, `1-cos ≈ |Δ|²/2` while `|Δ|` itself scales as 1/N. The calibration lineage (`2026-08-11-doc-semantic-drift-diff-scoped-embedding.md`) proposed a **0.3996** threshold from single-window samples, so under pooling a complete section rewrite in a long doc lands two orders of magnitude below the cutoff — every long doc reads "trivial". That is the inversion of what the signal claims, and fails the CLAUDE.md metric quality gate at step 3 (theory anchor) and step 4 (live-data sanity).

Max-pair reduces **exactly** to `1 - cos(a, b)` when each side is one chunk, so existing scores and that threshold keep their meaning.

## Schema / bus / API changes

- Added: `change_kind` (`added`/`modified`/`deleted`/null), `chunk_count_removed`, `chunk_count_added`.
- Removed: `possibly_truncated`.
- Behavior changed: `diff_scoped_embedding_diff` is now max chunk-pair drift, not a single pairwise cosine over (possibly clipped) whole hunks. Identical for single-chunk hunks.
- Compatibility: `orion:substrate:doc_semantic_drift` has `consumer_services: []` and no SQL model — nothing reads this payload today, so removing a field from an `extra="forbid"` model breaks no live consumer. Verified.

## Env/config changes

- Renamed: `COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_TRUNCATION_CHAR_THRESHOLD` → `COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_CHUNK_CHAR_SIZE` (2048 → 1200, `gt=0`).
- `.env_example` updated: yes.
- local `.env` synced: yes — **by hand, in both the primary checkout and this worktree.**
- Skipped keys requiring operator action: none.

**Real gap worth recording:** `scripts/sync_local_env_from_example.py` reads `.env_example` from the *primary checkout*, not from the worktree it is invoked in. A key renamed in a worktree is therefore invisible to it and it silently no-ops. It reported nothing and changed nothing; the rename only landed because it was done manually. Any worktree-based env change has this hole.

## Tests run

```text
.venv/bin/python -m pytest services/orion-cocreation-signals/tests/ orion/structural_mass/tests/ -q
140 passed, 15 warnings in 7.63s
```

Pre-existing, unrelated, not fixed here: `orion/schemas/tests/test_context_provenance.py::test_static_ctx_assignments_covered` fails on clean `main` too.

## Evals run

No eval harness exists for this producer. The closest thing is `scripts/analysis/measure_doc_semantic_drift.py`, which is now **stale**: it has its own self-contained hunk extraction with no chunking, so re-running it computes a number the producer no longer produces. See Risks.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-cocreation-signals build     -> Image built
scripts/safe_docker_build.sh orion-cocreation-signals up -d     -> Recreated, Started
docker exec ... printenv COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_CHUNK_CHAR_SIZE
  -> 1200
docker inspect ... -> Created=2026-08-12T23:03:28Z RestartCount=0
```

Live log, confirming the earlier durable-state fix survived this redeploy:

```text
23:03:32 cocreation_doc_semantic_drift_resumed_from_durable_state last_sha=f30a94e7c...
```

No errors, no tracebacks. No `..._published` line yet — `main`'s HEAD has not moved since the resume point, so there is genuinely nothing new to score. **The new scoring path has not yet published a real live event**; the 0.2401 figure above came from driving the live embedding host directly over a real historical diff, which is strong evidence but not the same as a published event.

## Review findings fixed

- Finding (F3, blocking): 2048 chars still exceeded the 512-token ceiling on 66.5% of real chunks, while the same commit deleted `possibly_truncated`, the only disclosure — the patch made truncation *less* visible without fixing it.
  - Fix: `CHUNK_CHAR_SIZE` 1200, `Field(gt=0)`.
  - Evidence: re-tokenized 500 real chunks at 1200 → 0 exceed 512, max 461.
- Finding (F1, blocking): new fabricated-score path. orion-vector-host reports failure as a well-formed `EmbeddingResultV1` with `embedding=[]` plus an extra `error` key (`app/main.py:540-571`); `EmbeddingResultV1` is `extra="ignore"`, so `error` was dropped and the result read as success. `_mean_pool` then discarded the empty vector and pooled the survivors. A **regression** — the old code passed `embedding=[]` to `_cosine_similarity` and got an honest `None`.
  - Fix: reject a truthy `error` key and reject empty embeddings, both with a log line.
  - Evidence: `test_embedding_host_error_reply_is_not_treated_as_a_success` constructs the exact real reply shape; `test_one_failed_chunk_nulls_the_side_rather_than_scoring_the_survivors` covers the multi-chunk consequence.
- Finding (F2): mean-pooling makes long-doc and short-doc scores non-comparable.
  - Fix: symmetric max chunk-pair drift (Juniper's call between the options).
  - Evidence: dilution table above; real-doc comparison 0.2401 vs 0.0464.
- Finding (F4): the `change_kind` docs claimed `None` + `modified` is an alarm. False on **34.8%** of real modified-doc changes (56/161 over 300 commits) — a pure-append edit is status `M` with an empty removed side under `--unified=0`.
  - Fix: docs corrected to point at the chunk counts, which do disambiguate. The test asserting this was a tautology (`x == "modified" or x is not None`, unfailable) and is now real.
  - Evidence: `test_pure_append_to_existing_doc_is_undefined_not_an_alarm`.

Two bugs in my own test fixtures, found by the tests failing rather than by reading them:

- A "pooled score reflects all chunks" test used `[0,1]` and `[0,-1]` as the differing tails — mirror-symmetric about the removed-side vector, so both scored identically and the assertion could never distinguish them.
- A "length-invariant" test used a surrounding chunk 45° from the changed pair, which let the changed chunk find a partial match. Fixing it surfaced a **real property of max-pair** that is now documented and tested rather than papered over (below).


### Second review round (verification pass on the fixes above)

The re-review confirmed F1/F3/F4 independently — including reproducing the 1200-char measurement (600 chunks, 0 exceeding 512) and enumerating every reply shape `orion-vector-host` can emit to confirm the F1 guards are exhaustive. It then found four more, two of them real behavior bugs:

- Finding: `graphify-out/**` was in scope for doc scanning. It is machine-regenerated on a post-commit hook, so scoring it measures a tool's output churn rather than co-creation. `GRAPH_REPORT.md` alone yields ~182 removed / ~175 added 1200-char chunks; at the embedding host's real ~0.6s serial throughput that is ~214s against a 30s per-request timeout — the file could **never** be scored, while monopolizing the shared embedding host that live chat memory depends on (3748 real requests in 24h). The 2048→1200 change made this 1.75x worse.
  - Fix: `_EXCLUDED_DOC_PATHSPECS = (":(exclude)graphify-out/**",)`.
  - Evidence: `test_generated_graphify_output_is_not_scored_as_doc_drift`.
- Finding: embedding fan-out was one unbounded `gather` over every chunk, against a host that consumes serially (`Hunter` built without `concurrent_handlers`, `hf` backend on CPU).
  - Fix: `_EMBED_CONCURRENCY = 4`, shared across both hunk sides.
  - Evidence: `test_embedding_requests_are_capped_in_flight` asserts observed peak ≤ cap and > 1 (so it can't pass by nothing overlapping).
- Finding: `_max_pair_drift` ran synchronously on the event loop shared with the heartbeat chassis and five other producers; measured 0.13s at N=M=20, 0.55s at N=M=40.
  - Fix: `asyncio.to_thread`.
- Finding: **the "length-invariant" claim was false.** A max over `2*(N+M)` non-negative terms is an extreme-value statistic, so more chunks biases the score high — measured over 30 real hunk pairs: median 0.1684 at N=1, 0.2237 at N=2-3, 0.2861 at N=4+. The test asserting invariance used padding vectors byte-identical on both sides, so every padding term was exactly `1 - 1.0 = 0` and could not detect the bias by construction — and matched no real input shape, since `--unified=0` hunks contain no unchanged chunks at all.
  - Fix: claim removed from `_max_pair_drift` and the schema; replaced with the measured bias and its direction.
  - Evidence: `test_extra_chunks_bias_the_score_high_not_low`, plus the pre-existing counter-case test.
- Finding: the schema's comparability note said `1 = ... reduces exactly to 1 - cos(a, b)`, but that requires **both** counts to be 1 — a 1-removed/2-added event is already a max-pair score. A consumer filtering on one count would build a non-comparable set.
  - Fix: corrected to require both.
- Finding: `_score_change`'s own docstring still described the deleted mean-pool.
  - Fix: corrected.

**Three fixture bugs across two rounds, all mine, all found by tests failing rather than by reading them.** The invariance one is the worst: it asserted a property that does not hold, and its fixture was constructed so it could never fail. My first replacement for it then claimed extra chunks can only *raise* the score — directly disproven by the test immediately below it. The net effect is content-dependent; both directions are now pinned by tests and the docstring says so.

## Risks / concerns

- Severity: medium
- Concern: `CHUNK_CHAR_SIZE` is a *measured* chars-per-token proxy, not a live token count. A chunk denser than the observed 2.81 minimum could still exceed 512 tokens and be clipped inside the model with nothing in the payload disclosing it — the same class of failure as before, just with the odds moved from 66.5% to near zero on the corpus measured.
- Mitigation: the honest fix is `input_token_count` on `EmbeddingResultV1` so the producer knows real truncation instead of estimating it. Deliberately deferred — it touches a second shared service. Recorded in the schema docstring so it is not silently forgotten.

---

- Severity: medium
- Concern: max chunk-pair drift is **not length-invariant**, and multi-chunk events are biased high relative to the single-window 0.3996 threshold (median 0.1684 at N=1 vs 0.2861 at N=4+ over 30 real hunk pairs). A consumer applying that cutoff across chunk counts will over-flag long docs. The bias is a tendency rather than a guarantee — an extra chunk adds a term to the max but also adds a match candidate for the other side, so the net on any single event is content-dependent.
- Mitigation: documented in `_max_pair_drift` and in the schema, with both directions pinned by tests. Far smaller than pooling's ~54x collapse and in the safer direction (over-reports rather than hides change). **A multi-chunk threshold must be derived separately — the 0.3996 figure applies only to events where both chunk counts are 1.**

---

- Severity: low
- Concern: `scripts/analysis/measure_doc_semantic_drift.py` is now stale — its own hunk extraction has no chunking, so it computes a metric the producer no longer computes. Anyone re-deriving a threshold from it would calibrate against the wrong thing.
- Mitigation: none applied. Follow-up.

---

- Severity: low
- Concern: chunking roughly doubles embedding requests per doc change (2.08x measured over 300 commits). Each request also unconditionally persists a vector-store document **and** calls `feed_tissue()`, injecting doc-diff chunks into live OrionTissue state — the producer's docstring disclosed the former but not the latter, and chunking multiplies both by N.
- Mitigation: fan-out is now capped at 4 in flight and `graphify-out/**` (by far the largest contributor) is excluded, so the pathological case is gone. The tissue-feed side effect is disclosed here but not otherwise addressed — worth a look before this producer's output is consumed.

---

- Severity: low
- Concern: the vector-store `doc_semantic_drift` collection now holds mid-document fragments rather than one document per hunk side, so semantic search over it returns windows, not whole hunks.

## Restart required

Already deployed as part of this patch:

```text
scripts/safe_docker_build.sh orion-cocreation-signals up -d --build
```

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/doc-semantic-drift-chunking

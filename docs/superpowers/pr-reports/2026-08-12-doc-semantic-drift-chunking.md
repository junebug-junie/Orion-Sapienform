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
138 passed, 15 warnings in 7.09s
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

## Risks / concerns

- Severity: medium
- Concern: `CHUNK_CHAR_SIZE` is a *measured* chars-per-token proxy, not a live token count. A chunk denser than the observed 2.81 minimum could still exceed 512 tokens and be clipped inside the model with nothing in the payload disclosing it — the same class of failure as before, just with the odds moved from 66.5% to near zero on the corpus measured.
- Mitigation: the honest fix is `input_token_count` on `EmbeddingResultV1` so the producer knows real truncation instead of estimating it. Deliberately deferred — it touches a second shared service. Recorded in the schema docstring so it is not silently forgotten.

---

- Severity: medium
- Concern: max chunk-pair drift is not strictly length-invariant. Symmetry contains most of the problem (an unmatched chunk contributes its own high term, so it cannot quietly drag the score down), but a second changed section resembling the first section's changed text can still give that chunk a better match than its true counterpart and lower the reported drift.
- Mitigation: documented in `_max_pair_drift`'s docstring and pinned by two tests asserting both directions of the behavior. Far weaker than pooling's ~1/N² collapse, which fired on every long doc regardless of content.

---

- Severity: low
- Concern: `scripts/analysis/measure_doc_semantic_drift.py` is now stale — its own hunk extraction has no chunking, so it computes a metric the producer no longer computes. Anyone re-deriving a threshold from it would calibrate against the wrong thing.
- Mitigation: none applied. Follow-up.

---

- Severity: low
- Concern: chunking roughly doubles embedding requests per doc change (measured 2.08x over 300 commits, max 20 chunks for one file), and up to 20 concurrent RPCs now fan out from one `_score_change` against a host that embeds sequentially with a 30s per-request timeout. A large doc's last chunk queues behind the rest, so larger docs are now *more* likely to time out — and one timeout nulls that whole side.
- Mitigation: none applied; not load-tested, so whether 20 chunks actually trips 30s is unverified. Worth a bounded concurrency cap if it shows up live.

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

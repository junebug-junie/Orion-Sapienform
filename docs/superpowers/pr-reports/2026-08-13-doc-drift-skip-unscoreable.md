# doc-semantic-drift: stop publishing unscoreable events

## Summary

- The first real live batch of the new scoring showed **6 of 8 events carrying `diff=None`** — all new PR reports and specs, where one hunk side is empty and cosine simply does not exist.
- That ratio is structural, not a small sample: this repo's docs are mostly created once and never revised, so waiting for data would only produce more nulls.
- Skip those changes entirely, **before embedding** rather than after.
- `change_kind` is removed from the payload as a consequence — it became a constant.
- Net effect: `diff=None` in a published event now means exactly one thing, a real embedding failure.

## Outcome moved

The channel stops carrying mostly-null traffic, and stops paying for it. A 15-chunk new doc was costing 15 real embedding requests — each one a vector-store write **and** a tissue feed on a shared, serial, CPU-bound host — to publish a guaranteed `None`. On the observed batch that was ~72 wasted embeddings out of ~80.

## Current architecture

`doc_semantic_drift_loop` polls this repo's `*.md` changes every 300s, extracts diff hunks, embeds each side in ≤1200-char chunks via `orion:embedding:generate`, and publishes max chunk-pair drift on `orion:substrate:doc_semantic_drift` (`consumer_services: []`, still shadow-write).

Before this patch it published an event for every changed doc, including ones where a hunk side was empty. Those carried `diff=None` plus `change_kind` and `chunk_count_*` fields added specifically so a consumer could tell "structurally undefined" from "measurement failed."

## Architecture touched

`orion-cocreation-signals`'s `doc_semantic_drift` producer and its schema. No channel, service, or consumer changes.

## Files changed

- `services/orion-cocreation-signals/app/producers/doc_semantic_drift.py`: added `_is_unscoreable()`; the loop `continue`s on it before `_score_change`. Dropped `change_kind` from the published event and the log line.
- `orion/schemas/doc_semantic_drift.py`: removed `change_kind`.
- `orion/structural_mass/doc_semantic_drift.py`: docstrings — see review finding 3.
- tests: `test_doc_semantic_drift_chunking.py` (5 new, 3 removed), `test_doc_semantic_drift_producer.py` (fixture).

## The live evidence this rests on

First real batch through the new scoring path, `sha=c92f4f679`, 2026-08-13:

```text
kind=modified  diff=0.3558  chunks=1/5     <- real score
kind=modified  diff=0.2956  chunks=1/2     <- real score
kind=added     diff=None    chunks=0/14
kind=added     diff=None    chunks=0/15
kind=added     diff=None    chunks=0/15
kind=added     diff=None    chunks=0/10
kind=added     diff=None    chunks=0/15
kind=added     diff=None    chunks=0/3
```

75% unscoreable. Confirmed against history rather than assumed — over 600 real commits / 587 real `*.md` changes:

```text
one side empty (now skipped):        357  (60.8%)
both sides non-empty (published):    230  (39.2%)
```

## Why keyed on hunk text, not on `change_kind == "added"`

A *modified* file can be equally unscoreable: a pure append has no removed lines under `--unified=0`, and that is **34.8%** of real modified-doc changes (56/161 over 300 commits). The status letter does not reveal that; the text does.

## Why `change_kind` was removed

After the skip, every published event is an `M` with real text on both sides — the field became a constant, which is the same reason `possibly_truncated` was removed two commits earlier. A constant is not a signal.

Verified by construction rather than by reasoning from a comment (the comment's premise was wrong — it claimed renames present as adds because no `-M` is passed, but `diff.renames` has defaulted to true since git 2.9). Real repos, real git:

```text
pure rename:    R100  a.md -> b.md   removed_len=0  added_len=1249  -> skipped
rename + edit:  R097  b.md -> c.md   removed_len=0  added_len=1256  -> skipped
```

The conclusion survives for a different reason than the comment gave: `diff_hunks()` re-runs `git diff` with a pathspec limited to the *new* path, so rename detection has no delete to pair with. `A` can never produce a `-` line, `D` is excluded by `--diff-filter=ACMR`, `C` needs `--find-copies` and is never emitted.

`change_kind` stays on the internal `DocHunkChange`, where the skip log line uses it.

## Schema / bus / API changes

- Removed: `change_kind`.
- Behavior changed: unscoreable changes publish no event at all.
- Compatibility: `consumer_services: []` and no SQL model — nothing reads this payload. Note the schema is `extra="forbid"`, so any event published between 2026-08-12 and this patch that still carries `change_kind` will fail validation on replay. Zero impact today; recorded so a future replay isn't surprised.

## Env/config changes

None.

## Tests run

```text
.venv/bin/python -m pytest services/orion-cocreation-signals/tests/ orion/structural_mass/tests/ -q
143 passed, 15 warnings in 8.22s
```

Pre-existing, unrelated: `orion/schemas/tests/test_context_provenance.py::test_static_ctx_assignments_covered` fails on clean `main`.

## Evals run

No eval harness for this producer. `scripts/analysis/measure_doc_semantic_drift.py` remains stale (its own hunk extraction, no chunking) — unchanged by this patch, still a follow-up.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-cocreation-signals build   -> Image built
scripts/safe_docker_build.sh orion-cocreation-signals up -d   -> Started
docker inspect -> Created=2026-08-13T05:21:51Z RestartCount=0
```

```text
05:21:57 cocreation_doc_semantic_drift_resumed_from_durable_state last_sha=c92f4f679...
```

No errors. No `..._published` or `..._skipped_unscoreable` line yet — `main` has not moved since the resume point, so there is nothing new to score. **The skip has not yet fired on live traffic**; it is proven by tests and by the historical measurement above, not by a live event.

## Review findings fixed

- Finding: `_is_unscoreable` used an emptiness test, so a side of only blank lines or spaces passed the guard and then failed at embed time — orion-vector-host rejects whitespace-only text with `error="missing_text"` (`app/main.py:540`), surfacing as a published `diff=None`: a permanent alarm on a structurally unscoreable change, and the one hole in this patch's own "a published None always means a real failure" claim.
  - Fix: `.strip()` on both sides.
  - Evidence: `test_whitespace_only_side_is_unscoreable`. Measured at **zero occurrences** over 600 real commits, so this is insurance rather than an observed defect — reported as such rather than as a live bug.
- Finding: the loop test could pass vacuously. If the cold-start branch ever ran (a change to `_load_last_sha` or `FakeRedis`), `last_sha` would be seeded and saved with zero changes ever examined, and all three assertions would still hold.
  - Fix: count real `doc_semantic_drift_changes` invocations and assert on it.
  - Evidence: reviewer demonstrated both branches passing the original assertions.
- Finding: `orion/structural_mass/doc_semantic_drift.py:70,163` still pointed at the deleted `DocSemanticDriftV1.change_kind`, and `changed_doc_files_with_status`'s rationale actively argued *against* this patch — "rather than inferring the kind from whether a hunk side came back empty" — which is exactly what `_is_unscoreable` now does.
  - Fix: rewritten to say why the hunk text is the stronger signal.
- Verified clean by the reviewer, by construction rather than by reading: the skip / `all_published` / `last_sha` interaction (a skip-only tick advances the baseline; a mixed tick with a publish failure holds it and re-scores only the scoreable file on retry), and no path by which a real scoreable change is silently lost.

## Restart required

Already deployed as part of this patch.

## Risks / concerns

- Severity: medium
- Concern: this narrows the signal to ~39% of doc changes. New docs — the majority of this repo's doc output — now produce no event at all, so `doc_semantic_drift` is silent about the single most common thing that happens to documentation here.
- Mitigation: deliberate. Drift between a before and an after is not defined for a file with no before, and publishing a null said nothing a consumer could use. If "how unlike our existing corpus is this new doc?" turns out to be the more valuable question, that is a **novelty** metric with its own field and its own calibration — not a reuse of `diff_scoped_embedding_diff`.

---

- Severity: medium
- Concern: still no valid threshold. Both real scores so far (0.3558, 0.2956) are multi-chunk events, which are biased high relative to the 0.3996 single-window figure, and neither is threshold-comparable (that needs both chunk counts == 1). Zero comparable events have been observed.
- Mitigation: none possible yet — needs accumulated samples. **Nothing should consume this signal until a multi-chunk threshold is derived from real data.**

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/doc-drift-skip-unscoreable

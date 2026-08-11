# Doc semantic drift: embedding-diff pre-filter calibration finds a real, disqualifying defect

Status: **DONE_WITH_CONCERNS**. Implements
`docs/superpowers/specs/2026-07-30-doc-semantic-drift-design.md`'s own "Recommended next patch"
-- the embedding-diff pre-filter alone, read-only, against real historical doc commits,
calibrating a real threshold before touching graphify Part B escalation or the
`orion/concepts/` registry shape at all. The real result: **the pre-filter as the design doc's
Phase-1 framing implicitly specifies it (whole-document embedding) does not work** on this
repo's real docs, for a specific, confirmed, structural reason -- not "no separation found," a
genuine architectural defect. This is the correct, honest output of a calibration exercise, not
a stall.

## Existing-mechanism check (design doc's own "Missing questions")

The doc asked: "what non-frontier embedding model is acceptable to run inline -- routed through
`orion-llm-gateway`'s existing model routing, or a new lightweight local dependency?" Real
answer, confirmed live 2026-08-11: `orion-vector-host` is already running in production
(`docker ps` confirms `orion-athena-vector-host` up), HF backend, model
`BAAI/bge-large-en-v1.5` -- a real, already-vetted, non-frontier embedding model this repo
already depends on. This calibration script reuses that exact model by calling directly into
the live container's already-loaded `Embedder` (`docker exec`), never adding a new embedding
dependency to this repo or duplicating a multi-GB model download.

## The real finding

Ran against 5 real historical doc commits from this repo's own git history (1-line PR-link edit
up to a 36-line real README addition -- a real size spread, not synthetic text):

| sha | path | shortstat | embedding_diff | expected |
|---|---|---|---|---|
| 9e34ee88 | ...ewma-successor-pr.md | 1 insertion, 1 deletion | -0.0000 | trivial |
| dc375dca | orion-attention-runtime/README.md | 17 insertions | 0.0000 | real |
| 2144f852 | orion-equilibrium-service/README.md | 24 ins, 4 del | 0.0000 | real |
| f88dc088 | orion-substrate-runtime/README.md | 3 ins, 1 del | 0.0000 | trivial |
| 121185a3 | orion-substrate-runtime/README.md | 36 insertions | 0.0000 | real |

**Every sample, trivial or substantial, scored `embedding_diff ~= 0.0000`.** Root-caused, not
assumed: `BAAI/bge-large-en-v1.5`'s real tokenizer has a **512-token hard limit** (confirmed live
via the actual tokenizer object inside the running container -- `model_max_length`, not a
guessed constant), and `Embedder._embed_hf` calls it with `truncation=True` and no explicit
`max_length`, so real text past 512 tokens never reaches the model at all. Real docs in this
repo are commonly 5KB-74KB -- `services/orion-substrate-runtime/README.md` is 71,795 real
characters, and the real edit in commit `121185a3` is a paragraph appended near the end of the
file, far past the truncation point. A truncated "before" and truncated "after" that share the
same (also truncated) leading content produce a near-zero embedding diff regardless of how real
the actual edit was.

**This is not a negative result about embedding-diff as a signal in the abstract -- it's a
structural defect in whole-document embedding as a technique**, for docs at the size most real
READMEs in this repo actually are. The design doc's Phase-1 framing ("embedding-diff between
before/after text for any docs/README files touched in a commit/PR") implicitly assumed
whole-document embedding would work; it doesn't, for this model, on this corpus.

## Review-caught improvement: real per-sample truncation measurement, not a char-count guess

First draft flagged truncation risk via a chars-per-token approximation (`len(text) > 512 * 4`).
Code review correctly flagged this as a real risk on this exact corpus: this repo's docs are
heavy in code blocks, file paths, and identifiers, which tokenize denser than English prose, so
a char-count proxy could false-negative on a doc that reads as "short enough" but is actually
truncated. Fixed by having the script's own `docker exec` call ask the container's real
tokenizer for a real token count per text (`truncation=False`, compared against
`tokenizer.model_max_length`) and returning that alongside each embedding -- a live-measured
fact, not an approximation. Re-ran with the fix: same result (5/5 truncated), now on solid
ground rather than a heuristic.

## What's NOT built (per the design doc's own explicit non-goals)

- No `services/orion-cocreation-signals/app/producers/doc_semantic_drift.py` -- the doc's own
  Phase-1 gate ("before touching graphify Part B integration... at all") isn't cleared, since
  the pre-filter it depends on doesn't work yet.
- No `orion/concepts/` registry changes.
- No graphify Part B escalation wiring.
- No self-narrative-coherence comparison against `structural_mass`.

## Recommended real fix (not implemented this patch -- a genuine follow-up, not scope creep)

Two credible directions, neither picked unilaterally:

- **Diff-scoped embedding**: embed only the changed hunks/sections (from the real git diff),
  not the whole document. Fits inside 512 tokens for the overwhelming majority of real doc
  edits, and is arguably a more precise signal anyway (measures what actually changed, not the
  whole document's drift from itself).
- **A longer-context embedding model**: `orion-vector-host` would need a second embedding
  profile/model for this use case, since swapping its primary model would affect every existing
  consumer (chat memory, RAG, topic foundry) that already depends on `BAAI/bge-large-en-v1.5`'s
  current dimensionality/behavior.

Diff-scoped embedding is the cheaper, more targeted fix and the natural next patch -- but is a
real architecture decision, not something to default into silently.

## Files changed

- `scripts/analysis/measure_doc_semantic_drift.py` (new): the calibration script.
- `scripts/analysis/tests/test_measure_doc_semantic_drift.py` (new): 7 tests for the pure logic
  (commit-prefix classification, cosine similarity, the truncation-combining rule).

## Tests run

```text
venv/bin/python3 -m pytest scripts/analysis/tests/test_measure_doc_semantic_drift.py -q
7 passed
```

## Evals run

`scripts/analysis/measure_doc_semantic_drift.py` against 5 real historical doc commits, using
the real live `orion-vector-host` model -- run twice (once with the char-count truncation
heuristic, once after fixing it to a real live tokenizer measurement), same conclusion both
times. This replay *is* the eval the design doc's own acceptance check required.

## Docker/build/smoke checks

None -- read-only script, calls into the already-running `orion-vector-host` container via
`docker exec`, no service/container/compose changes.

## Review findings fixed

- Finding: the truncation-risk heuristic used a chars-per-token average (4 chars/token), which
  risks false negatives on this repo's real corpus (code-block/path/identifier-heavy docs
  tokenize denser than prose).
  - Fix: `_embed_batch_via_vector_host` now also returns a real per-text token count from the
    container's own tokenizer (`truncation=False`, compared against `model_max_length`),
    replacing the char-count proxy entirely.
  - Evidence: re-ran the full calibration with the fix; identical conclusion (5/5 truncated), now
    from a real measurement instead of an approximation.
- Finding: the report's "too few non-truncated samples to propose a real threshold" sentence was
  hardcoded prose, not actually conditioned on the sample count -- would stay wrong if this
  script is rerun later with a genuinely larger clean sample.
  - Fix: added an explicit `MIN_SAMPLES_PER_SIDE_FOR_REAL_THRESHOLD` check that drives the
    reported confidence language from the real counts.
  - Evidence: `test_sample_truncated_true_if_either_side_is_truncated` and the surrounding logic
    change; re-ran, report text reflects the real (still-too-small) counts correctly.
- Finding (not fixed, disclosed): `_git_show`'s blanket `returncode != 0 -> ""` conflates
  "genuinely didn't exist at that revision" with other git failures (bad SHA, I/O error). Low
  severity given the fixed, already-verified `REAL_SAMPLE` list; not fixed this patch.

## Restart required

No restart required -- read-only script, no running service changed.

## Risks / concerns

- Severity: should-fix before this arc continues.
  - Concern: the doc-semantic-drift design doc's Phase-1 framing needs a real update reflecting
    this finding -- whole-document embedding doesn't work, diff-scoped embedding (or a
    longer-context model) is the real next architecture decision.
  - Mitigation: this report documents the finding; the design doc itself is not edited in this
    patch (a docs update is a small, separate, low-risk follow-up if useful).
- Severity: note.
  - Concern: 5 real samples is a small calibration set, and this patch didn't reach the point of
    testing the (unbuilt) diff-scoped-embedding fix against a larger sample.
  - Mitigation: natural next step once diff-scoped embedding is decided and built.

## PR link

(added after push)

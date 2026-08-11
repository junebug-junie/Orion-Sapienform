# Doc semantic drift: diff-scoped embedding confirms the fix works

Status: **DONE**. Follow-up to `docs/superpowers/pr-reports/2026-08-11-doc-semantic-drift-embedding-diff-calibration.md`
(PR #1560, merged), which found that whole-document embedding is structurally broken for this
repo's real docs (a 512-token model limit vs. real 5KB-74KB READMEs truncates both before/after
to the same leading content, producing `embedding_diff ~= 0.0000` regardless of edit size). This
patch implements and tests the recommended fix -- diff-scoped embedding, embedding only the
changed hunk lines instead of the whole document -- and confirms with real evidence that it
works.

## What changed

- `_git_diff_hunks(sha, path)`: extracts just the changed lines from a real unified diff
  (`git diff --unified=0`), not the whole before/after document. File headers and hunk markers
  excluded; only real `+`/`-` content lines.
- `measure_doc_semantic_drift.py`'s `main()` now embeds 4 texts per sample in the same batch
  call (whole-document before/after, kept for contrast, plus hunk-scoped before/after) and
  reports both.

## Real result

Re-ran the same 5 real historical doc commits from the prior patch:

| sha | hunk removed/added chars | diff_scoped_embedding_diff | expected | truncated |
|---|---|---|---|---|
| f88dc088 | 717/1547 | 0.0473 | trivial | no |
| 2144f852 | 672/3940 | 0.1687 | real | **YES** |
| 9e34ee88 | 9/59 | 0.3841 | trivial | no |
| dc375dca | 0/1342 | 0.4150 | real | no |
| 121185a3 | 0/2674 | 0.4356 | real | **YES** |

**Real separation found** on the 3 non-truncated samples: max `trivial` (0.3841) < min `real`
(0.4150). This confirms the prior patch's diagnosis directly -- the embedding-diff *signal*
works; whole-document embedding was the broken part, not the technique.

**Honest limitation, not glossed over**: 2 of 5 samples are *still* flagged truncated even after
diff-scoping -- a single hunk on a large real diff (24 insertions/4 deletions, 36 insertions) can
itself exceed 512 tokens. Diff-scoping fixes the common case (small, localized edits) but not
every case. The proposed threshold (0.3996, midpoint of the 3 clean samples) is explicitly
reported as "only a midpoint," not a calibrated cutoff -- 3 samples is far short of a real
sample size.

## Review findings fixed

- Docstring referenced a function (`embedding_diff_for_hunks`) that doesn't exist -- the scoring
  is inline in `main()`, not a separate function. Fixed the docstring to describe the real
  `text or " "` fallback pattern instead.
- Added a documented (not fixed -- currently unhit) limitation: a renamed file (no `-M` passed
  to `git diff`) shows as a full delete-at-old-path + full-add-at-new-path, degenerating hunk
  text back into whole-document size. Moot for the current hand-picked `REAL_SAMPLE` (no renames
  in it), but a real gap to close with an explicit guard before extending the sample set.
  Binary-file diffs noted similarly (no `+`/`-` lines, silently yields two empty strings).
- Added the missing symmetric test: a pure-deletion case (empty `added_text`) alongside the
  existing pure-addition test.

## Files changed

- `scripts/analysis/measure_doc_semantic_drift.py`: `_git_diff_hunks()`, 4-text-per-sample
  batching, new report section.
- `scripts/analysis/tests/test_measure_doc_semantic_drift.py`: 3 new tests (2 using real
  throwaway git repos to verify hunk extraction against real git output, not mocked diffs; 1
  pure-deletion symmetry test).

## Tests run

```text
venv/bin/python3 -m pytest scripts/analysis/tests/test_measure_doc_semantic_drift.py -q
10 passed
```

## Evals run

Real replay against the same 5 real historical doc commits, via the live `orion-vector-host`
container -- see "Real result" above. This *is* the eval: confirms diff-scoped embedding
recovers real separation on the sample where whole-document embedding failed completely.

## Docker/build/smoke checks

None -- read-only script, calls into the already-running `orion-vector-host` container, no
service/container changes.

## Restart required

No restart required.

## Risks / concerns

- Severity: note. 3 non-truncated samples is a small basis for the proposed threshold
  (0.3996) -- explicitly reported as a midpoint, not a calibrated cutoff. A real threshold needs
  a genuinely larger sample (the script's own `MIN_SAMPLES_PER_SIDE_FOR_REAL_THRESHOLD = 5`
  documents this bar).
- Severity: note. Large single-hunk diffs can still exceed the token limit even diff-scoped --
  diff-scoping narrows the truncation problem, doesn't eliminate it. A future patch could chunk
  an oversized hunk rather than truncating it wholesale.
- Severity: note. Still no producer, no service, no `orion/concepts/` registry wiring -- this
  patch clears the calibration gate the design doc's own Phase-1 required, but building the
  live producer is a separate, not-yet-started next step.

## PR link

(added after push)

# feat(scripts): add scripts/diag.py — phi corpus variance/gate diagnostic

**Status:** IMPLEMENTED, tested, reviewed. Builds a tool referenced by two
spec docs' acceptance checks that was never actually created — a gap
discovered when the operator went to run it and it didn't exist.

## Summary

- New `scripts/diag.py`: read-only diagnostic for the phi training corpus.
  Reports per-dimension variance/liveness and the three corpus gates
  (`min_rows`, `min_hours`, `variance_fraction`) as JSON, exit code 0/1.
- Reuses `scripts/fit_phi_encoder.py`'s row loading/filtering/variance-gate
  logic directly (`_load_jsonl`, `_filter_training_rows`, `_hours_span`,
  `_variance_gate`, `features_version`, `input_features`) — no duplicated
  logic. Never trains, never writes artifacts.
- Default `--features-version` is `seed-v4` (the currently-live corpus-write
  version), distinct from `fit_phi_encoder.py`'s own conservative
  `DEFAULT_FEATURES_VERSION` (stays `seed-v3` until a seed-v4 encoder passes
  the promote gate).

## Why this was needed

`docs/superpowers/specs/2026-07-09-phi-truthful-corpus-overview.md` and
`2026-07-09-phi-seedv4-feature-set-design.md` both name `scripts/diag.py` as
an acceptance check ("≥8 dims var>1e-6 on ≥4h fresh corpus"). It never
existed as a real, committed script — an earlier session's ad hoc scratchpad
reproduction of the same logic was referenced in the spec but never made it
into the repo. Now that `PUBLISH_REASONING_TELEMETRY` and
`INNER_FEATURES_VERSION=seed-v4` are both live (PR #922) and the corpus is
starting to accrue seed-v4 rows, this is the actual next tool needed.

## Review finding fixed (self-caught before commit)

- Finding: `run_diag()`'s docstring claimed "never raises," but it called
  `_load_jsonl()` unguarded — that function raises `ValueError` on a
  malformed JSONL line, which would crash the whole diagnostic instead of
  reporting a clean failure.
  - Fix: wrapped the call in `try/except`, degrading to
    `{"ok": False, "reason": "corpus load failed: ..."}`.
  - Evidence: `test_diag_never_raises_on_malformed_jsonl_line`, passing.

## Files changed

- `scripts/diag.py` (new)
- `tests/test_phi_corpus_diag_script.py` (new) — 7 tests: missing corpus,
  fully-live seed-v4 corpus passes all gates, mostly-frozen corpus fails the
  variance gate with correct got/need counts, too-few-rows fails min_rows,
  CLI subprocess exit code matches gate result (0/1), default features
  version, malformed-JSONL-line never raises.

## Tests run

```text
pytest tests/test_phi_corpus_diag_script.py tests/test_phi_encoder_fit_script.py tests/test_corpus_gate.py -q
  → 22 passed
```

Also manually verified against the real corpus:
```bash
python scripts/diag.py --corpus /mnt/telemetry/phi/corpus/inner_state.jsonl
```
→ correctly reports 12,100 total rows, 0 matching `seed-v4` (expected — the
version flip only just landed), all three gates failing with an honest
reason each. No crash, no false-positive readiness claim.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Docker/build/smoke checks

N/A — CLI script, no service/runtime change.

## Restart required

```text
No restart required.
```

## Risks / concerns

None — additive, read-only tooling; does not touch any live service or the
corpus itself.

## PR link

Branch pushed: `feat/phi-corpus-diag-script`.
Compare: https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/phi-corpus-diag-script

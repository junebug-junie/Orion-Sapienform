# feat(fit-phi-encoder): relaxed variance-gate policy for seed-v4's optional reasoning dims

**Status:** IMPLEMENTED, tested, reviewed. Follow-up to the deploy of specs
1-3 + the grammar-health gate fix — the corpus is now accruing correctly,
but the promote gate as originally specified can never pass in a deployment
without an active thinking-capable model.

## Summary

seed-v4's variance gate required ≥7 of 8 trainable dims live
(`ceil(0.8 * 8)`). Confirmed live (45 min post-deploy): 6 of 8 dims already
show real variance, but `reasoning_present` and `reasoning_load` are both
structurally dead here — no thinking-capable model is active, so no call
ever produces reasoning content, and `reasoning_load` additionally has no
provider-side thinking-token count to read yet (unrelated to this patch,
already documented as reserved in the reasoning-telemetry-adapter spec).
With both permanently dead, the corpus could accrue forever and never reach
7/8.

**New policy (Juniper's explicit call):**
- **6/8 required** when both `reasoning_present` and `reasoning_load` are dead.
- **7/8 required** the moment either one shows real signal — the bar goes
  back up as soon as it's actually achievable, rather than permanently
  discounting both.

## Files changed

- `scripts/fit_phi_encoder.py` — `_variance_gate()` gains an optional
  `feature_names` param. When provided and it contains **both**
  `reasoning_present` and `reasoning_load` (seed-v4's unique signature —
  seed-v3 also has `reasoning_present` but never `reasoning_load`, so the
  policy requires the full set to avoid silently activating for seed-v3
  too), switches to the fixed 6-or-7 policy. Without `feature_names`, or for
  any corpus lacking both dims, behavior is byte-identical to the old
  fraction-based gate (verified by test). `check_corpus_gates()`/`cmd_train()`
  thread `feature_names` through.
- `scripts/diag.py` — passes `feature_names` too, so the diagnostic tool's
  reported gate matches the real promote gate exactly (no more silent
  drift between what `diag.py` reports and what `fit_phi_encoder.py` would
  actually do).
- `tests/test_phi_encoder_fit_script.py` — 6 new tests on `_variance_gate`
  directly: 6/8 passes when both reasoning dims dead, 7/8 required once one
  is live, still fails below 6, still passes at 8/8, backward-compat when
  `feature_names` omitted, and — importantly — seed-v3's own feature names
  fall through to the plain fraction gate untouched.
- `tests/test_phi_corpus_diag_script.py` — updated one existing test's
  expected `need` value (7 → 6) to match the new policy for its fixture
  (both reasoning dims dead in that synthetic corpus).

## Review finding fixed (self-caught before commit)

- Finding: first draft activated the relaxed policy whenever **any** of the
  two optional dims appeared in `feature_names` — but seed-v3's own feature
  set legitimately includes `reasoning_present` (shared cognitive slot name
  with seed-v4), so the relaxed 6-dim policy would have silently applied to
  seed-v3 corpora too, weakening its gate without anyone asking for that.
  - Fix: require the **full set** (`SEEDV4_OPTIONAL_VARIANCE_DIMS.issubset(feature_names)`),
    which only seed-v4 satisfies (seed-v3 never has `reasoning_load`).
  - Evidence: `test_variance_gate_seedv3_feature_names_unaffected_by_seedv4_policy`
    — expects seed-v3's own gate math (`need == ceil(0.8 * 11) == 9`); against
    the buggy first draft it got `7` (the relaxed seed-v4 policy leaking in),
    failing the assertion. Passes after the `issubset` fix.

## Tests run

```text
pytest tests/test_phi_encoder_fit_script.py tests/test_phi_corpus_diag_script.py tests/test_corpus_gate.py -q
  → 29 passed
```

Verified against the real live corpus:
```bash
python scripts/diag.py --corpus /mnt/telemetry/phi/corpus/inner_state.jsonl
```
Before this patch: `{"variance": {"need": 7, "got": 6, "ok": false}}` (stuck,
unreachable). After: `{"variance": {"need": 6, "got": 6, "ok": true}}`. Only
`min_hours` remains failing now (needs ≥4h accrual — just time).

## Schema / bus / API changes

None.

## Env/config changes

None.

## Docker/build/smoke checks

N/A — CLI script logic only, no service/runtime change.

## Restart required

```text
No restart required. Takes effect the next time scripts/diag.py or
scripts/fit_phi_encoder.py is run.
```

## Risks / concerns

- Severity: low. This only relaxes the gate for the two dims that are
  genuinely unmeasurable in the current deployment; it does not touch the
  gate for any other dim, and the bar rises back to 7/8 automatically the
  moment either reasoning dim shows real signal (e.g. after a
  thinking-capable model is deployed).

## PR link

Branch pushed: `fix/phi-seedv4-variance-gate-optional-reasoning-dims`.
Compare: https://github.com/junebug-junie/Orion-Sapienform/compare/main...fix/phi-seedv4-variance-gate-optional-reasoning-dims

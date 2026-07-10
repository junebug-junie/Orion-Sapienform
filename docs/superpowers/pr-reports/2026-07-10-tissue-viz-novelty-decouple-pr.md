# fix(spark-introspector): decouple tissue-viz novelty from dead SelfStateV1 uncertainty

**Status:** IMPLEMENTED, tested, reviewed. Fixes a live production symptom
reported directly ("novelty and arousal in Hub are showing 0 and sticking to
it") — novelty half fixed here; arousal diagnosed but deliberately not fixed
yet (see below).

## Summary

Hub's tissue-viz (the novelty/arousal/valence/phi EKG display, served from
`services/orion-spark-introspector/app/static/tissue_viz.js`) showed novelty
stuck at exactly `0.0`. Verified live via a direct websocket connection to
`/ws/tissue` — confirmed across multiple consecutive ticks.

Traced the full chain:

- `_phi_from_self_state()` computes
  `novelty = (0.6*uncertainty + 0.4*introspection_pressure) * (0.3 + 0.7*coherence)`
  from the incoming `SelfStateV1` payload.
- Live `/latest` snapshot from `orion-self-state-runtime` confirmed:
  `uncertainty.score = 0.0` and `introspection_pressure.score = 0.0` — both
  **real, correctly-computed readings**, not missing/defaulted.
- `uncertainty` comes from `orion/self_state/scoring.py::uncertainty_score()`
  = `salience * (1 - coherence)` — zeroed because that service's own
  `coherence` was saturated at `1.0` (its own formula only drops on active
  failure/friction channels).
- `introspection_pressure` is fed solely by `egress_confidence_deficit`
  (`1 - egress_confidence`), which only registers nonzero when an execution
  **fails to emit** — the same "0.3% signal" structural sparsity already
  found and excised from φ's `exec_step_fail_rate`/`execution_friction` pair
  earlier in this initiative.

**This is not dead/fake code** — `uncertainty` and `introspection_pressure`
are real alarm-style signals (quiet during health, spike on trouble) being
repurposed for a display that wants continuous ambient variation. Wrong tool
for the job, not theater.

## Fix

Per explicit direction ("excise the fake math theater and find a real
replacement" → "decouple novelty from coherence entirely" → source it from
the already-correct embedding-cosine-distance novelty): tissue-viz novelty
now comes from `handle_semantic_upsert`'s real chat-triggered novelty
(confirmed live: healthy 0.1–0.5 variance), held in a new module-level
`_LAST_EMBEDDING_NOVELTY` and read through a new `_novelty_stat()` helper —
mirroring the exact existing `_INNER_LAST_HEADLINE`/`_headline_stat()`
precedent already in this file.

Applied to **both** broadcast sites that were reading the dead value:
- `handle_self_state`'s self-state-tick broadcast (fires continuously, every
  tick — the dominant source of the "stuck" symptom since it fires far more
  often than actual chat activity).
- `handle_trace`'s heartbeat branch (same `_get_phi_stats()` →
  `_phi_from_self_state()` chain, same bug).

The other two `tissue.update` broadcast sites (`_broadcast_tissue_update`,
and the trace-path candidate broadcast) already correctly used real
appraisal/embedding-based novelty with skip-if-unknown guards — untouched.

**Deliberately not touched:** `orion/self_state/scoring.py` — the shared
formulas themselves are correct for their original alarm-signal purpose and
also feed `agency_readiness_score`; changing them would be a much larger,
cognition-touching change out of scope for this display-layer fix.

## Arousal — diagnosed, not fixed (explicit scope decision)

Also traced why `arousal` sticks at 0: `resource_pressure` MAX-aggregates 7
heterogeneous channels (`transport_pressure`, `cpu_pressure`, `gpu_pressure`,
`memory_pressure`, `disk_pressure`, `thermal_pressure`, and a generic
`pressure` channel from the capability field graph). Live evidence:
`cpu_pressure=0.92` (real, high) but a separate `pressure=1.00` channel
(sourced from a specific capability's saturation in `orion-field-digester`'s
field graph, exact producer not traced to completion this pass) dominates
via `max()`, permanently zeroing `resource_cap = 1 - resource_pressure` and
therefore the whole `energy` formula regardless of what any other channel
reads. Presented three fix options (weighted-average aggregation, exclude
the untraced generic channel, diagnose further); Juniper chose "diagnose
further" — not implemented in this patch.

## Files changed

- `services/orion-spark-introspector/app/worker.py` — `_LAST_EMBEDDING_NOVELTY`
  global, `_novelty_stat()` helper, both broadcast sites updated.
- `services/orion-spark-introspector/tests/test_tissue_viz_novelty.py` (new)
  — 5 tests: `_novelty_stat()` defaults/reflects the held value;
  `_phi_from_self_state()` reproduces the exact live-incident zero (direct
  proof of the bug mechanism); `handle_semantic_upsert` updates the held
  value; end-to-end regression proving `handle_self_state`'s broadcast uses
  the embedding novelty instead of the dead SelfStateV1-derived one even when
  coherence is saturated.
- `services/orion-spark-introspector/tests/test_inner_state_emit.py` — one
  unrelated pre-existing test fixed in passing:
  `test_inner_features_settings_defaults` asserted
  `orion_phi_encoder_enabled is False`, stale since the seed-v4 encoder was
  promoted and the flag flipped `true` earlier this session (same
  live-`.env`-reading fragility already patched once this session for
  `inner_features_version`).

## Tests run

```text
pytest services/orion-spark-introspector/tests -q
  → 109 passed, 1 pre-existing unrelated failure (test_phi_reward_emitted_when_encoder_ok,
    confirmed present on main before this branch)
```

Regression verified by reverting the fix and confirming 4 of 5 new tests
fail (missing attribute / dead-value assertion), then reapplying.

## Review findings fixed

Self-reviewed (medium effort). No correctness findings. One accuracy
nitpick in my own comments, fixed before commit: `_LAST_EMBEDDING_NOVELTY`'s
docstring initially said "embedding-cosine-distance novelty" but the value
actually stored is `handle_semantic_upsert`'s *display* novelty (prefers
cached appraisal score via `_display_novelty_for_corr`, falls back to raw
embedding cosine distance) — comment corrected for precision.

## Docker/build/smoke checks

Not run against live containers in this environment. Live-verified the
*bug* directly against the running deployment (`/ws/tissue` websocket, three
consecutive ticks all showing `novelty: 0.0`) before writing the fix.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
```
After restart, reconnect to `/ws/tissue` (or reload the tissue-viz page) and
confirm `novelty` varies with real chat activity instead of holding at 0.

## Risks / concerns

- Severity: low. Additive display-layer fix; the underlying `uncertainty`/
  `introspection_pressure`/`coherence` formulas and every other consumer of
  them (φ, `agency_readiness_score`) are completely untouched.
- `arousal` is still stuck at 0 pending the resource_pressure aggregation
  redesign decision — known, disclosed, not silently left unaddressed.

## PR link

Branch pushed: `fix/tissue-viz-novelty-decouple-uncertainty`.
Compare: https://github.com/junebug-junie/Orion-Sapienform/compare/main...fix/tissue-viz-novelty-decouple-uncertainty

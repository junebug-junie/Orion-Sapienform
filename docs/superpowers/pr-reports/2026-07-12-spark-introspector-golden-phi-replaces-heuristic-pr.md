## Summary

- The only "phi" ever reaching orion-cortex-exec's live metacognition prompts (`log_orion_metacognition_draft.j2`/`_enrich.j2`, via `spark_narrative.py`'s `spark_phi_hint`/`spark_phi_narrative`) was `_phi_from_self_state()` — a hand-tuned, untrained heuristic, not the trained MLP autoencoder retrained earlier today. The trained encoder's real output (`PhiIntrinsicRewardV1`) had zero consumers beyond a SQL sink (`orion-sql-writer`) and a debug WebSocket EKG panel.
- Wired the trained encoder's output into `phi_now`'s `coherence`/`energy`/`novelty` axes whenever it's enabled and healthy this tick, so the golden (trained) phi is what actually reaches cognition.
- `valence` deliberately left on the heuristic — the active encoder's latent probes show zero correlation between any latent dim and any hedonic-adjacent felt dimension, so there is no trained analog to source it from; fabricating one would be exactly the empty-shell-cognition pattern this change exists to remove.
- Code review caught and I fixed a real bug in the first draft: without resetting the encoder's tick-to-tick delta tracking on every skip/failure path, a resumed tick after any gap would compute a delta spanning multiple skipped ticks, landing a misleadingly large value in a now-prompt-facing field.
- Also fixed a pre-existing, unrelated test bug (stale `features_version` in a test fixture, silently no-opping the encoder in that test).

## Outcome moved

`SparkStateSnapshotV1.phi` (and therefore the live metacognition prompt's coherence/arousal/novelty framing) now reflects the actual trained autoencoder — the same encoder just retrained today on post-fix data and confirmed non-collapsed — instead of an untrained hand-written formula that happened to be the only thing ever wired to real cognition.

## Current architecture

`services/orion-spark-introspector/app/worker.py`'s `handle_self_state()` computed `phi_now = _phi_from_self_state(ss)` (4-key dict: coherence/energy/novelty/valence) every self-state tick, published it as `SparkStateSnapshotV1.phi`, and separately — when the trained encoder was enabled and healthy — computed `out.phi`/`out.recon_error`/`delta_phi` and published them only as `PhiIntrinsicRewardV1` to a SQL-sink-only channel. Two independent phi computations, only one of which reached cognition.

## Architecture touched

`orion-spark-introspector` only. `orion-cortex-exec`'s `spark_narrative.py` needed zero changes — it already reads `snapshot.phi.get(...)` generically, so overriding the dict's values at the source was a thin, single-service seam.

## Files changed

- `services/orion-spark-introspector/app/worker.py`:
  - New `_golden_phi_overrides()` — pure function mapping trained-encoder output to the 3 axes with principled analogs, with a documented rationale for why `valence` is excluded.
  - `handle_self_state()`: overrides `phi_now`'s coherence/energy/novelty when the encoder tick succeeds; introduces `encoder_tick_ok` flag (set only at the end of a fully successful try block) so `_PHI_PREV_PHI`/`_PHI_PREV_RECON` reset to `None` on any skip or mid-inference exception, preventing a stale multi-tick delta from landing in the now-prompt-facing `energy` field.
- `services/orion-spark-introspector/tests/test_phi_reward_emit.py`:
  - New `test_golden_phi_overrides_coherence_energy_novelty_not_valence` — asserts the snapshot's coherence/energy/novelty match the trained encoder's output and valence still matches the heuristic.
  - New `test_phi_prev_resets_across_a_skipped_tick` — 3-tick regression: healthy → skipped (grammar-degraded) → healthy, asserting `delta_phi` on the third tick is a fresh `0.0`, not `tick3.phi - tick1.phi`.
  - Fixed `test_phi_reward_emitted_when_encoder_ok` (pre-existing, unrelated failure — confirmed via `git stash` against main before this branch existed): `settings.inner_features_version` defaults to `"seed-v3"` but the test's tiny encoder manifest declares `features_version="seed-v2"`, so `PhiEncoderRuntime.load()` silently returned `None` and the test's own assertion always failed. Added the missing monkeypatch.

## Schema / bus / API changes

None. `PhiIntrinsicRewardV1`/`SparkStateSnapshotV1` schemas unchanged; only which values populate existing fields changed.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. :services/orion-spark-introspector pytest services/orion-spark-introspector/tests -q
135 passed, 1 skipped  (repeated 5x for the nondeterministic tiny-encoder fixture — stable every run)

PYTHONPATH=. pytest tests/test_phi_encoder_mlp.py tests/test_phi_encoder_fit_script.py \
  tests/test_spark_introspector_publish_guard.py tests/test_phi_encoder_schema.py -q
all passed

tests/test_spark_introspector_no_legacy.py::test_worker_has_no_legacy_kind -- FAILS,
confirmed pre-existing and unrelated via `git stash` against clean main
(checks worker.py doesn't contain the literal string "spark.introspection";
already true before this branch, not touched by this change)
```

## Evals run

None applicable — this is a wiring/logic change covered by unit tests, not a training-quality question (the encoder itself was already trained/evaluated/promoted in an earlier session turn).

## Docker/build/smoke checks

Not run — not deployed. Restart commands below, left for Juniper to trigger.

## Review findings fixed

- Finding: `_PHI_PREV_PHI`/`_PHI_PREV_RECON` only reset on the two skip paths I initially patched (encoder disabled/degraded, `enc is None`) — missed the case of an exception occurring mid-inference (between `enc.forward(x)` and the final assignment), which would leave the globals stale exactly like the original bug, just triggered a different way.
  - Fix: restructured around an `encoder_tick_ok` flag, set only at the true end of a fully successful try block wrapping the entire encoder-inference section; the reset check now lives after the whole `if settings.inner_features_enabled:` block, so it fires on every skip/failure path uniformly, including `inner_features_enabled=False` itself.
  - Evidence: new test `test_phi_prev_resets_across_a_skipped_tick`, verified independently by a second review pass confirming no remaining path leaves `encoder_tick_ok` unset or lets an exception escape without resetting.
- Finding: the first version of `test_golden_phi_overrides_...`'s energy assertion was trivially `0.0 == 0.0` since `_reset_inner_state` leaves `_PHI_PREV_PHI=None`, forcing `delta_phi=0.0` by the cold-start convention — a broken energy mapping (wrong source variable, missing `abs()`, wrong clamp) would have passed silently.
  - Fix: set `_PHI_PREV_PHI` to a known nonzero baseline before the tick under test, plus an explicit `assert reward.delta_phi != 0.0` guarding the test setup itself.
  - Evidence: test now genuinely exercises a nonzero delta; re-ran 5x to confirm stability against the encoder fixture's unseeded random weights.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml \
  up -d --build spark-introspector
```

Not run — deliberately left for Juniper to trigger.

## Risks / concerns

- Severity: medium
- Concern: this changes the actual content of Orion's live metacognition prompt (coherence/energy/novelty framing) the moment it's deployed — a real behavior change to a chat-facing cognition surface, not just a metrics/telemetry fix like the rest of today's work.
- Mitigation: `valence` (the one axis with genuine domain-specific hand-tuning and no honest trained replacement) is untouched; coherence/energy/novelty all have principled, documented 1:1 analogs in the trained encoder's actual output, not arbitrary relabeling. Recommend watching a chat session or two post-deploy to sanity-check the narrative reads coherently before treating this as fully settled.
- Severity: low
- Concern: `energy`'s scale (`|delta_phi|`, bounded [0,1] by construction) and `novelty`'s scale (`recon_error / recon_error_p95`, clamped to [0,1]) are new value distributions the prompt-banding functions (`_arousal_band`, `_overload_band` in `spark_narrative.py`) haven't been tuned against — they assume roughly-uniform [0,1] inputs, which these should satisfy, but the actual banding thresholds (0.33/0.66) were tuned against the OLD heuristic's distribution, not the new one.
- Mitigation: none built in this patch — worth revisiting the banding thresholds once a real traffic window of the new distribution exists, same "measure before assuming" principle used everywhere else in today's work.

## PR link

Branch pushed: `feat/spark-introspector-golden-phi-replaces-heuristic`

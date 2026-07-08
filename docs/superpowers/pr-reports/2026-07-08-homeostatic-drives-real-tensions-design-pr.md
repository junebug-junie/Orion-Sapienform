# PR: Homeostatic drives + real deviation-driven tensions

**Status:** IMPLEMENTED (substrate) + design docs. Tasks 1–5, 7, 8 built with
tests and a passing end-to-end eval; Task 6 wires leaky math into the production
engine and stages the signal source; the live consumer subscription + host
verification (Tasks 6-live/9/10) is the remaining step and needs the running host.

## Summary

- **Kills the flat-0.731 pin.** `DriveEngine.update` is now a wall-clock leaky
  integrator `p ← clamp01(p·e^(−Δt/τ) + impulse·(1−base))` — rests at zero,
  cadence-invariant, no fixed point. Legacy `soft_saturate` kept behind a flag.
- **Deviation gate** (`DeviationGate`): per-`(signal_kind,dimension)` EWMA baseline;
  impulses only on worse-direction z-excess. Steady input (the 55/s `scene_state`
  flood) settles to its mean and mints zero.
- **Structural map** (`config/autonomy/signal_drive_map.yaml` + typed loader):
  closed `signal_kind→drive_impacts`, no free text; a grep-guard test forbids
  lexical inspection. Corrected to real bus dims (biometrics emits dynamic
  `<metric>_level/_volatility` — added structural suffix matching).
- **Adapter** (`signal_tension.py`): OrionSignal→tension (gated), failure→tension
  (direct — a failure is itself the deviation), equilibrium→tension (edge-triggered).
  All degrade to None, never raise.
- **Rate limiter**: per-`(kind,drive-signature)` sliding-window cap, bounded LRU state.
- **Production wire**: worker `DriveEngine` built with `leaky_math_enabled=cfg`;
  `SignalTensionSource` constructed on the worker.

## Outcome moved

Drive pressure stops being a tick-cadence artifact. End-to-end eval evidence
(`run_homeostatic_drives_eval.py`, replay through the real pipeline):

```
flood signals seen : 66000
flood tensions     : 0        (55/s scene_state fully starved)
total tensions kept: 4        (real events mint real tensions)
dominant histogram : {capability: 4}   (NOT alphabetical "autonomy")
pressure trajectory:
  t= 419s (mid-strain)  continuity 0.673, capability 0.673   (differentiated)
  t= 700s               capability 0.468, continuity 0.264, coherence 0.194
  t=1199s (quiet)       capability 0.089, continuity 0.050 …  (resting to zero)
RESULT: PASS  (all 5 checks)
```

## Current architecture (before)

`orion/spark/concept_induction/drives.py:36` — `_soft_saturate` = `1−exp(−gain·p)`,
stable non-zero fixed point ~0.731 → every drive pinned identically under frequent
ticks. Tensions fired ~0.064% of ticks; dominant drive was an alphabetical artifact.

## Architecture touched

- `orion/spark/concept_induction/drives.py` — leaky integrator + `leaky_math_enabled`.
- `orion/autonomy/deviation_gate.py`, `signal_drive_map.py`, `signal_tension.py`,
  `tension_ratelimit.py` — new pure modules.
- `config/autonomy/signal_drive_map.yaml` — structural map.
- `orion/spark/concept_induction/bus_worker.py` — engine leaky flag + staged source.
- `orion/spark/concept_induction/settings.py` + service `.env_example` — flags/params.
- `orion/autonomy/evals/run_homeostatic_drives_eval.py` — end-to-end eval.

## Files changed

- Real code: 6 modules + eval + 5 test files (see commits).
- Docs: spec, plan, this report.

## Schema / bus / API changes

- Reused `TensionEventV1`/`OrionSignalV1`/`compute_tick_attribution`/`is_stub_signal`
  unchanged. New tension `kind` strings (`tension.signal.v1`/`.failure.v1`/`.health.v1`)
  are free values on the already-registered `TensionEventV1` model — no per-kind
  registry entry (that would be a keyword-cathedral with no runtime behavior).
- Consumed channels (for the live wire): `orion:signals:biometrics`,
  `orion:signals:spark`, `orion:signals:equilibrium` — specific organs, NOT the
  `orion:signals:*` wildcard, so the `scene_state` flood is excluded at subscription.

## Env/config changes

- Added keys: `ORION_HOMEOSTATIC_DRIVES_ENABLED=true`, `ORION_DRIVE_LEAKY_MATH_ENABLED=true`,
  `DEVIATION_EWMA_ALPHA`, `DEVIATION_Z_THRESHOLD`, `DEVIATION_SIGMA_FLOOR`,
  `SIGNAL_TENSION_IMPULSE_K`, `SIGNAL_TENSION_CAP_PER_WINDOW`, `SIGNAL_TENSION_WINDOW_SEC`.
- `.env_example` updated; settings↔example parity verified (all 8 keys both sides).
- Local `.env` sync: worktree has no local `.env` (gitignored, on host); run
  `python scripts/sync_local_env_from_example.py` on the deployment host post-merge.

## Tests run

```text
pytest orion/spark/concept_induction/tests orion/autonomy/tests -q
254 passed  (incl. 30 new: leaky math 5, deviation gate 7, map 8, adapter 8, ratelimit 4)
python orion/autonomy/evals/run_homeostatic_drives_eval.py  → RESULT: PASS (5/5)
```

## Review findings fixed

```text
Code review (Task 9) pending on the implementation before final DONE.
```

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: medium — leaky math changes `policy_act` pressure reads. Gated by
  `ORION_DRIVE_LEAKY_MATH_ENABLED`; roll back to legacy without disabling the source.
- Consumer subscription not yet live: the signal→drive path is built + eval-proven
  but the worker does not yet subscribe to the organ channels. That wire needs a
  dedicated drive-only handler (so signals don't trip concept induction) + live
  verification against the real 55/s reality (Task 10) on the host.

## PR link

<to be filled by `gh pr create` / GitHub UI>

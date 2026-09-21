# Pre-registration: lattice vs thermometer (heartbeat v0)

**Date:** 2026-09-19
**Status:** run complete 2026-09-20 — lattice, not thermometer
**Run report:** `docs/research/preregistration/2026-09-19-heartbeat-lattice-vs-thermometer-run.md`
**Service:** `orion-heartbeat` (live `EnsembleSubstrate` + `route_atom` + `absorb`)
**Not:** the May 2026 fidelity-H1 formula (already shown tautological for a pure MPS — see `reconstruction.py`)

## Question

Does the running MPS still distinguish *which organ* produced an atom, or is `/h1` a busy-ness meter with extra math?

Mean ratio is already known capacity-saturated at χ=4. This test asks whether the **9-cut entropy profile** (and the kick's locality) still carry organ/site structure.

## What this is not

- Not a cognition-loop consumer. Nothing is wired.
- Not July's "gain organ / lion" overlay.
- Not a retune of verdict bands.
- Not raising χ.

## Arms (same atom sequence, same ensemble config, same base_seed=1000)

Production `EnsembleConfig` (N=8, γ=0.2, decay 0.15, reheat_strength 0.08). Decay/reheat ticks follow real inter-event gaps at `HEARTBEAT_DECAY_REHEAT_INTERVAL_SEC=2`. Reheat probability is the live bus_synaptic-derived value used by the calibration harness (or 0.0 if FalkorDB is unreachable — recorded in the report, not silently swapped).

**A. As-is.** Replay allowlisted `atom_emitted` grammar rows in time order through `route_atom` + `EnsembleSubstrate.absorb`.

**B. Shuffle.** Same rows, same timestamps, `source_service` remapped by a **fixed cyclic permutation** of the five v0 organs (site 0 traffic → site 1, …, site 4 → site 0):

```
orion-hub         → orion-biometrics
orion-biometrics  → orion-cortex-exec
orion-cortex-exec → orion-bus
orion-bus         → orion-cortex-orch
orion-cortex-orch → orion-hub
```

Event *counts* per original organ are preserved; only the site they land on changes. That is the organ-identity test.

**C. Kick.** After arm A warmup, snapshot the mean 9-cut profile, then absorb `KICK_N=50` strong chat-site atoms (`orion-hub`, `signal`, confidence=1, salience=1, uncertainty=0) with **no** decay between them. Snapshot again. The live `absorb()` already walks every hop from the organ site to the chain end with `_HOP_DECAY=0.7`, so a smear is possible by construction; the question is whether far cuts still move as much as near cuts.

## Sample

Live `grammar_events` where `event_kind='atom_emitted'`, last **2 hours**, routed through the v0 allowlist.

A 2-hour window is ~17k allowlisted atoms on this mesh (counted 2026-09-19 before the run). Full replay through 8 quimb trajectories is not a tick-loop job. **Pre-registered cap:** keep the **most recent 800** allowlisted atoms in that window (still live grammar, not synthetic). If the window has fewer than 400 routed atoms, extend until 400 or 6 hours, whichever first. If still <400 → UNVERIFIED.

Record: window hours, raw row count, routed count, skipped-organ, skipped-atom-type, whether the 800 cap fired.

## Scores

Profiles are mean (across 8 trajectories) entropy at cuts 1..9, then divided by `log2(χ)=2` so each cut is a ratio in [0, 1], matching `/h1`.

- `d_shuffle` = Euclidean distance between arm A and arm B final profiles (9-vector).
- `rel_shuffle` = `d_shuffle / rms(profile_A)`.
- Verdict A vs B at the final tick.
- Kick: `Δ = profile_after - profile_before` (same 9-vector of ratios).
  - `near` = mean |Δ| at cuts 1–2
  - `far`  = mean |Δ| at cuts 8–9
  - `smear` = `far / near` (undefined / fail-open as smear if `near < 1e-6`)

## Fail (thermometer) — both must hold

1. **Shuffle null:** `rel_shuffle < 0.05` **and** final verdict A == verdict B **and** `|mean_ratio_A - mean_ratio_B| < 0.01` **and** `|std_ratio_A - std_ratio_B| < 0.005`.
2. **Kick smear:** `smear >= 0.5` (far cuts moved at least half as much as near cuts).

If **both** fail conditions hold → **thermometer.** Inspect-only or stop. No consumer.

If **either** fail condition does **not** hold → **still a research instrument.** Write what moved (which cuts, which verdict). Still no consumer. Next conversation is H2/H3, not wiring.

## Partial outcomes (pre-declared, not fishing)

- Shuffle null but kick local → sites are washed out under traffic, but a fresh pulse still has a near/far gradient. Report as mixed; do not promote to a heart.
- Shuffle moves, kick smears → organ identity is in the *mix*, not in a traveling pulse. Report as mixed.
- Either arm errors / <400 atoms → **UNVERIFIED**, do not call thermometer or lattice.

## Artifact paths

- `/tmp/heartbeat-lattice-vs-thermometer/report.md`
- `/tmp/heartbeat-lattice-vs-thermometer/profiles.json`
- This document's commit hash is the pre-reg. The run report must cite it.

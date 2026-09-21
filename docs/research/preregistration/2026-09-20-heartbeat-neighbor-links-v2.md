# Pre-registration: neighbor links v2 (quiet means quiet)

**Date:** 2026-09-20
**Status:** run complete 2026-09-20 — relational (execution/bus held)
**Run report:** `docs/research/preregistration/2026-09-20-heartbeat-neighbor-links-v2-run.md`
**Service:** `orion-heartbeat` (offline replay, production `absorb`)
**Supersedes the split in:** `2026-09-20-heartbeat-neighbor-links.md` (v1 run was UNVERIFIED — median collapsed at zero). v1 file is not rewritten.

## Question

When two seated organs both actually fire in a short window, is the link between *their* seats higher than when both are silent and something else is busy?

v1 asked the same question with a median split. More than half the five-second bins had zero biometrics and zero execution, so “typical” was zero, every bin counted as co-fire, and the control pile was empty.

## Split (this is the whole patch)

Empty bins are still dropped (a bin exists only if it has at least one allowlisted atom).

- **co-fire:** `n_A ≥ 1` and `n_B ≥ 1` (both actually talked)
- **elsewhere:** `n_A = 0` and `n_B = 0` and `n_other ≥ 4` (both silent, other traffic busy)
- **neither:** everything else (one of the pair talked, or others too thin)

No medians. Zero is quiet.

`n_other` is the count of v0 organs in that window that are not A or B.

## Pairs and which one can decide

Same two neighbor pairs:

1. biometrics — execution (seats 1–2)
2. execution — bus (seats 2–3) ← **deciding pair**

Pulses only walk right along the chain. Bus traffic cannot touch the biometrics–execution seats. So pair 1’s elsewhere pile (both silent, bus busy) is expected to look dark. Pair 1 holding is geometry. It is reported. It cannot call the probe relational by itself.

Pair 2 can fail: biometrics sits to the left of execution and the bus, and a biometrics pulse *does* walk through those seats. If busy-biometrics / silent-exec-and-bus lights the exec–bus link as much as those two talking, that is smear. That is the thermometer call.

## Windows

Last **2 hours** of live `atom_emitted`, allowlisted organs, **5-second** bins. Fresh `EnsembleSubstrate` per bin (`EnsembleConfig` defaults, `base_seed=1000`). Decay/reheat off. Same two-seat `I` as v1 (`partial_trace_exact`, bits, mean of 8 trajectories). If one 2-seat measurement takes more than 2 seconds → UNVERIFIED.

No 10-second first pass. v1 already showed 10-second bins starve the quiet cell.

Write every bin’s organ counts and both `I` values to `windows.jsonl` so a later split does not require another hour of absorb.

## Fail (thermometer)

The deciding pair (execution — bus) has both cells ≥ 10 windows and `delta` below its threshold:

- If `max(mean I_cofire, mean I_elsewhere) ≥ 0.05`: need `delta ≥ 0.05`
- If both cell means `< 0.05`: need `delta ≥ 0.25 * max(mean I_elsewhere, 1e-4)`

Biometrics traffic smears into the exec–bus seats as much as those two talking.

## Hold (still an instrument)

Deciding pair beats its threshold, both of its cells ≥ 10 windows.

Write the numbers. Still no consumer. Pair 1 is a check, not the headline.

## Partial (pre-declared)

- Deciding pair < 10 windows in either cell → **UNVERIFIED**
- Pair 1 cells empty → record it; do **not** UNVERIFIED the whole probe
- Pair 1 holds and pair 2 fails → **thermometer** (geometry on the left, smear on the deciding pair)
- Pair 1 fails and pair 2 holds → **relational** (unexpected on pair 1; still the deciding pair won)
- 2-seat measurement too slow or errors → **UNVERIFIED**

## What this is not

Wiring. Chat. Raising χ. Retuning verdict bands. Quietly editing v1 after the fact.

## Artifact paths

- `/tmp/heartbeat-neighbor-links-v2/report.md`
- `/tmp/heartbeat-neighbor-links-v2/windows.json`
- `/tmp/heartbeat-neighbor-links-v2/windows.jsonl`
- This document is the pre-reg. The run report must cite it.

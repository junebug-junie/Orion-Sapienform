# Pre-registration: neighbor links (who talks together)

**Date:** 2026-09-20
**Status:** run complete 2026-09-20 — UNVERIFIED (median split collapsed at zero)
**Run report:** `docs/research/preregistration/2026-09-20-heartbeat-neighbor-links-run.md`
**Service:** `orion-heartbeat` (offline replay of live grammar through production `absorb`)
**Not:** the May 2026 H2 recipe (four weeks, Kraskov, vision/planner/social pairs). Those organs are not seated. Heartbeat does not save a per-seat history.

## Question

When two seated organs are busy in the same short window, does the link between *their* neighboring seats rise — or does any busy window heat that link the same amount?

This is the “who talks to whom” test. It is not a consumer. It does not change `/h1`.

## Why these pairs

Live 2-hour census (2026-09-19 dump, 17,492 `atom_emitted` rows):

- `orion-biometrics`: ~7,800
- `orion-bus`: ~5,200
- `orion-cortex-exec`: ~4,000
- `orion-cortex-orch`: ~170
- `orion-hub`: 8

Chat is too quiet. Routing is thin. The only neighbor pairs with enough traffic:

1. biometrics — execution (seats 1–2)
2. execution — bus (seats 2–3)

We do **not** declare these “friends” from last summer. We measure whether *their* co-firing moves *their* seat link more than busy-elsewhere traffic.

Neighbor-only on purpose: absorb walks rightward, so far-apart seats always look weaker. Comparing chat-to-routing would just rediscover chain distance.

## Arms (same production ensemble, fresh vacuum per window)

**Window.** 10 seconds of live `atom_emitted` grammar, last **2 hours**, allowlisted organs only. Empty windows dropped.

**Fallback.** If either pair has fewer than 10 windows in the co-fire cell **or** the elsewhere cell, rerun at **5 seconds**. If still short → **UNVERIFIED**.

**Fresh ensemble per window.** Each window starts from a new `EnsembleSubstrate` (`EnsembleConfig` defaults, `base_seed=1000`). A running chain saturates and stops moving. We are asking what *this* mix of atoms does, not what 17k atoms left behind.

**Link.** Two-seat mutual information in bits, mean across the 8 trajectories:

`I(i:i+1) = S(i) + S(i+1) - S(i,i+1)`

von Neumann entropies from `partial_trace_exact` on the 1-site and 2-site reduced states. Offline only. If a single 2-seat measurement takes more than 2 seconds, stop and call UNVERIFIED — do not swap in cut entropy.

Live-box smoke before this run: 24 mixed absorbs from vacuum, exact 2-site `I ≈ 0.002` bits, ~20 ms. A 10-second window is that size. An absolute 0.05-bit bar would be unreachable on that scale, so the fail line below is scale-adaptive (locked here, not after seeing the live cells).

Decay/reheat is **off** inside a 10-second window (too short to matter; keeps the window a pure absorb mix).

## Cells (median split, computed on the windows in this run)

For a pair `(A, B)` and `other` = all other v0 organs in that window (`hub`, `orch`, and the leftover of {bio, exec, bus}):

- **co-fire:** `n_A ≥ median(n_A)` and `n_B ≥ median(n_B)`
- **elsewhere:** `n_A < median(n_A)` and `n_B < median(n_B)` and `n_other ≥ median(n_other)`

Medians are the medians of those counts over kept windows. Pre-registered as a median split, not a searched threshold.

Record per pair: window counts, mean `I` in each cell, `delta = mean(I|co-fire) - mean(I|elsewhere)`.

## Fail (thermometer)

Both testable pairs fail their threshold:

- If `max(mean I_cofire, mean I_elsewhere) ≥ 0.05`: need `delta ≥ 0.05`.
- If both cell means are `< 0.05`: need `delta ≥ 0.25 * max(mean I_elsewhere, 1e-4)`.

Busy-elsewhere heats the neighbor link as much as those two talking.

## Hold (still an instrument)

At least one testable pair beats its threshold and both cells have ≥ 10 windows.

Write which pair moved. Still no consumer.

## Partial (pre-declared)

- One pair holds, one fails → **mixed**. Name the pair. Do not wire.
- 2-seat measurement too slow or errors → **UNVERIFIED**.
- Either cell < 10 after the 5-second fallback → **UNVERIFIED**.
- `n_A` and `n_B` correlate so hard that the elsewhere cell is empty even at 5 seconds → **UNVERIFIED**, and say so (those two names may be the same traffic).

## What this is not

- Not wiring into mood, memory, AST/HOT, or any cognition loop.
- Not raising χ, not retuning verdict bands, not a chat test.
- Not May H2. Not a friends-and-strangers list.

## Artifact paths

- `/tmp/heartbeat-neighbor-links/report.md`
- `/tmp/heartbeat-neighbor-links/windows.json`
- This document’s commit hash is the pre-reg. The run report must cite it.

# Pre-registration: surprise after a situation change (heartbeat v0)

**Date:** 2026-09-20
**Status:** run complete 2026-09-20 — mixed (start is louder; no session-specific decline)
**Run report:** `docs/research/preregistration/2026-09-20-heartbeat-surprise-run.md`
**Service:** `orion-heartbeat` (offline replay, production `absorb`)
**Not:** the May 2026 H4 formula. There is no forecast of the next boundary state. No free-energy loop. This test asks whether the *reading* settles in a real chat stretch and jumps at the start of one, compared with random stretches of the same shape.

## Question

After Juniper starts talking, does heartbeat’s picture stop thrashing and then stay quieter — more than it does in a random stretch of mesh traffic that is not a chat start?

## What we cannot use

- Charter `KL(p_forecast || p_observed)` — no `p_forecast` exists.
- World-pulse `new_topic` rows — 25 of them in the last day, all stamped within one millisecond of `12:00:16`. That is a batch publish, not 25 situation changes.
- Hub presence — one snapshot row, no history.
- Chat as a 2-hour stream — a handful of hub atoms. Sessions are visible only over days.

## Surprise (disclosed stand-in)

Replay a **running** ensemble (`EnsembleConfig` defaults, `base_seed=1000`). Every 10 absorbed atoms, snapshot the mean 9-cut ratio profile (same vector as the lattice probe).

`surprise_t = L2(profile_t − profile_{t−1})`

The last reading is the guess. That is a change detector, not a mind that predicted the next tick. If one snapshot takes more than 2 seconds → UNVERIFIED.

Decay/reheat follows real inter-event gaps, same rule as the lattice probe. Reheat from live bus graph, or 0.0 if unreachable (recorded).

## Contexts

**Chat session.** Hub `atom_emitted` rows, last **7 days**. A new session starts when the gap since the previous hub atom is **> 30 minutes**. Keep a session if it lasts **≥ 10 minutes** and has **≥ 8** hub atoms. Use the **most recent 12** that qualify (runtime bound). Need **≥ 8** or UNVERIFIED.

**Warmup.** For each kept session, absorb the allowlisted atoms in the 10 minutes *before* the first hub atom, cap 200. Then absorb session atoms from the start, cap 400. Surprise is scored only on session snapshots, not warmup. Warmup exists so we are not measuring “vacuum thermalizes,” which always slopes down.

**Control.** 12 windows of 20 minutes that do not overlap a kept session and sit **≥ 30 minutes** from any kept session’s start or end. Same warmup (10 minutes before the window) and same 200/400 caps. Need **≥ 8** or UNVERIFIED. If fewer exist, UNVERIFIED — do not loosen the gap.

## Scores

For each window (session or control), after warmup:

- Snapshots every 10 session absorbs. Need **≥ 8** surprise points or drop that window.
- `slope` = ordinary least-squares of surprise against snapshot index (0, 1, 2, …).
- `drop` = mean(first 3 surprises) − mean(last 3 surprises). Positive drop = started louder than it ended.

## Fail (this is not situation-tracking)

Either:

1. **Settling is not session-specific.** Median session slope is **not** `< 0`, or fewer than 75% of sessions have slope `< 0`. **Or** the controls settle the same way (median control slope `< 0` **and** ≥ 75% of controls negative). Saturation looking like a story.
2. **No extra jump at chat start.** Median session `drop` does not exceed median control `drop` by **≥ 0.05**.

Charter H4 needed both a within-context decline *and* a shift elevation. Same here. Fail if either clause fails.

## Hold (still an instrument)

Sessions settle (median slope `< 0` and ≥ 75% negative), controls do **not** meet that same pair of facts, **and** median session drop ≥ median control drop + 0.05.

Write the numbers. Still no consumer.

## Partial (pre-declared)

- < 8 sessions or < 8 controls or < 8 surprise points after drops → **UNVERIFIED**
- Both sides settle → **UNVERIFIED: saturation**, not a thermometer call and not a hold
- Sessions settle, drop too small → **mixed**: quiets down, but not more than a random stretch at the start
- Drop holds, settling fails → **mixed**: a bump at chat start, no decline after

## What this is not

Wiring. Raising χ. Retuning verdict bands. Pretending this is the May forecast KL.

## Artifact paths

- `/tmp/heartbeat-surprise/report.md`
- `/tmp/heartbeat-surprise/windows.json`
- This document is the pre-reg. The run report must cite it.

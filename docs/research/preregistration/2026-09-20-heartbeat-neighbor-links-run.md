# Heartbeat neighbor links

Pre-reg: `docs/research/preregistration/2026-09-20-heartbeat-neighbor-links.md`

Generated: 2026-09-20T06:06:17.495795+00:00

- window: 2.0h, raw 17386, routed 17148, 1147 bins of 5s (5s fallback)

## orion-biometrics — orion-cortex-exec (seats 1–2)

co-fire n=1147 I=1.1906; elsewhere n=0 I=0.0000; delta=1.1906

UNVERIFIED: need ≥10 windows in each cell, got cofire=1147 elsewhere=0

median bio=0, exec=0, other=7

## orion-cortex-exec — orion-bus (seats 2–3)

co-fire n=680 I=0.2793; elsewhere n=0 I=0.0000; delta=0.2793

UNVERIFIED: need ≥10 windows in each cell, got cofire=680 elsewhere=0

median exec=0, bus=7, other=0

## Decision

**UNVERIFIED: at least one pair lacked both cells**

thermometer=False relational=False mixed=False

## Plain reading

The test could not decide. In a five-second slice, biometrics and execution are quiet more than half the time, so the “typical amount” was zero. The rule treated zero as “talking together,” so every slice landed in the co-fire pile and the busy-elsewhere pile was empty. That is a hole in the split, not proof they are friends and not proof they are a thermometer.

Do not wire anything. Keep publishing the heartbeat verdict.

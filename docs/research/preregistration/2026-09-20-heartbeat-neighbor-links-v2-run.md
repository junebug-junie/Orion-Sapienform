# Heartbeat neighbor links v2

Pre-reg: `docs/research/preregistration/2026-09-20-heartbeat-neighbor-links-v2.md`

Generated: 2026-09-20T20:28:09.283775+00:00

- window: 2.0h, raw 17574, routed 17377, 1145 bins of 5s, quiet-means-quiet split

## orion-biometrics — orion-cortex-exec (seats 1–2)

co-fire n=69 I=0.8844; elsewhere n=410 I=1.4215; delta=-0.5371

thermometer-like: delta=-0.5371 (need ≥0.0500)

## orion-cortex-exec — orion-bus (seats 2–3)

co-fire n=170 I=1.0115; elsewhere n=127 I=0.6875; delta=0.3240

holds: delta=0.3240 (need ≥0.0500)

## Decision

**relational: execution/bus link rose when those two talked**

thermometer=False relational=True mixed=False

## Plain reading

Quiet now means quiet. Execution and the bus both talking raised their seat-link more than when they were silent and something else (mostly biometrics) was busy. That is the pair that *can* smear, and it still held.

Biometrics and execution went the other way: their link was *higher* when both were silent. That pair cannot call this by itself (pulses do not walk backward). Do not treat that invert as a second headline.

Nothing was wired. Keep publishing the heartbeat verdict.


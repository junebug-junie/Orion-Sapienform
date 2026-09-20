# Heartbeat lattice vs thermometer

Pre-reg: `docs/research/preregistration/2026-09-19-heartbeat-lattice-vs-thermometer.md`

Generated: 2026-09-20T03:23:06.731735+00:00

- window: 2.0h, raw atoms 17492, routed 800, 800-cap yes
- reheat_prob=0.0076 (raw_z=1.1408)

## Arm A (as-is)

mean=0.8260 std=0.0720 bulk=0.8674 verdict=mixed

## Arm B (cyclic organ shuffle)

mean=0.8663 std=0.0724 bulk=0.8612 verdict=mixed

rel_shuffle=1.2381 null=False

## Arm C (kick site 0 × 50)

near=0.4650 far=0.0319 smear=0.06864441169668505 smeared=False

## Decision

**lattice: shuffle moved and kick stayed local**

thermometer=False

## Plain reading

Heartbeat still notices which organ is talking. Swapping seats changed the picture. A poke at the chat seat stayed nearby. Do not treat this as a busyness meter. Do not wire it into cognition. Keep publishing the verdict.

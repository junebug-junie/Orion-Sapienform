# Whole-body structure replay (#3290): keep/drop verdicts

Run 2026-11-24 by `scripts/whole_body_structure_replay.py` (seed 3290). Raw numbers are in
`results.json`, which holds aggregates only (counts and channel names, never content). Every number
below is in that file.

## What was measured

Orion's channels in 5-minute buckets:
- **Body:** 16 machine pressures, plus the bus's own body-telemetry channels (`orion:biometrics*`).
- **Mind:** bus traffic per channel, attention traces, acts per actor, turns, curiosity runs.
- **World:** AC watts. Juniper's messages, Hub attention and camera events were loaded too, but
  none is active in 90% of buckets, so none is a channel. Juniper's messages and Hub attention are
  used to pick quiet nights.

Each channel becomes **surprise**: its departure from its own hour-of-day average over the
previous 14 days, using earlier days only. Each hour, the six structure features are computed over
the last 24 h. Each is compared against 20 shifted copies in which every channel comes from a
different time, keeping each channel's own rhythm and breaking the links between channels.

In the whole-body block, the channels in a shifted copy are at least 19 buckets (95 min) apart. A
link between channels at a lag longer than that survives the shift, so the null is conservative.

Channel rules (mechanical, no hand list):
- A count channel must be active (nonzero) in at least 90% of buckets.
- A level channel must be present in at least 90% of buckets.
- Channels that are near-exact copies (|r| ≥ 0.95, sign-aligned) are merged into one channel.

**Why the copy merge.** One bus message often lights several channels at once: a request and its
reply, or the three `biometrics` induction, sample and summary channels. Counted apart, they read
as coupling even on a quiet night. The merge took the whole-body block from 67 channels to 55.

Two blocks:
- **Whole body:** 2026-11-16 04:00 to 11-24 04:00 UTC, 8 days (the span with bus history plus 3
  days of profile). 168 windows, 55 channels: 18 body, 36 mind, 1 world.
- **Body only:** 30 days, 624 windows, 16 channels.

## Gate (spec acceptance check 2)

Gate rules:
- **Beats null:** the feature is beyond every shifted copy in at least 50% of windows.
- **Rest:** on a quiet night it comes within 1 null SD of the null. A quiet night is 00:00-06:00
  Mountain with no Juniper message and no Hub attention.
- **Copy:** |Spearman| > 0.9 with a kept feature means the feature is dropped.

Drift is not scored at rest. Successive quiet nights are about 24 h apart, while its null compares
windows 1 h apart.

### Whole body (8 days; 3 of 8 nights were quiet)

| Feature | Plain meaning | Beats null | Rest: quiet nights within 1 SD | Verdict |
|---|---|---|---|---|
| coupling | how much the channels move as one | 100% (median z 12.3) | 0 of 3 (closest z 8.3) | **hold: fails rest** |
| redundancy_synergy | parts repeat each other (+) or hold information only together (−) | 98% (z 5.4, always redundant) | 0 of 3 | **hold: fails rest** |
| body_mind | how tied mind traffic is to the body | 98% (z 4.8) | 2 of 3 | **keep, provisional** (see below) |
| modes_effective | how many separate things are going on | 100% | 0 of 3 | **drop: copy of coupling** (ρ 0.93) |
| leading_share | how dominant the top mode is | 33% | 1 of 3 | **drop** |
| structure_drift | how fast the pattern changes | 24% | not scored | **drop** |

Leading modes are **not** stable from day to day. The median matched similarity for ranks 1-4 is
0.56, 0.63, 0.47 and 0.51, against the spec bar of 0.7.

**Coupling cannot read "not coupled".** On all 3 quiet nights, Orion's channels stay far more
coupled than chance. Two explanations fit, and this replay cannot tell them apart:
- **Orion never rests.** Autonomous loops (curiosity, dreams, world pulse) keep the organs working
  together while Juniper is away.
- **Pipeline stages stay coupled.** A pipeline's stages (embedding → GPU pool → introspection,
  |r| 0.83-0.88) are not copies, but they are still one process. Their coupling is a floor that no
  night removes.

Telling them apart needs a real idle stretch, or a split of coupling into within-pipeline and
between-pipeline parts.

**Body-mind coupling is provisional.** It passes the gate, but its rest test rests on only 3 quiet
nights. Its strongest pair is the bus's body telemetry ↔ the spark signal (Layer 1: r 0.78). The
spark signal may be computed from the same body telemetry. That would make it a second reading of
one source, not the mind reacting to the body. The metric-gate independence check needs that causal
chain traced in code before this feature is wired anywhere. Until then it is keep-for-display only.

### Body only (30 days; 15 of 27 nights were quiet)

| Feature | Beats null | Rest: quiet nights within 1 SD | Verdict |
|---|---|---|---|
| coupling | 54% (median z 2.4) | 7 of 15 | **keep** |
| modes_effective | 35% | 10 of 15 | **drop** (and a copy of coupling, ρ 0.94) |
| leading_share | 18% | 9 of 15 | **drop** |
| redundancy_synergy | 18% | 9 of 15 | **drop** |
| structure_drift | 17% | not scored | **drop** |

Body modes are also not stable from day to day: 0.66 and 0.67 for ranks 1-2, just under 0.7.

### Layer 1 (spec acceptance check 3)

1,485 channel pairs were scored with lagged correlation and the time-shift null. **46 of the top 50
links are not body hardware: pass.** The top links are pipeline stages:
- GPU actuate request ↔ result (r 0.91)
- embedding ↔ introspection candidate (0.88)
- equilibrium snapshot ↔ spark signal (0.65)
- embedding ↔ GPU pool event (0.83)

## Organs for the integration rung (#3296) and the field demon's first read

The integration rung needs **stable organs**. No mode passed the day-to-day bar, so there are none
yet. Φ over the partition lattice has nothing stable to cut between at this resolution.

As a labelled **preview**, the field demon ran on the top 6 whole-body modes. They are unstable and
fitted on the whole span, so this is not a verdict.

- **Skill** over the better of "the usual at today's level" and "nothing changes": 20% at 5 min,
  9% at 15 min, 4% at 25 min, 1% at 45 min.
- **Horizon:** skill stays above 0 in every bootstrap all the way to the 2 h ceiling. So "the first
  step where skill reaches 0" saturates on slow modes, even though the useful foresight is gone by
  about 45 min. The skill half-life is about 10-15 min. This is a definition problem in the field
  demon spec, raised on #3296.
- **Full cut:** each mode forecasts only from its own past. That costs 0.044 of skill at 5 min (22%
  of the whole model's 0.203), and 0.005 at 60 min. That is above the null mean by 0.058 at 5 min,
  and above every shifted copy at every step. Links across modes carry real foresight.
- **Horizon cost:** 0 minutes, because both models hit the ceiling. On this data, the cost curve is
  the informative readout, not the horizon cost in minutes.

### Bug found in the field demon (fixed in this PR)

The demon's "clock" baseline forecast a raw residual of 0, the 14-day time-of-day profile. The
model's own long-range forecast is the last day's average. When today's level sat off the 14-day
profile, the model beat the clock at every horizon with no dynamics at all. Before the fix, this
replay read a flat ~6% skill out to 2 h. The clock baseline is now the training-window mean ("the
usual, at today's level"). The regression test is `test_a_level_offset_is_not_foresight`.

## Verdict summary

- **Keep:**
  - body coupling (30 days)
  - Layer 1 (46 of 50 non-body)
  - whole-body body_mind, provisional, display only, until its spark-signal link is traced
- **Hold (beats null, fails rest):** whole-body coupling and O-information.
- **Drop:**
  - effective modes (a copy of coupling)
  - leading share
  - structure drift
  - every body-only feature except coupling
- **Organs for Φ:** none stable yet (best rank 0.63 against the 0.7 bar).

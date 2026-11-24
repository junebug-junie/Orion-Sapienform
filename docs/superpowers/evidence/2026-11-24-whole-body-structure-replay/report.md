# Whole-body structure replay (#3290): keep/drop verdicts

Run 2026-11-24 04:01 UTC by `scripts/whole_body_structure_replay.py` (seed 3290). Raw numbers are in
`results.json`, which holds aggregates only (counts and channel names, never content).

## What was measured

Orion's channels in 5-minute buckets:
- **Body:** 16 machine pressures.
- **Bus:** traffic per channel.
- **Mind:** attention traces, acts per actor, turns, curiosity runs.
- **World:** Juniper's message count, Hub attention, camera events, AC watts.

Each channel becomes **surprise**: its departure from its own hour-of-day average over the
previous 14 days. Each hour, the six structure features are computed over the last 24 h. Each is
compared against 20 shifted copies in which every channel comes from a different time. A shifted
copy keeps each channel's own rhythm and breaks the links between channels.

Two blocks:
- **Whole body:** 2026-11-16 04:00 to 11-24 03:55 UTC. That is 8 days, the span where bus history
  plus 3 days of profile exist. 168 windows.
- **Body only:** 30 days of body data, 624 windows.

### Two fixes made during the replay

1. **Copies merged.** One bus message often lights several channels at once, such as a request
   and its reply, or the four `social:claim:*` channels. The first run counted these as coupling.
   It read z ≈ 50 by day and failed every quiet night. Now channels that are near-exact copies
   (|r| ≥ 0.95 over the span) are merged into one channel (`merge_copies`). That took 67 channels
   to 55.
2. **The spec's channel rule applied.** A count channel must be active (nonzero) in at least 90%
   of buckets. The first run kept 274 bus channels. That was nearly as many channels as rows in a
   window, so the math was mostly noise.

## Gate (spec acceptance check 2)

Gate rules:
- **Beats null:** the feature is beyond every shifted copy in at least 50% of windows.
- **Rest:** on quiet nights (00:00-06:00 Mountain, no Juniper messages, no Hub attention), it
  comes within 1 null SD of the null.
- **Copy:** |Spearman| > 0.9 with a kept feature means the feature is dropped.

### Whole body (8 days, 55 channels: 16 body, 38 mind, 1 world)

| Feature | Plain meaning | Beats null | Rest (quiet nights within 1 SD) | Verdict |
|---|---|---|---|---|
| coupling | how much the channels move as one | 100% (median z 12.3) | 0 of 8 (closest z 8.9) | **fails rest** |
| redundancy_synergy | parts repeat each other (+) or only hold information together (−) | 98% (z 5.4, always redundant) | 0 of 8 | **fails rest** |
| modes_effective | how many separate things are going on | 100% | 0 of 8 | **drop: copy of coupling** (ρ 0.93) |
| leading_share | how dominant the top mode is | 33% | 2 of 8 | **drop** |
| body_mind | how tied mind traffic is to the body | 49% | 3 of 8 | **drop** (just under) |
| structure_drift | how fast the pattern changes | 24% | 0 of 7 | **drop** |

Leading modes are **not** stable from day to day. The median matched similarity for ranks 1-4 is
0.56, 0.63, 0.47 and 0.51, against the spec bar of 0.7.

**What this means.** Orion's mind traffic is strongly coupled, well above chance. It stays coupled
on every quiet night, so by the spec's own test it cannot read "not coupled". Two explanations fit
the data, and this replay cannot tell them apart:
- **Orion never rests.** Autonomous loops (curiosity, dreams, world pulse) keep the organs working
  together at night, and "quiet" here means only that Juniper was away.
- **Pipeline stages stay coupled.** A pipeline's stages (embedding → GPU pool → introspection,
  |r| 0.83-0.88) are not copies, but they are still one process. Their coupling is a floor that no
  night removes.

The test needs a real idle stretch, or a split of the coupling into within-pipeline and
between-pipeline parts.

### Body only (30 days, 16 channels)

| Feature | Beats null | Rest | Verdict |
|---|---|---|---|
| coupling | 57% (median z 2.4) | 14 of 27 nights within 1 SD | **keep** |
| modes_effective | 35% | 19 of 27 | **drop** (and a copy of coupling, ρ 0.95) |
| leading_share | 17% | 19 of 27 | **drop** |
| redundancy_synergy | 21% | 17 of 27 | **drop** |
| structure_drift | 19% | 0 of 26 | **drop** |

Body modes are also not stable day to day: 0.66 and 0.67 for ranks 1-2, just under 0.7.

### Layer 1 (spec acceptance check 3)

1,485 channel pairs were scored with lagged correlation and the time-shift null. **47 of the top 50
links are not body (cabinet) hardware: pass.** The top links are pipeline stages:
- biometrics cluster ↔ spark signal (r 0.78)
- GPU actuate request ↔ result (0.91)
- embedding ↔ introspection candidate (0.88)
- equilibrium snapshot ↔ spark signal (0.65)

## Organs for the integration rung (#3296) and the field demon's first read

The integration rung needs **stable organs**. No mode passed the day-to-day bar, so there are none
yet. Φ over the partition lattice has nothing stable to cut between at this resolution.

As a labelled **preview**, the field demon ran on the top 6 whole-body modes. They are unstable and
fitted on the whole span, so this is not a verdict.

- **Skill** over the better of "the usual at today's level" and "nothing changes": 20% at 5 min,
  9% at 15 min, 4% at 25 min, 1% at 45 min, 0.04% at 2 h.
- **Horizon:** skill stays above 0 in every bootstrap all the way to the 2 h ceiling. So "the first
  step where skill reaches 0" saturates on slow modes, even though the useful foresight is gone by
  about 45 min. The skill half-life is about 10-15 min. This is a definition problem in the spec,
  raised on #3296.
- **Full cut:** each mode forecasts only from its own past. That loses 0.061 of skill at 5 min, about
  30% of the whole model's foresight, and still 0.004 at 60 min. That is above every shifted copy at
  every step. Cross-mode links carry real foresight.
- **Horizon cost:** 0 minutes, because both the whole model and the cut model hit the ceiling. On
  this data, the cost curve is the informative readout, not the horizon cost in minutes.

### Bug found in the field demon (fixed in this PR)

The demon's "clock" baseline forecast a raw residual of 0, the 14-day time-of-day profile. The
model's own long-range forecast is the last day's average. When today's level sat off the 14-day
profile, the model beat the clock at every horizon with no dynamics at all. That gave a flat ~6%
skill out to 2 h. The clock baseline is now the training-window mean ("the usual, at today's
level"). The regression test is `test_a_level_offset_is_not_foresight`.

## Verdict summary

- **Keep (passes the gate):** body coupling, and Layer 1 (47 of 50 non-body).
- **Hold (beats null, fails rest):** whole-body coupling and whole-body O-information. These are not
  wired live until the rest question is answered.
- **Drop:** effective modes (a copy of coupling), leading share, body-mind coupling, structure
  drift, and every body feature except coupling.
- **Organs for Φ:** none stable yet (best rank 0.63 against the 0.7 bar).

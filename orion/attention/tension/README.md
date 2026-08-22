# Field deviation tension sensing

Turns Orion's live interoceptive field into a continuously-varying, **scale-free** tension
signal and ranks it — no drive taxonomy, no categories, no hand-authored cross-channel weights.

**Read-only.** Nothing here publishes to the bus, registers a schema, feeds a prompt, or acts.

Design + live results: [`docs/superpowers/specs/2026-08-14-field-deviation-tension-sensing-design.md`](../../../docs/superpowers/specs/2026-08-14-field-deviation-tension-sensing-design.md)

## Run it

```bash
# From the host, against the live DB:
POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \
  make field-tension-report

make field-tension-report HOURS=72      # wider window
make field-tension-report Z=3.5         # tighter admission threshold
make field-tension-report LIMIT=10000   # newest N ticks only, fast
make field-tension-report JSON=1        # machine-readable
```

## What the numbers mean

### Admission

| Field | Meaning |
|---|---|
| `admission_rate` | Fraction of ticks that admitted **any** tension. The number to beat is **0.064%** — the drives-era rate (284 of 444,943), which is why every taxonomy built on it failed. Currently ~48%. |
| `mean_admissions_per_admitting_tick` | How many (node, channel) deviations cleared the gate on a tick that admitted anything. |

### Rank discrimination

| Field | Meaning |
|---|---|
| `top1_share` | Share of admitting ticks won by the single most frequent winner. **→1.0 is a monoculture** — the failure that killed the old drive economy at 96% `relational`. Currently ~50%. |
| `distinct_winners` | How many targets ever win. |
| `scorer_disagreement_rate` | Fraction of ticks where two channels' own top picks differ. Low disagreement means the channels are not adding independent information. |

### Channels

`never_admitted` is **not** a health verdict. Most entries are channels in the field
digester's `NODE_DECAY_CHANNELS` sitting at their designed resting state. Use the liveness
fields below instead.

### Liveness — two different questions, do not conflate them

| Field | Question it answers |
|---|---|
| `liveness_counts` | *"Is this channel telling me anything?"* — from `orion.field.channel_glossary.classify_channel_series` (`live` / `quiet` / `dead` / `ratchet_suspect` / `never_produced`). Reused, not re-derived. |
| `producer_liveness_counts` | *"Is anyone still **writing** this?"* — from [`liveness.py`](liveness.py) (`refreshed` / `silent_producer` / `at_floor` / `insufficient_data`). |

**`silent_producer` is the finding that matters.** It exists because of a real defect a
per-channel ratio check structurally cannot see: when `services/orion-biometrics` goes quiet,
every input to `resource_pressure` decays toward 0, the dimension reads *calm*, and because
`config/feedback/feedback_policy.v1.yaml` lists `resource_pressure: decrease` under
`positive_delta_channels`, **an in-flight action gets credited with a positive outcome for a
producer outage.** The discriminator: the decay loop can only make a value *smaller*, so a
series that is monotonically non-increasing across a long window received no producer writes
at all.

**`at_floor` is a blind spot, not a clean bill of health.** Once a channel has finished
decaying, its shape can no longer distinguish a dead producer from a calm one. As of
2026-08-14 that is **~90 of 149 series** — the majority. Closing that gap needs producer-side
heartbeats, not better series inference.

### Data quality

| Field | Meaning |
|---|---|
| `decay_undesigned` / `pinned_undesigned` | **Findings.** A channel decaying that is *not* in `NODE_DECAY_CHANNELS` — the `prediction_error` shape CLAUDE.md documents. Currently zero. |
| `decay_by_design_count` / `pinned_by_design_count` | **Not findings.** Channels decaying toward rest exactly as designed. Reported for visibility only. |
| `subnormal_distinct_series` | Distinct series, **not** the observation count — the observation count is this × tick count and is not independent evidence of breadth. |

## Reuse the pieces

Everything is pure and importable; none of it needs the measurement script.

```python
from orion.attention.tension import (
    DeviationGate,             # EWMA baseline + z-threshold admission
    FieldTensionCompetition,   # gate + Borda ranking over field ticks
    classify_producer_liveness,
    geometric_decay_ratio,
    load_direction_map,
)

comp = FieldTensionCompetition()
result = comp.observe_tick(field_json)   # chronological order matters
if result.any_admitted:
    print(result.borda.winner, result.borda.ranking)
```

`observe_tick` returns `borda=None` on a quiet tick — "nothing is happening" is a real
representable state, not something you infer from small numbers.

## Design invariants worth not breaking

- **Never combine channels with weights.** There is no exchange rate between thermal degrees
  and memory fraction. Combination is rank-space only, via
  [`orion/attention/rank_aggregation.py`](../rank_aggregation.py).
- **Every threshold is relative, never absolute.** An absolute `sigma_floor` shipped once and
  silently made small-variance channels structurally unable to admit — `memory_pressure`
  admitted 0 of 10,528 ticks while varying continuously.
- **Polarity is derived, not listed here.** `DirectionMap` seeds "down is worse" from
  `orion.field.pressure.HIGHER_IS_BETTER_CHANNELS` and raises if the YAML contradicts it. A
  hand-written copy shipped once as a strict subset, missing two channels.
- **Decaying ≠ broken.** Check `NODE_DECAY_CHANNELS` before calling any decay a defect. A
  first version of this instrument had a 100% false-positive rate for skipping that.

## Tuning surface

Three global scalars and one sign bit per channel. **No cross-channel exchange rates** — that
is the property that matters.

| Knob | Notes |
|---|---|
| `z_threshold` | Single objective: admission rate. Monotonic — 1.5→49.9%, 8.0→6.5%. |
| `alpha` | EWMA memory; derivable from channel autocorrelation time. |
| `relative_sigma_floor` | Fraction of the channel's own \|mean\|. Must stay relative. |
| `worse` direction | One bit per channel; falsifiable (get it backwards and the metric reads inverted). |

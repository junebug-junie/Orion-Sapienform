# Level-aware significance — wiring the unused half of the regime detector

Status: DESIGN ONLY. Not implemented. Per CLAUDE.md's proposal-mode rule for
cognition-affecting changes, this is presented for direction before any code.

## Arsonist summary

Orion's only significance signal today (`deviation_pressure`) is a change-detector. Its own
metric-gate writeup says so: it fires on deviation from an *adapted* baseline, which means a
channel that has been steadily bad for hours re-centers its baseline and reads calm. Juniper
named this precisely: *"never hits a calm and that is good; looks peaceful but is running
high load and that is steady state; is at some decayed floor because the process isn't
running by design at that moment."*

The fix already exists in this repo and is unused. `orion/field/regime.py::channel_regime()`
(PR #1622/#1633, live since before this arc) computes exactly the missing axis — level and
dispersion as SEPARATE readings, composed into regime labels `loaded_steady`,
`loaded_volatile`, `calm`, `quiet_volatile`, `pinned_max`/`pinned_min`. `loaded_steady` is
*literally* "looks peaceful but running high load." It has real live-validated thresholds
(`LOADED_LEVEL=0.70`, checked against 208 real windows, documented as a convention with no
natural gap rather than a fake discovery). **It has zero consumers outside one Hub debug
panel.** Never wired into attention, proposals, or dispatch. This design wires it in.

## Current architecture

- `channel_regime(channel, values, window_seconds, baseline=None, updated_at=None,
  window_start=None) -> ChannelRegime` — pure function. Takes an explicit **batch window**
  of values (a `list[float]`) plus an optional longer baseline list for the relative
  readings (`level_percentile`, `drift`, `dispersion_ratio`). This is architecturally
  different from `DeviationGate`, which is incremental (one EWMA update per observation,
  no stored window).
- Only real consumer: `services/orion-hub/scripts/field_channel_glossary_routes.py`'s
  `/health` endpoint, which pulls 1/6/24h of `substrate_field_state` per HTTP request and
  computes `channel_regime()` fresh, per request, for all 38 raw channels. Nothing persists
  the result; it exists only for the duration of that one response.
- `orion/field/credit_integrity.py` imports only the internal `_refresh_from_timestamps`
  helper, not `channel_regime()` itself — confirmed not a real second consumer.
- `orion.attention.rank_aggregation.aggregate_borda` — the existing, proven,
  no-hand-tuned-weights mechanism that already solved "combine N heterogeneous per-channel
  ballots into one ranking" for `deviation_pressure`. Real candidate for reuse here with a
  different per-channel vote definition.

## Missing questions (the reason this isn't code yet)

1. **Where does this run, and how often?** `channel_regime()` needs a *window* of history
   (`statistics.median`/`pstdev` over N samples), not a single incremental update like
   `DeviationGate`. Field-digester's hot loop runs every ~2s; recomputing a real window
   from Postgres inside that loop either means (a) a new incremental rolling-buffer data
   structure (real new state, real new failure surface) or (b) a much slower cadence
   (every 30-60s?) as a side-process, not inside the tick loop at all. This is the single
   decision that shapes everything else and I don't have a strong answer yet.
2. **Scope: all 38 raw channels × all nodes, or a subset?** All-channels is
   `~4 nodes x 38 = ~150` regime computations per cycle — cheap per computation, but the
   question of whether ALL of them are meaningful competition inputs (vs. noise) is open.
3. **What baseline window for a live producer?** The Hub panel lets an operator pick
   1/6/24h. A live producer needs one fixed, disclosed window — and that choice needs its
   own real live-data check (does `loaded_steady` actually occur at a genuine,
   non-degenerate rate at that window size, for real channels, not just in theory).
4. **Combination mechanism.** Proposed: reuse `aggregate_borda` exactly as `deviation_
   pressure` does — each channel ranks nodes by `pressure_equivalent_level` (or votes only
   when in a `loaded_*` regime), same "scorers rank targets, no cross-scorer exchange rate"
   shape. Not yet validated against real data the way the tension package's Borda use was.
5. **Independence from `deviation_pressure`, checked or assumed?** Expected to be low
   correlation (they answer different questions), but "expected" isn't "checked" — CLAUDE.md
   0A's independence-check item needs a real number here before this ships, not an
   assumption carried over from a different metric's clean bill.

## Proposed schema / API changes (sketch — not final until Q1-Q4 above are answered)

- A new `PRESSURE_DIMENSIONS` entry (working name `sustained_load_pressure`), with its own
  **derived** variance floor — not borrowed from `deviation_pressure`'s, per this repo's own
  recorded lesson that borrowed calibrated constants silently re-break across domains.
- New `FieldStateV1` field(s) to persist the computed value each cycle, mirroring the
  `tension_deviation_pressure` pattern, plus a winner-identity field if Borda reuse is
  confirmed.
- A new producer — shape depends entirely on the answer to Missing Question 1. Options on
  the table, not decided: a slower periodic task inside `orion-field-digester`; a
  standalone small script/service on its own cadence; or computed on-demand by whichever
  consumer needs it (closest to how the Hub panel already works, cheapest to build, but
  means no live persisted history for anything else to read).

## Files likely to touch (once the above is resolved)

`orion/field/regime.py` (reused, probably unchanged), a new producer (location TBD),
`orion/schemas/field_state.py`, `orion/field/pressure.py`, `orion/proposals/scoring.py`,
and — only in a LATER patch, matching this arc's own precedent — `config/proposals/
proposal_policy.v1.yaml` for actual action wiring.

## Non-goals (for a first patch, matching this arc's own established pattern)

- **No consumer/action wiring in the same patch.** PR #1699 shipped sensing only and PR
  #1701/outreach wired action to it two PRs later, only after the sensing layer had real
  live-data proof. Same staging here: a sensing-only patch first, validated against real
  data, before anything acts on it.
- **No fusion with `deviation_pressure` into one scalar.** Stays a separate,
  independently-competing `PRESSURE_DIMENSIONS` entry — the combination-without-hand-tuning
  answer here is "let them compete independently in the arena that already does this," not
  "invent a weighted formula."
- **No new leaky-integrator/accumulator state**, same reasoning as the outreach design.

## Acceptance checks (for whichever patch actually gets built)

1. Real live-data metric gate on the chosen aggregate: does `loaded_steady` (or whatever the
   final vocabulary is) occur at a real, non-degenerate rate; is genuine calm reachable; is
   it measurably independent of `deviation_pressure` (a real correlation check, not an
   assumption).
2. If Borda reuse is confirmed: a scale-freedom test mirroring the tension package's own
   (monotonic rescaling of one channel must not change the ranking), same rigor already
   proven out.
3. A blast-radius report in the same style as the sensing-layer spec's own.

## Recommended next patch

Investigate Missing Question 1 first — specifically, whether `orion-field-digester` already
has (or could cheaply gain) a slower periodic-task lane separate from its 2s hot loop
(worth checking `services/orion-field-digester/app/worker.py` for an existing pattern before
assuming one needs to be built), since that answer determines whether this is a small
addition to an existing mechanism or a genuinely new one. Then a sensing-only patch,
metric-gated against real data, with zero action wiring — same shape as PR #1699's own first
half.

## Exact question for Juniper

Given the two real open decisions above (Q1: where/how often this runs; Q4/Q5: Borda reuse
and the independence check) — go ahead and investigate Q1 now, or do you want to weigh in on
the cadence/architecture call first?

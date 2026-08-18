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

1. **Where does this run, and how often? ANSWERED, 2026-08-18.** `FieldDigesterWorker`
   (`services/orion-field-digester/app/worker.py`) already runs FIVE independent asyncio
   loops off one `start()`, each on its own interval, each wrapping its tick in
   `asyncio.to_thread` + a broad `except Exception: logger.exception(...)`: `_poll_loop`
   (the 2s hot loop), `_prune_loop`, `_health_loop`, `_causal_geometry_producer_loop`
   (hourly), `_anomaly_loop`. A sixth loop on the same pattern (`_significance_loop`, its
   own `field_significance_check_interval_sec`) is not a new architecture, it's the
   established idiom, already proven safe five times over.
   More specifically, **the exact "new rolling-buffer data structure" option (a) worried
   about is not new either** — `FieldChannelAnomalyScorer` (`app/anomaly_scorer.py`) already
   does precisely this: `append_row()` is called cheaply from the hot loop on every tick
   (the row is already computed there for the corpus sink), pushed into a bounded
   `deque(maxlen=window_size + margin)`; a *separate*, slower `_anomaly_loop` timer reads
   that buffer and does the expensive computation. Same shape works here directly: append
   the same per-tick `channel_pressures` row into a second rolling buffer, and let
   `_significance_loop` compute `channel_regime()` per channel from it on a slow cadence —
   zero new DB round-trips in the hot loop, zero new failure-handling pattern.
   One real wrinkle: today `channel_pressures` is only computed when `_FIELD_CHANNEL_SINK.
   enabled or self._anomaly_scorer is not None` (`worker.py` `_tick()`); wiring in
   significance means widening that `or` to include it too — a small, disclosed, real cost
   (one extra `collect_field_channel_pressures()` call per hot tick when only significance
   is enabled), not a hidden one.
2. **Scope: all 38 raw channels × all nodes, or a subset? PARTIALLY ANSWERED, 2026-08-18.**
   Real live distribution at `hours=1` (1,419 real rows, 2026-08-18): of 39 channels,
   14 `quiet_volatile`, 11 `pinned_min`, 10 `no_new_input`, 2 `calm`, 1 `loaded_volatile`,
   1 `loaded_steady`. Most channels sit in structurally uninteresting states
   (`pinned_min`/`no_new_input`) most of the time — real evidence for scoping the Borda
   vote (Q4) to channels currently in a `loaded_*`/`calm` regime, mirroring how `deviation_
   pressure` already scopes its own vote, rather than forcing all ~150 channel×node
   combinations to cast a ballot every cycle regardless of whether they carry information.
3. **What baseline window for a live producer? PARTIALLY ANSWERED, 2026-08-18 — plus one
   unrelated bug found along the way.** Bug: Hub's `/api/field-channel-glossary/health`
   panel offers 1/6/24h, but the query is capped at `row_cap=6000` rows
   (`field_channel_glossary_routes.py`) — at the live ~2.5s cadence that's **~3.3 hours**.
   Confirmed live: `hours=6` and `hours=24` both returned `row_count=6000,
   truncated=true` and IDENTICAL regime-label distributions across all 39 channels — the
   panel's 6h/24h options are silently the same effective window today, not what their
   labels claim. Out of scope to fix here (a debug-panel-only issue, not this design), but
   worth its own follow-up ticket.
   For the actual question: ran `channel_regime()` directly against a real 15-minute window
   (350 real rows, no baseline) — the timescale a live significance producer would plausibly
   use, not the debug panel's hour-scale presets. Result was non-degenerate and, more
   importantly, demonstrated real independence between level and dispersion on live data:
   `disk_capacity_pressure` (level=0.7655, dispersion=0.00079) read `loaded_steady` —
   literally "looks peaceful but running high load", Juniper's own example — while `power_
   pressure` at a similar level (0.868) but much higher dispersion (0.146) read `loaded_
   volatile`, and `memory_pressure` read genuine `calm`. Level and dispersion are
   demonstrably NOT the same axis on real data at this timescale, which is the entire
   premise this design rests on. Not yet answered: whether 15 minutes specifically (vs.
   10/20/30) is the right final choice, or what baseline window (for `level_percentile`/
   `dispersion_ratio`/`drift`) pairs with it — that still needs its own pass once a
   producer is actually being built, same as `MIN_RUN_LENGTH` was tuned from real replay
   data rather than picked in the abstract.
4. **Combination mechanism.** Proposed: reuse `aggregate_borda` exactly as `deviation_
   pressure` does — each channel ranks nodes by `pressure_equivalent_level` (or votes only
   when in a `loaded_*` regime), same "scorers rank targets, no cross-scorer exchange rate"
   shape. Not yet validated against real data the way the tension package's Borda use was.
5. **Independence from `deviation_pressure`, checked or assumed?** Expected to be low
   correlation (they answer different questions), but "expected" isn't "checked" — CLAUDE.md
   0A's independence-check item needs a real number here before this ships, not an
   assumption carried over from a different metric's clean bill. Not yet run: the 15-minute
   spot-check above didn't compute this correlation, only regime labels.

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

Q1 is answered: a sixth `worker.py` loop (`_significance_loop`) on `FieldChannelAnomalyScorer`'s
exact append-to-a-bounded-deque-then-slow-read pattern, no new architecture. Q2 and Q3 have
real, live, non-degenerate evidence behind them, though the final window size and baseline
pairing still need their own tuning pass once a producer actually exists. Remaining before a
sensing-only patch (same staged shape as PR #1699's own first half — no consumer/action
wiring in the same patch):

1. Build the `_significance_loop`/rolling-buffer producer per Q1's answer, computing
   `channel_regime()` per channel scoped to `loaded_*`/`calm` regimes per Q2's evidence.
2. Pick and disclose a final window (15 minutes is a reasonable starting point per the
   live check above, not yet a locked answer) and a baseline window for the relative axes.
3. Run the real independence check against `deviation_pressure` (Q5) and the Borda
   scale-freedom test (Q4) before anything downstream reads this.
4. New `PRESSURE_DIMENSIONS` entry + `FieldStateV1` field(s), metric-gated against real
   data per CLAUDE.md 0A, same discipline as `tension_deviation_pressure`.

## Exact question for Juniper

The architecture/cadence call (Q1) is resolved with real evidence, not a guess — reuse the
anomaly scorer's rolling-buffer pattern. Remaining before implementation: the Borda-reuse
validation (Q4) and the independence check against `deviation_pressure` (Q5), both of which
need to be run against real data as part of building the sensing-only patch itself, not
answerable from investigation alone. Ready to build that sensing-only patch, or hold for
explicit go-ahead per CLAUDE.md's proposal-mode rule for cognition-loop changes?

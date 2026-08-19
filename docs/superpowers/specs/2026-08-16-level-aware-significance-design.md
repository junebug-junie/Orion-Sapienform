# Level-aware significance — wiring the unused half of the regime detector

Status: SENSING-ONLY PATCH SHIPPED (2026-08-18), per this doc's own "Recommended
next patch" and "Non-goals" sections below. `orion.field.significance` computes
`sustained_load_pressure`, wired as a 6th `PRESSURE_DIMENSIONS` entry with a
real-data-derived variance floor (`orion/proposals/scoring.py`'s own metric-gate
comment has the full writeup, 24h/1,395-point/34,316-real-row replay). No
consumer/action wiring (no target-binding, no policy-template scoring on it) --
that stays deferred, same staged shape PR #1699/#1701's own tension package
used. Q1 (where/how often this runs) turned out to have a BETTER answer than
either option this doc originally proposed -- see "What changed from this
doc's original proposal" near the end.

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

1. Build the `_significance_loop`/rolling-buffer producer per Q1's answer, computing
   `channel_regime()` per channel scoped to `loaded_*`/`calm` regimes per Q2's evidence.
2. Pick and disclose a final window (15 minutes is a reasonable starting point per the
   live check above, not yet a locked answer) and a baseline window for the relative axes.
3. Run the real independence check against `deviation_pressure` (Q5) and the Borda
   scale-freedom test (Q4) before anything downstream reads this.
4. New `PRESSURE_DIMENSIONS` entry + `FieldStateV1` field(s), metric-gated against real
   data per CLAUDE.md 0A, same discipline as `tension_deviation_pressure`.

## What changed from this doc's original proposal (2026-08-18, during implementation)

Q1's answer above (a 6th `worker.py` loop reusing `FieldChannelAnomalyScorer`'s
rolling-buffer pattern) was correct that a proven pattern exists, but wrong about which one.
Reading `services/orion-field-digester/app/digestion/tension.py` closely (the ALREADY-SHIPPED
sibling producer, not just the anomaly scorer) surfaced a better answer:

- `substrate_field_state` is APPEND-ONLY tick history (`app/store.py::save_field` INSERTs a
  fresh row with a new `tick_id` every hot tick, not an UPDATE of one mutable "current" row).
  A separate slow-cadence loop calling its own `save_field()` would either insert a wasteful
  near-duplicate near-empty tick, or race the hot loop over which row is "latest".
- `channel_regime()` needs a real window of raw values (`statistics.median`/`pstdev`), unlike
  `tension.py`'s EWMA baseline (which is genuinely O(1)-updatable and needs no buffer at all)
  — but that window does NOT need an in-memory structure either. Postgres already has the
  history: `FieldDigesterStore.load_recent_field_json()` (new) queries the last
  `window_seconds` directly, same shape `tension_outreach_trigger.py`/`field_channel_
  glossary_routes.py` already use for exactly this kind of read.
- So the shipped design is simpler than either option this doc originally weighed: `orion/
  field/significance.py::compute_tick()` runs INLINE in the hot tick (`app/tensor/update_
  rules.py::run_digestion_tick`, right after `update_tension_pressure`, before `update_
  dimension_precision_baseline` — same ordering requirement tension already has, for the
  same reason), throttled by a persisted `sustained_load_computed_at` timestamp (round-
  tripped through Postgres every tick, same as `tension_baseline_mu` already is) rather than
  held on the worker process instance. No new async loop. No new worker-instance state. No
  in-memory buffer to reset on restart.

One real production bug was found and fixed along the way, in a DIFFERENT, already-merged
module: `tension_outreach_trigger.py`'s own lookback query used `make_interval(mins =>
:mins)` with a float, which raises `UndefinedFunction` in real Postgres (`mins`/`hours` are
integer-typed there) — silently swallowed by that module's own broad except, so the
tension-driven outreach trigger had never actually been able to fire since PR #1707 deployed.
Shipped as its own hotfix PR (#1715), not bundled here — see that PR for the full account.
Caught only by running this patch's own analysis script against real Postgres and hitting the
identical bug in a query written the same way.

## Acceptance checks (actual, 2026-08-18)

1. Real live-data metric gate: 24h replay (1,395 points, 34,316 real rows,
   `scripts/analysis/measure_sustained_load_pressure.py`) — 95.8% nonzero, 96 distinct
   values, population variance 2.337162e-02, genuine rest reachable (min=0.0, 58 points),
   Pearson r vs `deviation_pressure` = -0.0313 (genuinely independent). Full writeup in
   `orion/proposals/scoring.py`'s `PRESSURE_DIMENSIONS` comment.
2. `orion/field/significance.py` — 8 hand-computed tests (`tests/test_field_significance.py`).
3. `services/orion-field-digester/app/digestion/significance.py` — 6 tests
   (`services/orion-field-digester/tests/test_digestion_significance.py`): first-tick-always-
   recomputes, within-interval-skips, at-boundary-recomputes, empty-window-still-advances-
   the-throttle, store-failure-does-not-advance-the-throttle.
4. Existing suites updated for the new always-present dimension (same blast-radius pattern
   `deviation_pressure`'s own addition already went through): `test_dimension_precision_
   baseline.py`, `test_field_pressure_provenance.py` (4 spots), `test_proposal_scoring.py`'s
   cold-start jump-magnitude sweep (needs every `PRESSURE_DIMENSIONS` member's own real
   measured max). `_LEGACY_EMPTY_DIMENSIONS_FALLBACK` deliberately NOT touched — same
   decoupling `deviation_pressure`'s own code review already established.
5. `check_service_env_compose_parity.py orion-field-digester`: OK, all keys exposed.
6. `check_definition_drift.py --update`: 1 new metric locked (`sustained_load_pressure`),
   0 high severity.

## Non-goals confirmed held

No `PRESSURE_DIMENSIONS` scoring template declares `sustained_load_pressure` (checked live
against `config/proposals/proposal_policy.v1.yaml`). No winner/target-identity field shipped
— unlike `tension_borda_winner_target_id`, there is no consumer yet that needs identity, only
magnitude; when one exists, add it then, same precedent that field itself already set. No
fusion with `deviation_pressure` into one scalar — separate, independently-competing
dimension. No Hub glossary-panel entry added (checked: `tension_deviation_pressure`'s own
addition to the glossary wasn't required by anything downstream either — real, optional
follow-up, not required for this patch to be complete). No Borda ranking, unlike the sibling
tension package this reuses `channel_regime()`/`iter_observations()` from — `orion/field/
significance.py`'s own module docstring has the full account: `deviation_pressure()`'s own
scalar doesn't use Borda either, and building the ranking machinery here would be real code
with zero real callers until a winner/target field actually ships. `max()` over `loaded_
steady` ballots directly, same as `deviation_pressure()` already does.

## Review findings fixed (2026-08-18)

Code review (6 finder agents) surfaced 8 findings. One (a claim that `run_digestion_tick`'s
new required kwargs broke 5 pre-existing test call sites in 4 named files) was verified
FALSE — those files do not exist anywhere in this repo; discarded as a hallucinated finding,
not acted on. The other 7 were real:

- **`load_recent_field_json`'s `ORDER BY ASC LIMIT` truncation bug**: silently kept the
  OLDEST rows and dropped the newest ones once `row_cap` actually triggers — a sibling query
  (`field_channel_glossary_routes.py`) already carries a documented `DESC + reverse()` fix for
  this exact failure mode, and this function had copied the unfixed form instead. Fixed:
  matched the established pattern. Currently masked in practice (900s window at ~0.4 rows/sec
  sits well under the 4000-row default cap), but `FIELD_SIGNIFICANCE_WINDOW_SECONDS` is an
  exposed, unbounded operator env knob.
- **Duplicate-sample EWMA precision bias**: `sustained_load_pressure` is throttled (~30s), but
  was unconditionally present in `field_pressures()` every ~2s hot tick, feeding the identical
  carried-forward float into precision tracking as a "fresh" observation ~15x per real
  computation. Fixed: present in `field_pressures()` ONLY on the tick that actually recomputed
  it (`sustained_load_computed_at == generated_at`) — same "absent this tick" convention the
  original 4 channel-merge dimensions and this module's own consumers already support; unlike
  `deviation_pressure`, which genuinely IS fresh every tick, so its own "always present" stays
  correct and untouched.
- **Unjustified Borda ceremony**: fixed by removing it — see Non-goals above.
- **Independence-check rigor for excluding `loaded_volatile`**: the claim that including it
  "would blur this metric back into redundancy" was reasoned, not measured. Measured now
  (`scripts/analysis/measure_sustained_load_pressure.py --include-volatile`, made a real,
  reusable flag rather than a one-off snippet): 24h replay, `loaded_steady`-only gives
  r=-0.0313 vs `deviation_pressure`; widening to also include `loaded_volatile` gives
  r=-0.0021 — both are essentially zero, so the DATA does not actually distinguish the two
  scopes on independence grounds. The real justification for `loaded_steady`-only is the
  conceptual one already in the module docstring (a volatile-and-loaded channel is the kind of
  thing a change-detector can plausibly already catch), stated as reasoned, not measured — the
  correlation number does not carry that specific claim, and this doc no longer implies it
  does.
- **Flat-repeat channels misclassify as `no_new_input`**: disclosed as a known, real,
  unfixed-in-this-patch limitation — `orion/field/significance.py`'s own docstring has the
  full account, including why the live driver (disk_capacity_pressure) does not trigger it
  today (real jitter: 7 distinct values, longest identical run 96 of 347 samples in a real
  15-minute window) but a more coarsely-quantized channel could.
- **`extra="forbid"` rollback fragility**: confirmed pre-existing, not new to this patch —
  every additive `FieldStateV1` field ever shipped (including `tension_deviation_pressure`/
  `tension_baseline_mu` etc.) already carries this exact risk class under the schema's
  existing `ConfigDict(extra="forbid")`. This patch adds 2 more fields to an already-accepted
  risk, not a new one; schema versioning is out of scope for a sensing-only patch.
- **3 near-duplicate hand-rolled SQL fetches** (this patch's `store.py` method, this patch's
  own analysis script, and the pre-existing `field_channel_glossary_routes.py`): real
  duplication, disclosed as known debt rather than consolidated into a shared cross-service
  helper in this patch — that helper would need to live in the shared `orion/` package and
  touch an already-merged file (`field_channel_glossary_routes.py`) this patch has no other
  reason to change, which is scope creep beyond a sensing-only slice.

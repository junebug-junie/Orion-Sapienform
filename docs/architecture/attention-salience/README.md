# Attention salience: Candidate A / Candidate B

Status: **both live**, disjoint target universes, no fallback to the killed
hand-weighted formula. Last updated 2026-07-30.

This is the docs home for Orion's Layer 5 attention salience system — what
decides which real target (a substrate domain, a physical host, a
capability, or a system-level signal) Orion's attention actually goes to on
a given tick. It replaces `compute_salience()`, a 23-hand-picked-weight
linear blend that had no real theory behind it (killed outright, no
fallback, per CLAUDE.md §0A "kill means kill" — see [History](#history)).

Code lives in `orion/attention/field_attention/` (not moved for this
write-up — see `docs/superpowers/pr-reports/` for why a rename was
considered and deferred). This folder is documentation only.

## Why two candidates, not one theory

The hand-weighted formula scored every target the same way regardless of
what data actually existed for it. Once killed, it became clear that two
genuinely different real target universes exist, each with a different
theory-appropriate replacement — not a single uniform substitute:

- **Real historical time series exist** for exactly 5 targets (the
  `node:substrate.*` reducers). These get **Candidate A**.
- **No historical time series exists** for physical hosts or capabilities —
  only a single current-tick snapshot. These get **Candidate B**
  (novelty-only slice).

The two never compete for the same target. `select_node_targets()` (A) and
`select_host_targets()`/`select_capability_targets()` (B) partition the
real target space; `build_attention_frame()` just concatenates their
outputs. See [Target universe map](#target-universe-map).

## Candidate A — precision-weighted prediction-error salience

**Theory:** Feldman & Friston 2010, "Attention, Uncertainty, and
Free-Energy." `salience = precision × |prediction_error|`, where
`precision = 1/variance` of a target's own recent real error history. A
deviation stands out more when it breaks an otherwise-consistent pattern,
less when the target has always been noisy.

**Code:** `orion/attention/field_attention/candidate_precision_weighted.py`
(`precision_weighted_salience()`, `normalize_across_targets()`),
consumed by `select_node_targets()` in `selectors.py`.

**Real input:** `prediction_error` values written by `orion-substrate-runtime`'s
reducers into `substrate_reduction_receipts` (Postgres) on every tick —
JSONB path `receipt_json -> 'state_deltas' -> 0 -> 'after' -> 'pressure_hints'
->> 'prediction_error'`, filtered by `reducer_id = 'substrate.{reducer_key}'`.
Fetched via `AttentionRuntimeStore.load_prediction_error_history()`
(`services/orion-attention-runtime/app/store.py`), most recent 200 rows
(`ATTENTION_PREDICTION_ERROR_HISTORY_LIMIT`), oldest-to-newest.

**Real input quality:**
- Retention is a rolling ~30-minute window
  (`ORION_RECEIPT_RETENTION_SUCCESS_MINUTES`) — never full history, always
  recent-past only.
- Below `QUALIFYING_MIN_ROWS = 20` real samples, `confidence_score` scales
  down proportionally (`n_samples / 20`) rather than claiming false
  certainty.
- A target with **zero** real history is excluded entirely from this
  tick's targets, not scored `0.0` — "no data" and "confidently calm" are
  different claims, and conflating them was itself part of the disease
  being fixed here.

**Coverage — exactly 5 targets, no others, ever, without new real
grounding being built:**

```python
PREDICTION_ERROR_NATIVE_TARGETS: dict[str, str] = {
    "node:substrate.biometrics": "node_biometrics",
    "node:substrate.execution":  "execution_trajectory",
    "node:substrate.chat":       "chat_session",
    "node:substrate.route":      "route_arbitration",
    "node:substrate.bus_synaptic": "bus_synaptic",
}
```

`node:substrate.transport` is deliberately excluded — its prediction-error
write was permanently retired 2026-07-26 (it was a narrow, non-representative
2-Redis-Stream census, not real bus traffic; `bus_synaptic` is its real
successor). Physical hosts and capabilities have no real prediction-error
series at all — their vector's `prediction_error` field is a hardcoded
`0.0` placeholder, not a tracked signal.

**Output:** `salience_score` is min-max normalized across whichever of the
5 targets qualify *this tick* (0.0 for the weakest real competitor, 1.0 for
the strongest — a documented tradeoff: the weakest of N qualifying targets
always floors to 0.0 regardless of its own absolute magnitude).
`confidence_score = n_samples / 20` (clamped). `dominant_channels =
{"prediction_error": current_error}`.

## Candidate B — Global Workspace / Society-of-Mind, novelty-only

**Theory:** Baars 1988 / Dehaene 2014 (Global Workspace), Minsky 1986
(Society of Mind). The full design called for three independent scorers
combined by Borda-count rank-aggregation: magnitude, novelty, dwell. **Only
novelty is live.** This is a real, disclosed scope narrowing, not the full
candidate:

- `magnitude_scorer()` doesn't apply — no real prediction-error history
  exists for hosts/capabilities, the same reason Candidate A excludes them.
  Wiring it would mean fabricating input, not reusing real signal.
- `dwell_scorer()` was built but never wired live — confirmed 2837/2840
  (99.9%) of real `substrate_coalition_dwell_log` rows over a 24h window
  had `attended_node_ids = []`. Building the cross-service wiring for a
  signal that would almost never contribute a real vote wasn't a good
  first cut. Real, named follow-up, not solved.
- With exactly one real scorer, Borda aggregation has nothing to aggregate
  — the novelty ranking *is* the ranking.

**Code:** `orion/attention/field_attention/candidate_society_of_mind.py`
(`novelty_scorer()`), consumed by the shared `_novelty_targets()` in
`selectors.py`, called from `select_host_targets()` and
`select_capability_targets()`.

**Real input:** the raw pressure vector `orion-field-digester` computes per
target each tick (`field.node_vectors` / `field.capability_vectors` —
33 real channels for nodes, 8 for capabilities;
`services/orion-field-digester/app/tensor/channels.py`). A few channels are
single-observer (`stream_backlog_health`, `delivery_confidence` — only ever
observed from `node:athena`).

**Current-salience proxy:** `_current_pressure_proxy()` — `max()` over that
vector, an order statistic with zero free parameters (no weights, no
calibration). Five channels default to/sit near 1.0 when *healthy*
(`availability`, `delivery_confidence`, `stream_backlog_health` for nodes;
`confidence`, `available_capacity` for capabilities) — these are inverted
(`1 - value`) before the `max()` comparison, or the proxy would report
~1.0 for both a calm and a severely overloaded target whenever those
channels stayed near default (the normal case). This was a real, live bug
found and fixed by code review — see [Known gaps](#known-gaps-and-recent-fixes).

**Novelty — a one-tick diff, not a time series:**
`novelty = |this tick's proxy − last tick's own recorded salience for the
same target_id|`, read from the *previous persisted attention frame*
(searching all 5 of its real target buckets — `dominant_targets`,
`node_targets`, `capability_targets`, `system_targets`,
`suppressed_targets` — not just the "active" ones; also a real bug found
and fixed, see below). This is structurally shallower than Candidate A: a
delta against one prior observation, not a variance estimate over real
history.

**Output:** `salience_score = novelty_score` (already bounded 0–1, no
separate normalization needed — unlike Candidate A's unbounded raw
salience). `confidence_score = 1.0` only if the specific `target_id` had a
real entry in the previous frame, `0.0` otherwise (including "no previous
frame at all" and "target absent from an existing previous frame" — both
genuinely mean "no real prior context," now answered from the same real
5-bucket search Candidate B's own novelty math uses).

## Target universe map

```
FieldStateV1.node_vectors keys:
  node:substrate.biometrics  ─┐
  node:substrate.execution    │  Candidate A (precision-weighted,
  node:substrate.chat         │  real historical error series exists)
  node:substrate.route        │
  node:substrate.bus_synaptic ┘

  node:athena       ─┐
  node:atlas          │
  node:circe           │  Candidate B novelty-only
  node:prometheus      │  (no historical series, single-tick proxy)
  node:rpc_timeout    ┘
  node:substrate.transport  ── also lands here (catch-all: any node_vectors
                                key NOT in Candidate A's map), permanently
                                inert (0.0) since its own real channels
                                were retired 2026-07-26

FieldStateV1.capability_vectors keys:
  capability:graph
  capability:memory
  capability:vision           Candidate B novelty-only, ALL of them
  capability:storage          (no real per-capability prediction-error
  capability:transport        history exists for any capability, so
  capability:llm_inference    Candidate A never applies here at all)
  capability:orchestration
```

## Downstream: what consumes this

`build_attention_frame()` (`orion/attention/field_attention/builder.py`)
merges Candidate A's + Candidate B's outputs into one `FieldAttentionFrameV1`,
sorts by `salience_score`, applies `suppress_below`/`min_salience`
thresholds (`config/attention/field_attention_policy.v1.yaml`: `0.03`/`0.10`),
caps per-kind and total, and persists it.

`orion/proposals/builder.py`'s `ATTENTION_FIRST_TARGET_BINDING =
"attention.dominant_targets[0]"` binds proposal templates to the single
top-ranked target across *both* candidates. This is a real, live path —
confirmed this session to produce actual dispatched actions
(`cortex_verb: "substrate.inspect"`, `dispatch_status: "dispatched"`) for
`read_only`-gated deterministic templates through the proposal arena and
`execution_dispatch`.

**Related, not yet fixed:** the proposal arena's own top-level scoring
formula, `proposal_priority()` (`orion/proposals/scoring.py`), is still a
hand-picked linear blend (`base_priority + 0.4 * match_score + 0.2 *
urgency + 0.1 * confidence`) — same disease this whole system was built to
remove, one layer downstream, still open. A sibling piece,
`dimension_confidence()` in the same file, *did* get a real precision-
weighted fix (EWMA baseline, 2026-07-28/29) — `proposal_priority()` itself
did not.

## Upstream: where the raw numbers come from

- Candidate A's `prediction_error` values: written by `orion-substrate-runtime`'s
  reducers (`_prediction_error_receipt()` call sites in that service's
  `worker.py`) into `substrate_reduction_receipts`.
- Candidate B's raw channel vectors: written by `orion-field-digester`'s
  tensor builder into `FieldStateV1`, persisted to `substrate_field_state`.
  `orion-attention-runtime` reads the latest row each tick.

## Head-to-head comparison (Candidate A vs. Candidate B's magnitude scorer)

`scripts/analysis/measure_candidate_a_vs_b_head_to_head.py` — a real,
documented comparison closing a promise made in the original tentative-plan
design doc (which shipped Candidate A live before running it). **Important
caveat: this is a controlled replay, not a description of live behavior** —
live, A and B never compete for the same target (see the universe map
above). The script exists to answer a narrower question: on the one real
target set where *both* theories could in principle be computed (Candidate
A's 5 `node:substrate.*` reducers, replaying Candidate B's magnitude scorer
— raw prediction-error passthrough — against the same real historical
data), do the two theories actually agree on which tick is most salient?

Method: expanding-then-capped window per reducer (`values[max(0, i+1-200):i+1]`,
matching production's real 200-row cap), only after `QUALIFYING_MIN_ROWS = 20`
real samples accumulate.

**Result:** 2 of 5 reducers had enough real qualifying data. On both, the two
theories picked a **different** tick as most salient (0/2 agreement) — real
evidence the theories are meaningfully different, not evidence either one
is "more correct."

## Known gaps and recent fixes

- **Fixed (code review, 2026-07-30):** `_current_pressure_proxy()`'s
  `max()` was directionally blind — a calm and a severely overloaded
  target could both read ~1.0 whenever the 5 higher-is-better channels sat
  near their healthy default. Fixed by inverting those channels before the
  comparison.
- **Fixed (code review, 2026-07-30, two rounds):** `confidence_score` for
  Candidate B targets first conflated "any previous frame exists" with
  "this specific target had a real entry in it" (round 1), then — after a
  verification pass caught the fix was still incomplete — was found to
  check only 2 of the 5 real buckets a prior entry can live in, missing
  `suppressed_targets` (round 2). Both fixed; confidence and novelty now
  always answer from the same real 5-bucket search
  (`target_had_real_prior_entry()` / `prior_salience_for_target()` in
  `scoring.py`).
- **Fixed (found while investigating a related review finding):**
  `AttentionRuntimeStore.load_prediction_error_history()` fetched the
  *oldest* N rows under its limit instead of the newest — a real, latent
  production bug (not yet triggered at observed row volumes, but would
  silently return permanently-stale "current" readings once any reducer
  exceeded 200 real rows within its retention window).
- **Disclosed, self-resolving, not fixed:** the first live tick after this
  system deploys diffs Candidate B's new proxy against whatever the
  *previously-deployed, old-formula* frame recorded — an artificially
  large one-time "novelty" reading reflecting the formula changeover, not
  a real event. Resolves itself after exactly one tick.
- **Open, not part of this system:** `orion/proposals/scoring.py`'s
  `proposal_priority()` — see [Downstream](#downstream-what-consumes-this)
  above.
- **Deferred, named:** Candidate B's `dwell_scorer()` — built, real theory
  (coalition dwell duration), never wired live because real data shows it
  would almost never contribute a vote (99.9% empty).

## History

- `docs/superpowers/specs/2026-07-21-attention-salience-cathedral-replacement-tentative-plan.md`
  — original A/B design doc, later updated with a note on the head-to-head
  gap (Candidate A shipped live before the promised comparison ran).
- `docs/superpowers/specs/2026-07-30-candidate-b-hosts-capabilities-live-wiring.md`
  — Candidate B's own design doc, including live-data verification before
  and after both confidence/proxy fixes.
- PR #1484 (merged) — killed `compute_salience()`, wired Candidate A live,
  wired Candidate B (novelty-only) live for hosts/capabilities, shipped the
  head-to-head script.
- PR #1488 (merged) — the suppressed-bucket confidence fix, landed as a
  follow-up PR since #1484 had already merged by the time a verification
  review caught the gap.

## File map

```
orion/attention/field_attention/
  candidate_precision_weighted.py   Candidate A: precision_weighted_salience(),
                                     normalize_across_targets()
  candidate_society_of_mind.py      Candidate B: novelty_scorer(), magnitude_scorer()
                                     (unwired), dwell_scorer() (unwired)
  scoring.py                        shared: clamp01(), prior_salience_for_target(),
                                     novelty_for_target(), target_had_real_prior_entry()
  selectors.py                      select_node_targets() (A), select_host_targets()/
                                     select_capability_targets() (B),
                                     select_system_targets() (unrelated, EWMA-zscore),
                                     _current_pressure_proxy(), _novelty_targets()
  builder.py                        build_attention_frame() — merges everything
  policy.py                         FieldAttentionPolicyV1 (thresholds/limits only,
                                     no scoring weights since the kill)

config/attention/field_attention_policy.v1.yaml   thresholds, limits, observation modes
services/orion-attention-runtime/                 live service: worker.py ticks
                                                   build_attention_frame(), store.py reads/
                                                   writes Postgres
scripts/analysis/measure_candidate_a_vs_b_head_to_head.py   the head-to-head script
tests/test_attention_field_selectors.py                     selector-level tests
tests/test_attention_frame_builder.py                       builder-level tests
tests/test_measure_candidate_a_vs_b_head_to_head.py          head-to-head script tests
```

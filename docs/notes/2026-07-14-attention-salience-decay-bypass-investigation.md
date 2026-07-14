# Attention salience: decay-bypass + dwell-scoping investigation

Context: `fix/attention-loop-recency-never-decays` (PR #1052, merged) fixed one
dead-feature bug in rung-3's salience scoring (`_recency()`'s half-life decay
was correctly implemented but never fed real `first_seen_at` data, so it
returned `1.0` forever). That fix was found by hand-verifying a live reverie
narration claim against real Postgres data — a loop verdicted `resolved`
(2026-07-08) and `dismissed` (2026-07-10) via the Hub was still winning
`attention_broadcast`'s coalition selection on 2026-07-14.

PR #1055 (merged same day) separately fixed the *narration* symptom: reverie's
chain-refractory suppression had a theme-key format mismatch (`"loop:<id>"`
vs. the bare id everything else uses), and its prompt had no visibility into
`attention_loop_outcome` verdicts at all. Both fixes are real and live, but
they operate entirely inside `services/orion-thought/app/{chain,reverie,store}.py`
— the narration/chain-trigger layer. Neither touches
`orion/substrate/attention_broadcast.py` / `scoring.py` / `salience.py`, the
rung-3 coalition-*selection* path that actually decides
`AttentionBroadcastProjectionV1.selected_open_loop_id`.

This doc covers the follow-up investigation into why the *selection* layer
still let that same loop dominate indefinitely even with recency now fixed,
and what else in the same salience pipeline shares the "computed correctly
somewhere, never actually consumed" bug shape.

## Evidence: live Postgres data, not speculation

`attention_salience_trace` (written by reverie as a side effect of reading
each broadcast, `scope='reverie'`) has 2,166 rows for
`loop_id = open-loop-9d84d08cddf5` (`description = 'substrate:node:substrate.transport'`),
spanning 2026-07-08 19:56 through 2026-07-14 22:41 — after both #1052 and
#1055 had deployed:

```sql
SELECT min(created_at), max(created_at), count(*),
  count(DISTINCT features->>'evidence_strength') AS distinct_ev,
  count(DISTINCT features->>'recency') AS distinct_recency
FROM attention_salience_trace WHERE loop_id LIKE '%9d84d08cddf5%';
```

```
min=2026-07-08 19:56:31   max=2026-07-14 22:41:07   count=2166
distinct_evidence_strength_values = 1     -- pinned at exactly 1.0, always
distinct_recency_values           = 15    -- varies (recency fix confirmed working)
```

`attention_loop_outcome` confirms this loop was explicitly closed by Juniper
twice (`resolved` 07-08 20:32, `dismissed` 07-10 21:49, `habituation=1.0` —
fully saturated — at the second verdict) and kept winning anyway.

Even after the recency fix deployed, the loop's `salience` only dropped from
~0.59 to ~0.566 by 22:41 today — nowhere near enough to lose, because
`evidence_strength` (combiner weight `+0.30`, the largest single weight)
never moved.

## Finding A (primary root cause, being fixed): `_node_salience()` races raw vs. decayed and always picks raw

`orion/substrate/attention_broadcast.py:102-115`:

```python
def _node_salience(metadata: dict[str, Any]) -> tuple[float, str]:
    pressure = _f("dynamic_pressure")
    prediction_error = _f("prediction_error")
    if prediction_error >= pressure:
        return prediction_error, "prediction_error"
    return pressure, "pressure"
```

`dynamic_pressure` is properly time-decayed: `SubstrateDynamicsEngine.tick()`
(`orion/substrate/dynamics.py`, live, `SUBSTRATE_DYNAMICS_TICK_ENABLED=true`,
runs every 30s in prod) recomputes it every tick via
`prediction_error_pressure()` (`orion/substrate/pressure.py`), which applies
`weight=0.6` and a linear decay to zero over `prediction_error_decay_horizon_seconds`
(1800s / 30min) measured from the node's `observed_at`.

But `dynamic_pressure = raw_prediction_error * 0.6 * decay(<=1)` can **never
exceed** the raw `metadata["prediction_error"]` value, by construction. So
`prediction_error >= pressure` is true on every tick, forever, for any node
that has ever had a nonzero `prediction_error` written. The dynamics engine's
decay is computed correctly and then unconditionally discarded by the one
function that actually decides workspace salience. Same bug *shape* as the
recency fix (a real decay mechanism exists; the consumer doesn't use its
output) — third instance of this pattern found this session (recency, dwell
below, now this).

The raw value's origin: `_write_prediction_error_node()`
(`services/orion-substrate-runtime/app/worker.py:664`) upserts a single
fixed-identity node per source (re-writes collapse, no growth — that part is
sound), called from `_transport_tick()` (`worker.py:1687`) whenever
`transport_prediction_error(prev, curr)` (`orion/substrate/prediction_error.py`,
mean delta across `bus_health`/`delivery_confidence`/`transport_pressure`
over `_THRESHOLD=0.30`, clamped to `[0,1]`) returns `>0`. Whether this specific
node's raw value is a single stale write from 07-08 that was never revisited,
or an actively-recurring ~1.0 delta on every batch with events, wasn't
distinguished — out of scope for this patch (see Non-goals).

## Finding B (secondary, being fixed): `dwell` habituation is a global scalar, not loop-scoped

`orion/substrate/attention/salience.py:116-120`:

```python
def _habituation(theme_key: str, history: SalienceHistory) -> float:
    recurrence = min(1.0, history.recent_theme_counts.get(theme_key, 0) / RECURRENCE_NORM)
    dwell = min(1.0, history.dwell_ticks / DWELL_NORM)
    resonance = 1.0 if theme_key in history.resonance_theme_keys else 0.0
    return bounded(0.5 * recurrence + 0.3 * dwell + 0.2 * resonance)
```

`history.dwell_ticks` is a single `int` (`attention_broadcast.py`'s
module-level `_dwell_ticks`, tracking how long the *current active coalition*
has persisted) — not keyed by `theme_key`. `scoring.build_open_loops()`
constructs one `SalienceHistory` per tick and passes the *same* instance into
`compute_salience()` for every competing loop, so `dwell` contributes the
identical value to every candidate's habituation score in a given tick,
including loops that are not part of the dwelling coalition at all. It cannot
demote the specific stuck loop relative to its competitors — a uniform
per-tick offset changes nothing about who wins.

Net weight math confirms this is real but small: direct combiner weight
`dwell=+0.10`, indirect via habituation `-0.35 * 0.3 = -0.105` → net `-0.005`
regardless of which loop is scored. (For comparison: `recurrence` nets
`-0.025`; `resonance`, the only feature with no offsetting positive weight,
nets `-0.07`. All three together, even fully saturated, max out around
`-0.10` to `-0.20` — nowhere near enough to counter a pinned
`evidence_strength=+0.30` on its own, which is why Finding A is the one that
actually explains the live incident.)

## Decision

Fix both in this patch — same file family, same investigative session, both
independently real and low-risk:

1. `_node_salience()` should seed pressure from `prediction_error`, not race
   an undecayed copy of it against the properly-decayed output. Concretely:
   stop reading `metadata["prediction_error"]` directly in
   `attention_broadcast.py` at all — `dynamic_pressure` is already the
   decayed, authoritative signal (`prediction_error_pressure()` is exactly
   "prediction_error seeded into pressure, with decay"). Use `dynamic_pressure`
   alone.
2. Scope the `dwell` habituation term to loops that are members of the
   current active coalition (`_current_active_coalition`) instead of applying
   the module-level tick counter to every candidate uniformly.

## Non-goals (explicitly deferred)

- **Verdict-aware selection** (excluding `resolved`/`dismissed` loops from
  `build_open_loops`/`compute_salience` entirely): raised and explicitly
  *not* chosen for this patch — it's a real behavior change to the selection
  path, not a bug-shaped fix, and deserves its own scoping decision. Reverie's
  *narration* about a verdicted loop is already truthful (#1055); this patch
  only stops that loop's `evidence_strength` from being artificially pinned
  at max, which independently gives it a real (if not absolute) chance to
  lose the competition once other candidates have comparable evidence.
- **Root-causing why `substrate.transport`'s raw `prediction_error` reached /
  stayed at 1.0** (passive single stale write vs. actively-recurring
  near-1.0 transport-bus delta): not distinguished. Worth a follow-up if the
  same node (or another transport-adjacent one) keeps re-pinning after this
  fix — would show up as `dynamic_pressure` staying near its ceiling despite
  decay, rather than the `evidence_strength` never moving at all as observed
  here.
- No change to `recurrence` or `resonance` weighting — both are already
  correctly loop-scoped (keyed by `theme_key`), just small in magnitude.

## Acceptance checks

- A node whose `dynamic_pressure` has decayed well below its raw
  `prediction_error` seed no longer reports maximal salience from that raw
  value alone.
- A loop outside the current active coalition receives zero dwell
  contribution to its habituation score; a loop inside it still receives the
  existing dwell signal.
- Existing recency/habituation/dwell test suites (`test_attention_broadcast_recency.py`,
  `test_broadcast_habituation.py`, `test_attention_broadcast_dwell.py`,
  `test_scoring_salience_wiring.py`) still pass.

# PR report — attention salience decay-bypass + dwell not loop-scoped

PR: (create at https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/attention-salience-decay-bypass — `gh` unauthenticated in this environment)
Branch: `fix/attention-salience-decay-bypass`
Status: **DONE**

## Summary

- Follow-up to `fix/attention-loop-recency-never-decays` (PR #1052) and
  `fix/reverie-verdict-aware-narration` (PR #1055): those fixed one dead
  feature (recency) and the narration-layer symptom, but left the rung-3
  *selection* competition itself still permanently favoring a verdicted-dead
  loop. Investigated why, using live Postgres data, and found two more
  independent dead-feature bugs in the same salience-scoring pipeline.
- **Root cause (primary)**: `_node_salience()` raced a raw, never-decaying
  `metadata["prediction_error"]` field against the dynamics engine's
  properly time-decayed `metadata["dynamic_pressure"]` and always picked the
  raw value — `dynamic_pressure = raw * weight(0.6) * decay(<=1)` can never
  mathematically exceed the raw seed, so the decay was computed correctly by
  `SubstrateDynamicsEngine` every 30s and then silently discarded by its own
  consumer, forever, for any node that ever had a `prediction_error` written.
  Confirmed live: a verdicted-`resolved`/`dismissed` loop's
  `evidence_strength` sat at exactly `1.0` across 2,166 real trace rows over
  6 days.
- **Second bug**: `dwell` habituation used a single module-level tick
  counter applied uniformly to every competing loop in a tick — a per-tick
  offset shared by everyone changes nothing about who wins, so it could
  never demote the specific dwelling loop relative to its competitors.
- Both fixed. Code review (high effort, 8 finder angles) found and fixed 4
  more issues surfaced by the change (3 test files silently losing dwell
  coverage they used to have, one dead conditional, one unguarded edge
  case), and surfaced 3 real findings deliberately left as documented,
  accepted follow-ups.

## Outcome moved

The rung-3 workspace-competition scorer now uses the dynamics engine's own
decayed pressure as the sole magnitude signal, and dwell habituation
actually targets the loop that's dwelling instead of applying a flat,
rank-irrelevant offset to every candidate. A loop with a permanently pinned
raw `prediction_error` no longer gets to bypass decay by construction.

## Current architecture

`orion/substrate/attention_broadcast.py` runs the rung-3 continuous
workspace competition (`build_substrate_attention_frame` → `build_open_loops`
→ `select_actions`, then `broadcast_projection_from_frame` records the
winner and updates hysteresis/dwell/habituation state).
`orion/substrate/attention/salience.py`'s `compute_features()` computes 7
features combined via `LinearSalienceCombiner` (seed weights:
`evidence_strength=0.30`, `recency=0.13`, `habituation=-0.35`, `dwell=0.10`,
others smaller). `SubstrateDynamicsEngine` (`orion/substrate/dynamics.py`)
runs on its own 30s-default loop, recomputing `metadata["dynamic_pressure"]`
per node with a real, working, time-decaying model; `metadata["prediction_error"]`
is a separate, raw, write-once-then-upserted field that nothing ever clears
or decays on its own.

## Architecture touched

`orion/substrate/attention_broadcast.py`, `orion/substrate/attention/salience.py`
(both live production modules, `orion-substrate-runtime`), plus 6 test
files. No schema/env/Docker changes — pure scoring-logic fix.

## Files changed

- `orion/substrate/attention_broadcast.py`: `_node_salience()` now returns
  `dynamic_pressure` alone as magnitude (raw `prediction_error` only decides
  the `kind` string for anomaly-vs-concept typing). New
  `_current_dwelling_loop_id` module global, tracked in lockstep with the
  existing `_current_active_coalition`/`_dwell_ticks` transition logic in
  `broadcast_projection_from_frame`, guarded the same way
  `attended_node_ids` already is (only set when the selected loop actually
  resolves against `frame.open_loops`).
- `orion/substrate/attention/salience.py`: `SalienceHistory` gained
  `dwelling_loop_id: str | None = None`. New `_loop_dwell(theme_key,
  history)` helper returns 0 unless `theme_key == history.dwelling_loop_id`;
  used by both `_habituation()` and `compute_features()`'s standalone
  `dwell` feature, replacing the old unconditional
  `min(1.0, history.dwell_ticks / DWELL_NORM)`.
- `orion/substrate/tests/test_attention_broadcast.py`: fixed
  `test_prediction_error_beats_equal_plain_pressure`'s fixture (it modeled
  an unrealistic state — raw `prediction_error` set with no `dynamic_pressure`
  at all, exactly what let the old raw-value race win); new
  `test_salience_uses_decayed_pressure_not_raw_prediction_error` regression
  test.
- `orion/substrate/tests/test_scoring_salience_wiring.py`: new
  `test_dwell_scoped_to_dwelling_loop_only`.
- `orion/substrate/tests/test_attention_broadcast_dwell.py`,
  `test_broadcast_habituation.py`, `test_attention_broadcast_recency.py`:
  fixtures updated to also reset `_current_dwelling_loop_id`.
- `orion/substrate/tests/test_rumination_replay.py`,
  `test_salience_combiner.py`: pre-existing `SalienceHistory(dwell_ticks=N,
  ...)` constructions updated to also set `dwelling_loop_id`, restoring dwell
  coverage the dwell-scoping fix would otherwise have silently zeroed.
- `docs/notes/2026-07-14-attention-salience-decay-bypass-investigation.md`
  (new): full investigation, live-data evidence, decision, non-goals,
  review findings.

## Schema / bus / API changes

None. `SalienceHistory.dwelling_loop_id` is purely additive with a safe
default (`None`) — every existing caller that doesn't set it gets the same
behavior it would have gotten from an unset dwelling loop (dwell=0),
matching the "no history / Phase 1 shadow behavior" convention this
dataclass already documents for its other fields.

## Env/config changes

None.

## Tests run

```text
.venv/bin/python -m pytest orion/substrate/tests/ -q
=> 203 passed

.venv/bin/python -m pytest \
  orion/substrate/tests/test_rumination_replay.py \
  orion/substrate/tests/test_salience_combiner.py \
  orion/substrate/tests/test_attention_broadcast_recency.py -v
=> all 17 passed individually (confirms the review-fixed dwell/reset
   regressions are actually exercised, not just passing by omission)
```

## Evals run

None applicable — same as PR #1052, no eval harness exists for
`orion/substrate/attention/`. The live Postgres query against
`attention_salience_trace`/`attention_loop_outcome` (2,166 real rows, 6-day
span) that found this bug is the closest thing to an eval here — informal,
by hand, not automated. Same gap flagged in #1052's report: this class of
bug (a dead/bypassed feature) would benefit from a periodic live check.

## Docker/build/smoke checks

Not run — pure Python scoring-logic change inside `orion-substrate-runtime`'s
existing code path, no config/dependency/compose changes to validate.

## Review findings fixed

Code-review skill, high effort (3 correctness + 3 cleanup + altitude +
conventions = 8 finder angles, verified):

- **Finding**: `test_attention_broadcast_recency.py`'s reset fixture was the
  third file touching `attention_broadcast`'s module globals but wasn't
  updated to clear `_current_dwelling_loop_id` — found independently by 2 of
  8 angles.
  - **Fix**: added the reset to both setup and teardown.
  - **Evidence**: all 4 tests in that file still pass individually.
- **Finding**: `test_rumination_replay.py` and `test_salience_combiner.py`
  (both pre-existing, untouched by the original patch) construct
  `SalienceHistory(dwell_ticks=N, ...)` directly without the new
  `dwelling_loop_id` field — found independently by 3 of 8 angles,
  live-verified: both tests still passed, but the dwell term's 0.3 combiner
  weight silently contributed nothing, shrinking
  `test_rumination_replay`'s own documented margin (`stuckN=0.45` vs
  `freshN=0.47`) to an accidental ~0.015 instead of the ~0.02 the file's
  docstring claims.
  - **Fix**: set `dwelling_loop_id` on both fixtures.
  - **Evidence**: both tests still pass, now exercising the dwell term for
    real again.
- **Finding**: `_loop_dwell()`'s `history.dwelling_loop_id is None` guard
  clause was dead code — `theme_key != None` already evaluates `True` on its
  own for any real string `theme_key`.
  - **Fix**: removed the redundant clause.
  - **Evidence**: full suite still passes (behavior-neutral simplification).
- **Finding**: `_current_dwelling_loop_id` was set from
  `selected.open_loop_id` unconditionally, even when that id didn't resolve
  to a real loop in `frame.open_loops` (`selected_loop=None`) — inconsistent
  with `attended_node_ids`'s existing guard one line above.
  - **Fix**: guard `_current_dwelling_loop_id` the same way.
  - **Evidence**: full suite still passes; behavior was already fail-safe
    (dwell silently stayed 0) but is now consistent with the existing
    pattern.

Not fixed, accepted and documented (in the working doc and above) — would
expand this patch's scope beyond its two files:

- A freshly-written `prediction_error` node reports 0 salience (filtered
  below `min_salience`) until the next dynamics tick (default 30s interval)
  populates `dynamic_pressure`. Direct, bounded, minor consequence of no
  longer trusting the undecaying raw value — the tradeoff that fixes the
  actual live incident.
- `kind`/`target_type_hint` stays pinned to `"prediction_error"`/`"anomaly"`
  forever once any nonzero raw `prediction_error` was ever written, even
  after `dynamic_pressure` is driven by an unrelated source. Affects
  display/typing only, not salience ranking. A real fix needs
  `dynamics.py`'s `_compute_pressures()` to persist its own per-node
  `reason` onto metadata instead of discarding it after `tick()`.
- **New finding**: `orion/substrate/endogenous_curiosity.py`'s
  `_prediction_error_candidates()` reads the same raw, never-decaying
  `metadata["prediction_error"]` field directly as a "sustained prediction
  error" signal, with no staleness check — the identical bug pattern found
  and fixed here, in a sibling consumer. Currently dormant behind
  `ORION_ENDOGENOUS_CURIOSITY_ENABLED=false`; worth revisiting before that
  flag is ever flipped on.

Also explicitly deferred (not a review finding, a scoping decision recorded
in the working doc): wiring `attention_loop_outcome` verdicts into
`build_open_loops`/`compute_salience` so resolved/dismissed loops stop
competing entirely, rather than merely losing their artificial magnitude
boost. Raised, discussed, and deliberately not chosen for this patch.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```
Not run this session.

## Risks / concerns

- Severity: low
- Concern: a freshly-seeded `prediction_error` node is invisible to the
  workspace competition for up to one dynamics-tick interval (default 30s)
  instead of immediately, since magnitude no longer comes from the raw
  (undecaying) field. Intentional tradeoff, not treated as a regression —
  see "not fixed, accepted" above.
- Severity: low
- Concern: this patch does not guarantee a verdicted-dead loop can never win
  again — it removes the artificial `evidence_strength=1.0` pin, giving
  other candidates a real chance, but if the underlying transport-bus
  reducer (`transport_prediction_error()`) is genuinely, repeatedly
  re-seeding a near-1.0 raw value on every tick with events (vs. a one-time
  stale write — not distinguished this session, see working doc), the same
  loop could still win on legitimately-recomputed (if not stale) pressure.
  Worth a live re-check after this deploys, same as #1052's own open
  question about the habituation/resonance safety valve.

## PR link

`gh` is unauthenticated in this environment — open manually at:
https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/attention-salience-decay-bypass

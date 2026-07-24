# `field_salience_only` branch starvation — finding, 2026-07-24

Read-only measurement, per Sentience Striving Program charter section 7 ("measure before
minting"). Follow-up to a live harness-turn + reducer-replay sanity check run the same day
against real post-Postgres-rebuild data (8h window, 9208 ticks,
`scripts/analysis/measure_ast_hot_reducer.py`), which found the new Active-Inference
confidence formula (`_aggregate_prediction_error_confidence`, PR #1301/#1304) fired on only
4 of 9208 ticks (0.04%).

## Question

Is that 0.04% a bug (the new formula silently failing to run) or a structural consequence of
how `reduce_attention_self_model()` (`orion/substrate/attention_self_model.py`) branches?

## Method

`scripts/analysis/measure_attention_reason_branch_starvation.py` (new, this patch) — a pure,
unit-tested function (`analyze_branch_starvation`) reading the real per-tick
`attention_reason`/`broadcast_lane_present`/`broadcast_lane_stale` columns already written by
`measure_ast_hot_reducer.py`'s `ticks.csv` artifact. No new DB access — this is a secondary
analysis pass over that script's own real-data replay output, not a duplicate pipeline.

## Finding

**Structural, not a bug.** The reducer's `attention_reason` branches in strict elif order:
`top_down_override` > `bottom_up_salience` > `field_salience_only` > `no_data`
(`attention_self_model.py:242-303`). `field_salience_only` — the only branch that computes
the new Active-Inference confidence — is only reached when the broadcast/GWT-dispatch lane is
absent or stale (>60s old). On the real 8h window measured:

- Broadcast lane fresh (present and not stale): **9204/9208 ticks (99.9566%)**.
- `bottom_up_salience` fired on 9204 ticks (99.96%) — every tick where the broadcast lane was
  fresh, with zero exceptions (0 elif-ordering contradictions, confirmed against the real data).
- `field_salience_only` fired on exactly the 4 remaining ticks.

The broadcast lane runs on a 30s cadence and is essentially always fresh in the live system —
so the older `broadcast.coalition_stability_score`-based confidence wins the branch nearly
every time, and the new prediction-error-based confidence formula is almost never the one
actually driving `AttentionSelfModelV1.confidence` in production, even though it is correct
whenever it does run (unit-tested; live-verified this session against real data).

## Why this matters

Sentience Striving Program charter, section 6, objective 2 (the AST/HOT reducer) is gated as
"proven" before objective 3 can retire the drives bucket-vote layer. "Proven" has been checked
in the sense of "computed correctly when exercised" (unit tests, live replay) but not in the
sense of "actually exercised by live traffic at any meaningful rate." This finding closes that
second gap: the new confidence formula is real but functionally dormant today, shadowed by an
older branch that almost always wins first.

This is not presented as something to fix in this patch — no behavior changes here, this is
measurement only, per charter section 7's own discipline (a new signal doesn't get built on
until it's measured, and measuring first here means being honest that it's barely running,
not assuming it's fine because it passed a unit test).

## Non-goals

- Not deciding whether to reorder the elif branches, blend both confidence sources, or change
  the broadcast-staleness threshold — that's a real design choice affecting
  `AttentionSelfModelV1`'s live semantics and needs its own sign-off per this repo's proposal-
  mode rule for cognition-loop-adjacent changes.
- Not re-running the paused predicted_shift TEST-set validation (separate, already-tracked
  open item, blocked on insufficient post-Postgres-rebuild history).

## Artifacts

- `scripts/analysis/measure_attention_reason_branch_starvation.py` (new)
- `scripts/analysis/tests/test_measure_attention_reason_branch_starvation.py` (new, 7 tests)
- Source data: `/tmp/ast-hot-reducer/ticks.csv` (from the 2026-07-24 live replay run)

## Related

- `docs/superpowers/specs/2026-07-22-l6-self-model-ast-hot-active-inference-design.md`
- `orion/sentience_striving_program/README.md` section 6, objective 2
- PR #1301 (confidence/predicted_shift implementation), PR #1304 (predicted_shift reversion fix)

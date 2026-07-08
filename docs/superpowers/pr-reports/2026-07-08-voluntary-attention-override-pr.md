# PR: Voluntary attention override — Step 2 (DEFAULT-OFF)

**Status:** IMPLEMENTED + reviewed, default-OFF. "Choosing to attend against your
own salience." Ships behind `ORION_ATTENTION_TOPDOWN_ENABLED=false`; off = pure
bottom-up, byte-identical to today. Not gated on the measurement (0(a)/0(b) gate
only Steps 1 & 4) and independent of φ — the one arc rung buildable now.

## Summary

- **Top-down bias combiner** (`orion/substrate/attention/top_down.py`): an active
  goal projects a bias `b(c) = priority × relevance(drive_origin, loop)` onto
  attention candidates, competing with bottom-up salience under a bounded **effort
  budget** (`combined = s + gain·applied_b`, `applied_b ≤ remaining_E`). Biased
  competition (Desimone & Duncan), not override-by-fiat. Pure, never raises.
- **Goal→attention wire** (the scope-doubler): `orion/substrate/attention/goal_context.py`
  — a module-level active-goal store (`GoalProposalV1 → GoalContext`), mirroring
  the existing `_recent_selected_counts` module-state pattern. The substrate had
  **zero** goal references before this.
- **Override trace**: when top-down flips the winner, `VoluntaryOverrideV1` records
  what won, what it beat, both bottom-up scores, applied bias, effort spent — an
  inspectable act of will (or a measured 0 if it never fires).
- Layered onto `build_substrate_attention_frame` via `_apply_voluntary_attention`;
  all 10+ callers unchanged.

## Pre-build verification (before spawning)

- All 5 cited seams confirmed current: `LinearSalienceCombiner`/`SEED_WEIGHTS`,
  `OpenLoopV1` relevance fields, `scoring.py` populates them, the zero-goal-refs
  gap is real, `attention_broadcast` exists.
- Both in-flight attention branches (`computed-salience-actionable-attention`,
  `attention-frame-v1`) are **already merged** — no collision.

## Architecture / deviation

Uses the goal-context **module store** (not a new param threaded through 10
callers) to stay back-compatible, consistent with the file's existing module
state. Real agency-readiness scaling is stubbed at 1.0 (frame builder has no
self-state input) — a documented follow-on wire.

## Contracts (all additive / back-compat)

- `OpenLoopV1.top_down_bias`, `OpenLoopV1.combined_salience` (default 0.0).
- `VoluntaryOverrideV1` (new, embedded — no new channel).
- `AttentionFrameV1.voluntary_override` (None), `effort_budget_used` (0.0).
- No registry churn (additive optional fields on already-registered models).

## Env/config

- `ORION_ATTENTION_TOPDOWN_ENABLED=false` + `ATTENTION_TOPDOWN_GAIN/EFFORT_MAX/
  EFFORT_SCALE_BY_AGENCY/TOPDOWN_WEIGHTS_VERSION` in substrate-runtime `.env_example`.
  Read via `os.getenv` (same pattern as the salience flags).

## Tests / eval

```text
tests/test_top_down.py (10) + tests/test_voluntary_attention_wiring.py (8) → 18 passed
regression (attention frame/schema/builder + substrate + relational) → 319 passed
run_topdown_eval.py → PASS: override rate rises with priority (1.0), falls under
  effort scarcity (0.0), strong bottom-up beats weak bias, no-goal→no-override.
```

## Review findings fixed (Task 9, subagent)

- **MAJOR** — the goal store never cleared a terminal goal → a completed/failed
  goal would bias attention forever. **Fixed**: clears when the held goal's
  `artifact_id` goes terminal. Regression test added.
- **MINOR** — top-down now gated on `salience_v2_enabled()` (its bottom-up basis
  is `loop.salience`, the v2 output; with v2 off the override winner could
  disagree with real selection).
- **MINOR** — import block moved inside the never-raise `try`.
- **MINOR** — override trace recorded only when the winner has an action to
  re-point to (so `chosen_loop_id` can't disagree with `selected_action`).

## Enable / rollback

- Enable only after review: `ORION_ATTENTION_TOPDOWN_ENABLED=true` (+ verify with
  `salience_v2` on) and restart substrate-runtime. Off → pure bottom-up, exact
  current behavior.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
# (only if/when the flag is enabled)
```

## Risks / concerns

- Low while flag off (default). Inert until enabled.
- Real agency-readiness scaling stubbed at 1.0 (follow-on); the goal store is a
  latest-active-goal MVP proxy (a full priority-ranked live-goal set is a follow-on).

## PR link

<to be filled by `gh pr create` / GitHub UI>

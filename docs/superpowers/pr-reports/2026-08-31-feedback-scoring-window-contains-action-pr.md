# The scoring window has to contain the action

Branch `fix/feedback-window-contains-action`.

## Summary

- `orion-feedback-runtime` sampled the field before a dispatch and again after a **fixed
  15s** `action_settle_sec`. That constant was chosen on 2026-08-22 to fix this exact
  defect — for a population of **1.2–5.4s** actions.
- `express` runs **~50s**, so its "after" sample landed **35s before the action
  finished**, and three consecutive live outcomes read `baseline == observed_after` to
  four decimal places.
- The window now follows the action's **measured** latency:
  `min(action_settle_sec + max(latency), action_settle_max_sec)`, frame-wide max.
- A frame whose window has not closed is **deferred, not consumed** — scoring it early
  skips every candidate *and* clears `feedback_pending`, losing the measurement forever.
- A frame whose action outlasts the ceiling is **refused, not mis-scored**.

## Outcome moved

Live, on the deployed image:

| | baseline | observed_after | delta | latency_ms |
|---|---|---|---|---|
| **before** (15s) | 0.3525 | 0.3525 | **0.0000** | 50882 |
| | 0.3525 | 0.3525 | **0.0000** | 49944 |
| | 0.3519 | 0.3519 | **0.0000** | 50610 |
| **after** (65s) | 0.3522 | 0.3417 | **−0.0105** | 50730 |
| | 0.3518 | 0.3522 | **+0.0004** | 52304 |

A real distribution with sign variation, instead of a structural zero three times running.

Deferring also repairs `store.load_latest_field_after` (`:416`), which takes the first
tick in `(t, t+30s]`. At the measured minimum lag of 0.12s that tick may not exist yet, so
the fast tail was losing `field_after` in the **feedback frame itself**, not only in the
scoring window.

## Why the constant could not hold

A fix expressed as a constant only holds for the population it was sized against, and
nothing re-checks it when a new member joins. `baseline == observed_after` to full
precision across several samples is never "no effect" — it means the window missed. Worse
than a weak measurement: it is an **unbiased estimate of a quantity null by
construction**, so variance collapses *with confidence* (0.25 cold prior → 0.01266 at
n=3) and the action is retired below the allocator's information floor for a reason never
measured.

`latency_ms` was populated on 0 of 5,739 rows until 2026-08-21, which is why it was a
constant. It is real data now, so the window can follow the action.

**The signal was checked before the window was blamed.** `resource_pressure` over 400
live ticks: min 0.2952, max 0.3543, sd 0.01773, 23 distinct values. Live and moving.

## Why deferring was required, not optional

Widening alone would have made things worse. Scoring before the window closes returns
`(None, None)` → every candidate skipped as `missing_field_window` → and
`save_feedback_frame` still clears `feedback_pending`, so the dispatch is never rescored.

Frame-scoring lag over 10,261 frames: p50 **94.5s**, p95 172.5s, min **0.1s**. The
docstring's "waiting costs nothing" is true at the median and false at the tail.

**Head-of-line blocking is bounded, not free** — an earlier draft of this patch's comment
claimed the latter and was wrong. The queue is oldest-first (`_PENDING_SQL`,
`ORDER BY generated_at ASC`), so under a single constant everything behind the head was
equally unscoreable and deferring really did cost nothing. Per-frame settle breaks that: a
younger frame with a smaller settle *can* be scoreable and does get blocked. Measured over
24h: **170 of 857** dispatching frames defer, worst head-block **14.8s**, 1,702s total
(2.0% of wall time). It stays small because execution-dispatch-runtime inserts the frame
row *after* its sends, so a frame's age at first visibility already covers its own latency
and the wait is ≈ `base`, not ≈ `settle`.

## Files changed

- `services/orion-feedback-runtime/app/worker.py`: `_scoring_settle_sec()` → `(settle, clamped)`;
  `load_cortex_result_evidence` moved ahead of the window choice; the defer; the clamp refusal.
- `services/orion-feedback-runtime/app/settings.py`: `action_settle_max_sec`; `action_settle_sec` redefined as a margin.
- `services/orion-feedback-runtime/docker-compose.yml`: `ORION_ACTION_SETTLE_MAX_SEC` on the environment allowlist.
- `services/orion-feedback-runtime/app/store.py`: docstring corrected — its lag claim was measured false at the tail.
- `.env_example` + live `.env`.
- `services/orion-feedback-runtime/tests/test_scoring_window_contains_action.py`: **new**, 19 tests.

## Tests run

```text
pytest services/orion-feedback-runtime/tests -q   ->  27 passed
```

### Mutation tests (real file, targets asserted present first)

| mutation | result |
|---|---|
| settle reverts to the constant at the call site | 2 failed |
| defer guard → `if False:` | 1 failed |
| negative-age guard removed (`0.0 <=` dropped) | 1 failed |
| clamp refusal → `if False:` | 1 failed |
| `max(latencies)` → `min(...)` | 1 failed |
| clamp removed | 1 failed |
| `if not latencies:` → `if False:` | 1 failed |

## Review findings fixed

- **BLOCKING — the new knob never reached the container.** `docker-compose.yml:12` is an
  explicit `environment:` allowlist and `ORION_ACTION_SETTLE_MAX_SEC` was not on it;
  `.env`/`.env_example` were correctly synced, so the gap was invisible outside the
  container. Confirmed by `docker exec ... env | grep settle` returning only the old key —
  a kill switch absent from the container, a failure shape this repo has hit before.
  - **Fix + evidence:** added to the allowlist; post-deploy `env` now shows both, and
    `get_settings()` reads `180.0`.
- **The clamp silently rebuilt the defect, and my justification was factually wrong.** I
  wrote that 180s "clears the slowest real action by a wide margin (express ~50s)". But
  `execution_dispatch_policy.v1.yaml:117` and `:215` give `builder_prune` and
  `prune_dangling_images` `rpc_timeout_sec: 720`, and live max **success** latency over
  24h is **107.5s** — 50s is the p50, not the max. The clamp already binds live (3 failed
  express rows at `latency_ms=170113` → 185.1s → clamped to 180). When it bound, the code
  returned a window it knew was too short and scored anyway, with no warning.
  - **Fix:** `_scoring_settle_sec` returns `(settle, clamped)`; a clamped frame is **not
    scored** and logs `feedback_scoring_window_clamped` at WARNING. A confident wrong
    posterior is strictly worse than a gap.
- **A negative age defers forever.** The defer clears itself only because wall-clock age
  grows. `generated_at` is a bare `datetime` in the schema, so a naive value is legal; a
  backwards NTP step, a restore, or a manual insert would park the FIFO head permanently
  at one INFO line per 2s poll — structurally the 2026-07-22 stuck-head incident with a
  different cause, and `action_settle_max_sec` does not bound it (it clamps the settle,
  not the age).
  - **Fix:** `if 0.0 <= age_sec < settle_sec:`, plus a test that fails without it.
- **A test crashed incidentally instead of failing its own assertion.** Killing the defer
  guard failed `test_a_frame_younger_than_its_window_is_left_pending` with an
  `AttributeError` from the real frame builder, not on any of its four assertions — so the
  test could not actually distinguish "deferred" from "scored nothing", which is its
  entire stated purpose.
  - **Fix:** stub the builder there too; re-mutated, it now fails on the assertion.
- **A vacuous assertion.** `assert store.cleared == [] or store.saved` could never be
  False — the preceding line already asserts `store.saved`. Removed.
- **Over-claim corrected.** "min 0.1s over 10,261 frames" counts all frames, but ~98%
  dispatch nothing and have no measurement to lose. Restricted to frames that actually
  dispatched, only **2 of 861** in 24h were scored with lag < 15s. The defer prevents ~2
  lost measurements/day *today*; its real job is protecting the widened settle.
- **Frame-wide max cost re-measured.** Of 870 dispatching frames in 24h, 809 held one
  action; the 60 mixed frames widen 15s → ~33s, not → 65s. Express never shares a frame.

## Data invalidated (approved)

Snapshot at `/tmp/express-posterior-invalidation/` (`before_posterior.csv`,
`before_outcomes.csv`, `report.md`), then deleted 1 posterior row and 3 outcome rows for
`dispatch_kind='express'` — all measured through the broken window. Only express: other
kinds were measured against actions the 15s window genuinely contained.

## Static gates

All 10 pass (list derived from `.github/workflows/orion-static-gates.yml`).

## Restart required

```bash
scripts/safe_docker_build.sh orion-feedback-runtime up -d --build   # applied
```

## Risks / concerns

- **Severity: medium.** Throughput headroom is thin. Live: one dispatch frame per 2.096s
  against a 2.0s poll and one frame per tick = **95.4% utilization**, with a standing
  94-row `feedback_pending` backlog. The 2.0%/24h of deferral consumes ~43% of the
  remaining headroom. Modelled (not measured): ρ 0.954 → 0.973 moves mean queue length
  20.7 → 36 and p50 lag ~91s → ~150–160s. Self-limiting — more lag means fewer defers —
  but worth watching post-deploy.
- **Severity: low.** The defer path has fired **zero** times live so far: real lag (p50
  94.5s) exceeds the 65s settle, so it is a safety net for the fast tail. Covered by tests,
  not yet exercised in production.
- **Severity: low.** A deferred tick now issues three wasted queries per poll
  (`load_policy_frame`, `load_proposal_frame`, `load_cortex_result_evidence`) — ~850 extra
  queries/day. Negligible, disclosed.
- **Not fixed, worth a follow-up.** `store.load_cortex_result_evidence` (`:526`) filters on
  `dispatch_id` alone though `frame_id` exists on the table. Verified safe today across all
  279,366 rows (no dispatch_id spans two frames, no duplicates), but `AND frame_id = :frame_id`
  would make it structural rather than dependent on the id format holding.
- **Finding, not a defect.** The first real delta is **negative** (−0.0105) while
  `render_scene` declares `expected_direction: increase`. Two samples is nothing, but if it
  holds, express's declared claim is wrong — which is now a real measurement rather than an
  artifact.

## Status

DONE_WITH_CONCERNS — merged-ready, live-verified, four follow-ups above.

# PR report — endogenous curiosity prediction-error staleness

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1072
Branch: `fix/endogenous-curiosity-prediction-error-staleness`
Status: **DONE**

## Summary

- Fixed the sibling bug flagged in PR #1061's code review:
  `orion/substrate/endogenous_curiosity.py::_prediction_error_candidates()`
  read the raw, never-decaying `metadata["prediction_error"]` field directly
  and labeled the result `"sustained prediction error on {node_id}"` with no
  staleness check — the identical bug shape already fixed in
  `attention_broadcast.py::_node_salience()`.
- **Live-data correction to the task's own framing**: this path is not
  dormant. `ORION_ENDOGENOUS_CURIOSITY_ENABLED=true` has been live in the
  real `.env` since 2026-07-02 (explicit operator sign-off, commit message
  "Enable rung-5 endogenous curiosity on Athena after operator sign-off"),
  and Postgres confirms `node:substrate.transport` held
  `signal_strength=1.0` identically across all 1,428 persisted candidate
  sets in `substrate_endogenous_curiosity_candidates` over the prior 24h —
  the exact same stale node the PR #1061 investigation found pinned at
  `prediction_error=1.0` for 6+ days. This bug has been actively consuming
  the bounded per-cycle curiosity budget on the same phantom "surprise"
  every single tick since at least 2026-07-15, not a theoretical risk.
- **Decision (documented in code, not mechanically copied from #1061)**:
  did **not** switch to reading `dynamic_pressure` the way the salience fix
  did. `dynamic_pressure` is a composite of drive/prediction-error/
  contradiction pressure propagated across graph edges
  (`dynamics.py::_compute_pressures()`), so a node's `dynamic_pressure` can
  be driven entirely by an unrelated neighbor's pressure with zero
  prediction error of its own — reading it here would sometimes make
  `evidence_summary="sustained prediction error on {node_id}"` a literally
  false claim. Instead, the raw `prediction_error` value is decayed by the
  node's own age (`node.temporal.observed_at`), reusing the exact horizon
  (`PressureConfig().prediction_error_decay_horizon_seconds`, 1800s/30min)
  `prediction_error_pressure()` already applies to the identical raw field.
  This keeps the signal specifically about prediction error while fixing
  the actual bug (permanence), and lands on option (a) in spirit — "sustained"
  means "currently still surprising" — without conflating unrelated pressure
  sources under a mislabeled claim the way a literal `dynamic_pressure` swap
  would have.
- Code review (high effort) independently verified the `dynamic_pressure`
  composite-source claim against `dynamics.py`'s actual code (not just the
  comment's assertion), verified the decay formula is a faithful line-for-line
  port of `pressure.py`'s existing formula, and found one Important gap
  (an already-correct behavior change with no direct test) — fixed.

## Outcome moved

A substrate node that was surprising once, days ago, and hasn't been
re-observed since now decays to `signal_strength=0.0` after 30 minutes of
no re-surprise and stops generating `"sustained prediction error"`
curiosity candidates — freeing the bounded per-cycle endogenous-curiosity
budget (currently wasted on `node:substrate.transport` every tick) for
signals that are actually current. A node that keeps genuinely
re-triggering `transport_prediction_error()` (its `observed_at` keeps
advancing via `_write_prediction_error_node()`'s upsert) is unaffected —
decay only fires once re-observation actually stops.

## Current architecture

`orion/substrate/endogenous_curiosity.py` is rung 5 of the self-modeling
loop (`endogenous_curiosity_candidates()`), called from
`services/orion-substrate-runtime/app/worker.py::_endogenous_curiosity_tick()`
on the worker's regular tick loop when
`ORION_ENDOGENOUS_CURIOSITY_ENABLED=true` and the kill switch is off.
`_prediction_error_candidates()` is one of four candidate sources (the
others: repair-pressure appraisal, attention open-loops, world-coverage
gaps). Output rides `FrontierCuriosityEvaluator.evaluate()`'s existing
decision/plan path with `operator_requested=False`, and the bounded result
set is persisted to Postgres (`substrate_endogenous_curiosity_candidates`,
consumed by `services/orion-hub/scripts/substrate_observability_routes.py`
and `curiosity_hint.py`) — this is a live, wired, observed data path, not a
dead-end. `metadata["prediction_error"]` is written once per source via a
fixed-identity upsert (`_write_prediction_error_node()`,
`services/orion-substrate-runtime/app/worker.py:664-710`), refreshing
`temporal.observed_at` to "now" only when the transport reducer actually
re-fires with a new delta; it never decays on its own. Separately,
`SubstrateDynamicsEngine.tick()` (`orion/substrate/dynamics.py`) recomputes
`metadata["dynamic_pressure"]` from the same raw field via
`prediction_error_pressure()` (`orion/substrate/pressure.py`), applying
`weight=0.6` and a linear decay over the same 1800s horizon — but that
decayed value was never read by the curiosity module at all before this
fix.

## Architecture touched

`orion/substrate/endogenous_curiosity.py` (shared package code, consumed by
`orion-substrate-runtime`), plus its test file. No schema/env/Docker
changes — pure scoring-logic fix, same shape as PR #1061.

## Files changed

- `orion/substrate/endogenous_curiosity.py`: new
  `_prediction_error_staleness_decay()` helper (linear decay-to-zero by node
  age, mirroring `prediction_error_pressure()`'s formula exactly);
  `_prediction_error_candidates()` gained a required `now: datetime`
  parameter and now multiplies the raw clamped error by the decay factor
  before thresholding; `endogenous_curiosity_candidates()` gained an
  optional `now: datetime | None = None` parameter (defaults to
  `datetime.now(timezone.utc)`, backward compatible with the one real
  caller, which never passes it); module docstring's first bullet updated
  to reflect staleness decay; new module-level comment block documenting
  the live-data finding and the "why not `dynamic_pressure`" reasoning.
- `orion/substrate/tests/test_endogenous_curiosity.py`: `_node()` helper
  gained an optional `observed_at` parameter (omitted → no `.temporal`
  attribute at all → treated as unaged/decay=1.0, preserving every
  pre-existing test's behavior byte-for-byte); 3 new tests:
  `test_stale_prediction_error_decays_and_is_not_sustained` (the required
  regression — a node observed 4x the decay horizon in the past decays to
  `0.0` and, at the default `min_prediction_error=0.55` threshold, produces
  no candidate at all, vs. an identically-seeded fresh node which does),
  `test_prediction_error_staleness_decay_is_linear_within_horizon` (half-horizon
  age → `signal_strength≈0.5` for a raw error of `1.0`), and
  `test_missing_prediction_error_never_seeds_even_at_zero_threshold` (review
  finding: locks in the `raw_error <= 0.0` short-circuit, which fixed a real
  secondary bug — a zero/missing raw error could previously seed a spurious
  candidate at `signal_strength=0.0` when `min_prediction_error<=0.0`).

## Schema / bus / API changes

None. `_prediction_error_candidates()`'s new `now` parameter is required
but internal (private function, `_`-prefixed); `endogenous_curiosity_candidates()`'s
new `now` parameter is additive with a safe default, matching the existing
optional-parameter convention this module already uses elsewhere. No
consumer-visible payload shape change — `FrontierInvocationSignalV1.signal_strength`
stays in its existing `[0.0, 1.0]` range.

## Env/config changes

None. No new env keys. `_PREDICTION_ERROR_DECAY_HORIZON_SECONDS` reuses
`PressureConfig().prediction_error_decay_horizon_seconds`'s existing bare
default (1800s) rather than introducing a new knob — deliberately, to avoid
a config surface nothing yet needs (see code comment for the documented
future-landmine tradeoff if `PressureConfig` ever becomes env-driven).

## Live-data findings (before touching code)

- `ORION_ENDOGENOUS_CURIOSITY_ENABLED=true` in
  `services/orion-substrate-runtime/.env` (live, primary checkout) and in
  `.env_example` — enabled since 2026-07-02, not the "default false,
  dormant" state assumed by the module's own docstring and the task
  framing. `ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH=false`.
- `substrate_endogenous_curiosity_candidates` (Postgres,
  `localhost:55432/conjourney`) has 1,428 rows spanning
  `2026-07-15 01:03:02 UTC` → `2026-07-16 01:02:24 UTC` (roughly the
  service's tick interval × 24h, `ORION_ENDOGENOUS_CURIOSITY_TICK_INTERVAL_SEC=60`) —
  this path is actively firing and persisting, not a dead end.
- Every one of those 1,428 rows contains a `"sustained prediction error on
  node:substrate.transport"` candidate at `signal_strength=1.0`, byte-identical
  every time — direct live confirmation of the exact bug, on the exact node
  the PR #1061 investigation already found stuck at `prediction_error=1.0`
  for 6+ days in the salience path.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/tests/test_endogenous_curiosity.py -v
=> 16 passed (13 pre-existing + 3 new)

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/tests/ -q
=> 206 passed

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-substrate-runtime/tests/test_worker_endogenous_curiosity_tick.py services/orion-substrate-runtime/tests/test_worker_prediction_error_node.py -v
=> 12 passed
```

(No local `.venv` in this worktree; ran against the shared interpreter at
`/mnt/scripts/Orion-Sapienform/.venv` with `cwd` set to this worktree, same
source tree resolution as `pytest`'s `rootdir` discovery — confirmed by
`configfile: pyproject.toml` resolving to this worktree's own file.)

## Evals run

None applicable — no eval harness exists for `orion/substrate/endogenous_curiosity.py`,
same gap as PR #1061's report for its sibling module. The live Postgres
query against `substrate_endogenous_curiosity_candidates` (1,428 rows,
24h span) used to confirm this bug before touching code is the closest
thing to an eval — informal, by hand, not automated.

## Docker/build/smoke checks

Not run — pure Python scoring-logic change inside `orion-substrate-runtime`'s
existing code path, no config/dependency/compose changes to validate. A
live restart is required for this fix to take effect on the running
service (see Restart required below), since the bug is actively firing
right now.

## Review findings fixed

Code-review skill, high effort (correctness, edge cases, test coverage,
conventions, altitude/scope, integration-with-callers angles; dispatched as
a subagent, independently verified the `dynamic_pressure` composite-source
claim against `dynamics.py`'s real code rather than trusting the comment):

- **Finding (Important)**: the `raw_error <= 0.0: continue` short-circuit
  (`endogenous_curiosity.py:145-146`) is a real, correct secondary behavior
  change — previously a node with missing/zero `prediction_error` could
  pass the old `error < min_error` check and seed a spurious
  `"sustained prediction error"` candidate at `signal_strength=0.0` whenever
  a caller configured `min_prediction_error<=0.0` (which both new decay
  tests do, to isolate decay from thresholding) — but no test locked this in
  directly.
  - **Fix**: added
    `test_missing_prediction_error_never_seeds_even_at_zero_threshold`,
    covering both a node with `prediction_error=0.0` explicitly set and a
    node with the key absent entirely.
  - **Evidence**: `orion/substrate/tests/test_endogenous_curiosity.py` —
    17 passed (16 above + this one).
- **Finding (Minor)**: top-of-file comment block (~24 lines) was
  noticeably longer than the reference fix's equivalent docstring in
  `attention_broadcast.py`.
  - **Fix**: trimmed ~30%, kept the live-evidence numbers and the "why not
    `dynamic_pressure`" reasoning (judged load-bearing, not
    keyword-cathedral prose, per AGENTS.md's "deterministic gates over
    repeated yelling" principle — it's the explicit reasoning trail the
    task asked to see).
  - **Evidence**: full suite still passes; diff is comment-only.
- **Finding (Minor)**: `_PREDICTION_ERROR_DECAY_HORIZON_SECONDS` is
  captured once at module-import time from `PressureConfig()`'s bare
  default; a non-issue today (verified repo-wide: `PressureConfig()` is
  only ever instantiated with bare defaults, never from env, in exactly two
  places) but would silently diverge from `pressure.py`'s live value if
  that config is ever made env-configurable later.
  - **Fix**: added a one-line comment flagging this as a future landmine.
  - **Evidence**: comment-only change, full suite still passes.

Not fixed, accepted (Minor, review noted as pre-existing and copied
faithfully from `pressure.py`'s own identical behavior, not newly
introduced): `_prediction_error_staleness_decay()`'s `now` parameter is
never coerced to timezone-aware (only `observed` is) — a naive `now` passed
against an aware `observed` would raise `TypeError`. Every actual caller
(`worker.py`'s `datetime.now(timezone.utc)` default, both test files' aware
`_NOW`) already passes aware values.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```
Not run this session. This fix will not take effect on the live,
currently-firing bug until `orion-substrate-runtime` is rebuilt and
restarted with this branch merged.

## Risks / concerns

- Severity: low
- Concern: a freshly-written `prediction_error` node still reports its full
  raw strength immediately (decay factor 1.0 at age 0), same as before this
  fix — no new latency introduced, since (unlike the salience fix) this
  path never depended on a dynamics-tick interval to populate a value.
- Severity: low
- Concern: `node:substrate.transport` has been stuck at `signal_strength=1.0`
  for at least 24h (likely 6+ days, matching the PR #1061 investigation's
  finding on the same node in the salience path). Once this fix deploys,
  that candidate will decay to `0.0` and stop appearing — whether the
  underlying `substrate.transport` node is genuinely still being surprised
  (and `_write_prediction_error_node()` simply isn't being called to
  refresh `observed_at`, a possible separate bug) or has been legitimately
  quiet was not root-caused here, matching PR #1061's own explicitly
  deferred non-goal on the same question. Worth a live re-check after
  deploy: if `node:substrate.transport` candidates never reappear even
  though transport health is still degraded, that points to
  `_write_prediction_error_node()` not being re-invoked, a different bug.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1072

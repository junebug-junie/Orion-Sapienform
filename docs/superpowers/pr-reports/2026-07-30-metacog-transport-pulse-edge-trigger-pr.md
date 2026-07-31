# Metacog transport pulse: edge-triggered, not a per-poll level check

Branch: `fix/metacog-transport-pulse-edge-trigger`
Date: 2026-07-30
Status: **DONE_WITH_CONCERNS** (two live bugs found and disclosed, not fixed here)

## Summary

- The `bus_synaptic` branch of the `transport` metacog trigger was a pure **level** check on a 30s
  poll against a 30s cooldown lane — no effective rate limit — so one sustained condition re-drafted
  an LLM reflection on every tick.
- Live: `transport` wrote **1,812 `orion_metacog` rows in 24h**, ~48% from this one branch; 5,958
  rows in the table overall.
- Now fires on the **rising edge only** (episode start), with a hysteresis re-arm band.
- A staleness guard was written, then **removed** after live measurement proved the field it gated on
  is clobbered. Removing it is correct and covered by a test.
- Also reverts a live `.env` mistake of mine that had made this fire *more*, not less.

## Outcome moved

Metacognition entries for this branch now correspond to **events**, not polls. A sustained anomaly
that previously produced ~2,880 near-identical entries/day produces one. That is the difference
between "something notable happened" and "something is still the case", and it is the precondition
for any "what happened now, what happened next" reducer over this table: you cannot sequence
episodes when every tick is its own row.

## Architecture touched

`services/orion-equilibrium-service` only. No schema, no bus channel, no contract change. The
sibling transport branches (Option A `rpc_health`, Option C `rpc_timeout`) are untouched —
confirmed in review.

## Files changed

- `app/transport_metacog_gate.py` — rising-edge check; `episode_start` in `reason`; `transition`/
  `node_age_sec` in `upstream`; the "why there is NO staleness guard" note.
- `app/service.py` — edge state, `observed_at` fetch, `_node_age_sec()` helper, latch-on-publish.
- `app/settings.py` — `CLEAR_RATIO` (bounded `(0, 1]`).
- `.env_example`, `docker-compose.yml` — env parity.
- `tests/test_transport_metacog_gate.py` — gate contract.
- `tests/test_bus_synaptic_poll_state_machine.py` (new) — the state machine itself.

## The three mechanisms

1. **Rising edge.** Fires on the transition into anomaly; silent while it persists. This also makes
   `error_threshold` far less load-bearing — a mis-set threshold now costs one spurious entry per
   episode instead of one every 30s. That matters because the threshold calibration is still open
   (see Risks).
2. **Hysteresis re-arm** (`CLEAR_RATIO=0.8`). Rising-edge alone still re-fires on every crossing.
   Deliberately *not* solved by raising `EQUILIBRIUM_METACOG_TRANSPORT_COOLDOWN_SEC` — that lane is
   shared, and throttling it would starve the siblings (the exact bug per-kind lanes were introduced
   to fix).
3. **Latch on publish success.** See review finding 1 below.

## Two live bugs found while verifying — disclosed, not fixed

**1. `observed_at` / `recency_score` are clobbered on substrate nodes.** Sampling
`node:substrate.bus_synaptic` every 35s returned `observed_at` = `03:43:49` → `04:01:25` →
`03:43:49`, oscillating between fresh and ~18min stale, with `recency_score` flipping in lockstep
(0.715 → 0 → 0.999) while `prediction_error` moved normally the whole time. Independently reproduced
in review across 16 samples over 8 minutes, hitting the identical frozen snapshot timestamp.

Root cause identified in review: `SubstrateDynamicsEngine.tick()` (`orion/substrate/dynamics.py:189-192`)
re-upserts the whole node from its own start-of-tick snapshot with
`skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS`, which covers `prediction_error` (fixed
2026-07-29) but not the temporal fields. **Important correction for the follow-up:** `observed_at`
is *not* a metadata key — it is `node.temporal.observed_at`, so adding it to
`EXTERNALLY_OWNED_METADATA_KEYS` would not fix it. Needs a different mechanism.

This is the same bug class as the metadata-clobber campaign (PRs #1501/#1503/#1505/#1506/#1507), one
layer over.

**2. `rpc_health` (Option A) is now the largest transport polluter**, and has the same level-vs-edge
shape as the branch fixed here. 24h: rpc_health 820 rows vs bus_synaptic 466. Worth the same
treatment.

## Env/config changes

- Added: `EQUILIBRIUM_METACOG_TRANSPORT_BUS_SYNAPTIC_CLEAR_RATIO` (0.8, bounded `(0, 1]`) — present
  in `settings.py`, `.env_example`, `docker-compose.yml`, and the live `.env`; verified in the
  running container.
- Added then **removed** in the same branch: `..._MAX_NODE_AGE_SEC`. Deleted from all four surfaces
  rather than left unused — a config key with no consumer is one a future patch wires back up
  without rediscovering why it was abandoned.

**Live `.env` correction.** I had earlier set
`EQUILIBRIUM_METACOG_TRANSPORT_BUS_SYNAPTIC_ERROR_THRESHOLD=0.15` ahead of the fraction-metric branch
that justifies it. That branch is **not merged**; against the deployed magnitude metric `0.15` is a
*lower* bar than `1.0`, so it made this fire more, not less — 29 rows in the 51 minutes the container
ran with it. Reverted to `1.0`. It moves only in the same change that lands the fraction metric.
This was my error, not a deliberate operator override.

## Tests run

```text
services/orion-equilibrium-service$ pytest tests -q
2 failed, 169 passed, 14 warnings in 4.17s
```

The 2 failures are **pre-existing**, established against a scratch worktree at `origin/main`:

```text
origin/main baseline : 2
this branch          : 2
symmetric difference : (empty)
```

Teeth check on the new state-machine tests — replaying the pre-fix raw-level latching:

```text
OLD raw-level latching -> 1 publish attempted (episode start LOST, never retried)
NEW latch-on-publish   -> 3 attempted, publishes once the lane clears
```

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
Container orion-athena-equilibrium Recreated / Started

running config:  ERROR_THRESHOLD=1.0  CLEAR_RATIO=0.8  POLL_INTERVAL_SEC=30
                 MAX_NODE_AGE_SEC=(removed)
edge-trigger present in container: yes    staleness gate present: no
```

**Honest note on live verification.** An 8-minute post-deploy window showed 0 new rows, but that
proves nothing: the metric was reading `0.011`, far below the `1.0` threshold, so the *old* code
would also have fired zero times. The behavioral claim rests on the tests and the state-machine
teeth check above, not on that window. A real confirmation needs an actual anomaly episode.

## Review findings fixed

1. **must-fix — a cooldown-suppressed episode start was lost permanently.** Edge state latched from
   the raw level before publishing, and `_publish_metacog_trigger`'s return value was discarded. An
   episode start landing while the shared lane was shadowed by a sibling got dropped, latched anyway,
   then silenced forever by `previously_above`.
   - Fix: latch only on a real publish, so it retries until it lands.
   - Evidence: siblings publish ~859 rows/day against a 30s lane ≈ a quarter of the day shadowed.
     Teeth check above.
2. **should-fix — `clear_ratio` unbounded.** `<= 0` made the branch latch `True` after its first fire
   and never re-arm. Now `gt=0.0, le=1.0`.
3. **should-fix — my sibling-volume claim was ~8x wrong and load-bearing.** I wrote "~180 rows/week";
   real figure is 1,451/week. Corrected in place; conclusion unchanged but now correctly supported.
4. **should-fix — `upstream` never reaches the persisted row.** Verified 0 of 1,248 rows carry
   `transition`/`node_age_sec`; it feeds the LLM prompt only and is dropped under budget pressure.
   Comment corrected to credit `trigger_reason`, which is what a reducer can actually read.
5. **should-fix — stale comments** referencing the deleted staleness guard. Removed.
6. **should-fix — tests duplicated the state machine instead of exercising it.** Breaking hysteresis
   in `service.py` left all 20 gate tests green. New `test_bus_synaptic_poll_state_machine.py` drives
   the real bookkeeping (cooldown retry, sustained episode, hysteresis band, read failure, below
   threshold).
7. **nit — tautological assertion** on a hand-written list. Replaced with the real relationship.
8. **nit — `isinstance(..., (int, float))` accepted `bool`.** Tightened.

Confirmed clean by review and not re-litigated: sibling blast radius, env parity, and the
read-failure / restart / `error is None` paths (all hold state rather than re-arming, so none can
reintroduce spam).

## Restart required

Already applied. From the main checkout after merge:

```bash
git pull --ff-only
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
```

## Risks / concerns

- **Severity: note. No falling-edge event.** There is no "episode cleared" trigger, so episode
  duration is unrecoverable and an absence of rows is ambiguous between "still elevated", "restarted",
  and "poll died". The `True→False` transition is already computed, so emitting a resolution event is
  nearly free — and it is likely a real prerequisite for the now/next reducer. Deliberately out of
  scope here.
- **Severity: note. The threshold calibration is still open** (branch
  `fix/bus-synaptic-anomalous-fraction`, unmerged, needs 24h of data). Edge-triggering greatly
  reduces the cost of getting it wrong, which is why this patch does not wait on it.
- **Severity: note.** The hysteresis band `[0.8, 1.0)` is nearly inert at the current threshold — only
  3 of 1,248 historical readings landed in it, because the metric is bimodal (calm ≈ 0.011, anomaly
  ≥ 1.0). Cheap and harmless, and it becomes load-bearing if the fraction metric lands.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1533

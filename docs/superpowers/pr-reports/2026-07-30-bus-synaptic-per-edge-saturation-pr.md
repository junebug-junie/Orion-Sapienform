# orion-heartbeat reheat query: typed, recency-filtered, per-edge saturated

Branch: `fix/bus-synaptic-per-edge-saturation`
Date: 2026-07-30
Status: **DONE** (scope deliberately narrowed mid-PR — see "What was cut and why")

## Summary

- `orion-heartbeat`'s dissipation **reheat** signal was pinned at a permanent `1.0`, so reheat ran
  at constant maximum and gated nothing — the exact opposite of the "dynamically gated gain" the
  N-trajectory ensemble design exists to provide.
- Three separate defects in one query, all live-confirmed: **untyped** match (pulling in
  `CAUSALLY_FOLLOWED_BY` edges that carry `latency_zscore`, not `gap_zscore` — 190 of 438 matched
  rows contributed NULL), **no recency filter**, and **`avg()` over unbounded heavy-tailed values**.
- Fixed: typed `(:Organ)-[rel:PUBLISHES]->(:Channel)`, 1h `last_seen_epoch` filter, per-edge clamp
  before averaging, plus a `None` guard the recency filter newly makes reachable.
- Live-verified after deploy: reheat `signal` went from `1.0` to **`0.26`**, moving with real traffic.
- **Scope cut mid-PR after review**: the parallel fix to `bus_synaptic_prediction_error()` was backed
  out of this branch because it would have replaced a stuck-at-1.0 signal with a stuck-at-0.0 one.
  Details below — that is now a decision for Juniper, not a silent omission.

## Outcome moved

`reheat_prob` went from a constant `0.02` (its ceiling, every tick, regardless of bus activity) to a
live value tracking real publish-cadence anomaly. The dissipation loop's reheat term is now an actual
input rather than a constant, which is the precondition for the ensemble ever reaching a genuine rest
state — the open item at the top of
`docs/superpowers/specs/2026-07-28-precision-weighted-attention-organ-and-heartbeat-discrimination-design.md`.

## The measurement

Real `orion_bus_synapse` graph. Heartbeat's old query (untyped, unfiltered, unclamped) read
**29.278**, of which **28.6 came from a single edge** — `cortex-orch -> Channel`, `|z| = 7087.8`,
last fired **9 hours** earlier. Mean excluding that one edge: `0.701`.

```text
median(|z|)                  0.399
p90                          1.123     <- matches the design spec's stated "1.0-1.1 normal"
p99                          7.375
max                       7087.8
mean(|z|)      [old]        29.278     -> signal min(1, 29.3/3.0) = 1.0000  (pinned)
```

With the fix applied, run directly against live FalkorDB before deploy: **0.680**, matching the
offline computation of 0.687.

## Architecture touched

`services/orion-heartbeat` only. No contract, schema, bus, or env changes. Deliberately does **not**
import `orion.substrate.prediction_error` — that package drags in `requests` and heavier substrate
machinery this service exists to stay off, per its own module docstring.

## Files changed

- `services/orion-heartbeat/app/substrate/bus_synaptic.py`: `_build_query()` replaces the module-level
  query constant. Typed match, recency filter, per-edge `CASE WHEN` clamp, `None` guard on the result.
- `services/orion-heartbeat/tests/test_bus_synaptic.py`: five new tests (see below).

## The NULL guard is not defensive padding

Adding the recency filter makes "every edge has aged out" genuinely reachable. Confirmed live that
`avg()` over an empty match returns a real NULL:

```text
$ GRAPH.QUERY orion_bus_synapse "... rel.last_seen_epoch > <future> ... RETURN avg(...)"
[["avg(...)"],[[null]],["Cached execution: 0", ...]]
```

Without the guard `float(None)` raises every tick, gets swallowed by the existing broad `except`, and
logs a spurious `bus_synaptic_query_failed` warning — for what is actually a correct reading: no live
traffic, therefore no reheat.

## What was cut and why

The branch originally also changed `orion/substrate/prediction_error.py::bus_synaptic_prediction_error()`
to clamp per-edge, fixing `node:substrate.bus_synaptic`'s pinned `1.0` and the false
"Bus Anomaly Detected" alerts it drives. **Code review caught that this would have been a worse bug,
and independent verification against live data confirmed it.**

`bus_synaptic_prediction_error()` subtracts a calm floor of `sqrt(2/pi) = 0.7979` — the theoretical
`E|Z|` for a standard normal population. The live population is narrower than unit normal (its
z-scores are computed against an EWMA variance inflated by the same outliers), so after clamping:

```text
live substrate-tick edge set: n=222
  mean|z| raw         13.5025
  mean|z| clamped      0.5575   <- what the cut patch fed the formula
  calm floor           0.7979
  headroom above floor -0.2404  <- NEGATIVE: pinned at exactly 0.0
```

Response curve of the cut patch:

```text
 19 of 222 edges at 3 sigma  -> first non-zero reading at all
 50 of 222                   -> 0.1883
111 of 222                   -> 0.5331
222 of 222                   -> 1.0000
```

`orion-equilibrium-service`'s gate fires at `error >= 1.0` by default
(`EQUILIBRIUM_METACOG_TRANSPORT_BUS_SYNAPTIC_ERROR_THRESHOLD`), which under that patch requires
**all 222 edges simultaneously at 3 sigma** — structurally unreachable.

So the cut patch traded a permanently-firing detector for a permanently-silent one, and the PR report
originally cited `-> prediction_error 0.0000` as *evidence of success*. That is CLAUDE.md §0A metric
quality gate item 4 verbatim: *"a metric reading a suspiciously clean 0.0 is not automatically
'confirmed calm' either."* Caught by review, verified independently, backed out.

Fixing it properly requires recalibrating the calm floor against the real population **and** retuning
the equilibrium threshold in the same changeset (§6: payload meaning changing without its consumer
migrating). That changes what a live signal means and how a live alert behaves — a Juniper decision,
not one to make unilaterally inside a query fix. Tracked as the next patch, not silently dropped.

Note this half of the fix is unaffected by that problem: heartbeat's query deliberately does **not**
subtract the calm floor (it wants raw ambient hum), so it has no floor coupling, and its live reading
is healthy at `0.262`.

## Schema / bus / API changes

None.

## Env/config changes

None. `_STALE_CUTOFF_SEC` / `_MIN_EDGE_COUNT` mirror `orion-substrate-runtime`'s
`SUBSTRATE_BUS_SYNAPTIC_MAX_EDGE_AGE_SEC` / `_MIN_EDGE_COUNT` defaults, now enforced by a test that
parses that service's real values rather than a comment cross-reference.

## Tests run

```text
services/orion-heartbeat$ ORION_BUS_ENABLED=false pytest tests -q
62 passed, 14 warnings in 16.12s      (57 baseline + 5 new, zero regressions)
```

New tests:

- aged-out-to-NULL returns `0.0` (backed by the live `[[null]]` behavior above)
- query is typed, recency-filtered, and clamped per edge
- **executes `_build_query()` against a live FalkorDB** and asserts it parses
- **companion proving that test has teeth**: the malformed query that passed every substring
  assertion is rejected by FalkorDB
- stale-cutoff/min-count sync, parsing `orion-substrate-runtime/app/settings.py`'s actual defaults

## Evals run

```text
No eval harness exists for services/orion-heartbeat.
```

Flagged, not claimed.

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-heartbeat up -d --build
Container orion-athena-heartbeat Recreated / Started

$ curl -fsS http://localhost:7251/health | jq .last_reheat
raw mean|gap_z| (clamped, recency-filtered): 0.7800
signal  = min(1, raw/3.0)                  : 0.2600
reheat_prob                                : 0.00520
```

Was `raw 29.33 / signal 1.0000 / reheat_prob 0.02000`.

`orion-substrate-runtime` was deliberately **not** redeployed from this worktree: it currently runs
PR #1513's `worker.py` fix, which this branch does not contain, and deploying from here would
silently revert a live-verified fix — precisely the incident `scripts/safe_docker_build.sh` exists to
prevent.

## Review findings fixed

Code review ran in a subagent per CLAUDE.md §12.

- **Finding (must-fix): the `prediction_error.py` change pins the metric at exactly 0.0 and makes the
  equilibrium alert unreachable.**
  - Fix: backed out of this branch entirely (see "What was cut and why").
  - Evidence: independently reproduced — clamped mean `0.5575` vs floor `0.7979`, headroom `-0.2404`;
    19 of 222 edges must simultaneously hit 3σ for any non-zero reading; the default alert threshold
    of 1.0 requires all 222.
- **Finding (must-fix): consumer threshold not migrated in the same changeset.** Moot once the
  above was cut; recorded as the constraint the follow-up patch must satisfy.
- **Finding (should-fix): the stale-cutoff sync test was tautological** — it asserted
  `_STALE_CUTOFF_SEC == 3600.0`, comparing heartbeat's constant to a literal in the same file, and
  could never detect the drift it claimed to guard.
  - Fix: now parses `orion-substrate-runtime/app/settings.py`'s real defaults for both the window and
    the min-count.
  - Evidence: verified it reads `3600.0` / `5` from that file, not from a local literal.
- **Finding (should-fix): the query test passes on broken Cypher.** Review built a malformed query
  (missing `END` and a paren) and confirmed all six substring assertions passed while FalkorDB
  rejects it outright — the failure most likely to slip through, since the broad `except` would
  swallow it into a warning and leave reheat pinned at 0.0.
  - Fix: added a live-FalkorDB-gated test that actually executes the built query, plus a companion
    asserting the malformed variant is rejected.
  - Evidence: both pass against the live graph; skip cleanly when no FalkorDB is reachable.
- **Finding (should-fix): env-configurable vs hardcoded asymmetry.** substrate-runtime's window is
  operator-tunable; heartbeat's is a constant, so an operator tuning the documented knob gets a
  silent split.
  - Fix: the new sync test gates the **defaults**. The override case is called out as a known,
    accepted asymmetry in Risks rather than silently tolerated.
- **Finding (nit): `_MIN_EDGE_COUNT` interpolated bare while siblings used `!r`.** Fixed for a
  uniform "all interpolations are repr'd literals" invariant.
- **Finding (nit): the divergence rationale was already false** — the two services read different
  edge sets (substrate-runtime adds `CAUSALLY_FOLLOWED_BY` latency edges), not just different
  windows. Fixed at the constant's definition, not only in a test docstring.

Verified clean by review, not re-litigated here: typed match correctness against
`orion-bus-mirror/app/graph_writer.py`'s actual `MERGE`, `!r` float interpolation safety (including
FalkorDB parsing `1e-05`/`1e+20`), and no injection surface (both interpolated values are local).

## Restart required

Already applied:

```bash
scripts/safe_docker_build.sh orion-heartbeat up -d --build
```

No restart needed for any other service — nothing outside `orion-heartbeat` changed.

## Risks / concerns

- **Severity: should-know. `node:substrate.bus_synaptic` is still pinned at 1.0 and still driving
  false "Bus Anomaly Detected" alerts.** This PR does not fix that, by design (see "What was cut").
  The next patch needs a calm-floor recalibration plus an equilibrium-threshold retune, decided
  together.
- **Severity: note. Operator-override asymmetry.** Setting
  `SUBSTRATE_BUS_SYNAPTIC_MAX_EDGE_AGE_SEC` in a deployment moves substrate-runtime's window but not
  heartbeat's constant. The new test gates the defaults only. Promoting heartbeat's to real settings
  fields is the fuller fix; not done here because heartbeat has no current reason to diverge, and
  adding an env key with no operator use is its own smell.
- **Severity: note.** The `cortex-orch -> Channel` edge carrying `|z| = 7087.8` is now excluded by
  recency rather than explained. Whether it reflects a real pathology in `cortex-orch`'s publish
  cadence is a separate open question.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1516

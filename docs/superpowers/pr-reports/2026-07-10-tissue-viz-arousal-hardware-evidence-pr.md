# fix(spark-introspector): source tissue-viz arousal from hardware pressure evidence, not saturated resource_pressure

**Status:** IMPLEMENTED, tested, reviewed. Fixes the arousal half of the
"novelty and arousal in Hub are showing 0 and sticking to it" incident —
novelty was fixed in `389e049f`; arousal was diagnosed but explicitly
deferred at Juniper's direction ("diagnose further"). This patch replaces
the deferred diagnosis with a real fix per explicit direction: "we need to
replace it with something that isn't bullshit."

## Summary

Hub's tissue-viz `energy`/`arousal` stat was permanently stuck at `0.0`.

- `_phi_from_self_state()` in `services/orion-spark-introspector/app/worker.py`
  computes `energy = intensity * (resource_cap * execution_cap) ** 0.5`,
  where `resource_cap = 1.0 - resource_pressure.score`.
- `resource_pressure` (`orion/self_state/scoring.py::map_channels_to_dimensions`)
  MAX-aggregates 7 heterogeneous channels: real hardware load
  (`cpu_pressure`, `gpu_pressure`, `memory_pressure`, `disk_pressure`,
  `thermal_pressure`) alongside `transport_pressure` and a generic
  capability-graph `pressure` channel.
- Live evidence (this session): `cpu_pressure=0.92` (real, high but not
  maxed) while a separate `pressure=1.00` channel — sourced from an
  untraced capability in `orion-field-digester`'s field graph — dominates
  via `max()`, permanently pinning `resource_pressure.score` at `1.0`.
  `resource_cap` hard-zeroes to `0`, and multiplication zeroes `energy`
  regardless of what `intensity` or `execution_cap` read.
- The exact stuck capability was **not** traced to completion (field-digester
  has no debug/snapshot endpoint to inspect the raw graph — noted as a gap
  in the original diagnosis and still true).

## Fix

worker.py only receives the already-built `SelfStateV1` payload, not the raw
per-channel pressures internal to self-state-runtime's scoring — so this is
a display-layer fix, same scope discipline as the novelty fix (`scoring.py`
and `builder.py` untouched; still shared cognition machinery feeding
`agency_readiness_score`).

`orion/self_state/builder.py::evidence_for_dimension()` already puts a raw
per-channel breakdown on the wire via `SelfStateDimensionV1.dominant_evidence`
— up to 3 `"channel=value"` strings, sorted by value, for whichever channels
map to that dimension. New `_hardware_resource_pressure()` helper parses
`resource_pressure.dominant_evidence`, keeps only the 5 genuine hardware
channels (ignoring the generic `pressure` and `transport_pressure` entries
that can stick saturated), and returns their max. `_phi_from_self_state()`
uses that value for `resource_cap` when present, falling back to the old
(possibly-saturated) raw score when no hardware channel appears in the
evidence — honest no-signal, never fabricated.

**Known limitation, disclosed not fixed:** `dominant_evidence` only carries
the top 3 channels by value. In an edge case where `pressure` and 2+ other
non-hardware/noisier channels simultaneously outrank a real hardware channel,
that hardware channel falls out of the evidence list and the code falls back
to the old saturated score — no worse than before, but not a guaranteed fix
in that case. A complete fix means widening `evidence_for_dimension()`'s
limit or exposing raw channel pressures on the wire — a `builder.py`/schema
change, out of scope for this display-layer patch.

## Review findings fixed

Ran the code-review skill in a subagent (medium effort); it stalled twice on
its own internal finder-angle orchestration and had to be resumed via
SendMessage before it produced results. Once it did, 3 findings came back —
1 confirmed and unconditional, 2 real but currently guarded upstream:

- **Finding:** Trajectory momentum term (`energy += ... - 0.1 * max(0.0,
  _t("resource_pressure"))`) still read `dimension_trajectory["resource_pressure"]`
  — the delta of the raw, still-saturated aggregate score — even after the
  base term switched to hardware-evidence-derived pressure. If the generic
  `pressure` channel swings, the momentum term would move energy in response
  to it even though the base term no longer reflects it at all —
  reintroducing the same stuck-channel theater into the trajectory
  component alone.
  - **Fix:** the resource_pressure trajectory delta is now only applied when
    `hardware_pressure is None` (i.e. when the base term is also using the
    raw score). When hardware evidence drives the base term, that momentum
    component is omitted rather than mixing units.
  - **Evidence:** new test
    `test_energy_momentum_ignores_raw_resource_pressure_trajectory_when_hardware_evidence_used`
    sets `dimension_trajectory["resource_pressure"]=1.0` alongside hardware
    evidence and asserts `energy` is unaffected by it; a second test
    (`test_energy_momentum_still_uses_raw_trajectory_in_fallback_path`)
    confirms the trajectory term still applies in the no-evidence fallback
    path.
- **Finding:** `dominant_evidence` is an unvalidated `list[str]` crossing a
  service boundary (unlike `dim.score`, schema-bounded to `[0, 1]`). A
  NaN/inf string would pass `float()` without raising, and Python's
  `min(1.0, nan)`/`(negative)**0.5` semantics could silently fabricate
  `energy=1.0` or crash with a `TypeError` on a complex number, depending on
  the value. Currently guarded in practice by `clamp01()` calls upstream in
  `scoring.py` (the only known producer), but that invariant isn't enforced
  locally or by the schema.
  - **Fix:** parsed values are now checked with `math.isfinite()` and
    clamped to `[0, 1]` at this boundary, rather than trusting the upstream
    guarantee to hold forever.
  - **Evidence:** new tests
    `test_hardware_resource_pressure_ignores_non_finite_values` and
    `test_hardware_resource_pressure_clamps_out_of_range_values`.

## Files changed

- `services/orion-spark-introspector/app/worker.py` — `math` import;
  `_HARDWARE_RESOURCE_CHANNELS` frozenset; `_hardware_resource_pressure()`
  helper; `_phi_from_self_state()`'s energy calculation updated to use it
  with the trajectory-term fix.
- `services/orion-spark-introspector/tests/test_tissue_viz_arousal.py` (new)
  — 10 tests: helper unit tests (missing dim, generic-only evidence, mixed
  evidence, multiple hardware channels, non-finite values, out-of-range
  clamping); end-to-end regression reproducing the live incident's exact
  saturated-score-with-real-hardware-evidence case; honest-fallback
  regression (no hardware evidence → unchanged old behavior); both
  trajectory-term regressions from the review fix.

## Tests run

```text
/tmp/orion-test-venv/bin/pytest services/orion-spark-introspector/tests/test_tissue_viz_arousal.py -q
  → 10 passed

/tmp/orion-test-venv/bin/pytest services/orion-spark-introspector/tests -q
  → 119 passed, 1 pre-existing unrelated failure (test_phi_reward_emitted_when_encoder_ok,
    same failure already present on main before this branch, confirmed in the
    prior novelty-fix PR)
```

Regression verified by stashing the worker.py change and confirming 5 of 6
first-commit tests fail (missing attribute / dead-value assertion), then
restoring.

Note: the shared `/tmp/orion-test-venv` was missing `scipy` (a pre-existing
`orion-spark-introspector` dependency, unrelated to this change — the whole
module already imports it via `orion.spark.orion_tissue`); installed it into
that scratch venv to make tests importable at all. Not a project dependency
change (already in `requirements.txt`), no repo files touched for this.

## Evals run

No dedicated eval harness for `orion-spark-introspector`'s tissue-viz stats.
Not adding one in this patch — same disclosed gap as the novelty fix PR.

## Docker/build/smoke checks

Not run against live containers in this environment. This fix could not be
live-verified against `/ws/tissue` the way the novelty bug was (that
required an active incident with a live saturated `resource_pressure`
reading); verification is via the unit tests reproducing the exact
dimension/evidence shapes observed live in this session's earlier
investigation (`cpu_pressure=0.92`, `pressure=1.00`).

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
```
After restart, reconnect to `/ws/tissue` (or reload the tissue-viz page)
during a period of real CPU/memory load and confirm `arousal`/`energy`
varies instead of holding at 0. Note: if the untraced field-digester
capability channel is not currently saturated, or if hardware load happens
to fall outside `dominant_evidence`'s top-3 cutoff at a given moment,
`energy` may still occasionally read low/zero — that's the disclosed
known limitation above, not a regression.

## Risks / concerns

- Severity: low. Additive display-layer fix; `orion/self_state/scoring.py`
  and every other consumer of `resource_pressure` (φ, `agency_readiness_score`)
  are completely untouched.
- Known limitation: top-3 `dominant_evidence` truncation can occasionally
  still mask a real hardware channel (see above) — disclosed, not silently
  left unaddressed. A full fix requires a `builder.py`/schema change.
- The root stuck capability in `orion-field-digester`'s graph is still
  untraced. This patch makes arousal resilient to that channel's saturation
  rather than fixing the saturation itself — `resource_pressure` (as seen by
  φ and `agency_readiness_score`) remains pinned at 1.0 until that's found.

## PR link

Branch pushed: `fix/tissue-viz-arousal-hardware-evidence`.
Compare: https://github.com/junebug-junie/Orion-Sapienform/compare/main...fix/tissue-viz-arousal-hardware-evidence

`gh` CLI unauthenticated in this environment (same as the rest of this
session) — PR not opened via API. To open:

```bash
gh pr create --title "fix(spark-introspector): source tissue-viz arousal from hardware pressure evidence" \
  --base main --head fix/tissue-viz-arousal-hardware-evidence \
  --body-file docs/superpowers/pr-reports/2026-07-10-tissue-viz-arousal-hardware-evidence-pr.md
```

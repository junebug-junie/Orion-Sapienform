# PR: Self-state Phase 2 — deviation-probe measurement instrumentation

## Summary

- Ports `orion.autonomy.deviation_gate.DeviationGate` (proven EWMA-baseline + z-threshold mechanism, already live for chat/biometric tensions) into `orion-self-state-runtime`, per the recommended starting point from the 2026-07-12 brainstorm continuing the self-state/mesh substrate redesign.
- Each tick, every real `SelfStateV1` dimension is observed against its own learned baseline; the resulting deviation impulse is logged alongside the dimension's confidence value so the two can be compared on live data before deciding whether `channel_dimension_confidence()`'s cruder max-min-spread formula should be replaced.
- Deliberately log-only: no new schema field yet ("measure before you name it"), no gated behavior, no score changes.
- Medium-effort code review (3 angles) caught one real issue before it shipped and two smaller ones; all fixed in the same commit.

## Outcome moved

Layer 6 gains a real, proven mechanism for answering "has this dimension been sustained or is this a momentary spike," without inventing new statistics or duplicating `DriveEngine`'s already-fixed cadence-artifact bugs. Nothing is named yet — the point of this patch is to generate real data before shaping a schema field around it.

## Current architecture

Before this PR: Layer 6's only notion of history was a single-previous-tick diff (`dimension_trajectory`, age-gated at 300s) — no sense of "how long has this been true" beyond that. `DeviationGate` already existed and was proven elsewhere (`orion/spark/concept_induction/bus_worker.py`'s chat/biometric tension extraction) but had never been applied to self-state's own dimensions.

## Architecture touched

- `orion/self_state/` — new `deviation.py` module
- `services/orion-self-state-runtime/` — worker wiring, settings, env
- `config/self_state/self_state_policy.v1.yaml` — new `dimension_worse_direction` section
- `orion/self_state/policy.py` — new policy schema field

## Files changed

- `orion/self_state/deviation.py`: new — `observe_dimension_deviation(gate, state, policy)`, reuses `DeviationGate` directly, reads per-dimension "worse" direction from policy config (not a hardcoded table — see Review findings)
- `orion/self_state/policy.py`: new `dimension_worse_direction: dict[str, Literal["up","down"]]` field on `SelfStatePolicyV1`
- `config/self_state/self_state_policy.v1.yaml`: new `dimension_worse_direction` section, all 12 dimensions
- `services/orion-self-state-runtime/app/worker.py`: `DeviationGate` instantiated once in `__init__` (in-memory, per-process, dies with the process — same accepted limitation as `DriveEngine`'s pressure store); new `_log_deviation_probe()` called once per tick, gated by a settings flag
- `services/orion-self-state-runtime/app/settings.py`: new `self_state_deviation_probe_enabled` (default `True`)
- `services/orion-self-state-runtime/.env_example` + local `.env`: new `SELF_STATE_DEVIATION_PROBE_ENABLED=true`
- `tests/test_self_state_deviation.py`, `services/orion-self-state-runtime/tests/test_worker_deviation_probe.py`: new

## Schema / bus / API changes

- Added: `SelfStatePolicyV1.dimension_worse_direction` (config schema, not a bus/API contract)
- No changes to `SelfStateV1` itself — this PR is deliberately schema-silent per its own stated goal
- No bus channel or schema-registry changes

## Env/config changes

- Added keys: `SELF_STATE_DEVIATION_PROBE_ENABLED=true` (`services/orion-self-state-runtime/.env_example` + local `.env`, synced)
- Removed keys: none
- skipped keys requiring operator action: none

## Tests run

```text
pytest tests/test_self_state_builder.py tests/test_self_state_builder_hardening.py \
  tests/test_self_state_schemas.py tests/test_self_state_scoring.py \
  tests/test_self_state_policy_loader.py tests/test_self_state_prediction.py \
  tests/test_self_state_reliability_decontamination.py tests/test_self_state_runtime_store.py \
  tests/test_self_state_transport_dimension.py tests/test_self_state_deviation.py \
  tests/test_proposal_transport_readonly_candidates.py -q
→ 54 passed, 0 failed

pytest services/orion-self-state-runtime/tests/ -q
→ 23 passed, 0 failed
```

`git diff --check`: clean.

## Evals run

None — measurement-only instrumentation, no eval harness applicable yet. The point of this patch is itself to generate a measurement window before any eval-worthy schema/behavior exists.

## Docker/build/smoke checks

Not run — no live Docker environment available in this session. Restart command below is for the operator.

## Review findings fixed

Medium-effort code review (3 of 8 angles: line-by-line scan, removed-behavior/cross-file tracer, reuse/conventions — scoped down from the full 8-angle high-effort pass given this diff's small size).

- Finding: the per-dimension "worse" direction table (`WORSE_DIRECTION`) was a hardcoded Python dict, a second, independently-maintained source of a fact the repo already has an established config-driven pattern for (`config/autonomy/signal_drive_map.yaml`, whose own header states "This is the WHOLE mapping surface" for the same kind of fact) — exactly the "create another place for config to drift" bad seam named in `CLAUDE.md`.
  - Fix: moved to `config/self_state/self_state_policy.v1.yaml`'s new `dimension_worse_direction` section; `observe_dimension_deviation()` now takes `policy` as a parameter instead of importing a module-level table.
  - Evidence: `test_policy_dimension_worse_direction_covers_every_real_dimension` loads the real live YAML, not a Python constant.
- Finding: `DeviationGate.observe()`'s own `float(confidence)` coercion isn't guarded by its internal try/except (only the `x`/score coercion is), contradicting its module docstring's "never raises" claim — a pre-existing gap in the shared primitive, not introduced by this diff, but the new call site should defend anyway, matching the existing `orion/autonomy/signal_tension.py` caller's pattern for this same gate.
  - Fix: defensive `float(dim.confidence)` cast at the call site in `orion/self_state/deviation.py`, falling back to `1.0` on failure.
  - Evidence: `test_observe_dimension_deviation_defensively_casts_bad_confidence`.
- Finding: no way to disable the new per-tick log without a code change/redeploy if it proves noisy at production volume, unlike the sibling perception-input feature documented as explicitly toggleable.
  - Fix: new `SELF_STATE_DEVIATION_PROBE_ENABLED` setting (default `true`), checked at the top of `_log_deviation_probe()`.
  - Evidence: `test_log_deviation_probe_disabled_via_settings_does_not_log`.
- Finding: `test_worse_direction_matches_scoring_conventions` only re-asserted the hardcoded up/down values already in the table it was testing, rather than independently deriving them from real gate behavior — only one of 12 dimensions was actually behaviorally verified.
  - Fix: rewrote as `test_worse_direction_behaviorally_matches_scoring_conventions_for_every_dimension`, driving a real rise and a real fall through a fresh gate for all 12 dimensions.
  - Evidence: the new test itself; would have caught a flipped direction on any of the 11 previously-unverified dimensions.
- 1 finding considered and not implemented (documented, non-blocking): resetting `self._deviation_gate`'s baselines when the self-state policy changes, mirroring how `previous` (the stored prior self-state) is invalidated on a `policy_id` mismatch. Checked and confirmed inapplicable: `self._policy` is assigned exactly once, in `SelfStateRuntimeWorker.__init__`, with no reload path — the gate and the policy always construct and die together on process restart, unlike `previous`, which is loaded from Postgres and genuinely crosses restarts. The analogy doesn't transfer to this case.
- 1 pre-existing issue found but explicitly out of scope: `DeviationGate.observe()`'s own unguarded confidence coercion (see above) is a latent gap in a shared module used elsewhere too (`orion/spark/concept_induction/bus_worker.py`) — flagged as a candidate follow-up, not fixed here, since it's unrelated to this PR's actual integration and touching shared code beyond the new call site would widen scope unnecessarily.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-self-state-runtime/.env \
  -f services/orion-self-state-runtime/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
  Concern: `DeviationGate`'s baselines are in-memory only; a worker restart resets every dimension's learned baseline to cold-start (first observation only, zero warmup progress). Given the default `warmup=5`, the first ~5 ticks (~10s at the 2s poll interval) after any restart produce no impulses.
  Mitigation: accepted for this measurement-only pass, same as `DriveEngine`'s own pressure store; persistence is a candidate follow-up if this graduates past instrumentation.
- Severity: low
  Concern: `DeviationGate.observe()`'s own confidence-coercion gap (see Review findings) remains unfixed at the shared-module level.
  Mitigation: flagged as a follow-up; this PR's own call site defends against it, so no live risk from this integration specifically.

## PR link

<!-- filled in after push -->

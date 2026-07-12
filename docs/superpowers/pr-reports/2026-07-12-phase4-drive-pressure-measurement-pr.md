# PR: Phase 4 — DriveEngine vs AutonomyStateV2 pressure measurement probes

## Summary

- Measurement-only, log-only instrumentation (no schema change, no behavior change) comparing `DriveEngine`'s and `AutonomyStateV2`'s independently-computed drive-pressure vectors, per `docs/superpowers/plans/2026-07-12-self-state-mesh-substrate-redesign.md`'s Phase 4 ("Drive unification") first step: "log side-by-side on the same live traffic window... before deciding merge vs. keep-separate."
- Both computations key off the same six-drive taxonomy (`coherence, continuity, capability, relational, predictive, autonomy`) but are fed by completely different inputs (`DriveEngine`: self-state tensions + biometrics via `orion/spark/concept_induction/bus_worker.py`; `AutonomyStateV2`: chat-turn evidence only, via `services/orion-cortex-exec/app/chat_stance.py`, test-banned from touching self-state) — this is the first time either has been directly comparable.

## Outcome moved

The merge-vs-keep-separate decision from the redesign spec can now be made from real correlated data (grep both services' logs, join on `subject` + nearby timestamp) instead of the assumption either direction currently rests on.

## Architecture touched

- `orion/spark/concept_induction/bus_worker.py`
- `services/orion-cortex-exec/app/chat_stance.py`

## Files changed

- `orion/spark/concept_induction/bus_worker.py`: new `_log_drive_pressure_probe(self, subject, pressures)`, try/except-guarded (never breaks the drive-update rail, which runs on live bus traffic), called right after both `save_drive_state` call sites (~line 745, ~874)
- `services/orion-cortex-exec/app/chat_stance.py`: new module-level `_log_autonomy_pressure_probe(subject, pressures)`, same guard pattern, called right after `_run_autonomy_reducer`'s existing `save_autonomy_state_v2` try/except (mirrors that block's defensive style directly)
- `orion/spark/concept_induction/tests/test_drive_pressure_probe.py`: new, 5 tests (helper behavior, both call sites, never-raises for each)
- `services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2_pressure_probe.py`: new, 4 tests (probe fires with correct subject/pressures, pressures match the folded state exactly, never-raises through the full call path and directly on the helper)

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
pytest orion/spark/concept_induction/tests/ -q
→ 119 passed (114 pre-existing + 5 new)

pytest services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2_pressure_probe.py -q
→ 4 passed
```

Both verified directly by the orchestrator (not just the implementing agent) on the actual committed branch. `git diff --check`: clean.

Note: a broader `-k autonomy_v2`/`-k chat_stance` sweep across the whole `services/orion-cortex-exec/tests/` directory shows pre-existing collection errors (`ValueError: Verb already registered: legacy.plan`) and 2 pre-existing test failures unrelated to this change — confirmed present on `main` with this patch fully reverted, not introduced here.

## Evals run

None applicable — pure logging instrumentation.

## Docker/build/smoke checks

Not run this session. Restart commands below for when this merges.

## Review findings fixed

No dedicated code-review round for this PR — small, self-contained, measurement-only addition following an already-established pattern (mirrors `orion-self-state-runtime`'s Phase 2 deviation-probe logging exactly in style and defensive guarding). Verified directly instead: read both diffs in full, confirmed the try/except placement matches the existing defensive pattern immediately above each new call site, ran both test suites myself.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
  Concern: this is pure `logger.info` output on hot paths (every accepted bus event for `DriveEngine`, every chat turn for `AutonomyStateV2`) — adds log volume with no way to disable short of a redeploy, matching the same tradeoff already accepted for Phase 2's deviation probe (no feature flag was added there either, by the same reasoning: cheap, in-memory, fail-open).
  Mitigation: none needed yet; add a settings flag if it proves noisy in practice, same escalation path as Phase 2.
- Severity: informational
  This PR only produces the *data*; it does not decide merge-vs-keep-separate. That decision is Phase 4's next step, gated on actually collecting and comparing a real traffic window's worth of both logs.

## PR link

<!-- filled in after push -->

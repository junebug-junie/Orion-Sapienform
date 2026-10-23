# feat(body-economy): readers registry + "no unregistered veto" CI gate (Patch A1)

## Summary

- New registry `config/physiology/contract.yaml`. It lists every code site that reads Orion's regulation, arousal or thermal state, with a role for each: `spinal` (may veto), `bidder` (vetoes today, moves to `bid()` in Patch D), or `display` (reads, gates nothing). Modules that define or produce the state are listed under `sources`.
- New gate `scripts/check_body_economy_contract.py`. It scans the repo's Python by syntax tree (AST), not by a list of module names. It fails when a module reads the state but is not registered, when a registered module no longer reads it, or when the YAML is malformed. It runs in about 0.8 s.
- The gate runs in CI (`orion-static-gates.yml`) and locally (`make check-body-economy-contract`).
- Removed the zombie env keys `ORION_ATTENTION_HABITUATION_ENABLED` and `ORION_ATTENTION_SALIENCE_WEIGHTS` from `services/orion-cortex-exec/.env_example`. No code reads either key.
- The "synthetic goal in `capability_policy`" is already gone. Nothing was removed for it (details below).

## Outcome moved

Before this patch, nothing stopped a new `if arousal == "strained": return` from landing. The body-economy design says that this pile-up of strained vetoes is how every earlier economy decayed. Now any new module that reads the state fails CI. The failure message names the module and the fix. Phase 1 can retire the old vetoes one by one, and the pile cannot grow back in the meantime.

## Current architecture

Arousal: durable-runs writes one `RegulationStateV1` to Redis `orion:regulation:latest` on every regulate step. Readers use `orion.regulation.arousal_view` / `arousal_read`, or the Hub's `ArousalReader`, and each one refuses optional work with the reason `arousal_strained`. Heat: `orion.autonomy.thermal_gate` defines the 32 C and 34 C lines. `orion.autonomy.cabinet_heat` reads the cabinet sensor. Hardware-watch raises the `cabinet_hot` reflex and the GPU pool sheds on it. Before this patch, no registry of these readers existed.

## Architecture touched

- New contract file `config/physiology/contract.yaml` (Phase 2 "one source of truth", seeded with `protected_paths`, `readers`, `sources`; the `constraints` table lands with Phase 1 Budget).
- New static gate and CI step. No runtime code changed.
- One env template edit (cortex-exec), removal only.

### Readers table (found by the gate's own scan, not copied from the design doc)

54 modules read the state. 38 are readers and 16 are sources.

| role | count | modules |
|---|---|---|
| spinal | 4 | `orion/hardware_watch/rules.py`, `services/orion-hardware-watch/app/watcher.py`, `orion/gpu_pool/shed.py`, `services/orion-gpu-pool/app/runtime.py` |
| bidder | 17 | reverie `baseline.py`; proposal-runtime `worker.py`; `world_pulse_read/pull.py` + Hub `world_pulse_read_pipeline.py`; Hub `retro_leisure.py`; `mutation_pressure.py`, `mutation_worker.py`, Hub `api_routes.py` (`_mutation_arousal_level`); dream `cycle.py` + `main.py`; Hub `curiosity_investigation.py`, `endogenous_outreach.py`, `main.py` (wires the reader); thought `chain.py`, `visual_chain.py`; gpu-pool `guards.py`; `autonomy/self_shed.py` |
| display | 17 | thought `visual_chain_health_monitor.py`; Hub `temporal_self_routes.py`; substrate-runtime `worker.py`/`store.py`; proposal-runtime `store.py`; feedback-runtime `worker.py`/`store.py`; `feedback/world_settlement.py`; `hardware_watch/shed_effect.py`; `temporal_self/arcs.py`, `sources.py`, `stance_cue.py`; five `scripts/analysis/*` replay tools |
| sources | 16 | `thermal_gate.py`, `cabinet_heat.py`, `regulation/arousal.py`, `arousal_view.py`, `arousal_read.py`, `strain_news.py`, `schemas/regulation.py`, `schemas/registry.py`, `schemas/hardware_watch.py`, `inner_state_registry.py`, Hub `arousal_reader.py`, durable-runs `main.py`, `regulation_store.py`, `temporal_self_graph.py`, `temporal_self_driver.py`, `runner.py` |

How this differs from the design doc's list ("about a dozen" strained sites, under Fact 3):

- **The doc missed these sites:**
  - `services/orion-gpu-pool/app/guards.py`: it refuses a gpu2 model load at the 32 C line.
  - `orion/autonomy/self_shed.py`: Orion's own shed act is eligible only while the cabinet is warm.
  - Hub `api_routes.py`: the mutation-cycle arousal read.
  - Hub `main.py`: it constructs the reader.
  - dream `main.py`: it wires the Redis read.
  - `world_pulse_read_pipeline.py`: it feeds thermal state into the pull gate.
  - All of the display and source sites.
- **The doc miscounted one site:** the visual-chain health monitor is display, not a gate. Its own code says "Diagnostic read only (it gates nothing)" (`visual_chain_health_monitor.py:277`).
- **"world_pulse_read resonance":** `resonance.py` does not read arousal. Its "strained" wording is about a reading straining a belief. Only `pull.py` gates.
- **Needs Juniper's call:** gpu-pool `guards.py` is labelled `bidder` with a note. It is a hardware-flavoured guard at 32 C, not the 34 C reflex. Making it `spinal` needs the sign-off the design requires for spinal entries.

### How detection works (structural)

A module reads the state when any one of these is true:

1. It imports from `orion.regulation.{arousal,arousal_view,arousal_read,strain_news}`, `orion.schemas.regulation`, `orion.autonomy.{thermal_gate,cabinet_heat}`, or `orion.physiology`.
2. It references a state symbol such as `RegulationStateV1`, `ArousalView`, `ArousalReader`, `ThermalVerdict`, `CabinetHeatReading` or `PhysiologyStateV1`.
3. It holds one of these literals: `"orion:regulation:latest"`, `"arousal_level"`, `"arousal_strained"`/`"arousal_engaged"`/`"arousal_engaged_cooldown"`, `"cabinet_hot"`.
4. It compares a value against `"strained"`: `==`, `!=`, or `in` a literal tuple, list or set. This rule catches vetoes whose level arrives through an injected callable, such as `mutation_worker.py`.

Comments, docstrings and substring tests (`"strained" in text`) do not count. Tests and evals are not scanned.

Known limit: a module that receives the state only through an object passed in by its caller is not seen if it has no watched import, name or literal of its own. The wrapper's module and the caller that wires it are seen.

## Files changed

- `config/physiology/contract.yaml`: new registry.
- `scripts/check_body_economy_contract.py`: new gate.
- `tests/test_check_body_economy_contract.py`: 28 tests covering seeded fixtures, each detection rule, non-detections, malformed YAML, exit codes, and the real repo staying green.
- `Makefile`: `check-body-economy-contract` target.
- `.github/workflows/orion-static-gates.yml`: CI step.
- `services/orion-cortex-exec/.env_example`: zombie keys removed.
- `docs/superpowers/pr-reports/2026-10-23-body-economy-veto-gate-pr.md`: this report.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. Behavior changed: none at runtime.
- Compatibility notes: CI now fails a PR that adds an unregistered reader of regulation/thermal state.

## Env/config changes

- Added keys: none.
- Removed keys: `ORION_ATTENTION_HABITUATION_ENABLED`, `ORION_ATTENTION_SALIENCE_WEIGHTS` (orion-cortex-exec only; the root, orion-thought and orion-substrate-runtime templates already dropped them on 2026-07-31).
  - Evidence that nothing reads them: `rg ORION_ATTENTION_HABITUATION_ENABLED|ORION_ATTENTION_SALIENCE_WEIGHTS` outside docs finds only `.env_example` lines and comments. No Python, compose or shell file reads either key.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran it. It only adds keys and never removes them, so I removed the two dead lines from the primary checkout's `services/orion-cortex-exec/.env` by hand. That file is gitignored, and no code reads those keys.
- Skipped keys requiring operator action: none.

### Synthetic goal in `capability_policy` (investigated, not removed)

It is already gone. Commit `3354188e1` ("capability_policy.py reads real field-native goal") replaced the fabricated `GoalProposalV1` with `orion.autonomy.goal_state.get_active_goal()`. That function returns a `FieldGoalProvenanceV1 | None`, and `None` is an honest absence. `goal_proposal_from_episode_intent()` no longer exists anywhere in the repo. `CapabilityEvaluationContext.goal` is typed `FieldGoalProvenanceV1 | None` (`orion/autonomy/capability_policy.py:41-45`). Removing anything more would change runtime behaviour, so I did not touch it. The design doc's Retirements row for it is stale and should be marked done.

## Tests run

```text
python -m pytest tests/test_check_body_economy_contract.py -q            -> 28 passed
python -m pytest tests/scripts/test_sync_local_env_from_example.py scripts/tests/test_check_env_template_parity.py tests/test_check_body_economy_contract.py -q -> 66 passed
python scripts/check_body_economy_contract.py                            -> OK (54 state-reading modules, all registered), ~0.8 s
make check-body-economy-contract                                         -> OK
python scripts/check_env_template_parity.py                              -> PASS (95 services)
git diff --check origin/main..HEAD                                       -> clean
mutation check: disabling the "compares against 'strained'" rule turns 6 tests red (G1 fixture included)
```

`scripts/check_service_env_compose_parity.py orion-cortex-exec` crashes on the compose file's `!override` YAML tag. The same crash happens on main, so this branch did not cause it.

## Evals run

```text
None. This is a static CI gate with no runtime behaviour to evaluate. The design's G6 check
(run against the last 30 merged PRs) belongs to the full Phase 2 PR.
```

## Docker/build/smoke checks

```text
Not applicable: no service code, compose or Dockerfile changed.
```

## Review findings fixed

(filled in after the review subagent)

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: detection can miss a reader that gets the state only through an injected object. Mitigation: the caller that wires the object is detected. Rule 4 catches the `== "strained"` shape wherever it appears.
- Severity: low. Concern: a false positive costs one contract row. Mitigation: each row is one line, and the message says exactly what to add.
- Severity: decision. Concern: gpu-pool `guards.py` is `bidder` until Juniper decides whether the 32 C swap-load guard is spinal.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

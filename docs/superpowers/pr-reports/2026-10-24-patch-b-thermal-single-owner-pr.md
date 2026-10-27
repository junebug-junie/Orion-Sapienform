# PR report: body economy Patch B, track 2 -- one owner for the room's thermal lines; delete quota_budget.py

## Summary

- The room's temperature lines (hot 32.0 C, re-arm 30.5 C; critical 34 C / 33 C; elevated 29.5 / 28.0 C) now live in one place, `orion/autonomy/thermal_gate.py`. Every consumer reads them from there.
- orion-thought had its own copy as two env keys. Both keys are deleted. The visual chain imports the owner's constants.
- The prediction service's series registry had the 34 C and 32 C lines typed in as numbers. It now names them (`value_from: thermal_gate.critical` / `thermal_gate.hot`), and a typed-in number on the cabinet series is refused at load.
- The CI gate `scripts/check_env_key_single_source.py` now refuses any new copy of a thermal line (env key, compose, settings `Field`, Python constant, parameter default). It runs on every PR in `orion-static-gates`.
- `orion/autonomy/quota_budget.py` is deleted, with its replay script and tests. Nothing in the runtime imported it.

## Outcome moved

Before: retuning the hot line meant editing up to four places (thermal_gate.py, the thought settings default, the thought `.env_example`, the live `.env`, the prediction YAML), and nothing failed when one was missed. After: one edit in `thermal_gate.py`, and CI refuses a new copy.

## Current architecture

- `orion/autonomy/thermal_gate.py` defined the lines and `thermal_state()` (hysteresis; missing or >300 s stale reading = `unknown`, which allows).
- orion-thought: `settings.thermal_hot_c` / `thermal_hot_rearm_c` (`ORION_THERMAL_HOT_C=32.0`, `ORION_THERMAL_HOT_REARM_C=30.5`) passed into `thermal_state()`.
- gpu-pool swap guard: already read the owner through `orion.autonomy.cabinet_heat.read_cabinet_heat` (no numeric copy; docstring restated the numbers).
- Hub reading-pull gate: already called `thermal_state()` with its defaults (no copy).
- orion-prediction: `config/prediction/series.v1.yaml` had `value: 34.0` (shed) and `value: 32.0` (experiment_guard).
- hardware-watch replay eval: `CRITICAL_C = 34.0`, `ELEVATED_C = 29.5` (found by the new gate, not in the brief).

## Every copy found, and how it reads the owner now

| copy | before | now |
|---|---|---|
| orion-thought `settings.py` + `.env_example` + live `.env` | `ORION_THERMAL_HOT_C=32.0`, `ORION_THERMAL_HOT_REARM_C=30.5` | deleted; `visual_chain.py` imports `DEFAULT_HOT_C`, `DEFAULT_HOT_REARM_C` |
| `config/prediction/series.v1.yaml` | `value: 34.0`, `value: 32.0` | `value_from: thermal_gate.critical` / `thermal_gate.hot`, resolved in `services/orion-prediction/app/series.py` |
| `services/orion-hardware-watch/evals/run_rules_replay_eval.py` | `CRITICAL_C = 34.0`, `ELEVATED_C = 29.5` | `from orion.autonomy.thermal_gate import DEFAULT_CRITICAL_C as CRITICAL_C, DEFAULT_ELEVATED_C as ELEVATED_C` |
| gpu-pool `app/guards.py` | no numeric copy (via `read_cabinet_heat`) | unchanged code; docstring names the owner constants. Still refuses gpu2 at `hot` and on `unknown`. |
| Hub `world_pulse_read_pipeline.py` `_pull_thermal` | no numeric copy (`thermal_state()` defaults) | unchanged; decision table pinned |

Search method: AST scan of every `.py` for numeric constants 32/30.5/34/33/29.5/28 on lines naming a temperature line; env/compose/YAML scan; then the extended gate itself, which found the hardware-watch eval copy.

## Architecture touched

orion-thought, orion-prediction, orion-gpu-pool (docstring only), orion-hardware-watch (eval only), orion-cocreation-signals (`.env_example` comment only), shared `orion/autonomy`, `orion/dev_economics`, `orion/schemas/claude_limit.py` (docstrings), CI gates and workflows.

## Files changed

- `orion/autonomy/quota_budget.py`, `scripts/report_quota_budget.py`, `tests/test_quota_budget.py`, `tests/test_report_quota_budget.py`: deleted.
- `services/orion-thought/app/{settings.py,visual_chain.py}`, `.env_example`: keys deleted; read the owner.
- `services/orion-prediction/app/series.py`, `config/prediction/series.v1.yaml`, `README.md`: `value_from`.
- `services/orion-hardware-watch/evals/run_rules_replay_eval.py`: import the owner.
- `services/orion-gpu-pool/app/guards.py`: docstring.
- `scripts/check_env_key_single_source.py`: `MODULE_OWNED_LINES`, `RETIRED_KEYS`; files read once (runtime unchanged, ~8 s).
- `config/physiology/contract.yaml`: registers `services/orion-prediction/app/series.py` as `display` (it now imports thermal_gate).
- Pin tests: `services/orion-thought/tests/test_thermal_line_pin.py`, `services/orion-gpu-pool/tests/test_guard_thermal_line_pin.py`, `tests/test_rung3_reading_pull_thermal_pin.py`, `services/orion-prediction/tests/test_series.py`, `tests/test_env_key_single_source.py`.
- Workflows: `orion-static-gates.yml` (gate on every PR), `regulation-arousal.yml`, `thinking-rung3.yml`, `visual-baseline.yml`, `orion-prediction-tests.yml`.
- Docstring references to the deleted module: `orion/autonomy/thermal_gate.py`, `orion/bus/channels.yaml`, `orion/dev_economics/rate_limit_events.py`, `orion/schemas/claude_limit.py`, `tests/test_rate_limit_events.py`, `services/orion-cocreation-signals/.env_example`.

## Schema / bus / API changes

- Added: `value_from` on a series threshold in `config/prediction/series.v1.yaml` (allowed: `thermal_gate.hot`, `thermal_gate.critical`).
- Removed: none on the bus.
- Behavior changed: none. Every consumer's decision table is pinned by hand and passed before and after the change.
- Compatibility: a cabinet series threshold with a literal `value` is now refused at registry load.

## Env/config changes

- Removed keys: `ORION_THERMAL_HOT_C`, `ORION_THERMAL_HOT_REARM_C` (orion-thought).
- `.env_example` updated: yes (orion-thought).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. The sync never deletes, so the two lines were removed by hand from `/mnt/scripts/Orion-Sapienform/services/orion-thought/.env` (was lines 148-149: `ORION_THERMAL_HOT_C=32.0`, `ORION_THERMAL_HOT_REARM_C=30.5`). Live container env had the same values as the new owner (verified with `docker exec orion-athena-thought env`).
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-thought: test_thermal_line_pin, test_visual_chain_thermal_gate, test_visual_chain, test_visual_steps, test_settings_mind_enrichment -> 135 passed
  (pin table run against the old code first: passed, only the "no copy" assertion failed, as intended)
services/orion-gpu-pool tests -> 198 passed, 15 skipped
services/orion-prediction tests/test_series.py tests/test_effects.py -> 30 passed
  (full suite: 8 failures, all ModuleNotFoundError prophet/statsmodels or their downstream in the local venv; untouched code)
services/orion-hub tests/test_world_pulse_read_pipeline.py -> 54 passed
tests/test_rung3_reading_pull.py + tests/test_rung3_reading_pull_thermal_pin.py -> 31 passed
tests/test_rate_limit_events.py tests/test_thermal_gate.py orion/autonomy/tests -> 264 passed
tests/test_env_key_single_source.py -> 29 passed, 1 skipped (no local .env in worktree)
services/orion-hardware-watch tests -> 81 passed
```

Static gates: `check_env_key_single_source.py` OK; `make check-body-economy-contract` OK (69 modules); `check_definition_drift.py --gate` PASS; `check_metric_lineage.py --gate` PASS; `check_env_template_parity.py` PASS; `check_inner_state_registry.py` OK.

## Evals run

```text
services/orion-hardware-watch/evals/run_rules_replay_eval.py -> PASS (now reads CRITICAL_C / ELEVATED_C from the owner)
```

## Docker/build/smoke checks

Not run. No new dependency; orion-prediction already ships `orion/` in its image.

## Review findings fixed

Read-only review subagent. No behaviour change found in any consumer. When the gate was pointed at main's pre-patch code, it flagged all 6 expected copies.

- Finding (should): `scripts/analysis/fit_heat_sensitivity.py` loads `series.py` by path, and the new `orion` import broke it outside CI.
  - Fix: the script puts the repo root on `sys.path`. `body-economy-analysis.yml` now watches `orion/autonomy/thermal_gate.py`.
  - Evidence: `central_ac_proxy_spec()` loads from /tmp. `scripts/analysis/tests/test_body_economy_replay_and_fit.py`: 13 passed.
- Finding (should): the hardware-watch eval still had three literal `32.0` hot-line copies.
  - Fix: imports `DEFAULT_HOT_C as HOT_C`. The output label names (`hot_ge_32`) are unchanged.
  - Evidence: `run_rules_replay_eval.py` PASS.
- Finding (should): the gate missed call keywords (`thermal_state(..., hot_c=32.0)`), dict literals and negative numbers.
  - Fix: all three are scanned now. The name rule needs a line word (hot/crit/elevat/rearm/trip/limit/threshold/line), so readings such as `cabinet_temp_c=32.0` are not flagged.
  - Evidence: new parametrized cases in `tests/test_env_key_single_source.py`. 40 passed.
- Finding (should): `orion-gpu-pool-tests.yml` did not run when the owner changed.
  - Fix: added `orion/autonomy/thermal_gate.py` and `cabinet_heat.py` to its paths.
- Finding (nit): quoted compose defaults (`KEY: "${KEY:-32.0}"`) and `export KEY=` slipped through.
  - Fix: the regex covers both, with tests.
- Finding (nit): the prose-exemption test passed for the wrong reason.
  - Fix: it now runs the retired-key check (`_module_owned_failures`) on prose. It is paired with tests that refuse setting or reading a retired key.
- Not fixed (nit, pre-existing): the gate walks gitignored trees (`.venv`, nested worktrees) in the primary checkout (~90 s there; ~8 s in CI and in a worktree). Out of scope.
- Known gap: a bare comparison (`temp_c >= 32.0`) is not a named line, so the gate does not see it. Review caught the eval's instances by hand.

## Restart required

Rebuild order does not matter (no consumer changes its decision). From the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-prediction up -d --build
```

orion-gpu-pool, orion-hub, orion-hardware-watch: no runtime change (docstring / test / eval only); no rebuild needed.

## Risks / concerns

- Severity: low. Concern: prediction registry now imports `orion.autonomy.thermal_gate` at load. Mitigation: the image already copies `orion/`; module is stdlib-only.
- Severity: low. Concern: the gate matches names by a thermal word plus a `_c` suffix AND an owned value; an oddly named copy (e.g. `LIMIT = 32.0`) passes. Mitigation: the prediction registry refuses literal cabinet thresholds independently.

## PR link

PR_LINK

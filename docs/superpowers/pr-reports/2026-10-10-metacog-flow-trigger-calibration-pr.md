# Retire the metacog "flow" trigger: its plateau is the idle rest state

## Summary

- The "flow" metacog trigger was supposed to fire rarely, when Orion's prediction confidence held a genuine high, steady plateau. On live data it fired on 38% of windows (calibrated for 3.2%), and the only thing limiting it was its 30-minute cooldown: ~30 metacog entries a day, half of them exactly 1800 s apart.
- Checked whether a better threshold could rescue it. It can't. The confidence value is `1 - mean` prediction error over five domains, and three of those (chat, execution, route) read exactly 0 whenever nothing is happening. A high calm plateau is what idleness looks like. 93% of the calmest 3.2% of windows had no chat, execution, or route activity at all.
- So the trigger is **retired** (kill means kill): gate module, detector, settings, env and compose keys, cooldown lane, cortex-exec's `trigger_kind=="flow"` type override, and the inner-state registry consumer are all removed. The metric lock is regenerated.
- `orion/metacog/evidence_map.py`'s `map_flow` stays, documented as a reader for the 872 historical `flow` rows. Nothing produces new ones.
- `insight`, which reads the same field but looks for a dip followed by a recovery, is unchanged.

## Outcome moved

- About 30 metacog entries a day stop being written. Each one cost an LLM draft and was labelled "flow" while it was really describing idle time (`metacog_trigger`: 23–37 flow rows/day for 2026-10-02..09; `orion_metacog`: 863 flow rows all-time).
- Downstream, a CollapseMirror entry only gets `type="flow"` from real phi bands now, not from an idle-detector trigger.

## Current architecture

`orion-equilibrium-service._generative_metacog_poll_loop` read the newest 20 rows of `substrate_attention_self_model.prediction_error_confidence` every 30 s. It fed two gates. `insight` looks for a low→high recovery. `flow` (`detect_flow_regime`) fired when `min(window) >= 0.90` and `stdev(window) <= 0.02`, then published through its own 1800 s cooldown lane. cortex-exec mapped `trigger_kind=="flow"` to `CollapseMirrorEntryV2.type="flow"`.

## Architecture touched

- orion-equilibrium-service: the flow gate, its settings, env/compose keys, and cooldown lane are removed. The poll loop now runs insight only.
- `orion/substrate/metacog_trigger_signals.py`: `detect_flow_regime` / `FlowRegime` are removed.
- orion-cortex-exec: the `trigger_kind=="flow"` override branch is removed. The phi-band "flow" guess is untouched.
- `orion/inner_state_registry.py` plus `config/metrics/metric_definitions.lock.json`: the flow gate is removed as a declared consumer of `attention_self_model.v1`. Touched because the registry named the deleted module, so the drift gate requires both. Re-locked after the commit with `check_definition_drift.py --update`. The deltas are 12 x `routing_changed` (consumer list only).

## Files changed

- `services/orion-equilibrium-service/app/flow_metacog_gate.py`: deleted.
- `services/orion-equilibrium-service/app/service.py`: removed the flow gate evaluation, de-dupe key, cooldown lane, and fetch-limit term.
- `services/orion-equilibrium-service/app/settings.py`: removed the five `metacog_flow_*` fields.
- `services/orion-equilibrium-service/.env_example`, `docker-compose.yml`: removed the five `EQUILIBRIUM_METACOG_FLOW_*` keys and added a retirement note.
- `services/orion-equilibrium-service/README.md`: marked the trigger-table row retired, added a retirement note with numbers, and removed the flow env rows.
- `orion/substrate/metacog_trigger_signals.py`: removed the detector and dataclass, plus the unused `statistics` import.
- `services/orion-cortex-exec/app/executor.py`: removed the override branch.
- `orion/metacog/evidence_map.py`: added a comment saying `map_flow` is for historical rows only.
- `orion/schemas/telemetry/metacog_trigger.py`, `orion/schemas/attention_self_model.py`: doc text only.
- `orion/inner_state_registry.py`, `config/metrics/metric_definitions.lock.json`: consumer removal and re-lock.
- Tests:
  - New: `tests/test_flow_trigger_retired.py`, plus the fixture `tests/fixtures/prediction_error_confidence_live_2026-10-03.json` (240 real ticks).
  - Insight-only tests renamed or trimmed: `test_insight_metacog_gate.py`, `test_insight_separate_cooldown.py`, `test_generative_metacog_gate_evaluation.py`, `tests/test_metacog_generative_trigger_signals.py`, `orion/metacog/tests/test_evidence_map_producer_contract.py`, `services/orion-cortex-exec/tests/test_metacog_draft_trigger_kind_type_mapping.py`, `scripts/test_check_metric_dead_wiring.py`.

## Schema / bus / API changes

- Added: none
- Removed: no producer emits `trigger_kind="flow"` any more. `MetacogTriggerV1.trigger_kind` is a free string, so the schema is unchanged (description text updated).
- Renamed: none
- Behavior changed: `trigger_kind="flow"` no longer forces `type="flow"` in cortex-exec. A replayed historical flow row falls through to the phi-band guess.
- Compatibility notes: stored historical flow rows still map through `map_flow`. No bus channel changes.

## Env/config changes

- Added keys: none
- Removed keys:
  - `EQUILIBRIUM_METACOG_FLOW_TRIGGER_ENABLE`
  - `EQUILIBRIUM_METACOG_FLOW_COOLDOWN_SEC`
  - `EQUILIBRIUM_METACOG_FLOW_FLOOR`
  - `EQUILIBRIUM_METACOG_FLOW_MAX_STDEV`
  - `EQUILIBRIUM_METACOG_FLOW_MIN_TICKS`
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-equilibrium-service`: yes ("no changes needed"). The sync script does not delete keys, so the five retired keys were removed by hand from the primary checkout's `services/orion-equilibrium-service/.env`, after a backup at `.env.bak-2026-10-10` (gitignored).
- skipped keys requiring operator action: none

## Metric quality gate findings (the input: `prediction_error_confidence`)

1. **Provenance.** `orion/substrate/attention_self_model.py::_unconditional_prediction_error_confidence` computes `1 - mean(prediction_error_by_domain)` over `ACTIVE_INFERENCE_DOMAINS` = {biometrics, bus_synaptic, chat, execution, route}. orion-substrate-runtime persists it every ~30 s to `substrate_attention_self_model.self_model_json`.
2. **Independence.** flow and insight read the same field. flow's plateau is a direct function of three per-domain errors being 0, which is the absence of chat, execution and route activity. It is not an independent state.
3. **Theory anchor.** The design (`docs/superpowers/specs/2026-07-28-collapse-mirror-generative-triggers-design.md`) defined flow as a "sustained high-confidence, low-variance plateau". In active-inference terms, low prediction error is only meaningful when there is something being predicted. When the activity domains are idle, their error is 0 by construction, not by good prediction. So the anchor does not hold on this field.
4. **Live-data sanity** (bounded read-only queries, 2026-10-03 to 2026-10-10, 15,235 rows; the table holds 7 days):
   - Per-tick values: p1 0.683, p10 0.847, p50 0.980, p90 0.993, max 0.9999. Near the ceiling, but not flat.
   - Per-domain error. Biometrics: mean 0.038. bus_synaptic: mean 0.035. Chat: 0 on 6,514 of 8,584 rows. Execution: 0 on 12,515 of 15,230 rows. Route: 0 on 13,412 of 14,448 rows.
   - Using the gate's own window definition (20 rows, span ≤ 19×30×2 s), there were 14,990 windows. Window minimum: p50 0.875, p90 0.971, p95 0.976, p96.8 0.978.
   - Windows qualifying under the old rule (`min≥0.90`, `stdev≤0.02`): **38.0%**, against **3.2%** at calibration (2026-07-30, 71/2246 windows, `docs/superpowers/pr-reports/2026-07-30-collapse-mirror-insight-flow-gates-pr.md`). Per day it ranged from 23.6% to 47.2%.
   - Real firings (`metacog_trigger`, trigger_kind=flow) were 23–37 a day. A replay of the old gate plus its 1800 s cooldown over the same rows reproduced them (27/30/37/27/31/23/26 simulated vs 31/30/37/27/31/23/26 actual).
   - **Median gap between consecutive real flow publishes: 1800.6 s.** 109 of 204 gaps fall in 1800–1900 s. The cooldown sets the rate, not the data.
   - **Would recalibrating help?** A self-relative top-3.2% threshold (window min ≥ 0.978) selects windows with **93% zero chat/execution/route activity**. The old rule's qualifying windows were 76% zero-activity, against 30% across all windows. Mean biometric error in the selected windows was 0.014, versus 0.038 overall. A percentile would just pick the quietest idle stretches with the quietest hardware sensors. That is a "calm because nothing happened" detector, which repeats the known lesson that a signal which only updates on activity reads as calm during an outage.
5. **Existing mechanism.** The drift was caused upstream: the per-domain errors were redefined repeatedly after calibration (z-scored 08-19, route touched-only 09-25, chat touched-only 09-29, stale-domain omission 10-02). Any absolute constant here drifts again.
6. **Reversibility.** Retirement is cheap to undo (one revert). Leaving it running was the expensive path: ~30 LLM-drafted metacog entries a day.

**Verdict: RETIRE.** The input carries no "flow" information that any threshold can extract. A real "engaged and calm" detector would need an activity-present condition (chat, execution or route touched in the window) plus low error. That is new cognition design and needs proposal mode, so it is not in this patch.

## Tests run

```text
services/orion-equilibrium-service: pytest tests -q               -> 185 passed (also after review fixes)
tests/test_metacog_generative_trigger_signals.py + orion/metacog/tests/{test_evidence_map,test_service,test_capture_replay,test_evidence_map_producer_contract}.py
  + orion/schemas/tests/test_metacog_entry.py + scripts/test_check_metric_dead_wiring.py -> 203 passed
services/orion-cortex-exec: test_metacog_publish_lane, test_metacog_two_pass_draft,
  test_metacog_trend_cue_prompt_render, test_metacog_draft_trigger_kind_type_mapping -> 39 passed
scripts/check_metric_lineage.py --gate            -> PASS
scripts/check_definition_drift.py --gate          -> PASS (after commit + --update)
scripts/check_inner_state_registry.py             -> OK (15 entries)
scripts/check_sentience_instruments.py --static-only -> All claims hold
scripts/check_system_health_producers.py          -> OK
scripts/check_service_env_compose_parity.py orion-equilibrium-service -> OK (82 keys)
scripts/check_env_template_parity.py              -> PASS
```

Mutation check: I injected a flow publish (old floor/stdev rule) back into `_generative_metacog_poll_loop`. `test_poll_loop_publishes_no_flow_trigger_on_the_live_series` and `test_no_service_constructs_a_flow_trigger` both FAILED (the loop published `['flow','flow',...]`). With the injection reverted, all 7 tests pass.

## Evals run

```text
PYTHONPATH=. python orion/metacog/evals/run_capture_eval.py -> PASS (fixture's 25 historical flow rows still map via map_flow)
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-equilibrium-service build -> Image orion-equilibrium-service-equilibrium-service Built
docker run --rm --entrypoint python <image> -c "import app.service ..." -> no 'flow' cooldown lane, no metacog_flow_* settings
```

Not deployed.

## Review findings fixed

Review ran in a subagent against `git diff origin/main...HEAD`. It found no must-fix items.

- Finding (should-fix): the PR report that every retirement comment cites was not committed yet, so those references would dangle.
  - Fix: committed it (this file).
  - Evidence: `git ls-files docs/superpowers/pr-reports/2026-10-10-metacog-flow-trigger-calibration-pr.md`.
- Finding (should-fix): the poll-loop test could not fail on its own. The loop swallows exceptions, there was no positive control, and on main the flow flag defaulted False in code.
  - Fix: added positive controls. The test now asserts reader calls == windows+1, insight evaluations == windows, and zero `generative_metacog_poll_loop_failed` logs. The docstring now states the test's real scope: it guards against flow being re-added to the loop, not against the old default config.
  - Evidence: re-ran the mutation (old flow publish injected into the loop). The test fails on `assert 'flow' not in ['flow', 'flow', ...]`. With the mutation reverted, it passes.
- Finding (nit): README said "both conditions" for the freshness guard. Fix: reworded to "a gate condition".
- Finding (nit): the oracle test used a magic `19`. Fix: it now uses `_OLD_MIN_TICKS`.
- Finding (nit): the oracle test reads like code coverage. Fix: its docstring now says it is an oracle over frozen data.
- Reviewer side note: "where did the live flow flag come from, given the code default is False?" Answer: the primary `services/orion-equilibrium-service/.env` had `EQUILIBRIUM_METACOG_FLOW_TRIGGER_ENABLE=true`. I had already removed it (backup kept) before the review ran, which is why the reviewer saw no key.

## Restart required

After merge, deploy from the primary checkout on main:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-equilibrium-service/.env -f services/orion-equilibrium-service/docker-compose.yml up -d --build
```

cortex-exec only lost a branch that nothing will reach after the equilibrium restart, so restarting it is optional.

## Risks / concerns

- Severity: low
  - Concern: the "flow" CollapseMirror entry type is now produced only by phi bands, so there will be fewer flow-typed entries.
  - Mitigation: intended. The removed ones were idle-labelled-as-flow.
- Severity: low
  - Concern: the primary `.env` no longer has the flow keys, while the running container is still on the old code until it is redeployed.
  - Mitigation: if the old image restarts before this merges, its settings default `metacog_flow_trigger_enable=False`, so it just stops firing flow early. That is harmless.
- Severity: info
  - Concern: insight reads the same field and may have a related idle bias.
  - Mitigation: out of scope. It needs a dip to ≤0.70 first, which comes from real activity spikes. Worth its own check.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2578

🤖 Generated with [Claude Code](https://claude.com/claude-code)

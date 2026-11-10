## Summary

- Orion's autonomous web fetch and recall now need a goal that is about the world (chat, reading, camera surprise). A goal raised by a body prediction-error signal no longer unlocks them.
- Attended world winners may now emit goals. Before, only five body prediction-error nodes could.
- Each autonomy decision log line now names the goal's target, so the before/after mix can be measured.
- Goal-gate tests now run in CI (they ran in no workflow before).
- First build patch of the prediction-error suite rebuild spec (PR #3168), retire step 2, built per Juniper's Decision 1 ("both": content gate + world-source goals).

## Outcome moved

Half of Orion's permission to act came from a broken signal. Measured 2026-11-10, before this patch:

- 6,005 of 6,005 goals in the `orion-athena-attention-runtime` log window came from body signals: execution 2,389, bus_synaptic 2,321, chat node 1,105, biometrics 190. Zero from the world.
- `action_outcomes`: about 1 autonomous `web.fetch.readonly` per day for the past month (the world-pulse gap fill).

After: a body goal is denied with `goal_not_about_world`. Expect the daily gap-fill fetch to run only on days a world goal is active when world-pulse runs. That drop is the intended behavior, not a regression: the permission must come from something real.

## Current architecture

`orion-attention-runtime` picks a goal from `frame.node_targets` restricted to `PREDICTION_ERROR_NATIVE_TARGETS` (five body nodes) and emits `FieldGoalProvenanceV1` every tick a dominance streak holds. Consumers keep the latest goal (`goal_state`, 4 h hold). `evaluate_capability` checks only that a goal exists at status `proposed`; it never looked at what the goal named.

## Architecture touched

- `orion/attention/field_attention/goal_provenance.py`: goal eligibility = native body node marked internal (unchanged) OR attended world source (`is_world_source_id` + `source_kind:external`). Scans `dominant_targets` too, because chat is a `channel` target and never in `node_targets`. World-first off: unchanged (no markers, so no world target qualifies).
- `orion/autonomy/{models,capability_policy}.py` + `config/autonomy/capability_policy.v1.yaml`: new rule field `requires_world_goal` (default false), true on `web.fetch.readonly` and `recall.query.readonly`. Journal is not content-gated (spec step 2 gates the read-only capabilities).
- `services/orion-attention-runtime/app/store.py`: the competition read keeps `world:` refs (review finding).
- Logs: `goal_target=` added to `substrate_policy_act` and `world_pulse_curiosity_gate`.

## Files changed

- `orion/attention/field_attention/goal_provenance.py`: world winners may name goals.
- `orion/autonomy/models.py`: `CapabilityPolicyRuleV1.requires_world_goal`.
- `orion/autonomy/capability_policy.py`: `goal_not_about_world` denial.
- `config/autonomy/capability_policy.v1.yaml`: gate on fetch + recall.
- `orion/autonomy/policy_act.py`, `services/orion-world-pulse/app/services/curiosity.py`: log the goal target.
- `services/orion-attention-runtime/app/store.py`: competition read keeps `world:` refs.
- Tests: `services/orion-attention-runtime/tests/test_competing_loop_refs_postgres.py` (new), `orion/autonomy/tests/test_capability_policy.py`, `test_policy_act.py`, `services/orion-world-pulse/tests/test_curiosity.py`, `tests/test_attention_world_first_field_frame.py`.
- `.github/workflows/orion-attention-runtime-tests.yml`, `orion-world-pulse-tests.yml`: run the gate tests; trigger on `orion/autonomy/**` and `config/autonomy/**`.
- `orion/sentience_striving_program/instruments.yaml`: new claim `readonly_autonomy_needs_a_world_goal` on the O1 instrument, with the baseline.

## Schema / bus / API changes

- Added: `CapabilityPolicyRuleV1.requires_world_goal: bool = False` (config model, baked into the same images as the YAML; no bus payload change).
- Removed / Renamed: none.
- Behavior changed: goals may name `world:chat` (`target_kind="channel"`), `world:reading`, `node:substrate.perception`, `node:substrate.perception_walkway`. `FieldGoalProvenanceV1` is unchanged (`field_target_id`/`target_kind` are free strings).
- Compatibility notes: no change to any `extra="forbid"` bus model.

## Env/config changes

- Added / Removed / Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/autonomy/tests tests/test_attention_world_first_field_frame.py tests/test_attention_field_goal_provenance.py services/orion-attention-runtime/tests tests/test_attention_runtime_worker.py tests/test_attention_runtime_store.py services/orion-attention-runtime/evals -> 349 passed, 6 skipped
FOCUS_RUN_TEST_POSTGRES_URI=<throwaway postgres:15> pytest services/orion-attention-runtime/tests/test_competing_loop_refs_postgres.py -> 1 passed (fails with the old filter)
PYTHONPATH=.:services/orion-world-pulse pytest services/orion-world-pulse/tests                                    -> 158 passed
pytest services/orion-substrate-runtime/tests/test_goal_context_listener.py orion/substrate/tests/test_goal_context_target_carry.py tests/test_voluntary_attention_wiring.py -> 18 passed
pytest orion/sentience_striving_program/tests -> 24 passed; scripts/check_sentience_instruments.py --static-only -> All claims hold
Mutation check: requires_world_goal set false -> all 4 new regression tests fail (capability, policy_act end-to-end, world-pulse gap fill, shipped-policy pin)
Pre-existing failures, also on clean main: tests/test_autonomy_goals_bus_catalog.py::test_goal_proposal_schema_in_registry, orion/spark/concept_induction/tests/test_metabolism_hook.py (2)
```

## Evals run

```text
pytest services/orion-attention-runtime/evals (focus history replay) -> passed (in the 89 above)
No eval measures goal mix yet; the live 7-day before/after (Decision 1) is the eval. Baseline recorded above and in instruments.yaml.
```

## Docker/build/smoke checks

```text
UNVERIFIED live until deploy. After deploy, check:
docker logs --since 24h orion-athena-attention-runtime 2>&1 | grep -o "field_target_id=[^ ]*" | sort | uniq -c   # expect world:* rows
docker logs --since 24h orion-world-pulse 2>&1 | grep world_pulse_curiosity_gate                             # goal_target= on each decision
```

## Review findings fixed

- Finding (medium-high): the goal producer's competition read (`services/orion-attention-runtime/app/store.py`, `load_competing_loop_refs`) kept only `node:%` refs. A body node held by the substrate competition therefore beat an attended chat or reading winner on every tick, which would have kept goals mostly body-sourced.
  - Fix: filter is now `node:%` OR `world:%`.
  - Evidence: new real-Postgres test `services/orion-attention-runtime/tests/test_competing_loop_refs_postgres.py` passes with the fix and fails without it (throwaway `postgres:15` container); runs in CI's existing attention-runtime Postgres service. Live projection had no open loops at check time, so no live `world:chat` ref was observed (UNVERIFIED live).
- Finding (low): precedence and hysteresis untested with mixed world/body targets.
  - Fix: `test_competition_precedence_keeps_a_world_goal_it_holds`, `test_world_streak_is_held_when_competition_read_is_unknown`.
- Finding (low): world-pulse body-goal test only asserted an empty result, which any earlier denial also produces.
  - Fix: it now asserts `reason=goal_not_about_world` and `goal_target=node:substrate.bus_synaptic` in the log.

## Restart required

Rebuild from the clean primary main checkout after merge (post-pull rebuild covers it): `orion-attention-runtime`, `orion-world-pulse`, `orion-spark-concept-induction`, `orion-cortex-exec`.

## Risks / concerns

- Severity: medium. Concern: autonomous fetch/recall drop to near zero on days no world goal is active (goals are "latest wins"; body goals arrive every ~2 s). Mitigation: intended; re-measure in 7 days per the instrument claim. If world winners are too rare, the next lever is the attention contest, not loosening the gate.
- Severity: low. Concern: journal is still unlocked by body goals. Mitigation: retire steps 5-6 remove the broken body nodes themselves.
- Not done here (spec items for later patches): O1's demonstrated action-budget ceiling; `prediction.surprise.v1` for cabinet/office forecasts.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3169

🤖 Generated with [Claude Code](https://claude.com/claude-code)

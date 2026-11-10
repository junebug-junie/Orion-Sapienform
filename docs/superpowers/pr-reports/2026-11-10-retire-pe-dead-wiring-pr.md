# Retire dead prediction-error wiring (PE rebuild, retire step 1)

## Summary

- Deletes the Hub "Bus Anomaly Detected" notifier. Its only trigger stopped on 2026-09-30, so it could never fire.
- Stops stamping the bus prediction-error number onto every action outcome. Nothing read it. The module that fetched it (`orion/substrate/bus_synaptic_surprise.py`) is deleted.
- Deletes the unused "domain surprise" field in the autonomy permission policy. No rule ever set it.
- Removes `ORION_ACTION_OUTCOME_DB_URL` from orion-world-pulse and orion-spark-concept-induction. Its only use there was the stamp.
- **Retire step 4 (remove the `route` prediction-error node) is NOT in this PR.** The spec says route is stale and read as absent. Live data says it is not. Details below; it needs a decision from Juniper.

Spec: `docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md` (PR #3168, "Retire order" step 1). Approved by Juniper 2026-11-10.

## Outcome moved

Less code that pretends to measure surprise. Each autonomy act and each cooling shed no longer runs a Postgres read of the bus number for a value nobody uses. No runtime behavior changes: every removed piece was write-only or could not fire.

## Current architecture

- Hub subscribed to `orion:equilibrium:metacog:trigger` and posted an in-app warning when a `transport` trigger with `evidence_source=bus_synaptic_prediction_error` arrived. Equilibrium stopped publishing that trigger on 2026-09-30 (PR "retire bus synaptic transport trigger").
- `episode_fetch`, `policy_act`, `curiosity_reuse`, the durable-runs cooling act, world-pulse curiosity and spark concept-induction read `node:substrate.bus_synaptic.prediction_error` from `substrate_field_state` and wrote it into `ActionOutcomeRefV1.surprise` / `ActionOutcomeEmitV1.surprise`. Without a source they wrote a 0/1 success proxy.
- The same read fed `CapabilityEvaluationContext.domain_surprise_score`, which became a note on the capability decision and an optional gate (`required_domain_surprise_below`).

## Architecture touched

- orion-hub: notifier removed from startup and shutdown.
- `orion/autonomy`: `SurpriseSource`, `resolve_surprise`, every `surprise_source` parameter, the domain-surprise gate, note and policy field.
- orion-durable-runs: `CoolingActStore.latest_surprise` and its use.
- orion-world-pulse, orion-spark-concept-induction: the bus read, its Postgres engine and its env key.
- `orion/substrate/metacog_trend_signals.py` now imports its staleness horizon from the existing `orion/substrate/prediction_error_freshness.py` (same value) instead of the deleted module.

## Evidence that each removed piece was dead

All queries ran against the live `conjourney` database on 2026-11-10 (UTC).

1. **Hub notifier.**
   - `select max(timestamp), count(*) from metacog_trigger where upstream->>'evidence_source'='bus_synaptic_prediction_error'` -> `2026-09-30 06:31:54`, 3118. Nothing in 41 days.
   - `rg bus_synaptic_prediction_error services/orion-equilibrium-service` -> no publisher left (only the replay/evidence mappers in `orion/metacog/` read old rows).
   - `notify_requests` rows with `event_kind='bus_synaptic.transport_trigger'` or title "Bus Anomaly Detected": 0.
   - Hub log since its 06:01 restart today: notifier "started", 0 "notification published".
2. **Action-outcome surprise stamp.**
   - Written: `action_outcomes` kinds in the last 14 days carry the bus value (mean ~0.034, range 0.002-0.18). Column is `NOT NULL`.
   - Read: the only SQL reader is `orion/autonomy/action_outcomes.py::load_action_outcomes`; its only caller is cortex-exec `chat_stance._project_recent_dispatch_actions`, which keeps `kind`, `summary`, `success`, `observed_at` and drops `surprise`. The other reader is an offline analysis script (`scripts/analysis/measure_capability_surprise_success_correlation.py`), which still works on the stored history.
3. **Domain-surprise policy field.**
   - `rg required_domain_surprise_below config/` -> no rule sets it, so the gate always returned "no_surprise_gate".
   - `CapabilityDecisionV1.notes` from these calls are logged without notes and never persisted or read, so dropping the advisory note changes nothing.
4. **World-model `_last_*` caches.** Not removed. The spec says to remove them "for retired nodes". This PR retires no node (bus, execution and chat are out of scope; route is held, see below). Live check for the record: `SUBSTRATE_WORLD_MODEL_PUBLISH_TICK_ENABLED=false` in the running substrate-runtime, so the caches are write-only today. They go with their nodes.

## Why retire step 4 (route) is held

The spec says: "Remove route. Already stale; every consumer already reads it as absent." It also says no row since 11-07 15:57. Live data disagrees:

- `substrate_node_prediction_error_history`, `node:substrate.route`: 81 rows with `observed_at` after 2026-11-07 15:57, the latest at 2026-11-10 07:28. Rows every day 11-03 to 11-10 (7-48 per day). Values are 0 or 0.5.
- Live `substrate_field_state` holds `node:substrate.route.prediction_error = 0.0`, fresh.
- `substrate_attention_self_model`, last 24 h: route is in `prediction_error_by_domain` in 229 of 780 rows (non-zero in 37). That map feeds the `1 - mean(PE)` confidence (`ACTIVE_INFERENCE_DOMAINS`), which drives `insight` metacog triggers. Removing route changes the mean, for example 0.8988 -> 0.8735 on the newest row.
- Where it is already absent: attention frames (route won 134 frames, all before 2026-11-07 13:15, none since), curiosity seeds (14 in 7 days, last 2026-11-07 13:12), self-signal episodes (0 in 7 days), prompt lines (0 route lines in 7 days of stance, reverie and cockpit records, against 6 bus lines in reverie).

Removing route is therefore a behavior change to a live cognition input. It is not the no-op the approval assumed. That needs Juniper's decision (proposal mode, CLAUDE.md §0A): either remove route and accept the confidence shift, or hold it until step 5 rebuilds the aggregate confidence.

## Files changed

- `services/orion-hub/scripts/bus_synaptic_trigger_notifier.py`, `services/orion-hub/tests/test_bus_synaptic_trigger_notifier.py`: deleted.
- `services/orion-hub/scripts/main.py`: notifier import, global, start and stop removed.
- `services/orion-hub/scripts/endogenous_outreach.py`: comments no longer point at the deleted module.
- `services/orion-hub/tests/test_hub_listeners_resubscribe.py`: notifier case removed.
- `services/orion-hub/tests/test_bus_anomaly_notifier_retired.py`: new regression test.
- `orion/autonomy/episode_fetch.py`, `curiosity_reuse.py`, `policy_act.py`: surprise plumbing removed.
- `orion/autonomy/capability_policy.py`, `models.py`: domain-surprise gate, note and policy field removed; `ActionOutcomeRefV1` docstring marks `surprise` retired.
- `orion/autonomy/prediction.py`: docstring now in past tense about the stamp.
- `orion/substrate/bus_synaptic_surprise.py`, `orion/substrate/tests/test_bus_synaptic_surprise.py`: deleted.
- `orion/substrate/metacog_trend_signals.py`: staleness horizon from `prediction_error_freshness`.
- `orion/spark/concept_induction/bus_worker.py`, `settings.py`; `services/orion-spark-concept-induction/{.env_example,docker-compose.yml,README.md}`; `services/orion-spark-concept-induction/tests/test_bus_synaptic_surprise_source.py` (deleted).
- `services/orion-world-pulse/app/services/curiosity.py`, `app/settings.py`, `.env_example`, `README.md`; tests updated; `tests/test_curiosity_bus_synaptic_surprise.py` deleted.
- `services/orion-durable-runs/app/cooling_act.py`, `evals/cooling_act_replay.py`, `tests/test_cooling_act.py`, `requirements.txt` (comment only).
- Comment-only: `orion/attention/field_attention/candidate_precision_weighted.py`, `orion/substrate/mutation_proposals.py`, `services/orion-cortex-exec/app/metacog_trend_reader.py`, `services/orion-mind/app/recall_signal_resolver.py`.
- `orion/autonomy/tests/*`: surprise tests replaced by regression tests that the removed hooks stay gone.

## Schema / bus / API changes

- Added: none.
- Removed: `CapabilityPolicyRuleV1.required_domain_surprise_below` (set on no rule; the model is `extra="forbid"`, so a policy file that sets it now fails to load, on purpose). `CapabilityEvaluationContext.domain_surprise_score` / `domain_surprise_source` (in-process dataclass).
- Renamed: none.
- Behavior changed: `action.outcome.emit.v1` payloads now always carry `surprise: 0.0`. No reader.
- Compatibility notes: the `surprise` field and the `action_outcomes.surprise` column stay. Dropping them is a consumer-first two-step migration (forbid models, NOT NULL column) and is left for the step that rebuilds outcome surprise.

## Env/config changes

- Added keys: none.
- Removed keys: `ORION_ACTION_OUTCOME_DB_URL` from orion-world-pulse and orion-spark-concept-induction (orion-cortex-exec keeps it; it uses it for `load_action_outcomes`).
- Renamed keys: none.
- `.env_example` updated: yes, both services.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran (adds only). The removed key was also deleted by hand from both local `.env` files.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/autonomy/tests orion/substrate/tests/test_metacog_trend_signals.py   240 passed
pytest services/orion-world-pulse/tests (from repo root)                           152 passed
pytest services/orion-durable-runs/tests/test_cooling_act.py                       22 passed
pytest services/orion-hub/tests/test_hub_listeners_resubscribe.py
       services/orion-hub/tests/test_bus_anomaly_notifier_retired.py
       services/orion-hub/tests/test_endogenous_outreach*.py                       270 passed
pytest services/orion-cortex-exec/tests/test_metacog_trend_reader.py
       test_metacog_trend_cue_prompt_render.py
       test_chat_stance_recent_dispatch_actions_projection.py                      40 passed
pytest services/orion-spark-concept-induction/tests orion/spark                   91 passed, 4 failed
       (the same 4 falkor/graph tests fail on clean origin/main: 95 passed, 4 failed;
        the 4-test difference is the deleted surprise-source tests)
python scripts/check_env_template_parity.py                                       PASS
git diff --check                                                                  clean
```

## Evals run

```text
python services/orion-durable-runs/evals/cooling_act_replay.py   9/9 passed
```

No other touched seam has an eval harness for this removal. The regression tests are the gate.

## Docker/build/smoke checks

```text
Not run. No dependency or port change. The only image-level change is removed code and one removed env key.
```

## Review findings fixed

(filled in after review)

## Restart required

After merge, the post-pull rebuild redeploys the touched services. Nothing to do by hand.

## Risks / concerns

- Severity: medium. Concern: retire step 4 (route) is not done, because its premise failed the live check. Mitigation: evidence above; needs Juniper's decision.
- Severity: low. Concern: `action_outcomes.surprise` is now a column that is always 0.0 for new rows. Mitigation: documented on the model; drop it in the step that rebuilds outcome surprise.

## PR link

(filled in after PR creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

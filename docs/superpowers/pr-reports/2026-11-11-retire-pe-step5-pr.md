# Retire bus_synaptic and route prediction error; quiet the mood-arc trigger (PE rebuild, retire step 5)

**Merge after the 2026-11-17 goal-mix re-measure is recorded.** bus_synaptic still sources about 22% of attention goals. Removing it earlier would confound that measurement. One deploy, so Orion's confidence shifts once (Juniper, 2026-11-10).

## Summary

- Orion stops computing two fake "prediction error" signals: the bus one (`node:substrate.bus_synaptic`) and the route one (`node:substrate.route`). Neither was ever a forecast. The bus one is a lull-ending detector.
- Every reader of them is removed in the same patch. Readers that scan *all* prediction-error nodes now check one shared retired list, so a stored copy left in FalkorDB or in 7 days of history rows is never read again.
- The field digester prunes both nodes from stored field state, the way it already does for the retired transport node.
- The mood-arc anomaly encoder is **kept** (Juniper, 2026-11-11, for a retrain at the end of this arc). Its metacog trigger is switched off, its scorer keeps running, and its training-data writer is switched back on.
- The route reducer and the raw bus synaptic graph (heartbeat, recall causal-latency lines, hub graph pages) are untouched.

Spec: `docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md` (PR #3168), "Retire order" steps 4, 5 and 7.

## Outcome moved

- Orion's confidence number ("1 - mean prediction error") now averages execution, biometrics and chat, not five domains. On the 11-10 newest row, removing route alone moved it 0.899 -> 0.874. bus_synaptic sat on a ~0.27 floor that never read calm, so removing it raises the mean confidence on most ticks.
- bus_synaptic stops opening self-signal episodes (40% of them in the 11-08 audit) and stops winning attention frames and goals (22% of goals at the 11-10 baseline; 51% before #3169).
- The mood-arc trigger stops starting about 55 reflection passes a day (386 `telemetry_anomaly` triggers in the 7 days to 11-11) from a model that flags 25-28% of all minutes.
- No change to the goal gate, the route reducer, the heartbeat or recall's real causal-latency lines.

## Proposal mode (CLAUDE.md §0A)

- **Capability change:** Orion loses two self-readings and one reflection trigger. Confidence, attention, goals, curiosity, episodes and the mind digest no longer see bus or route prediction error.
- **Data touched:** FalkorDB nodes in graph `orion_substrate`; `substrate_node_prediction_error_history` (5,836 bus_synaptic + 203 route rows); `substrate_node_prediction_error_baseline`; `substrate_field_state` node vectors; Redis key `orion:lattice:transport_thresholds:v1`. Code never deletes stored rows; the cleanup commands below are for Juniper to run.
- **Privacy boundary:** none crossed. All data is Orion's own internal telemetry.
- **Trace that proves it worked:** see "Post-deploy checks".
- **Dangerous failure mode:** a stored node coming back as an attention target ("trap A": the selectors treat any field node vector not on the native-target list as a novelty target). Closed by the field digester pruning both nodes every reconcile and refusing late writes to them; covered by `tests/test_retired_pe_step5_readers.py` and `services/orion-field-digester/tests/test_retired_pseudo_nodes.py`.
- **Rollback:** `git revert` of the merge restores the producers. Stored history is untouched by the code. Mood-arc trigger: set `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TRIGGER_ENABLE=true`.

## Current architecture

- substrate-runtime `_bus_synaptic_tick` read FalkorDB edge z-scores every 30 s, wrote `node:substrate.bus_synaptic`, a receipt into the field, a history row, and a transport EWMA sample. `_route_tick` ran the route reducer, then wrote `node:substrate.route` from the share of routing labels that changed.
- Both were in `ACTIVE_INFERENCE_DOMAINS`, the glossary (prompt sites), attention native targets, self-reading wording, the hub domain list, the world-model execution slots and the metric lock. bus_synaptic also fed a topology edge into `capability:transport.pressure`, the transport lattice row and EWMA thresholds, the mind's "transport: bus_synaptic" digest line (through `recall_signal_resolver.py`), a self-signal outcome pair, a metacog evidence mapper and causal geometry's node query.
- Generic readers (curiosity, the attention contests, the PE history cache, the dynamics pressure seed, the outcome rollup, substrate-runtime PE magnitudes) took any `node:substrate.*` node with a value.
- The field digester never decays `prediction_error`, so a stopped producer leaves its last value in the field forever.
- Mood-arc v4 encoder scored the field every minute; equilibrium turned flagged windows into `telemetry_anomaly` metacog triggers. Its corpus writer was off since 09-08.

## Architecture touched

- **substrate-runtime:** bus_synaptic tick, client and settings removed; route tick keeps only the reducer; domain map is execution/biometrics/chat; world-model execution slots 2-3 become zero padding (dims unchanged at 16; the publish tick is off live).
- **Shared:** `RETIRED_PREDICTION_ERROR_NODE_IDS` + `is_retired_prediction_error_node()` in `orion/substrate/prediction_error_freshness.py`, checked in `world_first_fenced`, `NodePeHistoryCache.refresh`, endogenous curiosity, substrate-runtime `_prediction_error_magnitudes`, `pressure.prediction_error_pressure`, the equilibrium outcome rollup and the hub attention-organ rows.
- **field-digester:** both ids in `RETIRED_PSEUDO_NODES`; tension baselines for pruned nodes dropped; the bus_synaptic -> capability:transport edge deleted; a capability with edges but none into `pressure` now drops `pressure`, `confidence` and `available_capacity` instead of re-seeding a calm 0.0/1.0/1.0. Live, that only changes `capability:transport` (the reviewer checked every capability in the topology).
- **mind:** digest line and resolver deleted.
- **recall:** the publish-gap half of the bus adapter (text="" fragments that only the resolver read) retired; causal-latency fragments kept; fusion's signal_kind exemption removed.
- **hub:** transport pressure gate reads only the reliability half; domain lists updated.
- **equilibrium:** mood-arc trigger defaults off.
- **field-digester corpus:** writer on, `CORPUS_SINK_ROTATED_KEEP` 5 -> 40.
- **mood_arc:** enrichment no longer reads bus_synaptic.

## Files changed

129 files, +1419 / -4042. Main groups:

- `services/orion-substrate-runtime/app/{worker.py,settings.py,world_model_features.py}`, `.env_example`, `docker-compose.yml`: producers removed.
- `orion/substrate/{prediction_error.py,prediction_error_freshness.py,attention_self_model.py,endogenous_curiosity.py,pressure.py,causal_geometry_engine.py}`, `orion/substrate_ladder_liveness.py`.
- `orion/attention/{field_attention/selectors.py,self_readings.py,self_signal_outcomes.py,pe_history_cache.py}`, `orion/schemas/{attention_candidate.py,prediction_error_definitions.py}`.
- `services/orion-field-digester/app/{tensor/channels.py,tensor/reconcile.py,digestion/diffusion.py,settings.py}`, `.env_example`, `docker-compose.yml`, `README.md`.
- `config/field/{field_channel_glossary.v1.yaml,orion_field_topology.v1.yaml}`, `config/substrate-lattice/transport_lattice_policy.v1.yaml`, `config/metrics/metric_definitions.lock.json` (regenerated).
- Deleted: `orion/field/transport_thresholds.py`, `services/orion-mind/app/recall_signal_resolver.py` and their tests, `services/orion-substrate-runtime/tests/test_worker_bus_synaptic_tick.py`.
- `services/orion-mind/app/{evidence.py,engine.py,settings.py}`, `.env_example`, `docker-compose.yml`.
- `services/orion-recall/app/{storage/falkor_bus_synaptic_adapter.py,fusion.py}`.
- `services/orion-hub/scripts/{substrate_lattice_routes.py,attention_organ_routes.py}`, static JS, `.env_example`, `docker-compose.yml`.
- `services/orion-equilibrium-service/app/{settings.py,self_signal_outcome_rollup.py}`, `.env_example`, `docker-compose.yml`, `README.md`.
- `orion/metacog/{evidence_map.py,capture_replay.py}`: bus_synaptic mapper removed. The `telemetry_anomaly` mapper stays (encoder kept).
- `orion/mood_arc/{corpus_enrichment.py,README.md}`.
- `orion/sentience_striving_program/instruments.yaml`: bus and route claims marked historical.
- New guard tests (below) and `.github/workflows/retired-pe-guards.yml`.

## Schema / bus / API changes

- Added: none.
- Removed: metric URNs `node:substrate.bus_synaptic.prediction_error` and `node:substrate.route.prediction_error`; `route_arbitration` PE definition version; hub lattice `threshold_provenance` field.
- Renamed: none.
- Behavior changed: `capability:transport.pressure`, `.confidence` and `.available_capacity` read absent (unmeasured) instead of a number. Confidence aggregate over 3 domains. Mood-arc trigger off.
- Compatibility notes: no bus channel or schema registry entry changes. `field_channel_anomaly_score` keeps filling as the baseline.

## Env/config changes

- Removed keys:
  - orion-substrate-runtime: `SUBSTRATE_BUS_SYNAPTIC_TICK_ENABLED`, `SUBSTRATE_BUS_SYNAPTIC_TICK_INTERVAL_SEC`, `SUBSTRATE_BUS_SYNAPTIC_MIN_EDGE_COUNT`, `SUBSTRATE_BUS_SYNAPTIC_MAX_EDGE_AGE_SEC`, `FALKORDB_BUS_GRAPH`, and the 8 `TRANSPORT_THRESHOLDS_*` keys.
  - orion-hub: `TRANSPORT_THRESHOLDS_DERIVED_ENABLED`, `_FAST_HALF_LIFE_SEC`, `_SLOW_HALF_LIFE_SEC`, `_MIN_SAMPLES`, `_K_WATCH`, `_K_STEP`, `_MAX_STATE_AGE_SEC`, `_STATE_KEY`.
  - orion-mind: `RECALL_TRANSPORT_PG_DSN`, `RECALL_TRANSPORT_RENDER_GATE_THRESHOLD`, and the 8 `TRANSPORT_THRESHOLDS_*` keys.
- Changed defaults:
  - orion-equilibrium-service: `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TRIGGER_ENABLE` true -> false.
  - orion-field-digester: `FIELD_CHANNEL_CORPUS_PATH` empty -> `/mnt/telemetry/field_channels/corpus/field_channels.jsonl` (compose and .env_example; code default stays empty so tests never write to the host); `CORPUS_SINK_ROTATED_KEEP` 5 -> 40.
- `.env_example` updated: yes (substrate-runtime, hub, mind, equilibrium, field-digester).
- local `.env` synced: field-digester's two keys updated by hand on 2026-11-11 (harmless before deploy: it only starts data collection earlier). Removed keys left in local `.env` files; they are ignored.
- **Skipped key requiring operator action:** `services/orion-equilibrium-service/.env` still has `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TRIGGER_ENABLE=true`, which overrides the new default. Left on until merge so an unrelated rebuild cannot quiet the trigger before the 11-17 re-measure. Run this right before the merge:

```bash
sed -i 's/^EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TRIGGER_ENABLE=true$/EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TRIGGER_ENABLE=false/' /mnt/scripts/Orion-Sapienform/services/orion-equilibrium-service/.env
```

## Tests run

```text
field-digester (CI form)                                300 passed, 6 skipped
equilibrium tests + evals                               283 passed
orion-mind                                              234 passed
orion-thought                                           842 passed
attention-runtime tests + evals + capability_policy     77 passed, 6 skipped
mood_arc + action_warrant + pressure_dimensions         177 passed (with topology/readers/self-model)
step-5 guard tests (retired-pe-guards.yml set)          all passed
orion-hub                                               4 failed  (same 4 fail on origin/main f8bbb896d)
orion-recall                                            2 failed  (same 2 fail on origin/main)
substrate-runtime (minus Postgres integration test)     17 failed (same 17 fail on origin/main)
orion/substrate/tests + root attention/metric tests     5 failed  (same 5 fail on origin/main: 4 felt_state, 1 metric_lineage)
check_metric_lineage.py --gate                          PASS
check_definition_drift.py                               no definition changes (lock regenerated)
check_env_template_parity.py                            PASS
git diff --check                                        clean
docker compose config (substrate-runtime, hub, mind)    OK
```

## Evals run

```text
equilibrium evals (self-signal episode + outcome replay)   passed (in the 283)
attention-runtime evals                                     passed (in the 77)
```

No eval measures "retired node is never read" end to end; the guard tests are the gate. The live proof is the post-deploy checks below.

## Docker/build/smoke checks

```text
docker compose config: substrate-runtime, hub, mind OK. No image builds: no dependency changes.
```

## Review findings fixed

Code review ran in a subagent (read-only, high effort).

- Finding: `tests/test_field_topology_edges.py` used `capability:transport.pressure` as its "live channel" example; static-gates CI would go red.
  - Fix: example uses `capability:llm_inference`.
  - Evidence: test passes.
- Finding: live `.env` values override the new defaults (trigger stays on; `ROTATED_KEEP=5` would delete the 08-21 corpus file at the next rotation).
  - Fix: field-digester `.env` updated; equilibrium line is an explicit pre-merge operator step above.
- Finding: the new guard tests ran in no CI workflow.
  - Fix: `.github/workflows/retired-pe-guards.yml`.
- Finding: mood-arc enrichment still requested `prediction_error_bus_synaptic`, so a retrain would forward-fill stale values into fresh rows.
  - Fix: removed from `_PREDICTION_ERROR_DOMAINS`; tests updated.
- Finding: recall causal-latency fragments lost the fusion exemption with no test.
  - Fix: `services/orion-recall/tests/test_fusion_causal_fragment_survives.py`.
- Checked, no change: the `diffusion.py` capability rule changes only `capability:transport` live; no reader crashes on the missing keys or reads absent as calm.

## Restart required

The post-pull rebuild redeploys the touched services. Before the merge, run the equilibrium `.env` command above.

After deploy, optional cleanup (production writes, run by Juniper, not by code):

```bash
docker exec orion-athena-falkordb redis-cli GRAPH.QUERY orion_substrate "MATCH (n) WHERE n.node_id IN ['node:substrate.bus_synaptic','node:substrate.route'] DETACH DELETE n"
```

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "DELETE FROM substrate_node_prediction_error_baseline WHERE target_id IN ('node:substrate.bus_synaptic','node:substrate.route')"
```

```bash
docker exec orion-athena-falkordb redis-cli DEL orion:lattice:transport_thresholds:v1
```

History rows in `substrate_node_prediction_error_history` age out by the 168 h retention prune; readers already ignore them. The FalkorDB node may be re-created by cortex-exec's materializer (documented for transport); readers ignore it either way.

## Post-deploy checks

1. No new rows: `select node_id, max(observed_at) from substrate_node_prediction_error_history where node_id in ('node:substrate.bus_synaptic','node:substrate.route') group by 1` stays at the pre-deploy time.
2. Field pruned: latest `substrate_field_state.field_json->'node_vectors'` has neither key.
3. Self-model: newest `substrate_attention_self_model` rows list only execution/biometrics/chat in `prediction_error_by_domain`.
4. Goals: no `field_goal_provenance_published` line names either node.
5. Mood-arc: `field_channel_anomaly_score` keeps ~1,440 rows/day; no new `metacog_trigger` with `trigger_kind='telemetry_anomaly'`; `/mnt/telemetry/field_channels/corpus/field_channels.jsonl` grows.

## Risks / concerns

- Severity: medium. Concern: confidence and attention shift at once (intended, single deploy). Mitigation: merge after the 11-17 numbers; checks above.
- Severity: low. Concern: v4 encoder now gets a 0.0 bus_synaptic input; its scores are not trusted until retrained. Mitigation: trigger off; documented in `orion/mood_arc/README.md`.
- Severity: low. Concern: `capability:transport` now has no pressure. Mitigation: correct ("unmeasured"); a real bus forecast is the spec's successor.
- Severity: low. Concern: pre-existing failure `tests/test_metric_lineage.py::test_prediction_error_blast_radius_includes_endogenous_curiosity` on main. Not from this branch.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3201

🤖 Generated with [Claude Code](https://claude.com/claude-code)

# PR report: alarms are not prediction errors; retire the codebase PE node

## Summary

- Three signals fire on a fact, not on a missed forecast: the cabinet warming alarm, strain news, and street events (doorbell rings, graded handoffs). They keep competing for attention exactly as before. They stop being read as a "surprise size" by readers that expect one, and they stop being called "prediction error".
- One contract names them: `ALARM_NODE_IDS` in `orion/substrate/prediction_error_freshness.py`. A test pins it equal to the glossary's `prediction_error` entries with `value_kind: trigger` (the glossary already meant "it fired" by that kind; no new schema field).
- `node:substrate.codebase` is retired end to end: no graph node, no field receipt, on the shared retired deny-list and the field digester's prune list, glossary entry removed. The churn score itself stays in its own log for the Hub co-creation tab and Orion's self-study reader.
- The env flag that gated the codebase node write is renamed, because it no longer writes a node.
- Perception 0.0 leak (spec retire step 8): already fixed by #3171 and verified on live field history. No code change; its regression tests already exist.

## Outcome moved

- Curiosity no longer files a strain news pulse or a cabinet alarm as "sustained prediction error". Live, 7 days to 2026-11-13: strain made 56 such seeds (33 admitted), cabinet 6 (4 admitted). Both still reach curiosity through the attention open loop when they win attention (strain 30, cabinet 16, street 1 open-loop seeds in the same week, unchanged).
- A churn number that is not a forecast no longer wins Orion's attention. Live, 7 days: codebase won 3,432 of 122,958 attention frames (2.79%). Replayed: 3,146 of those frames become "no winner" (a calm body stays silent); 286 go to the next eligible source (strain 105, execution 69, chat 46, perception 45, reading 21). It also stops seeding curiosity (128 seeds, 53 admitted; 73 open-loop seeds) and stops opening metacog episodes (14 in 7 days).
- Goals: no shift. Goals may name only `node:substrate.biometrics`, `node:substrate.execution` or an attended world source (`goal_provenance._goal_eligible_node`). Cabinet, strain and codebase never could. Street is a world source and still can, unchanged.

## Current architecture

- Every `node:substrate.*` node with a `prediction_error` metadata value was treated as a prediction error by generic readers: attention contests, dynamics pressure, endogenous curiosity seeds, the metacog episode gate, the outcome rollup, self-readings, Hub's PE panel.
- Cabinet, strain and street were already glossary `value_kind: trigger`, and the alarm-aware readers (world-first judge, self-readings, outcome rollup) already branched on that. Curiosity's seed path did not.
- Codebase: `_handle_codebase_delta_message` scored each `CodebaseDeltaV1` with `codebase_prediction_error()` and wrote a graph node, a field receipt, the EWMA baseline and the raw delta log.

## Architecture touched

- Shared: `orion/substrate/prediction_error_freshness.py` (contract), `orion/substrate/endogenous_curiosity.py`, `orion/substrate/attention_broadcast.py`, `orion/attention/self_readings.py`, `config/field/field_channel_glossary.v1.yaml`, `orion/metrics/semantics.py`, metric lock.
- orion-substrate-runtime: codebase handler, node labels, settings/env/compose/README.
- orion-field-digester: `RETIRED_PSEUDO_NODES`.
- orion-attention-runtime, orion-equilibrium-service: candidate labels.
- orion-hub: attention organ PE panel.
- orion-thought eval: self-readings truth parser.

## Per-consumer decisions

| Consumer | Cabinet / strain / street | Codebase |
|---|---|---|
| Attention-runtime world-first contest | keep (it is the alarm path); label now "cabinet warming alarm" / "strain news" / "street event alarm" | gone (retired deny-list) |
| Substrate attention broadcast | keep; signal `signal_kind=substrate_alarm`, `signal_driver=alarm` (written, no reader matches on the old value) | gone |
| Dynamics pressure seed (`pressure.py`) | keep: the cabinet's documented route into the workspace (`cabinet_heat.py` docstring) | gone |
| Endogenous curiosity PE seeds | STOP (value is "it fired", not a size) | gone |
| Curiosity attention open-loop seeds | keep (follows attention) | gone (never wins) |
| Self-readings (orion-thought stance, reverie) | keep as fires per hour; ref now `[alarm:cabinet@...]` | gone (glossary entry removed) |
| Hourly outcome rollup | keep (on/off hours, already trigger-aware) | gone |
| Metacog episode gate | keep (judged by the same trigger rule); label says alarm | gone |
| Self-model PE confidence / domains | never included; now guard-tested | never included |
| Goal provenance | never eligible (street: world source, unchanged) | never eligible |
| Hub attention organ PE panel | STOP (not a PE domain) | removed from known domains |
| `substrate_node_prediction_error_history` | keep writing: attention's 7-day window and strain news's restart seed read it | no new rows |
| Field node vector | none (graph-only nodes) | digester prunes it |
| Hub co-creation tab, cortex-exec self-study, mood-arc corpus | n/a | keep: they read `substrate_codebase_delta_log.score` as churn, not PE |

## Perception 0.0 leak (spec retire step 8)

Already fixed by #3171 (merged 2026-11-10 08:02 UTC); the running digester and substrate-runtime contain it (`mode="clear"` in `state_deltas.py`, `missing_reason=` in `worker.py`). Live evidence from `substrate_field_state` (2026-11-10 05:53 to 11-13 06:17, 123,341 writes) and `substrate_perception_embedding_baseline`:

- Only one camera gap of 45 s or more since 2026-11-10 08:00: cam0 04:02:52 to 04:05:05 and walkway 04:02:51 to 04:05:13 on 2026-11-13.
- Field `prediction_error` for both camera nodes in that gap: ABSENT on 43 consecutive writes (04:03:48 to 04:05:18), never 0.0. That hour is the only one where the channel was missing (1,644 of 1,687 writes present); every other hour it is present on every write.
- The 0.0 run just before (04:00:34 to 04:03:18) is a held real reading: the history table has a real 0.0 row at 04:01:02 for both nodes, frames were still arriving, and it changed before the 180 s expiry.
- Regression tests already exist: `services/orion-field-digester/tests/test_prediction_signal_missing_is_absent.py` (missing tick drops the channel; stopped producer expires after 180 s). None added.

## Files changed

- `orion/substrate/prediction_error_freshness.py`: codebase on `RETIRED_PREDICTION_ERROR_NODE_IDS`; new `ALARM_NODE_IDS`, `ALARM_LABELS`, `is_alarm_node`, `node_signal_label`.
- `orion/substrate/endogenous_curiosity.py`: PE seeds skip alarms.
- `orion/substrate/attention_broadcast.py`: alarm signal kind/driver.
- `orion/attention/self_readings.py`: `alarm:` ref; codebase words removed.
- `orion/substrate/prediction_error.py`: docstring, codebase score no longer a node.
- `orion/schemas/context_provenance.py`: self_readings description mentions `[alarm:...]`.
- `config/field/field_channel_glossary.v1.yaml`: alarm meaning text, consumers; codebase entry removed.
- `orion/metrics/semantics.py`, `config/metrics/metric_definitions.lock.json`: codebase URN out of the prompt inventory; re-locked.
- `services/orion-substrate-runtime/app/{worker,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`: codebase node/receipt removed; alarm node labels; flag rename.
- `services/orion-field-digester/app/tensor/channels.py`: prune codebase.
- `services/orion-attention-runtime/app/worker.py`, `services/orion-equilibrium-service/app/self_signal_episode_gate.py`: labels.
- `services/orion-hub/scripts/attention_organ_routes.py`: PE panel drops codebase and alarms.
- `services/orion-thought/evals/self_readings_truth.py`: parses `[alarm:...]`.
- `scripts/sync_local_env_from_example.py`: `SUBSTRATE_CODEBASE_` prefix so the renamed key syncs.
- `.github/workflows/retired-pe-guards.yml`: runs the new tests.
- Tests: new `tests/test_pe_alarm_refile.py`; updated retired/fixture/eval tests that used codebase as their example event-written node (now execution).

## Schema / bus / API changes

- Added: `ALARM_NODE_IDS` contract (code + test against the glossary). No schema model, registry or channel change.
- Removed: `node:substrate.codebase` glossary entry and node.
- Renamed: env `SUBSTRATE_WRITE_CODEBASE_PREDICTION_ERROR_NODE` -> `SUBSTRATE_CODEBASE_DELTA_LISTENER_ENABLED`; setting `enable_codebase_prediction_error_node` -> `enable_codebase_delta_listener` (default now True, matching the live value).
- Behavior changed: broadcast signals for alarms carry `signal_kind="substrate_alarm"`; graph labels for alarm nodes; self-readings ref prefix for alarms.
- Compatibility notes: the `prediction_error` field and history rows of the alarm nodes are unchanged, so attention and strain news's restart seed behave the same. Deploy order does not matter.

## Env/config changes

- Added keys: `SUBSTRATE_CODEBASE_DELTA_LISTENER_ENABLED=true` (substrate-runtime).
- Removed keys: `SUBSTRATE_WRITE_CODEBASE_PREDICTION_ERROR_NODE` (renamed).
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`+SUBSTRATE_CODEBASE_DELTA_LISTENER_ENABLED='true'`). The old key is still in the local `.env`; it is ignored. Optional tidy: delete line `SUBSTRATE_WRITE_CODEBASE_PREDICTION_ERROR_NODE=true` from `services/orion-substrate-runtime/.env`.
- skipped keys requiring operator action: none.

## Metric gate record (for the alarm re-file)

No new metric. The three alarm values are unchanged in provenance (`cabinet_heat.py:176`, `strain_news.py:58`, `street_attention.py:247`); only the set of readers moved. Reversibility: drop an id from `ALARM_NODE_IDS` (and set its glossary kind to a measurement) to restore the old reads.

## Tests run

(filled below)

## Evals run

(filled below)

## Docker/build/smoke checks

(filled below)

## Review findings fixed

(filled below)

## Restart required

Normal post-pull rebuild from clean main: orion-substrate-runtime, orion-field-digester, orion-attention-runtime, orion-equilibrium-service, orion-hub, orion-thought (shared `orion/` and `config/`).

## Post-deploy cleanup (production writes, for Juniper to run; not run here)

Stop substrate-runtime first so its graph cache cannot write the node back:

```bash
docker stop orion-athena-substrate-runtime && docker exec orion-athena-falkordb redis-cli GRAPH.QUERY orion_substrate "MATCH (n) WHERE n.node_id = 'node:substrate.codebase' DETACH DELETE n" && docker start orion-athena-substrate-runtime
```

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "DELETE FROM substrate_node_prediction_error_history WHERE node_id = 'node:substrate.codebase'"
```

705 rows; they would also age out in 7 days, and every reader already skips them. No `substrate_node_prediction_error_baseline` row exists for codebase (checked). The field digester prunes the field vector by itself on its next reconcile.

Checks after deploy:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "SELECT max(observed_at) FROM substrate_node_prediction_error_history WHERE node_id='node:substrate.codebase'"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "SELECT count(*) FROM substrate_codebase_delta_log WHERE observed_at > now()-interval '1 day'"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "SELECT c->'focal_node_refs'->>0, count(*) FROM substrate_endogenous_curiosity_candidates t, jsonb_array_elements(t.candidates_json) c WHERE generated_at > now()-interval '1 day' AND c->>'evidence_summary' LIKE 'sustained prediction error%' GROUP BY 1"
```

Pass: first stays at the pre-deploy time; second keeps growing (listener alive); third lists no cabinet, strain, street or codebase.

## Risks / concerns

- Severity: low. Concern: the alarms still travel on a field named `prediction_error` and in a table with that name. Mitigation: `ALARM_NODE_IDS` is the one place readers ask; moving storage to an alarm channel is a larger migration and is filed as a follow-up issue.
- Severity: low. Concern: the codebase churn score (not a forecast) still feeds the Hub co-creation tab and the self-study reader. Mitigation: neither reads it as prediction error; whether to keep it is a separate decision, filed as an issue.

## PR link

(filled below)

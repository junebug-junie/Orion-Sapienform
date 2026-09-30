## Summary

- Retires the `transport` metacog trigger's third evidence source: equilibrium's 30s FalkorDB poll of `node:substrate.bus_synaptic`'s `prediction_error`, which fired `transport:bus_synaptic:episode_start:error=X` whenever the reading crossed 0.15.
- Ran CLAUDE.md's metric quality gate on it with live data for the first time. It fails steps 2-4 (details below), so it is removed outright, not disabled: builder, poll loop, FalkorDB client, 4 `EQUILIBRIUM_METACOG_TRANSPORT_BUS_SYNAPTIC_*` settings, `FALKORDB_URI`/`FALKORDB_SUBSTRATE_GRAPH` (only this poll used them), compose entries, its unit/e2e tests and its e2e eval.
- Adds `tests/test_bus_synaptic_transport_retired.py`, which fails on the pre-patch tree (4/4) and passes after, so a later patch cannot quietly bring it back.
- Out of scope, left alone: the producer (`orion-substrate-runtime`'s `_bus_synaptic_tick`) and the node's other readers (attention self-model, curiosity, world-model features, spark), and the Hub's "Bus Anomaly Detected" notifier, which now never receives a matching trigger.

## Outcome moved

Removes 50-155 content-free metacog reflections a day (642 in the last 7 days, e.g. "A synaptic prediction error exceeded threshold, signaling a shift in perceptual processing.") and the matching Hub "Bus Anomaly Detected" notifications. Real transport trouble stays covered by sources that measure it directly: rpc_health timeouts, rpc_timeout grammar atoms, and the per-hop EWMA baseline gate.

## Metric quality gate (live, 2026-09-29/30)

1. **Where the number comes from.** `orion-bus-mirror` (`app/main.py` mirror loop -> `graph_writer.py::record_publish`/`record_causal_hop`) keeps a running average and z-score (how unusual a value is) per bus edge in FalkorDB graph `orion_bus_synapse`: the gap between two messages on each organ->channel edge (`gap_zscore`) and the delay on each organ->organ hop (`latency_zscore`). Every 30s, `orion-substrate-runtime/app/worker.py::_bus_synaptic_tick` reads those z-scores for edges seen in the last hour with count > 5, and `orion/substrate/prediction_error.py::bus_synaptic_prediction_error` returns the fraction with |z| >= 3. That value is written to `node:substrate.bus_synaptic` in graph `orion_substrate`, which equilibrium polled.
2. **Is it an independent sensor?** No. As a detector of mesh-wide outages it overlaps the direct RPC sensors. The one thing it was meant to add, per its own design notes, was catching a single bespoke organ (the governor's long-poll RPC). Its own docstring shows that can't work: one organ failing completely reads about 0.05, which sits inside the normal noise, below the 0.15 threshold.
3. **Theory anchor.** The fraction of edges past 3 sigma is a real statistic, but the timestamps behind every z-score are taken when the mirror *reads* a message, not when it was published. The mirror handles one message at a time (a SQLite commit plus 2-4 FalkorDB round trips each) and falls behind. Measured live: its Redis pub/sub output buffer grew about 110 KB/s to 45 MB, then Redis dropped the connection at 23:14:59. The bus shows `client_output_buffer_limit_disconnections:424`, which matches the mirror's 425 container restarts, and the logs show 79 crash/reconnects in 24h, about one every 18 minutes. So every edge's gap shares one noise source: the observer's own backlog.
4. **Live-data sanity.**
   - **Rest state: yes.** The 2026-07-26 `sqrt(2/pi)` ~0.27 floor is gone, because the 2026-07-30 redefinition counts edges instead of averaging magnitudes. Live 30s receipts sit at 0.020-0.030 between spikes. The long-run baseline row (`substrate_node_prediction_error_baseline`, 125,275 observations) has an average of 0.040 with sd 0.036. So the 0.15 threshold is not firing on a calm floor.
   - **What it fires on.** 781 episodes in 14 days (daily 2-155; 51/64/84/70/111/155 over the last six days). Value at episode start: median 0.20, p90 0.28, max 0.73. They are spread evenly across the hours of the day (UTC hourly counts 15-46).
   - **Live spike captured** (FalkorDB sampled every 15s): the fraction went 0.041 -> 0.089 -> 0.147 -> 0.175 -> 0.152 -> 0.059 between 23:01 and 23:03. The trigger fired at 23:03:04 with error=0.176. At the peak, 44 of 290 edges were anomalous, and about 30 of them were different organs' `orion:system:health` heartbeat edges all at z 3-4 at once (average gap about 72s, sd about 22s, observed gap about 150s). Unrelated services on different hosts do not miss a heartbeat together. A shared observer-side stall does make this pattern. It also explains why only slow edges show it: fast edges overwrite their z-score with a normal value within seconds.
   - **Does it follow real incidents? No evidence of it.** During the 2026-09-28 14:00-19:00 RPC-timeout storm (52-70 other transport fires per hour), bus_synaptic fired once per hour. Overnight, in quiet hours, it fired 3-6 times per hour. Across 14 days the hourly correlation with other transport fires is -0.65. Only 22% of its fires had another transport fire within ±2 min, against 30% for random times. *Caveat:* all three sources share one 30s cooldown, which can hide a bus_synaptic episode during a storm, so the negative correlation is partly produced by that cooldown. It is not evidence that the metric runs backwards; it only rules out "tracks storms".
   - **Link to mirror crashes.** 32% of fires land within 2 minutes after a mirror crash, against 12% for random times (2.7x). Not every crash produces a spike: the 23:14:59 disconnect did not.
5. **Does it already exist elsewhere?** Yes, the direct transport sensors above. A per-organ signal, the right tool for single-organ loss, belongs at the Hub `/propagate` seam named in the producer's docstring. It does not belong here.
6. **Reversibility.** Cheap. Nothing persists this trigger's state and no schema changes. `orion/metacog/evidence_map.py` and `capture_replay.py` still map historical `bus_synaptic_prediction_error` rows, so the stored `orion_metacog` history keeps reading correctly.

**Verdict: RETIRE.** No threshold can be fitted from this data: there is no labeled set of real incidents to fit it against, and what dominates the signal comes from the observer, not the mesh.

## Current architecture

`orion-equilibrium-service` had three `transport` evidence sources sharing one 30s cooldown: (A) `RpcHealthSnapshotV1` timeouts, (C) `rpc_transport_timeout` grammar atoms, and (bus_synaptic) a FalkorDB poll with rising-edge detection, hysteresis, and latch-on-publish.

## Architecture touched

`services/orion-equilibrium-service` only, plus one producer-contract test in `orion/metacog/tests`. No bus, schema, or registry changes.

## Files changed

- `services/orion-equilibrium-service/app/service.py`: removed the import, `_node_age_sec`, the FalkorDB client, the rising-edge state, `_bus_synaptic_poll_loop`, and the task create/cancel/gather lines.
- `services/orion-equilibrium-service/app/transport_metacog_gate.py`: removed `build_transport_metacog_trigger_from_bus_synaptic`; left a retirement note in its place.
- `services/orion-equilibrium-service/app/settings.py`: removed the 4 bus_synaptic fields and the 2 falkordb fields.
- `services/orion-equilibrium-service/docker-compose.yml`, `.env_example`, `README.md`: removed the keys and docs; added a retirement note.
- `services/orion-equilibrium-service/tests/test_transport_metacog_gate.py`: removed the bus_synaptic tests.
- `services/orion-equilibrium-service/tests/test_bus_synaptic_poll_e2e.py`, `test_bus_synaptic_poll_state_machine.py`, `evals/run_bus_synaptic_poll_e2e_eval.py`: deleted.
- `services/orion-equilibrium-service/tests/test_bus_synaptic_transport_retired.py`: new regression gate.
- `orion/metacog/tests/test_evidence_map_producer_contract.py`: dropped the contract test for the removed builder.

## Schema / bus / API changes

- Added: none
- Removed: equilibrium no longer emits `MetacogTriggerV1` with `upstream.evidence_source=bus_synaptic_prediction_error`.
- Renamed: none
- Behavior changed: the `transport` kind now has two evidence sources.
- Compatibility notes: the Hub notifier and `evidence_map` still accept the old shape. Nothing breaks; the Hub notifier just goes quiet.

## Env/config changes

- Added keys: none
- Removed keys: `EQUILIBRIUM_METACOG_TRANSPORT_BUS_SYNAPTIC_POLL_ENABLE`, `_POLL_INTERVAL_SEC`, `_ERROR_THRESHOLD`, `_CLEAR_RATIO`, `FALKORDB_URI`, `FALKORDB_SUBSTRATE_GRAPH` (equilibrium only)
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-equilibrium-service`: ran, "No changes needed". The sync script does not delete keys, so the 6 retired keys and their comments were removed from the primary checkout's `services/orion-equilibrium-service/.env` by hand, with a backup in the session scratchpad.
- skipped keys requiring operator action: none

## Tests run

```text
services/orion-equilibrium-service: pytest tests -q            -> 168 passed
  pre-patch tree + new regression test                          -> 4 failed (expected), 184 passed
orion/metacog/tests + orion/schemas/tests/test_metacog_entry.py -> 221 passed
python scripts/check_env_template_parity.py                     -> PASS
git diff --check                                                -> clean
```

## Evals run

```text
PYTHONPATH=. python orion/metacog/evals/run_capture_eval.py -> PASS
```
The deleted `run_bus_synaptic_poll_e2e_eval.py` covered only the retired path. There is no replacement to add.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-equilibrium-service build -> Built (not brought up)
docker run --rm --network none <image> python -c 'import app.service ...' -> poll False builder False
```

## Review findings fixed

- Finding: the PR report cited by the gate note, .env_example, README and regression test was not committed yet.
  - Fix: this file committed.
  - Evidence: `ls docs/superpowers/pr-reports/2026-09-30-retire-bus-synaptic-transport-trigger-pr.md`
- Finding (nit): the README still said "All three evidence sources share this one lane".
  - Fix: now says both remaining sources (A/C).
  - Evidence: README.md transport section.
- Finding (nit): the requirements.txt pin comment and the settings.py ship-disabled comment still pointed at the removed falkor import / bus_synaptic precedent.
  - Fix: reworded both.
  - Evidence: diff.
- Finding (not fixed, out of scope): stale comment/doc references in other services (`orion-substrate-runtime/README.md:718` still names equilibrium's poll as a consumer; orion-bus-mirror README; orion-mind/orion-hub comments cite the removed threshold key). None of them run as code. Left for a follow-up because this patch is limited to equilibrium.
- Reviewer confirmed: no dangling code references anywhere in the repo (including CI, Makefile and scripts); no non-bus_synaptic code removed; rpc_health, grammar-atom and baseline-gate paths untouched. `evidence_map.py`'s historical-row mapping should stay, because the capture-replay fixture and in-flight triggers depend on it.

## Restart required

```bash
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build   # from a worktree at merged main
```

## Risks / concerns

- Severity: medium. Concern: `orion-bus-mirror` is a chronically slow pub/sub consumer. Redis disconnects it about every 18 min, and it stamps edge timing at read time. So `node:substrate.bus_synaptic` stays questionable for its other readers (attention self-model, endogenous curiosity, world-model features, spark concept induction). Mitigation: follow-up to measure the mirror's throughput and switch to publish timestamps, then re-run the metric gate for those readers.
- Severity: low. Concern: the Hub's `bus_synaptic_trigger_notifier.py` is now dead code. Mitigation: follow-up PR in orion-hub to delete it (outside this patch's allowed scope).
- Severity: low. Concern: during the investigation, one read-only SQLite query on the mirror's 25 GB `bus_mirror.sqlite` held a shared lock long enough that the mirror's writer hit `database is locked` and crashed once (23:04:59, back up within 1s). Mitigation: do not query that file in place. Copy it or use a replica.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2421

🤖 Generated with [Claude Code](https://claude.com/claude-code)

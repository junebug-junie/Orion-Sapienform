## Summary

- The GPU pool now publishes each lifecycle event (admitted, granted, released, ...) on its own bus correlation id, not on the id of the chat turn that asked for the GPU.
- The pool's grammar events (the exception cases: unavailable, recalled, aborted, ...) do the same.
- The turn is still findable: every event keeps `turn_correlation_id` in its payload, and every grammar event keeps the turn in its payload `correlation_id`. Every consumer already joins on those payload fields; none reads the envelope id.
- The envelope id is the event's own `event_id` read as a UUID, so a bus message can be matched to its `gpu_pool_events` row by hand.
- Juniper chose this on 2026-09-29 ("fresh event IDs"). It was option (a) in the Risks section of the stage-3 loose-ends report (`2026-09-29-gpu-pool-stage3-loose-ends-pr.md`).

## Outcome moved

Bus-mirror (the service that turns bus traffic into a live graph of which service follows which) links any two messages that share a correlation id into a "service A caused service B" edge. Because pool events carried the turn's id, the pool showed up inside every turn's causal chain. That broke the parent spec's rule that waiting in line for a GPU is not transport. After deploy, the pool drops out of turn chains, and the 36 pool edges stop growing.

## Current architecture

- `services/orion-gpu-pool/app/runtime.py::_emit` published `orion:gpu_pool:event` with `correlation_id = event.turn_correlation_id`. For grammar-worthy events, `_grammar` called `publish_grammar_event` with no explicit id. That helper falls back to `GrammarEventV1.correlation_id`, which is also the turn.
- The lease RPC itself already used a fresh id (`orion/gpu_pool/client.py::lease_rpc`, `correlation_id=uuid.uuid4()`). The replies echo the request's id (`services/orion-gpu-pool/app/main.py::_reply`). State snapshots (`publish_state`) and actuator requests passed `None` and already got a fresh uuid4.
- So the only turn-id leak was `_emit` plus `_grammar`.

## Architecture touched

- `orion-gpu-pool` runtime only. No schema, channel, env, or consumer change.

## Files changed

- `services/orion-gpu-pool/app/runtime.py`: new `event_envelope_correlation(event)` (the `event_id` as a UUID, uuid4 fallback). `_emit` and `_grammar` both use it.
- `services/orion-gpu-pool/tests/test_runtime.py`: two regression tests (see Tests run).
- `docs/superpowers/pr-reports/2026-09-29-gpu-pool-event-fresh-corr-pr.md`: this report.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: the envelope `correlation_id` on `orion:gpu_pool:event` and on pool-sourced `orion:grammar:event` messages is now per-event, not the turn's.
- Compatibility notes: payload fields are unchanged. The lease RPC request/reply id pairing is untouched, because `rpc_request` matching uses the client's fresh request id, which `_reply` echoes.

### Consumers checked

None of these reads the envelope correlation id of a pool event.

| Consumer | What it keys on | Evidence |
|---|---|---|
| `orion/gpu_pool/client.py::_wait_for_grant` (gateway, world-model, thought, diffusion, durable-runs via client) | payload `lease_id` | `payload.get("lease_id") != lease_id` |
| `services/orion-durable-runs/app/main.py` → `admission.on_pool_event(env.payload)` | payload only | passes `env.payload` |
| `services/orion-hub/scripts/gpu_pool_routes.py` (SSE + history) | payload only (`absorb(channel, payload)`); history SQL selects `turn_correlation_id` column | |
| `services/orion-sql-writer` → `gpu_pool_events` | table has **no** envelope-correlation column | live `information_schema`: event_id … turn_correlation_id … detail; last hour 332 rows, 300 with `turn_correlation_id` |
| `services/orion-sql-writer` → `grammar_events` (trace `gpu_pool.lease:*`) | payload `GrammarEventV1.correlation_id` (`orion/grammar/ledger.py:107`) | last day 257 rows, 237 with `correlation_id`. These stay the turn. |
| sql-writer grammar persist fallback (`worker.py::handle_envelope` `corr_id = env.correlation_id or payload…`) | only used for fallback/error rows on persist timeout | these fallback rows will now carry the event's id instead of the turn's. This is diagnostic only. |

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Metric gate (bus_synaptic_prediction_error): this removes an artefact, it adds no metric

1. **Provenance.** The edges come from `services/orion-bus-mirror/app/graph_writer.py::ChainTracker.observe`, which links consecutive envelopes that share a `correlation_id` across different organs. The metric is `orion/substrate/prediction_error.py::bus_synaptic_prediction_error`: the fraction of edges with |z| ≥ 3. The edges it reads come from `services/orion-substrate-runtime/app/worker.py::_BUS_SYNAPTIC_EDGE_QUERIES`, filtered to count > 5 and seen within the last `SUBSTRATE_BUS_SYNAPTIC_MAX_EDGE_AGE_SEC=3600` (live env).
2. **Independence.** The pool edges were not independent signal. They are a relabelling of the same turn chains already measured through cortex-exec→llm-gateway and similar edges, split into pieces by the pool's interleaved publish.
3. **Theory anchor.** The parent spec (`docs/superpowers/specs/2026-09-24-gpu-pool-design.md`, "Transport-metric and reader impacts" item 1) says waiting for a GPU is not transport.
4. **Live data, read-only, FalkorDB `orion_bus_synapse`, 2026-09-29 ~21:05 UTC.**
   - All-time: 36 `CAUSALLY_FOLLOWED_BY` edges touch `orion-gpu-pool`, summing 134,482 observations, out of 401 edges and 13.2M observations mesh-wide. Largest: durable-runs→pool 27,093; cortex-exec→pool 22,014; pool→llm-gateway 21,873 (EWMA 7.07 s); llm-gateway→pool 20,354; http:anthropic→pool 20,049; topic-foundry→pool 17,727.
   - In the metric's live population (count > 5, seen in the last hour): 70 causal edges plus 204 publish edges, 10 anomalous. So the metric reads about 10/274 = 0.036.
   - 12 of those 70 causal edges are pool edges, and 0 of the 12 are anomalous. Removing them gives about 10/262 = 0.038.
   - The reading barely moves. The change is to what the metric's population means, not to its level.
   - The pool's own `PUBLISHES` (gap z-score) edges are unaffected. The pool still publishes.
5. **Existing mechanism.** The lease client already used fresh ids for the RPC. This patch applies the same rule to events.
6. **Reversibility.** A one-line revert. No schema or stored default changes.

**How the old edges go away.** Bus-mirror never deletes edges. The 36 pool edges stay in FalkorDB with frozen counts. They leave the metric's population on their own 1 hour after the last pool event on the old code, because of the `last_seen_epoch` recency filter. The Hub anomaly view and the recall adapter (`services/orion-recall/app/storage/falkor_bus_synaptic_adapter.py`) may still list them as stale edges. Deleting them is an optional manual FalkorDB cleanup. It was not done here, since it is a production write.

**Known consequence (accepted by choosing option a).** When a turn is not interrupted by a pool event, bus-mirror sees cortex-exec followed directly by llm-gateway. That edge's latency will now include the time spent waiting in the GPU queue as well as the LLM call.
- Baseline before deploy: `cortex-exec→llm-gateway`, count 487,222, EWMA 4.16 s, variance 46.9, z −0.13, last seen about 48 minutes before the query. It was barely updating, because the pool was stealing that hop.
- Expect it to update often again. Expect its EWMA to drift up by the typical queue wait.
- Watch it for a z-score burst in the first hour after deploy while EWMA alpha 0.2 re-learns.

## Tests run

```text
cd services/orion-gpu-pool && PYTHONPATH=<worktree> python -m pytest tests -q -p no:cacheprovider
  92 passed, 7 skipped
  new: test_pool_events_travel_on_their_own_correlation_id_not_the_turns
       (envelope corr != turn, == UUID(event_id), unique per event; grammar envelope != turn,
        payload correlation_id == turn) -- verified FAILING on the old _emit/_grammar code
  new: test_event_envelope_correlation_falls_back_to_fresh_uuid_for_a_non_hex_event_id
cd services/orion-durable-runs && python -m pytest tests -q          188 passed, 64 skipped
cd services/orion-hub && python -m pytest tests/test_gpu_pool_routes.py tests/test_curiosity_gpu_lease.py tests/test_unified_turn_gpu_placement.py -q
  25 passed, 1 skipped
cd services/orion-sql-writer && python -m pytest tests/test_gpu_pool_event_sql_shape.py -q   4 passed
python scripts/check_metric_lineage.py --gate        PASS
python scripts/check_definition_drift.py --gate      PASS
python scripts/check_sentience_instruments.py --static-only   All claims hold
python -m pytest tests/test_grammar_event_producer_catalog.py -q   3 passed
git diff --check                                     clean
```

## Evals run

```text
python services/orion-gpu-pool/evals/run_pool_day_eval.py   VERDICT: PASS
```

## Docker/build/smoke checks

```text
Not deployed (per task). No build: no dependency, Dockerfile, or compose change.
Live read-only checks: FalkorDB edge counts and Postgres consumer columns (above).
Post-deploy proof is UNVERIFIED until run:
  MATCH (a:Organ)-[e:CAUSALLY_FOLLOWED_BY]->(b:Organ)
  WHERE a.organ_id='orion-gpu-pool' OR b.organ_id='orion-gpu-pool'
  RETURN a.organ_id, b.organ_id, e.count, e.last_seen_epoch
  -> every last_seen_epoch should stop at the deploy time and counts should stop growing.
```

## Review findings fixed

Review ran in a separate subagent, reading the diff only. It found no blockers and two nits.

- Finding: the `_emit` docstring said `turn_correlation_id` is "the only field every consumer reads". That is wrong: the client keys on `lease_id` and `event`.
  - Fix: the docstring now says "No consumer reads the envelope correlation_id".
  - Evidence: `runtime.py::_emit` docstring.
- Finding: sql-writer's grammar fallback rows (`BusFallbackLog.correlation_id`, `worker.py:2079-2091`) will now carry the event id, not the turn.
  - Fix: none needed. The turn is still inside the stored payload, and no code looks these rows up by turn. It is disclosed under Risks.
  - Evidence: reviewer grep; consumers table above.
- Also confirmed by the reviewer:
  - A pool event and its grammar event share one envelope id but have the same `source.name`. `ChainTracker.observe` returns None for a repeat from the same organ, so no edge forms.
  - The one-off chain entries are filtered out by `real_hop_count >= 1` and expire after 120 s.
  - Checked against `origin/main`'s `runtime.py`, the new test fails at the first envelope assertion.

## Restart required

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
```

Only orion-gpu-pool. No consumer needs a restart: no consumer reads the field that changed.

## Risks / concerns

- Severity: low. cortex-exec→llm-gateway latency now includes GPU queue wait (see Metric gate). Mitigation: this was the accepted trade in option (a). Watch the edge's z-score after deploy.
- Severity: low. The 36 stale pool edges stay in FalkorDB forever. They are out of the metric after 1 h, but still visible in graph views. Mitigation: optional manual delete, which needs Juniper's approval because it is a production write.
- Severity: low. sql-writer grammar fallback rows for pool events carry the event id, not the turn. They are diagnostic only.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2407

🤖 Generated with [Claude Code](https://claude.com/claude-code)

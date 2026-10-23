# Body economy A2: ActCostV2, and every act close records what it cost

## Summary

- Every act Orion closes now records what it spent, where that can be measured. Until now `grammar_acts.cost_json` was empty on every row (221 of 221 on 2026-10-23).
- New `ActCostV2` = `ActCostV1` plus `gpu_slot_s_by_lane`, `wh_by_host`, `juniper_msgs`, `claude_limit_events`. All optional; None means "not measured", never 0.
- The act ledger (sql-writer) fills cost on close: wall seconds for a producer close, and GPU slot-seconds per lane from the pool's lease events for the act's turn. Producer-reported values win; a cost reported at open survives the close.
- Consumer-first: every reader accepts V2 in this PR; no producer sends V2 fields. The writer-side fill never crosses the bus.

## Outcome moved

The act ledger can now answer "what did this act cost?" Replaying the new rule over the live ledger (read-only SQL, 2026-10-23) shows 199 of 221 closes would get a non-null cost:

```text
actor                 close source  closes  seconds  gpu  non-null
hub.chat_reply        resolver          34        0   34        34
hub.outreach          resolver           8        0    8         8
hub.world_pulse_read  producer         116      116    0       116
hub.world_pulse_read  expired           15        0    0         0
thought.dream_hop     producer          18       12    0        12   (6 closed before their open arrived)
thought.painting      producer          30       29    0        29
```

The remaining 22 have nothing measurable. They are expired closes, which have no producer finish time, and closes that arrived before their open, which have no wall time. They stay NULL rather than recording a cost of zeros.

## Current architecture

`ActV1.cost` was `ActCostV1 | None` with `{seconds, gpu_seconds, tokens, watts}`, `extra="forbid"`. No producer filled it. `close_act` wrote `cost_json = COALESCE(:producer_cost, cost_json)`, so the column stayed NULL. `substrate_action_outcomes.latency_ms` came from `cost.seconds` when a producer sent one, which none did.

## Architecture touched

- Contract: `orion/schemas/grammar.py` (`ActCostV2`; `ActV1.cost: ActCostV1 | ActCostV2 | None`).
- Writer: `orion/grammar/act_ledger.py` (`fill_close_cost`, called from `close_act`), which runs inside orion-sql-writer.
- No bus channel, env, registry or table change. `cost_json` is an existing jsonb column.

## Files changed

- `orion/schemas/grammar.py`: `ActCostV2` (non-negative maps, `from_any`), and the widened `ActV1.cost` union.
- `orion/grammar/act_ledger.py`: `fill_close_cost`, the GPU lease join (`_GPU_LEASES_FOR_TURN_SQL`, in its own savepoint), the stored-open-cost merge, and a `log_prefix` on `_isolated`.
- `orion/grammar/acts.py`: a comment pinning `act_closed_event(cost=)` to V1 until the producer phase.
- `tests/test_act_ledger_contract.py`: union parsing, V2 round trip, forbid and negative values, the pinned consumer-first hazard.
- `services/orion-sql-writer/tests/test_act_ledger_integration.py`: the regression that `cost_json` is filled on close, plus a missing GPU table, the resolver and expired paths, close-before-open, producer cost wins, and the open cost surviving the close.
- `services/orion-sql-writer/README.md`: an act ledger paragraph.

## Schema / bus / API changes

- Added: `ActCostV2` (nested model; not a bus payload of its own).
- Removed: none.
- Renamed: none.
- Behavior changed:
  - `grammar_acts.cost_json` is now written on close. The merge order is: the close's own cost, then the cost stored at open, then the writer fill.
  - `seconds` is filled only on a producer close.
  - `gpu_slot_s_by_lane` is filled from `gpu_pool_events` where `turn_correlation_id` equals the act's `correlation_id`. The window is open minus 1 h to close plus 120 s, and each lease counts once (the max `held_ms`).
  - `latency_ms` in L10 still comes only from a producer-reported `seconds`, so L10 is unchanged.
- Compatibility notes (rollout):
  1. **This PR.** All readers get the V2-aware model. Producers are unchanged and send V1 or no cost. The fill writes only `grammar_acts.cost_json`, never `grammar_events.event_json` or the bus. So every deploy order is safe: an old reader never sees a V2 payload.
  2. **Later PR, producers.** A producer may send V2 fields only after every reader of `orion:grammar:event` / `grammar_events` runs this code. Today that is sql-writer (persist, fallback drain, `_persisted_close`) and substrate-runtime (`store.py` reads `grammar_events.event_json`), plus anything else built on `orion/schemas/grammar.py`. V1 and `ActV1` are `extra="forbid"`, so an old reader rejects the whole event (the #2126 lesson). `test_v1_reader_model_rejects_v2_fields_which_is_why_producers_wait` pins this.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Metric quality gate

1. **Provenance.**
   - `seconds` is `grammar_acts.closed_at - opened_at`, both stamped by the producer's own act events.
   - `gpu_slot_s_by_lane` is `gpu_pool_events.held_ms`, which the pool stamps at release (`services/orion-gpu-pool/app/runtime.py` `_emit_row`). The join key is `turn_correlation_id`.
2. **Independence.**
   - `seconds` and GPU slot-seconds overlap for a GPU act: a long act holds a lease longer. But wall time also counts queueing and non-GPU work, so they are related, not one transform of the other.
   - Neither feeds any model yet.
3. **Theory anchor.** These are the design's ActCostV2 currencies (Organ 3), measured in the units the Body organ budgets: slot-seconds per lane, and wall time.
4. **Live data.**
   - Join coverage was checked live: 34 of 34 chat replies and 8 of 8 outreach acts join to their turn's leases. Live, a chat reply's leases end at most about 7.5 min before its act opens, so the 1 h lookback is enough.
   - Degenerate case found and fixed in review: on resolver and expired closes, `seconds` would have equalled `horizon_sec` exactly, a constant. These closes now get no `seconds`.
5. **Existing mechanism.** The fill reuses the existing `cost_json` column, `_isolated` savepoints and pool lease events. No new table.
6. **Reversibility.** The change is cheap to undo. Nothing reads `cost_json` yet, and the fill is one function.

## Tests run

```text
PYTHONPATH=. pytest tests/test_act_ledger_contract.py orion/grammar/tests -q          48 passed
ORION_SQL_WRITER_TEST_DATABASE_URI=<throwaway pg16 container> \
  pytest -m integration services/orion-sql-writer/tests/test_act_ledger_integration.py \
         services/orion-sql-writer/tests/test_act_workspace_binding_integration.py -q  30 passed, 0 skipped
PYTHONPATH=.:services/orion-sql-writer pytest services/orion-sql-writer/tests/test_act_ledger_wiring.py \
  services/orion-sql-writer/evals/test_act_ledger_calibration_eval.py \
  services/orion-sql-writer/tests/test_grammar_ledger_sql_shape.py -q                  8 passed
PYTHONPATH=.:services/orion-substrate-runtime:services/orion-sql-writer/tests \
  pytest -m integration services/orion-substrate-runtime/tests/test_grammar_consumer_integration.py   3 passed
pytest tests/test_agent_trace_schema_registry.py tests/test_grammar_event_producer_catalog.py  5 passed
python scripts/check_definition_drift.py --gate      no definition changes
python scripts/check_metric_lineage.py --gate        PASS
python scripts/check_inner_state_registry.py         OK
body-economy contract gate (#2862's checker, run against this branch)   OK
```

The Postgres integration tests ran here against a disposable local `postgres:16-alpine` container, not the live database. CI runs the same two files in its "must not skip" step.

## Evals run

```text
services/orion-sql-writer/evals/test_act_ledger_calibration_eval.py   passed (unchanged behavior: L10 grading untouched)
Live replay of the fill rule over grammar_acts (read-only SQL, table above): 199/221 closes non-null.
```

## Docker/build/smoke checks

```text
No compose or env change. Runtime path: orion-sql-writer's grammar persistence. No Docker build was run.
UNVERIFIED live: the first new close writing a non-null cost_json happens only after sql-writer is rebuilt.
```

## Review findings fixed

The independent review (orion-repo-agent, read-only) found no must-fix issues. It confirmed the rollout is safe in any deploy order. Fixed:

- Finding: resolver and expired closes wrote a made-up `seconds`. It equalled `horizon_sec` exactly, which Organ 6 would later learn as cost.
  - Fix: `seconds` is filled only when `outcome.source == "producer"`. The docstring is updated.
  - Evidence: `test_resolver_and_expired_closes_do_not_invent_seconds`. A resolver chat reply keeps its real GPU cost; an expired look stays NULL.
- Finding: the close overwrote the cost stored at open. This was latent, since no producer reports cost yet.
  - Fix: `cost_json` is read in the close's `FOR UPDATE` and merged in this order: the close's cost, then the open's cost, then the fill.
  - Evidence: `test_cost_reported_at_open_survives_the_close`.
- Finding: the reviewer's own environment skipped the new Postgres integration tests.
  - Fix: I ran them here against a throwaway Postgres: 30 passed, 0 skipped.
- Nit: the widened `act_closed_event` signature invited an early V2 producer.
  - Fix: the signature stays V1, with a comment explaining the consumer-first rule.
- Nit: a shared correlation_id would charge every act the whole turn's GPU time. This is now documented in the comment block. Live, no correlation_id is shared.
- Nit: GPU-join failures were logged as `act_workspace_gpu_cost_failed`. They are now `act_cost_gpu_failed`.

## Restart required

From the clean primary `main` checkout after merge and `git pull`:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

To verify after the first act closes: `SELECT actor, outcome_source, cost_json FROM grammar_acts WHERE closed_at > now() - interval '1 hour' ORDER BY closed_at DESC LIMIT 5;`. Producer closes should show `seconds`, and chat-reply closes should show `gpu_slot_s_by_lane`. Other services pick up the V2-aware schema on their next rebuild; nothing requires it yet.

## Risks / concerns

- Severity: low. Concern: GPU slot-seconds count a durable-run hold and its child calls as separate leases. Mitigation: documented. The Body organ (#2863) excludes children via `gpu_pool_leases.hold_lease_id`, and the same exclusion can be applied here once a GPU-heavy durable act closes on the ledger.
- Severity: low. Concern: `wh_by_host`, `juniper_msgs` and `claude_limit_events` stay None. No measurement is wired yet, and by design that is not 0.
- Severity: low. Concern: this is independent of #2862 and #2863 and is based on main. It touches `orion/grammar/act_ledger.py`, which #2862's contract lists as a protected path (organ: learning). That listing is informational and adds no gate.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

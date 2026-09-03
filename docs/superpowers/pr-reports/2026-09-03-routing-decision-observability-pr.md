# Record how Orion routes a chat turn, and whether its own gate demoted it

## Summary

- Orion's mutation loop can turn one dial about its own behavior: `chat_reflective_lane_threshold`, the confidence it needs before acting instead of just replying. Until now the gate that dial drives fired and left no trace — no record existed of what it actually did to a turn.
- Adds a full contract (schema, bus channel, producer, consumer) that records, per auto-routed turn: the depth Orion wanted before the gate, the depth after, its confidence, the threshold it was judged against, and whether the gate demoted it.
- Fixed in review, before this ever ran live: the consumer wasn't subscribed (would have produced zero rows forever), the publish could have stalled a chat turn up to two minutes on a slow bus, and a free-text `reason` field could have leaked verbatim user content onto the bus.
- **This closes the observability gap only.** It does not yet fix that Orion's mutation loop is watching the wrong evidence to justify changing this dial — see Outstanding below and `project_mutation_loop_signal_action_mismatch_2026-09-03.md`.

## Outcome moved

Before: zero record of Orion's own routing decisions existed anywhere. After: every auto-routed turn (including the menu-shortcut path, which the gate can never touch) publishes a `RoutingDecisionRecordV1` that `orion-sql-writer` persists to a new `routing_decision` table.

## Current architecture

- `services/orion-cortex-orch/app/decision_router.py` — `DecisionRouter.route()` is the only place `chat_reflective_lane_threshold` is read (`get_chat_reflective_lane_threshold()`) and the only place the demotion gate fires.
- The gate previously wrote its effect into `rewritten.options["routing_threshold_gate"]`, a dict nothing downstream read.
- `orion/substrate/mutation_detectors.py` maps zone `autonomy_graph` → surface `routing`, and the mutation pipeline's only patch for surface `routing` targets `chat_reflective_lane_threshold`. But nothing in that pipeline ever measured actual routing behavior — the "evidence" was graph-review telemetry, unrelated to what the dial controls. **That mismatch is not fixed by this PR.**

## Files changed

- `orion/schemas/routing_decision.py` — `RoutingDecisionRecordV1` (new)
- `orion/schemas/registry.py` — registered in **both** `_REGISTRY` and `SCHEMA_REGISTRY` (two separate registries in this repo; `resolve()` only reads `_REGISTRY` — see gotcha below)
- `orion/bus/channels.yaml` — new channel `orion:routing:decision`, producer `orion-cortex-orch`, consumer `orion-sql-writer`
- `services/orion-cortex-orch/app/decision_router.py` — emits the record; fire-and-forget with a 5s deadline; sanitizes `reason` to structured tags only
- `services/orion-sql-writer/app/models/routing_decision.py` — `RoutingDecisionSQL` (new)
- `services/orion-sql-writer/app/settings.py` — route-map entry **and** subscribe-channel entry (see gotcha below — these are two different lists and both are required)
- `services/orion-sql-writer/.env_example` + local `.env` — subscribe list updated, synced
- `orion/schemas/tests/test_routing_decision_contract.py`, `services/orion-cortex-orch/tests/test_routing_decision_emit.py`, `services/orion-sql-writer/tests/test_subscription_matches_declared_consumers.py` — new
- `config/metrics/metric_definitions.lock.json` — re-locked for the new bus-channel metric

## Schema / bus / API changes

- Added: `RoutingDecisionRecordV1` schema, `orion:routing:decision` channel, `routing_decision` SQL table.
- Removed / Renamed: none.
- Behavior changed: `DecisionRouter.route()` now does a fire-and-forget bus publish on every auto-routed turn (including the menu-shortcut early-return path).
- Compatibility notes: additive only.

## Env/config changes

- Added keys: none new env vars; `SQL_WRITER_SUBSCRIBE_CHANNELS` in `.env_example` gained the new channel.
- `.env_example` updated: yes.
- local `.env` synced: yes, by hand (see gotcha — `sync_local_env_from_example.py` writes to the **primary checkout**, not this worktree; re-copy the result in).
- skipped keys requiring operator action: none.

## Tests run

```
cd services/orion-cortex-orch && pytest tests/test_routing_decision_emit.py -q   → 5 passed
cd services/orion-sql-writer && pytest tests/test_subscription_matches_declared_consumers.py -q → 3 passed
pytest orion/schemas/tests/test_routing_decision_contract.py -q → 6 passed
python scripts/check_definition_drift.py --gate → exit 0 (post re-lock)
python scripts/check_metric_lineage.py --gate → exit 0
python scripts/check_inner_state_registry.py → exit 0
python scripts/check_control_surface_store_parity.py → exit 0
```

Full `orion-cortex-orch`/`orion-sql-writer` suites were **not** re-run end to end this session (both have large pre-existing unrelated failure counts on `main` — baseline this before trusting a raw pass/fail count).

## Evals run

None — no eval harness exists for either service's routing/bus-consumer behavior. Flagged as a gap, not fixed here.

## Docker/build/smoke checks

Not run this session. `orion-sql-writer` needs a restart to pick up the new subscribe channel; nothing has been deployed.

## Review findings fixed

- Finding: consumer never subscribed to the new channel → zero rows, forever, silently.
  - Fix: added to `sql_writer_subscribe_channels` in both `settings.py` and `.env_example`.
  - Evidence: `test_subscription_matches_declared_consumers.py`.
- Finding: publish was a blocking bus call inline in the hot chat-routing path — a slow/stalled Redis could add up to ~2 minutes to a turn.
  - Fix: fire-and-forget via `asyncio.create_task` with a 5s timeout guard; failures logged, never raised.
  - Evidence: `test_routing_decision_emit.py::test_a_failing_bus_does_not_break_the_turn`.
- Finding: `reason` field on the emitted record could carry verbatim free-text user content if the LLM router path populates it.
  - Fix: `_safe_routing_reason()` — allowlists a fixed charset (`a-z0-9_:+.-`), 120-char cap; anything else is dropped, not truncated-and-kept.
  - Evidence: `test_routing_decision_emit.py::test_free_text_reason_never_reaches_the_bus`.
- Finding: two separate registries (`_REGISTRY`, `SCHEMA_REGISTRY`) — registering in only one leaves `resolve()` unable to find the schema.
  - Fix: registered in both.
  - Evidence: `test_routing_decision_contract.py`.

## Restart required

```
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
```

Neither has been run. **Do not deploy without re-verifying against current main** — this branch was rebased once already mid-session as main moved fast (PRs #2063–#2070 landed while this was in flight).

## Risks / concerns

- Severity: medium — This is plumbing only. See "Outstanding / next adversarial pass" below: the loop this feeds is watching evidence unrelated to what this dial controls, and separately, the hardcoded patch value (`0.58`) is *below every heuristic confidence the router ever emits* (min observed: `0.82`), meaning the gate may be structurally unreachable at that value. Neither is fixed by this PR — do not treat "routing is now observable" as "routing self-modification now works end to end."
- Severity: low — full downstream test suites for both touched services were not re-run this session; only the new/targeted tests were confirmed green post-rebase.

## PR link

<PR link filled in below after creation>

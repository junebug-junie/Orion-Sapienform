## Summary

- Each dream sleep that starts replay is now one act in the act ledger: actor `dream.cycle`, verb `make` (body economy Patch A5, spec `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`, "Patch A results" producer split item 2).
- The act opens when the cycle passes its gates and has material to replay, and closes once: 1 for a saved, completed sleep with at least one hypothesis, 0 otherwise. Any exception or cancellation after the open (including during the open publish) closes it at 0 and re-raises.
- While the sleep runs, every recombination LLM call carries one correlation id derived from the cycle id. The GPU pool records it as each lease's `turn_correlation_id`, so the ledger's close-time cost fill can charge the cycle for its GPU time. Before this, each call had its own random id (live 10-16..23: 60 dream leases, 60 distinct ids), so no join was possible.
- Flag `DREAM_ACT_EMIT_ENABLED`, on by default. Off = no act, and each LLM call keeps its own id.
- `orion:grammar:event` lists `orion-dream` as a producer. The metric definition-drift lock is refreshed for that routing change.

## Outcome moved

The dream family has a graded act per sleep in `grammar_acts`, with wall seconds and (once live) GPU slot-seconds. The economy needs >= 30 graded closes per family before a cutover (spec Patch D).

Live rate (dream_cycle, last 7 days to 2026-10-23): 15 non-empty cycles (8 completed, 7 failed), about **2.1 acts/day**. Zero `empty` cycles in that window; empty cycles open no act.

## Current architecture

`run_cycle_once` (`services/orion-dream/app/cycle.py`) read pressure, gated, replayed, recombined over the LLM gateway, and persisted a `DreamCycleV1`. Dream *hops* (`thought.dream_hop`, orion-thought) already emitted acts; the cycle itself did not.

## Architecture touched

- orion-dream only, plus the `orion:grammar:event` producer list.
- The event is built in `services/orion-dream/app/act_emit.py`, not with `orion.grammar.acts.act_opened_event`: that helper's `_provenance` only accepts orion-hub and orion-thought, and the shared module was out of scope for this patch. `build_act` and `stable_act_id` are reused unchanged. The literal `GrammarProvenanceV1(source_service="orion-dream")` is what `tests/test_grammar_event_producer_catalog.py` scans.
- `orion/grammar/act_ledger.py` is untouched.

## Files changed

- `services/orion-dream/app/act_emit.py`: new. Act builder, open/close events, outcome rule, cycle correlation id, lease-correlation contextvar.
- `services/orion-dream/app/cycle.py`: `CycleDeps.publish_act`; `CycleActTrace`; open/close around replay → persist in `run_cycle_once`.
- `services/orion-dream/app/llm.py`: envelope correlation id = the cycle's id while a cycle runs; the reply channel stays unique per call.
- `services/orion-dream/app/main.py`: `_publish_act` wired behind the flag.
- `services/orion-dream/app/settings.py`, `.env_example`, `docker-compose.yml`: `DREAM_ACT_EMIT_ENABLED=true`.
- `services/orion-dream/README.md`: section "Every sleep is one act in the act ledger".
- `services/orion-dream/tests/test_dream_cycle_act.py`: new, 13 tests.
- `orion/bus/channels.yaml`: `orion-dream` in `orion:grammar:event` producer_services.
- `config/metrics/metric_definitions.lock.json`: re-locked (routing_changed for the producer list).

## Schema / bus / API changes

- Added: orion-dream publishes `grammar.event.v1` (`act_opened` / `act_closed`) on `orion:grammar:event`. Existing schema; no new fields.
- Behavior changed: during a sleep, recombination `llm.chat.request` envelopes share one correlation id (`uuid5(NAMESPACE_URL, "orion:dream.cycle:<cycle_id>")`).
- Compatibility: orion-sql-writer already persists acts from any producer. The RPC pending key is `(reply_channel, corr)`, and the reply channel is still unique per call.
- Side effect: orion-llm-gateway keys its assistant-reply embedding `doc_id` on the envelope correlation id. A cycle's recombination replies now upsert one embedding doc per cycle (the last reply) instead of one per call. Hub chat turns already share one id across calls the same way.

## Act contract

- Key: cycle id → `act_id = stable_act_id("dream.cycle", "make", cycle_id)`. Open once, close once (`CycleActTrace.closed` guard; the writer is also first-close-wins). A restart mid-cycle leaves the act open; the writer expires it at 0 at its horizon. The next sleep has a new cycle id.
- `expect`: `artifact_produced`, horizon = (DREAM_HYPOTHESES_PER_CYCLE + DREAM_CONTROL_PER_CYCLE) × DREAM_LLM_TIMEOUT_SEC + 120 s = 480 s by default. This is an upper bound: recombination is one sequential call per pair. Live 30 d wall time: p90 9.7 s, max 190 s (n=81).
- Outcome: 1 when the cycle is persisted, `completed`, and has ≥ 1 hypothesis (live 30 d: 3 of 81 completed with 0 hypotheses). The note carries the status, persisted flag, hypothesis count, llm_failures and no_link.
- Cost: `ActCostV1(seconds=<open→close wall time>)`. No ActCostV2-only fields.
- correlation_id = the id the cycle's GPU leases carry (see above).

## Env/config changes

- Added keys: `DREAM_ACT_EMIT_ENABLED` (default true).
- `.env_example` updated: yes.
- Local `.env` synced: `python scripts/sync_local_env_from_example.py --all-keys orion-dream` → `+DREAM_ACT_EMIT_ENABLED='true'` in the primary checkout's `services/orion-dream/.env`. The default (feature-keys-only) mode reported no change because this key is not on its list.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-dream: pytest tests -q -> 283 passed, 7 skipped
tests/test_grammar_event_producer_catalog.py -> 3 passed
make check-body-economy-contract -> OK (66 state-reading modules, all registered)
CI static gates run locally (from .github/workflows/orion-static-gates.yml): no_personal_terms, env sync/parity tests,
check_env_template_parity, check_service_env_compose_parity orion-dream, compose_logging_journald, ci_service_image_sources,
check_metric_lineage --gate, check_definition_drift --gate (after re-lock), inner_state_registry, body_economy_contract + test,
scripts_dir_no_stdlib_shadow, service_hostname_refs, compose_no_relative_mounts, journal_dispatch_registry,
daily_schedule_collisions, sentience_instruments --static-only, system_health_producers, async_routes_not_blocking,
chat_route_poachers -> all PASS
Mutation check: making the lease-correlation reset a no-op fails 2 tests.
```

## Evals run

```text
None. orion-dream has evals/, but none cover act emission. The real evaluation is live: graded closes per family in grammar_acts.
```

## Docker/build/smoke checks

```text
Not run (no image dependency change). Live smoke after deploy is listed under Risks.
```

## Review findings fixed

- Finding: the lease-correlation reset was untestable. The tests checked the contextvar after `asyncio.run`, which runs in a copied context.
  - Fix: both checks now run inside the cycle's own task.
  - Evidence: with the reset made a no-op, 2 tests fail.
- Finding: a cancel during `act.open()` left the act open until it expired.
  - Fix: the open and the bind moved inside the `try`. New test `test_cancel_while_the_open_is_published_still_closes`.
- Finding: shared correlation id → the gateway embedding `doc_id` overwrites per cycle.
  - Fix: documented in `act_emit.py` and in this report. Accepted: the cost join needs the shared id, and hub turns already behave this way.
- Finding: the horizon basis said "own deadline" but the DB reads have none.
  - Fix: the basis now says "upper bound" and names the slack.
- Finding: the `run_cycle_once` docstring said "Never raises".
  - Fix: the docstring now describes the raise path.
- Finding (documented, not fixed here): the cost fill runs once, at close. A lease released after that query is missed.
  - Fix: recorded as a gap (below). A deferred refill belongs in `act_ledger.py`, which another agent owns.

## Restart required

```bash
scripts/safe_docker_build.sh orion-dream up -d --build
```
(Run from the clean primary `main` checkout after merge.)

## Risks / concerns

- Severity: low. Concern: GPU cost join is UNVERIFIED live. The fill can miss a lease released after the close is processed. Mitigation: after the first sleep post-deploy, compare `grammar_acts.cost_json->'gpu_slot_s_by_lane'` with `sum(held_ms)` in `gpu_pool_events` for that `turn_correlation_id`.
- Severity: low. Concern: merge conflict with the other A5 producer PRs in `orion/bus/channels.yaml` and `metric_definitions.lock.json`. Mitigation: the producer entries sit at different list positions; re-run `check_definition_drift.py --update` after rebasing.
- Live verification: UNVERIFIED until deploy. Query: `SELECT act_id, opened_at, closed_at, outcome_value, cost_json FROM grammar_acts WHERE actor='dream.cycle' ORDER BY opened_at DESC LIMIT 5;`

## PR link

(filled on open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

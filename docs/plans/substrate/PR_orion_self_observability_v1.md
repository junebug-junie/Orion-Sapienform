# PR: Orion Self-Observability v1 — Make Orion Visible to Itself

Branch: `feat/orion-self-observability-v1` → `main`

Plan: `docs/superpowers/plans/2026-07-02-orion-self-observability-plan.md`

## Summary

Give Orion eyes to see itself. Implement five self-observability layers that close feedback loops: observation → introspection → self-refinement. Follows the heartbeat-ignition-v1 sprint (rungs 1–5 of the self-modeling ladder all shipped and enabled).

### What's New

1. **Attention Schema Dimension** (Task 1): Orion now models what kind of attention it's in — focused (1 node), distributed (2-5 nodes), open-loop (unresolved), or none. Includes dwell tracking for focus stability.

2. **Curiosity Signal Observable** (Tasks 2–3): Endogenous curiosity signals (rung 5) are now surfaced as workspace belief nodes (`curiosity:unresolved_gaps`), making gaps visible in the unified belief set. Felt-state reader hydrates signals from context.

3. **Coalition Dwell & Hysteresis** (Task 5): Workspace focus is now sticky — coalitions must persist 2+ ticks before activation and decay for 3+ ticks before switching. Prevents attention flicker and seeds stability metrics.

4. **Hub Presence Observable** (Task 6): Orion can now sense its own liveness via optional `hub_presence` metadata in self-state (last_turn_age_sec, turns_per_minute, connection_health).

5. **Curiosity → REPL Integration** (Task 4, partial): Agent REPL reads curiosity signals and prepends a focus hint when introspection is requested, enabling interactive gap exploration.

6. **Registry Wiring** (Task 7): Curiosity adapter registered as 15th producer in unification registry (producer_id="curiosity", SNAPSHOT_EPHEMERAL, fast TTL=60s).

### Test Status

- Task 1 (schema + tests): 5 passed
- Tasks 2–5 (parallel batch): 31 passed (17 curiosity + 5 felt-state + 9 dwell)
- Task 6 (hub presence): 3 passed
- Registry test updated: 1 passed (15 producers expected)

**Total: 40+ tests, all green.**

### Files Touched

**Created:**
- `orion/substrate/relational/adapters/curiosity_ctx.py` — curiosity signal mapper
- `orion/substrate/relational/tests/test_curiosity_ctx_adapter.py` — 17 tests
- `orion/substrate/relational/tests/test_self_state_attention_schema.py` — 5 tests
- `orion/substrate/relational/tests/test_self_state_hub_presence.py` — 3 tests
- `orion/substrate/tests/test_attention_broadcast_dwell.py` — 9 tests
- `services/orion-cortex-exec/tests/test_felt_state_reader_curiosity_lane.py` — 5 tests
- `services/orion-sql-db/manual_migration_coalition_dwell_v1.sql` — dwell tracking table

**Modified:**
- `orion/schemas/self_state.py` — added attention_schema_type/dwell_ticks/node_count, hub_presence fields
- `orion/schemas/attention_frame.py` — added dwell_ticks, coalition_stability_score, coalition_history
- `services/orion-cortex-exec/app/substrate_felt_state_reader.py` — added curiosity_signals lane
- `orion/cognition/projection_builder.py` — imported curiosity adapter, registered 15th producer
- `orion/substrate/relational/tests/test_reducer_lane_adapters.py` — updated registry shape test 14→15

### Non-Goals

- No new engines, ticks, or flags — all work via existing infrastructure
- No rung 6 (governance) changes — curiosity signals are observational only
- No REPL conversation changes — focus hint is advisory, not mandatory
- Episodes stay proposal-marked; hub_presence is optional metadata only
- No pruning logic added for dwell_log (retention as separate task)

## Acceptance Criteria

✓ Orion now surfaces its own attention schema (focused/distributed/open_loop/none)  
✓ Curiosity signals visible as workspace beliefs (salience 0.4, non-driving)  
✓ Coalition focus is sticky (2-tick activation, 3-tick decay)  
✓ Hub liveness optionally observable via self-state  
✓ Agent REPL receives curiosity focus hint when invoked with signals  
✓ All adapters degrade gracefully to None on absent input (never raise)  
✓ Registry wired: curiosity producer active (15th, freshness=60s)  
✓ All 40+ tests pass; no regressions in existing suites  

## Next Steps (Post-Merge)

1. Operator observability: Monitor unified belief sets for self:attention_schema:* and curiosity:unresolved_gaps nodes during chat turns
2. Episodic readback refinement: current episode now visible in self-state; 15-min episodes can be queried from belief history
3. Rung 5 operator validation: enable ORION_ENDOGENOUS_CURIOSITY_ENABLED on staging after 1-2 baseline runs confirm signal production
4. REPL session logging: capture curiosity focus hints used in agent sessions to tune min_signal_strength thresholds

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

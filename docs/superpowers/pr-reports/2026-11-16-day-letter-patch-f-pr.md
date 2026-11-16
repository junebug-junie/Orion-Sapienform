## Summary

Patch F of the day-letter fix plan, approved by Juniper ("go with patch f"). Refs #3393 #3395 #3396.

- The curiosity tick no longer passes a hardcoded "all calm" report. The four report-driven evaluator branches, the goal-pressure branch in consolidation, the `orion/graph_cognition` package and the unused `graph_view_basis` planner are deleted. Three signal types leave the contract.
- Orion's taught worldview writes are safe to repeat: findings `MERGE` on `finding_id`, a belief move `MERGE`s on the whole move, and `times_tested` moves once per run (guarded by a new `last_counted_run_id`). The correction lane and consult texts point at the same writes.
- One rule for "Orion's current answer": `newest_answer_per_question(require_evidence=)`. Reading resonance now offers only the current answer per question.
- `scripts/worldview_prior_unique_rule.py` tests the index + unique rule on a scratch FalkorDB and prints the operator command. The live rule is NOT applied.
- The three proposals now carry answers to Orion's 16 questions (reply turn `e24df41cfeba`, thread `01ae43c8a48b`).

## Outcome moved

- Reading resonance: was 24 of 32 lived-answer candidates already superseded (11-11..11-14). After deploy every one should be current (`scripts/analysis/check_resonance_current_answers.py --since <deploy date>`).
- A replayed write batch no longer doubles findings, revisions or the test count (scratch replay: same counts after the second batch).
- Dead code that could never fire is gone (1,393 lines).

## Current architecture

The tick built `FrontierCuriosityEvaluator` with `_neutral_frontier_metacog_inputs()`; four branches read that constant report. The kickoff prompt taught `CREATE` for findings and revisions and an unguarded `times_tested + 1`. The resonance pool offered every lived answer, old and new.

## Architecture touched

`orion/substrate` (frontier_curiosity, consolidation, query_planning), `orion/core/schemas/frontier_curiosity.py`, substrate-runtime worker, `orion/curiosity` (kickoff_prompt, self_inquiry, value comment), `orion/orion_day/corrections.py`, `orion/consult/beliefs.py`, `orion/world_pulse_read/resonance_records.py`, two scripts.

## Files changed

- `services/orion-substrate-runtime/app/worker.py`: stub removed.
- `orion/substrate/frontier_curiosity.py`, `consolidation.py`, `query_planning.py`: dead branches and arguments removed; delete-site comment names the `DURABLE_NODE_KINDS` whitelist.
- `orion/graph_cognition/`: deleted (no live importer).
- `orion/core/schemas/frontier_curiosity.py`: three signal types retired.
- `orion/curiosity/kickoff_prompt.py`, `orion/orion_day/corrections.py`, `orion/consult/beliefs.py`, `orion/autonomy/ask_claude_trigger.py` (docstring): repeat-safe writes.
- `orion/curiosity/self_inquiry.py`, `orion/world_pulse_read/resonance_records.py`: current-answer helper and its use.
- `scripts/worldview_prior_unique_rule.py`, `scripts/analysis/check_resonance_current_answers.py`: new.
- Tests: phase8/9/15/16 substrate tests, tick test (binds real signature), ctx adapter, worldview prompt tests, new `test_current_lived_answer.py`, `test_worldview_prior_unique_rule.py`; deleted `test_cognitive_substrate_phase5_graph_cognition.py`.
- `docs/superpowers/specs/2026-11-15-patch-f-*.md`: answers to Orion's questions.

## Schema / bus / API changes

- Removed: `concept_instability`, `contradiction_hotspot`, `unresolved_pressure_region` from `FrontierInvocationSignalTypeV1`. None stored in 30 days (all 16,109 candidates are `curiosity_candidate`); readers of old rows already skip invalid ones.
- Removed: `cognition_report` / `perception_brief` arguments of `FrontierCuriosityEvaluator.evaluate`, `FrontierCuriosityOrchestrator.run`, `GraphConsolidationEvaluator.consolidate`.
- Behavior changed: `times_tested` counts runs, not commands (6.3% of per-run tests moved it by more than 1). New graph property `Prior.last_counted_run_id`.

## Env/config changes

None. No `.env_example` changed.

## Tests run

```text
focused suites (substrate, curiosity, world_pulse_read, orion_day, consult, autonomy, worldview, tick):
2296 passed, 1 failed, 78 skipped, 1 xfailed  -- the 1 failure also fails on origin/main
(orion/substrate/tests/test_felt_state_lived_answers.py::test_hydrate_lived_answers_latest_per_pinned_concept_id)
post-review rerun: 605 passed
pre-existing on main, not touched: tests/test_cognitive_substrate_phase9_consolidation.py::test_outcomes_cover_reinforce_requeue_damp_retire_and_priority
check_integration_seam OK; check_metric_dead_wiring rc=0; check_metric_lineage rc=0; git diff --check clean
```

## Evals run

```text
python3 scripts/worldview_prior_unique_rule.py scratch   -> RESULT: PASS (scratch container, live image, graph module 41811)
  taught batch x2: Prior:2 PriorRevision:2 Finding:1 edges:1, times_tested 2 -> 3 once (both runs identical)
  rule PENDING -> OPERATIONAL with an id-less Prior present
  duplicate CREATE -> "unique constraint violation on node of type Prior"; second id-less CREATE allowed
python3 scripts/worldview_prior_unique_rule.py preflight (read-only, live) -> 0 duplicate ids, 1 id-less prior (node 3208), no constraints
python3 scripts/analysis/check_resonance_current_answers.py --since 2026-11-01 -> candidates=32 not_current=24 cited=3 cited_not_current=2 (pre-deploy baseline)
```

## Docker/build/smoke checks

```text
No image build run. Tick behavior is covered by the worker tick test binding real kwargs; live check after deploy:
no substrate_endogenous_curiosity_tick_failed in substrate-runtime logs for 24 h.
```

## Review findings fixed

- Finding: two ctx-adapter tests still built signals with retired types.
  - Fix: switched to live types. Evidence: `test_curiosity_ctx_adapter.py` passes.
- Finding: moving `last_run_id` into the guarded count command broke run attribution in value.py/corrections.py when a run skipped the count.
  - Fix: guard on new `last_counted_run_id`; the update keeps `last_run_id`. Evidence: prompt test + scratch replay PASS.
- Finding: `previous_answer` neighbour followed FalkorDB's null-first DESC order.
  - Fix: lists re-sorted with the helper's key; test with a null-dated row.
- Finding: check script's evidence test disagreed with `_evidence_list` on `"[]"`, dropped float `written_at`.
  - Fix: Cypher CASE treats `[]`, `''`, `'[]'` as none; `int(float())`. Evidence: baseline reproduces 32/24/3/2.

## Restart required

```bash
# after merge, from the clean primary main checkout (post-pull rebuild does this):
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Live unique rule, ONLY after Juniper approves (run preflight first):

```bash
docker exec orion-athena-falkordb redis-cli GRAPH.QUERY orion_worldview "CREATE INDEX FOR (p:Prior) ON (p.prior_id)"
docker exec orion-athena-falkordb redis-cli GRAPH.CONSTRAINT CREATE orion_worldview UNIQUE NODE Prior PROPERTIES 1 prior_id
docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_worldview "CALL db.constraints()"   # expect OPERATIONAL
```

## Risks / concerns

- Severity: low. Concern: Orion skips the second (count) command. Mitigation: value.py still reports it as moved-untested; corrections still attribute the move.
- Severity: low. Concern: an A->B, B->A, A->B sequence in one run folds the third move into the first. Mitigation: rare; documented.
- Severity: info. Live unique rule not applied; waits for Juniper. Issues opened: none. Issues closed: none (#3393 stays open for the self-node seed symptom).

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

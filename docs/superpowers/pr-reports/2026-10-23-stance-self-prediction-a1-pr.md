## Summary

- Stance now writes a private guess about how Orion will behave on the turn: will I ask a question, will I fetch an outside source, how many steps, how long a reply. Each guess carries a stated confidence. This is `ThoughtEventV1.self_prediction` (`TurnSelfPredictionV1`).
- The parser cleans the guess and never invents a value. Probabilities are clamped to 0.02–0.98. Out-of-range values, inverted ranges, half ranges and huge numbers are dropped. A malformed guess can never fail or defer the turn.
- The model that writes the reply and the finalize judge never see the guess. Tests enforce this on the motor prompt, the motor instruction, the full harness prompt and the judge context.
- The guess is stored in `thought_decision.self_prediction`, through an additive migration. The self-monitor organ (#2827) settles it later against `harness_turn_trace` (A3).
- `STANCE_SELF_PREDICTION_ENABLED=true` in orion-thought. Set to false, the stance prompt is byte-identical to before.

Design: PR #2869, `docs/superpowers/specs/2026-10-23-unified-turn-self-prediction-design.md` (patch A1).

## Outcome moved

Orion starts recording "what I expected to do" on every unified turn, in a form that code can check. Before this, stance wrote a plan nobody ever compared with what happened.

## Current architecture

- Stance (`orion/cognition/prompts/stance_react.j2`, one model call on the chat lane for Juniper's turns) wrote an imperative, a tone and a task slice.
- The imperative went to the motor as guidance and to the judge as a grading standard.
- Nothing recorded a checkable expectation.
- Mind's `expected_depth` (shallow/deep) is the only earlier self-prediction. It has no confidence, it is shown to the motor, and it defaults to "deep".

## Architecture touched

- `orion/schemas/thought.py`: new `TurnSelfPredictionV1`, plus optional `self_prediction` on `ThoughtEventV1` and `ThoughtDecisionRecordV1`. Registered in both registries; `resolve("TurnSelfPredictionV1")` verified.
- orion-thought: a flag sets the `self_prediction_enabled` key in the stance context.
- cortex-exec renders the prompt. No code change there.
- `orion/harness/finalize.py`: the judge's thought dump excludes `self_prediction`.
- orion-sql-writer: new column.

## Files changed

- `orion/schemas/thought.py`, `orion/schemas/registry.py`: the schema and its registration.
- `orion/cognition/prompts/stance_react.j2`: the flag-gated `self_prediction` block.
- `orion/thought/stance_react.py`: `normalize_self_prediction`, which drops and never fills; any error drops the bet.
- `orion/harness/finalize.py`: keeps the bet out of the judge.
- `services/orion-thought/app/{settings.py,bus_listener.py}`, `.env_example`: the flag and the context key.
- `services/orion-sql-writer/app/models/thought_decision.py`: the `self_prediction` jsonb column.
- `services/orion-sql-db/manual_migration_thought_decision_self_prediction_v1.sql`: additive, `REQUIRED-BY: orion-sql-writer`.
- Tests:
  - `orion/thought/tests/test_stance_self_prediction.py` (parser);
  - `orion/harness/tests/test_self_prediction_never_reaches_motor.py` (no leak);
  - `services/orion-thought/tests/test_stance_self_prediction_prompt.py` (prompt and flag);
  - `services/orion-sql-writer/tests/test_thought_decision_sql_shape.py` (persists the bet, and null);
  - `orion/thought/tests/test_stance_react_prompt_imperative_discipline.py` (the strict-render variable list).

## Schema / bus / API changes

- Added: `TurnSelfPredictionV1` (`turn.self_prediction.v1`), `ThoughtEventV1.self_prediction`, `ThoughtDecisionRecordV1.self_prediction`, and the `thought_decision.self_prediction` column.
- Removed / Renamed: none.
- Behavior changed: the stance LLM is asked for one more field.
- Compatibility notes:
  - Every consumer of `orion:thought:artifact` ignores extra fields (the review checked them), so any deploy order is safe for readers.
  - sql-writer needs the migration first. The post-pull applier runs it, because it is additive DDL with a REQUIRED-BY header.

## Env/config changes

- Added keys: `STANCE_SELF_PREDICTION_ENABLED=true` (orion-thought).
- Removed / Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced: the key is in the main checkout's `services/orion-thought/.env`. `scripts/sync_local_env_from_example.py orion-thought` was run.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/thought/tests + orion/harness/tests/test_self_prediction_never_reaches_motor.py: 107 passed
orion/harness/tests + orion/schemas/tests: passed (793 + new)
services/orion-thought/tests: 768 passed, 32 skipped
services/orion-sql-writer/tests: 14 failed, all also failing on clean origin/main (identical list, verified in a detached origin/main worktree); new tests pass
static gates: check_migration_required_by OK, check_env_template_parity PASS,
  check_sentience_instruments --static-only OK, check_inner_state_registry OK,
  check_body_economy_contract OK, tests/test_agent_trace_schema_registry.py + test_check_settings_defaults.py 26 passed,
  git diff --check clean
```

## Evals run

```text
No eval yet: A1 only records guesses. Acceptance is live (below). The calibration eval arrives with A3, when the organ grades them.
Baseline (7 d, chat-lane hold of the stance call, from gpu_pool_events):
  Juniper chat turns: p50 11.9 s, p90 13.9 s (n=33)
  Orion-started turns: chat p50 13.8 s (n=106); agent p50 95.1 s (n=296); agent-gpu2 p50 120.0 s (n=105)
```

## Docker/build/smoke checks

```text
Not run in the worktree (deploys go from main). After merge:
1. The migration is auto-applied by the post-pull applier (additive, REQUIRED-BY header).
2. Rebuild orion-thought and orion-sql-writer. Harness-governor (finalize.py) and cortex-exec (the prompt file) use the shared orion/ package, so rebuild them if the post-pull diff does not.
Live acceptance after deploy:
- self_prediction is non-null on 90% or more of new thought_decision rows;
- the stance chat-lane hold p50 on Juniper's turns grows by 3 s or less (baseline 11.9 s);
- agent-lane stance holds are measured too (the Qwen thinking caveat).
```

## Review findings fixed

- Finding: a huge JSON integer in the guess (`float()` raises `OverflowError`) failed the whole stance turn.
  - Fix: catch `OverflowError`, and wrap the guess cleanup so any error drops the guess with a warning.
  - Evidence: `test_huge_json_int_is_dropped_not_raised` and `test_any_bet_error_still_parses_the_turn` pass.
- Finding: the prompt said "nobody is shown it", but Juniper's cockpit shows the full thought.
  - Fix: the wording now says the model that writes the reply never sees it.
  - Evidence: the prompt test asserts the new wording.
- Not changed (informational): the Hub turn-trace route's explicit column list does not select `self_prediction`. Nothing breaks. Showing the guess there is a possible follow-up.
- Review verified there is no leak path to the motor, the judge, repair, tool retry, the governor, the role-teach splice or the introspect tools. The prepared-context path keeps the flag. With the flag off, the prompt is byte-identical under cortex-exec's renderer.

## Restart required

```bash
# after merge, from the clean primary main checkout (post-pull rebuild normally does this):
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: medium.
  - Concern: the Qwen thinking caveat. On the agent lanes (Orion's own runs), the extra fields may lengthen thinking.
  - Mitigation: measure agent-lane stance holds after deploy, and decide then (Juniper, 10-23).
- Severity: medium.
  - Concern: the guesses may be degenerate (flat p, huge ranges).
  - Mitigation: the metric gate re-runs on 7 days of live guesses before A3 reads them.
- Severity: low.
  - Concern: about 1–2 s more stance time on Juniper's turns.
  - Mitigation: B (fold stance into Mind) removes the ~12 s stance call.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)

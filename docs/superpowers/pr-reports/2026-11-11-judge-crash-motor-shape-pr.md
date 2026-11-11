## Summary

- The finalize 5b judge grades each draft reply. It failed on 74 of 1,689 judged turns (4.4%) in the 7 days to 2026-11-11. When it fails, the turn gets a fake "misaligned" verdict marked `degraded_llm_failure_fallback`.
- 46 of the 74 failures were the model not writing a judgement at all. Bonsai (agent-gpu2) thought until it ran out of tokens, or misspelled keys (`imperial`, `imprative`). Qwen3.8-Flash-Next copied the reading draft's own keys (`what_i_learned`, `summary`). The other 28 were transport (GPU pool hold not granted, timeouts). This PR does not touch transport.
- Root cause: the judge call had no output schema, no thinking limit, and no turn that asked for a grade. cortex-exec sent the whole prompt as one system message, with the draft in the middle and a recall digest of Orion's past replies at the end. The model read the draft and the request as its task.
- Fix: the judge call now uses the gateway's existing structured output (a grammar on llama.cpp, validate-and-refuse on Strata) with the judgement fields only. Thinking is off, as for `stance_react`. The draft and the request are quoted as data. A user turn asks for the grade. Recall is off for this verb.
- A failed judge is still marked as failed. The harness now also owns the provenance fields, so a model cannot label its own output a fallback or a quick-gate verdict.

## Outcome moved

Failure mode: `reflect_llm_failed` on the 5b judge.

Replay (`orion/harness/evals/judge_crash_replay_eval.py`, live lanes, PARTIAL at time of writing):
- new request on the real failed turns: 0 failures of 34 replayed (all 3 lanes; Bonsai, chat and agent).
- old request (with its stored recall digest) on the Strata failed turns: 3 failures of 9 (motor-shape copies reproduced).
- Control (verdict agreement with stored) and perception-claim buckets: still running; UNVERIFIED in this report.

## Current architecture

Hub unified turn -> harness governor -> `run_finalize_reflection` (orion/harness/finalize.py) -> cortex-exec verb `harness_finalize_reflect` -> llm-gateway -> the lane that owns the turn (chat or agent on Strata Qwen3.8-Flash-Next, or agent-gpu2 on llama.cpp Ternary Bonsai 2 27B).

Before this PR, cortex-exec built the call like this (`executor.py` `_build_hop_messages`, `_append_memory_digest`):

- one system message: the rendered prompt, then `# RELEVANT MEMORY` (recall ran on 1,029 of 1,369 traced judge calls), then a guardrail line;
- no user message;
- no `response_format`; `structured_output_requested: False` on all 1,368 traced calls;
- the lane's thinking default: on for agent and agent-gpu2 (median 2,818 / 3,884 completion tokens), off on chat;
- `max_tokens` 8,000.

### Failure mix (live Postgres, `harness_turn_trace` + `cognition_traces`)

7 days (2026-11-03..11): 1,689 judged turns, 74 failed (4.4%). Since A2 (#3097, 2026-11-04 19:00Z): 1,314 turns, 51 failed (3.9%).

| class | 7 days | since A2 | where |
|---|---|---|---|
| unparseable: the reasoning text shipped as the answer ("We need answer user's request...") | 23 | 15 | Bonsai; 12 of the traced ones hit `finish_reason=length` at 8,000 tokens with empty content |
| validation: missing fields or misspelled keys (`imperial`, `imprative`, `impressive`, `imperitive`) | 14 | 12 | Bonsai |
| validation: motor-shape copy (`what_i_learned` 4, `summary` 1) | 5 | 3 | Flash-Next: 4 on chat (thinking off), 1 on agent |
| unparseable: broken JSON (`{` only; cut strings) | 4 | 3 | Flash-Next |
| gpu_pool_unavailable | 21 | 13 | transport |
| timeout (`caller_budget_exhausted`, RPC timeout) | 7 | 5 | transport |

Per lane (traced calls): agent-gpu2 Bonsai 25 failed of 316 (7.9%); chat 6 of 195 (3.1%); agent 1 of 861 (0.1%). 42 failures have no judge trace (transport, or the step failed before a trace was written).

Callers of the traced failures: reading 9, curiosity digest items 10, other turns (chat, self-study probes) 13. Reading drafts are JSON with `what_i_learned`; that is the shape Flash-Next copied.

## Architecture touched

- `orion/harness/finalize.py`: the judge context (schema, thinking off, grade turn, `raw_user_text`), and provenance fields set after parse.
- `orion/cognition/prompts/harness_finalize_reflect.j2`: the draft and the request are quoted in tags as data; the required-fields list matches the schema, notes before verdict.
- `orion/cognition/verbs/harness_finalize_reflect.yaml`: `recall_enabled: false`.
- No change to cortex-exec or the gateway: both already forward and apply `structured_output_*` and `chat_template_kwargs`.

## Files changed

- `orion/harness/finalize.py`: `finalize_reflect_json_schema()`, `FINALIZE_REFLECT_USER_TURN`, context keys, harness-owned provenance.
- `orion/cognition/prompts/harness_finalize_reflect.j2`: grade-not-write framing, quoted inputs, field list.
- `orion/cognition/verbs/harness_finalize_reflect.yaml`: recall off, with the evidence.
- `orion/harness/tests/test_finalize_reflect_judge_contract.py`: schema, context, rendered prompt, motor-shape regression, provenance.
- `services/orion-cortex-exec/tests/test_finalize_reflect_request_shape.py`: the request reaches the gateway as `[system, user]` with the schema and thinking off; recall off.
- `orion/harness/evals/judge_crash_replay_eval.py`: live replay of the failed turns, old vs new request.
- `orion/harness/evals/test_judge_crash_replay_eval.py`: the eval's pure parts.
- `orion/harness/evals/fixtures/judge_crash_cases.json`: correlation ids and labels only (74 failed, 21 control).
- `.github/workflows/orion-gpu-pool-tests.yml`: runs the new tests in the consumers job (next to the stance thinking-off tests); adds `jsonschema`.
- `docs/superpowers/pr-reports/2026-11-11-judge-crash-motor-shape-pr.md`: this report.

## Schema / bus / API changes

- Added: none. `FinalizeReflectionV1` is unchanged.
- Removed: none.
- Renamed: none.
- Behavior changed: the judge call sends `structured_output_schema` (9 judgement fields, `additionalProperties: false`), `structured_output_method=json_object_schema`, `enable_thinking=false`, and a user message. The harness sets `schema_version`, `reflection_source`, `quick_lane_skipped_llm`, `finalize_changed` after parse.
- Compatibility notes: existing optional gateway options. Strata refuses an answer that breaks the schema (HTTP 502); that reaches the harness as a gateway failure and is marked failed, as before.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Tests run

```text
orion/harness/tests + orion/cognition/prompts/tests + orion/harness/evals: 774 passed
cortex-exec relevant suites (38 files): 354 passed, 3 failed (same 3 fail on main: route mapping, memory_graph budget, story_weave)
new: test_finalize_reflect_judge_contract.py 6 passed; test_finalize_reflect_request_shape.py 3 passed; test_judge_crash_replay_eval.py 4 passed
orion-static-gates.yml steps run locally: all pass
```

## Evals run

See "Outcome moved". Rerun: `python -m orion.harness.evals.judge_crash_replay_eval` (opt-in, live lanes).

## Docker/build/smoke checks

```text
No image build in this PR's checks. The change is Python + a prompt + a verb file, all read by
orion-harness-governor (it builds the plan with build_plan_for_verb). Live effect UNVERIFIED until deploy.
```

## Review findings fixed

- Finding: eval skipped the harness error-text check. Fix: classify() applies looks_like_error_text. Evidence: test added.
- Finding: thinking off may weaken the perception-claim check. Fix: new "perception" fixture bucket (16 stored catches) and a still-flags metric. Evidence: replay running, UNVERIFIED.
- Finding: motor-shape test also passes on main. Fix: documented as a failure-path regression guard; the context/seam tests fail on main.
- Declined: maxLength caps in the schema. Strata validates strictly, so caps would turn long-but-valid answers into 502 refusals.
- Not done: router-level "no RecallService RPC" test (metadata test only).

## Restart required

From the clean primary `main` checkout after merge (the post-pull rebuild normally does this):

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

### Post-deploy re-measure

Run 3 days after the deploy. Replace `<DEPLOY_UTC>` with the governor's start time (`docker inspect -f '{{.State.StartedAt}}' orion-harness-governor`).

```sql
-- 1. Failure rate and mix after the deploy (compare: 3.9% since A2, 33 of 51 non-transport).
SELECT count(*) AS judged_turns,
       count(*) FILTER (WHERE run_artifact->'reflection'->'alignment_notes'->>0 LIKE 'reflect_llm_failed:%') AS failed,
       count(*) FILTER (WHERE run_artifact->'reflection'->'alignment_notes'->>0 LIKE 'reflect_llm_failed:%gpu_pool_unavailable%'
                           OR run_artifact->'reflection'->'alignment_notes'->>0 ~* 'timeout') AS transport,
       count(*) FILTER (WHERE run_artifact->'reflection'->'alignment_notes'->>0 LIKE 'reflect_llm_failed:%validation error%') AS validation,
       count(*) FILTER (WHERE run_artifact->'reflection'->'alignment_notes'->>0 LIKE 'reflect_llm_failed:%Could not parse JSON%') AS unparseable,
       count(*) FILTER (WHERE run_artifact->'reflection'->'alignment_notes'->>0 LIKE 'reflect_llm_failed:%502%') AS strata_refused
FROM harness_turn_trace
WHERE created_at > '<DEPLOY_UTC>' AND run_artifact ? 'reflection';

-- 2. The new request shape is live: structured output requested, thinking off, no recall step.
SELECT (s->'result'->'LLMGatewayService'->>'model_used') AS model,
       count(*) AS calls,
       count(*) FILTER (WHERE s->'result'->'LLMGatewayService'->>'meta' LIKE '%''structured_output_requested'': True%') AS schema_on,
       count(*) FILTER (WHERE s->'result'->'LLMGatewayService'->>'meta' LIKE '%''enable_thinking'': False%') AS thinking_off
FROM cognition_traces t, jsonb_array_elements(t.steps::jsonb) s
WHERE t.verb = 'harness_finalize_reflect' AND t.timestamp > '<DEPLOY_UTC>'
  AND s->'result' ? 'LLMGatewayService'
GROUP BY 1;
SELECT count(*) AS judge_traces_with_recall
FROM cognition_traces t, jsonb_array_elements(t.steps::jsonb) s
WHERE t.verb = 'harness_finalize_reflect' AND t.timestamp > '<DEPLOY_UTC>' AND s->'result' ? 'RecallService';

-- 3. Verdict drift (before, since A2: aligned 82.7%, misaligned 10.9%, uncertain 6.4% of 1,182).
SELECT run_artifact->'reflection'->>'alignment_verdict' AS verdict, count(*),
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
FROM harness_turn_trace
WHERE created_at > '<DEPLOY_UTC>'
  AND run_artifact->'reflection'->>'reflection_source' = 'substrate_informed_pass'
GROUP BY 1 ORDER BY 2 DESC;
```

Run with `docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "<query>"`.

## Risks / concerns

- Severity: medium. Concern: thinking off may lower judge quality, mainly the perception-claim check (65 of 86 recent catches came from the thinking-on agent lane). Mitigation: perception bucket in the replay; post-deploy verdict-drift SQL.
- Severity: low. Concern: Strata does not constrain decoding; a non-schema answer becomes a 502 (still a labelled failure). Mitigation: framing + grade turn; watch strata_refused in the re-measure.
- Severity: low. Concern: transport failures (28 of 74) untouched.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)

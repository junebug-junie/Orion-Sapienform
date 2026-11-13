# Held Mind runs: the frontier judge no longer thinks its whole budget away

Fixes #3294

## Summary

- Root cause: the frontier judge was the only Mind phase that ran with thinking on. On the held seats the model spent the whole 3072-token budget reasoning:
  - Bonsai 27B (`agent-gpu2`): ~9 KB of reasoning, then the JSON was cut off at 3072 tokens (`json_parse_failed`), or the reasoning used every token and the reply was empty (`empty_llm_response`).
  - Strata flash-next (`agent`): it thought to its 2304-token thinking budget, the answer was cut at 3072, and Strata refused the incomplete structured output with HTTP 502 `structured output was incomplete (finish_reason=length)`. Mind saw `empty_llm_response` from model `Active-GGUF-Model`. That name is the gateway's default label (`ORION_DEFAULT_LLM_MODEL`) when no model answered.
- Fix (request side): a held run now sends the judge `enable_thinking=false`, like the semantic and stance phases. New key `MIND_LLM_THINKING_APPRAISAL_HELD` (default false; true = rollback). Unheld runs are unchanged.
- Fix (parse side, same approach as #3283): the judge prompt lists the allowed `matter_kind` / `recommended_effect` values; a claim kind written as `matter_kind` maps like a claim kind instead of becoming `turn_anchor`; replaced values stay in `metadata._orion_off_vocab_fields`. A malformed `deferred` / `suppressed` entry is dropped and counted instead of failing the reply. Object-shaped hazards keep their text.
- Telemetry now names the cause: `empty_llm_response:upstream_http_5xx:http_502`, and `frontier_schema_invalid:<field>`.
- No hollow fallback: an empty or cut-off reply still fails, and so does a matter without a label, summary, or claim id, or an out-of-range score.

## Outcome moved

Held Mind runs (world pulse, curiosity, journal, field notes) can finish the judge and the stance phase on both seats. Gate: captured judge requests from live held runs, replayed on the same seat under an operator hold.

| reply | seat | thinking | tokens | result |
|---|---|---|---|---|
| live held run b956b9cd | agent | on | cut at 3072 | `empty_llm_response` (Strata 502) |
| live held run 6f061ded | agent | on | cut at 3072 | `empty_llm_response` (Strata 502) |
| live held run 97383a30 | agent-gpu2 | on | 3072 (9,114 chars reasoning) | `json_parse_failed` |
| replay b956b9cd | agent | off | 753, 14 s | `ok`, 4 matters |
| replay 6f061ded | agent | off | 357, 9 s | `ok`, 2 matters |
| replay 97383a30 | agent-gpu2 | off | 846, 23 s | `ok`, 3 matters |
| stance after replay b956b9cd | agent | off | 411, 11 s | `ok`, stance brief valid |
| stance after replay 97383a30 | agent-gpu2 | off | 298, 13 s | `ok`, stance brief valid |

Replies with thinking on that did fit (2 Strata replays, 2724 and 2872 tokens) pass on both origin/main and this branch.

## Current architecture

`services/orion-mind/app/engine.py` runs three LLM phases: semantic synthesis, the active frontier judge (`app/appraisal.py`), and the stance handoff. Since #3238 a held run sends every phase to route `agent` with the turn's GPU hold. Semantic and stance send `enable_thinking=false`. The judge used `MIND_LLM_THINKING_APPRAISAL=true`, which leaves the model's own default (thinking) on. On the metacog lane that costs ~190 tokens. On the held seats it costs the whole budget. `guardrails.py:243` sets `invalid_handoff` when no frontier is selected.

## Architecture touched

- `services/orion-mind` only. No bus, schema, or route change.

## Files changed

- `services/orion-mind/app/engine.py`: held runs send the judge `MIND_LLM_THINKING_APPRAISAL_HELD`.
- `services/orion-mind/app/settings.py`, `.env_example`, `docker-compose.yml`: new key `MIND_LLM_THINKING_APPRAISAL_HELD=false`.
- `services/orion-mind/app/appraisal.py`: allowed values in the prompt; claim-kind mapping and effect aliases for matters, with a provenance record; side entries checked one by one; object hazards; telemetry error detail.
- `services/orion-mind/app/synthesis.py`: public name `RECOMMENDED_EFFECT_ALIASES` for the alias table appraisal reuses.
- `services/orion-mind/tests/test_held_frontier_judge.py`: regression tests on the captured replies, plus the cases that must still fail.
- `services/orion-mind/tests/fixtures/held_frontier_judge_replies_2026-11-13.json`: 7 captured replies (3 live, 4 replays). Content is a public arXiv abstract and a public repo PR; only `current_user_text` is replaced. Reasoning text is not kept, only its length.
- `services/orion-mind/README.md`: section under "Held turns run on their own GPU seat".

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: held runs send the judge `enable_thinking=false`. A matter may carry `metadata._orion_off_vocab_fields` (metadata is a free dict). Malformed deferred/suppressed entries are dropped and noted.
- Compatibility notes: no consumer reads the new metadata key or notes.

## Env/config changes

- Added keys: `MIND_LLM_THINKING_APPRAISAL_HELD` (default `false`).
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-mind`: yes (`services/orion-mind/.env` now has `MIND_LLM_THINKING_APPRAISAL_HELD=false`).
- skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-mind && python -m pytest tests -q
351 passed

New test file (32 tests) against origin/main app code: 17 failed, 15 passed.
python scripts/check_env_template_parity.py: PASS (89 services)
```

Two tests (`test_thinking_on_held_replies_ran_out_of_tokens`, `test_thinking_off_replies_fit_the_token_budget`) only pin facts of the captured evidence; they are not regression coverage.

## Evals run

```text
Gate script (scratchpad gate_judge.py): each captured reply through the real MindLLMClient.request_json
parse path and run_active_frontier_judge. Results in the table above. Before (origin/main) and after
give the same status per reply; the request-side change (thinking off) is what moves a held run from
the failing rows to the passing rows. After also maps off-vocab labels and names the gateway error.
```

The service has no eval harness for the judge; the tests above run on the real captured replies.

## Docker/build/smoke checks

```text
No image build: Python-only change, no dependency change. One new env key (compose default false).
Capture: passive bus tap (redis-cli psubscribe on orion:mind:llm:reply:* and the gateway intake).
Note: all three Mind phases of a run share one correlation id, so request/reply pairs were matched in order.
Replays: 6 judge calls + 2 stance calls through the gateway under operator holds on agent / agent-gpu2.
Strata log (circe, orion-circe-atlas-strata-agent): "thinking budget reached (2304 tokens) ... done: 3072 tokens (length)" on each held judge call.
```

## Review findings fixed

- Finding: a test reloaded `app.settings`, swapping the shared settings object, and read the local `.env`.
  - Fix: assert the field default on `Settings.model_fields`.
  - Evidence: `test_held_thinking_setting_defaults_off`.
- Finding: the provenance record logged `null` and whitespace-padded valid values as off-vocab.
  - Fix: record only non-null values that are not on the contract.
  - Evidence: `test_provenance_records_only_off_contract_values` (4 cases).
- Finding: a model could write the provenance key into deferred/suppressed metadata.
  - Fix: the key is removed from side entries.
  - Evidence: `test_model_cannot_forge_provenance_on_deferred_or_suppressed`.
- Finding: an object hazard without text became a JSON string handed to stance.
  - Fix: dropped.
  - Evidence: `test_object_hazard_without_text_is_dropped`.
- Finding (nit): private alias table imported across modules; unbounded gateway reason in telemetry.
  - Fix: public `RECOMMENDED_EFFECT_ALIASES`; error capped at 200 chars.

## Restart required

Deploy from the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

Post-deploy check (held runs since deploy; expect `meaningful_synthesis` and judge `ok` on both seats):

```sql
select coalesce(p->>'model','?') model, r.result_jsonb->'brief'->'machine_contract'->>'mind.quality' quality,
       p->>'status' judge_status, left(p->>'error',60) judge_error, count(*)
from mind_runs r, jsonb_array_elements(r.result_jsonb->'brief'->'machine_contract'->'mind.phase_telemetry') p
where r.created_at_utc > '<deploy time>' and p->>'phase_name' = 'active_frontier_judge' and p->>'route' = 'agent'
group by 1,2,3,4 order by 5 desc;
```

## Risks / concerns

- Severity: medium
  - Concern: the 3 live Strata `frontier_schema_invalid` failures (11-13 04:11, 04:44, 05:39) left no reply to read, and 2 thinking-on replays of Strata passed. Their cause is UNVERIFIED. The likely cause is a malformed `deferred` / `suppressed` entry (the prompt asks for reasons, and the suppress reason is a closed list the prompt never shows). This PR drops such entries instead of failing, and the telemetry row now names the failing field, so a repeat is diagnosable.
  - Mitigation: post-deploy SQL above; `frontier_schema_invalid:<field>` names it.
- Severity: low
  - Concern: the judge on held seats now decides without reasoning. Replayed frontiers look sound (real matters, correct uncertainty and identity-boundary kinds), but quality versus the thinking version is not measured.
  - Mitigation: `MIND_LLM_THINKING_APPRAISAL_HELD=true` restores the old behavior.
- Severity: low
  - Concern: the prompt change (allowed values) also reaches unheld metacog runs, which pass 21/21 today. UNVERIFIED on metacog.
  - Mitigation: the lines only restate the contract.

## PR link

<filled after push>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

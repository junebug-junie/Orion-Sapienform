# Held Mind runs: invented claim labels no longer sink the semantic phase

Fixes #3274

## Summary

- Root cause: the models on the held seats write good claims (real summary, real evidence refs) but make up three label fields: `claim_kind` (`event_claim`, `technical_detail`, `state_claim`), `anchor` (`event`, `technical`, `body`) and `recommended_effect` (`acknowledge`, `follow`, `monitor`). The contract only allows a fixed list. One bad label failed the whole reply as `semantic_schema_invalid`, so Mind gave the turn nothing (`fallback_contract_only`).
- It is not the route or its request options. The same request sent on the `metacog` route gives the same kind of invented labels. In the last 24 h, the metacog lane's Qwen3-8B also failed 29 of 58 semantic phases this way, and flash-next on metacog failed 6 of 6. Held runs failed 186 of 187 because both held-seat models invent labels on nearly every reply.
- Fix 1 (request side): the semantic system prompt now lists the allowed values for the three fields. The lists are read from the contract's `Literal` types, so they cannot drift.
- Fix 2 (parse side): `try_coerce_off_vocab_claim_fields` maps an invented label onto the contract: kind becomes `current_turn_claim` if a ref cites `current_turn:`, else `situation_claim`; anchor becomes `unknown`; effect becomes its alias or `no_effect`. The original values stay on the claim in `metadata._orion_off_vocab_fields`, and the run notes `normalized_off_vocab_claim_fields`. Appraisal already coerced its own labels the same way.
- No hollow fallback. A claim without a non-blank string `claim_id`/`label`/`summary`/`claim_kind`, with non-dict `metadata`, or with an out-of-range number still fails `semantic_schema_invalid`. Claims whose evidence refs are not in the pack are still dropped by the guardrail filter.

## Outcome moved

Held Mind runs (world pulse, curiosity, journal, field notes) can produce a semantic synthesis again. Gate on the three captured raw replies:

| reply | origin/main | this branch |
|---|---|---|
| `agent` seat (Strata flash-next), live held | `semantic_schema_invalid` | `ok`, 8 of 8 claims kept |
| `agent-gpu2` seat (Bonsai 27B), live held | `semantic_schema_invalid` | `ok`, 8 of 10 kept (2 cite a ref not in the test pack, dropped by the guardrail) |
| `metacog` probe, same request | `semantic_schema_invalid` | `ok`, 5 of 5 kept |

## Current architecture

`services/orion-mind/app/synthesis.py` `run_semantic_synthesis` asks the LLM for `SemanticSynthesisV1` JSON. `_validate_semantic_raw` validated it and had repair paths for a bare single claim, a top-level array, and the legacy `{"claim": ...}` shape. A normal `{"claims": [...]}` object with invented labels had no repair path, so it failed. Since PR #3238 (11-12 09:02Z) held runs go to route `agent` (`held_call_route`), where both seat models invent labels on almost every reply.

## Architecture touched

- `services/orion-mind/app/synthesis.py` only. No contract, bus, env, or route change.

## Files changed

- `services/orion-mind/app/synthesis.py`: allowed values in the prompt (from the contract Literals); new `try_coerce_off_vocab_claim_fields`, wired into `_validate_semantic_raw` after the legacy step; the singleton path now keeps the same provenance record.
- `services/orion-mind/tests/test_held_semantic_off_vocab.py`: regression tests on the captured replies, plus the failure cases that still have to fail.
- `services/orion-mind/tests/fixtures/held_semantic_replies_2026-11-13.json`: captured replies. The gpu2 reply is verbatim (a public news headline). The agent and metacog replies had personal and home-security details, so their label and summary text is replaced. Every label value, ref, and number is verbatim.
- `services/orion-mind/README.md`: section under "Held turns run on their own GPU seat".

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: a semantic claim with invented `claim_kind`/`anchor`/`recommended_effect` is now kept with mapped labels instead of failing the reply. `SemanticClaimV1.metadata` may carry `_orion_off_vocab_fields` (metadata is a free dict already).
- Compatibility notes: no consumer reads the new metadata key.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-mind && python -m pytest tests -q
309 passed

Same new test file against origin/main synthesis.py: 7 failed, 6 passed (the captured replies fail as in production).
```

## Evals run

```text
Gate (scratchpad script feeding the raw captured replies through run_semantic_synthesis):
BEFORE (origin/main)
agent (gpu1 Strata flash-next, live held): err=semantic_schema_invalid status=schema_invalid
agent-gpu2 (Bonsai 27B, live held):       err=semantic_schema_invalid status=schema_invalid
metacog probe (same request):              err=semantic_schema_invalid status=schema_invalid
AFTER
agent (gpu1 Strata flash-next, live held): err=None status=ok validation_ok=True raw=8 retained=8
agent-gpu2 (Bonsai 27B, live held):       err=None status=ok validation_ok=True raw=10 retained=8
   first claim: kind=current_turn_claim anchor=unknown effect=answer_directly
   off_vocab={'claim_kind': 'event_claim', 'anchor': 'event', 'recommended_effect': 'acknowledge'}
metacog probe (same request):              err=None status=ok validation_ok=True raw=5 retained=5
```

The service has no eval harness for semantic synthesis; the tests above run on the real captured replies.

## Docker/build/smoke checks

```text
No image build: Python-only change, no dependency or compose change.
Capture method: passive bus tap (redis-cli psubscribe on orion:mind:llm:reply:* and the gateway intake) caught the
real held requests and replies with no extra GPU load. One metacog probe was sent through the gateway from inside
the orion-mind container with the captured request. A second probe with the new prompt got gpu_pool_unavailable
twice (pool busy), so the prompt's effect on the models is UNVERIFIED; the parse-side fix does not depend on it.
```

## Review findings fixed

- Finding: a list or dict label value raised `TypeError` (unhashable) out of the Mind run.
  - Fix: membership tests only on strings; non-string kind fails, non-string anchor/effect is mapped and recorded.
  - Evidence: `test_non_string_label_values_do_not_crash` (9 cases).
- Finding: non-dict model metadata was silently replaced.
  - Fix: non-dict metadata now fails the reply, as before.
  - Evidence: `test_non_dict_metadata_still_fails`.
- Finding: the model could write or clobber the provenance key.
  - Fix: key is `_orion_off_vocab_fields`; model keys are kept beside it.
  - Evidence: `test_model_written_provenance_key_is_overwritten_by_the_real_record`.
- Finding: any claim with refs became `current_turn_claim` (a turn anchor in appraisal), even with only `situation_compact` refs.
  - Fix: `current_turn_claim` only when a ref starts with `current_turn:`, else `situation_claim`.
  - Evidence: `test_off_vocab_claim_citing_only_situation_is_not_promoted_to_turn_anchor`.
- Finding: a whitespace-only summary passed the content gate.
  - Fix: content fields must be non-blank strings.
  - Evidence: `test_off_vocab_claim_without_content_still_fails` (4 cases).
- Finding (nit): singleton path mapped labels without a record; weak tests (confidence, refs equality, prompt line match).
  - Fix: singleton path keeps the same record; tests strengthened.
  - Evidence: `test_singleton_root_with_invented_labels_keeps_provenance`; 309 passed.

## Restart required

Deploy from the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

Post-deploy check (held runs since deploy should show semantic phases that passed, on both seats):

```sql
select p->>'model' model, p->>'status' st, count(*)
from mind_runs m, jsonb_array_elements(m.result_jsonb->'brief'->'machine_contract'->'mind.phase_telemetry') p
where m.created_at_utc > '<deploy time>' and p->>'phase_name' = 'semantic_synthesis'
  and m.result_jsonb->'brief'->'machine_contract'->>'mind.semantic_route' = 'agent'
group by 1, 2 order by 3 desc;
```

## Risks / concerns

- Severity: medium
  - Concern: after semantic passes, held runs now reach the appraisal and stance phases on the held seats for the first time. Whether those phases pass there is UNVERIFIED (no held run got past semantic since 11-12). Appraisal already maps invented labels; stance has its own alias coercion.
  - Mitigation: the post-deploy SQL above, extended to `mind.quality`; any new failure shows in `mind.fallback_reason`.
- Severity: low
  - Concern: mapped labels are coarse (`no_effect`, `unknown`). Claims keep their real summary and refs; the original labels are kept in metadata.
  - Mitigation: the prompt now names the allowed values, so fewer replies should need mapping. Count `normalized_off_vocab_claim_fields` notes after deploy.
- Issues: closes #3274. No new issue opened.

## PR link

<filled in on PR creation>

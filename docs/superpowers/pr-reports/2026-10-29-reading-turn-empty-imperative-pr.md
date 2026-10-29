## Summary

Orion's reading turns sometimes stopped before any reading happened. The step that decides how to approach the task (the stance pass) returned no instruction. This PR fixes the two causes found in the real failed turns.

- The stance pass saw two output formats in one prompt: its own, and the JSON shape the reader must return. One model wrote the reader's fields instead of its own. Another model spent its whole token budget deciding between the two formats and returned nothing. Now the reader's format is fenced in the prompt, and the stance pass gets a copy of the task with that fenced part replaced by one line. The reader still gets the full prompt.
- One model (Bonsai-2-27B, a 2-bit quantized model) writes the field name `imperial` when it means `imperative`. Its own reasoning shows it knows the right spelling and still cannot produce it. When `imperial` holds real text, that text is now used. When it is empty or null, the turn still stops as before.
- No check was loosened. An empty instruction still stops the turn.

## Outcome moved

Failed `reading.turn` runs, 2026-10-26..28 (UTC), 102 of 558 runs. This patch targets 13 of the 102:

| Cause | Count | After this patch |
|---|---|---|
| `stance_react_failed: timeout:caller_budget_exhausted` (GPU capacity) | 65 | Not touched. Already treated as capacity, retried without using an attempt. |
| `turn_error: Could not parse JSON object from LLM text` (the reader's answer) | 14 | Not touched. |
| `turn_exception: TimeoutError` | 8 | Not touched. |
| `empty_imperative`: stance wrote the reader's fields (Qwen3.6-35B) | 6 | Expected fixed: the fields are no longer in the stance prompt. UNVERIFIED live. |
| `empty_imperative`: `imperial` with real text (Bonsai-2) | 2 | Fixed. Replayed from the real stored output: both now proceed. |
| `empty_imperative`: `imperial: null` (Bonsai-2) | 2 | Still stop, correctly (no instruction). One of the two weighed both formats in its reasoning, so it may improve. UNVERIFIED. |
| `missing thought payload`: reasoning used the whole 8000-token budget (Bonsai-2) | 3 | Expected to drop: 2 of 3 spent the budget weighing the two formats. UNVERIFIED live. |
| `gpu_pool_recalled:lease_lost`, truncated stance JSON | 2 | Not touched. |

The brief said these failures were mostly `empty_imperative`. They are not: GPU-capacity refusals are 65 of 102. Those already have their own handling (`orion/world_pulse_read/retry.py:is_capacity_deferral`).

Seed-level effect: failed turns are retried, so only 1 seed ended for good on `empty_imperative` in the window. The real cost is wasted stance calls (each 57-240 s of GPU time) and retry slots.

`world_pulse_read_seed` final failures in the same window are a separate problem: `no_read_evidence:thin_fetch` 21, `no_read_evidence` 18, `no_read_evidence:http_error` 18. These are the reader failing to fetch the source (WSJ, NYT, Reuters paywalls; x.com; pbs.org thin pages 9 of 21). They are not stance failures. Not touched here; pbs.org thin fetches look worth a follow-up.

## Evidence (real failing turns)

Raw stance text is stored in `cognition_traces` (verb `stance_react`), joined to `substrate_durable_run_state` by `turn_correlation_id`.

- Qwen3.6-35B: 6 of 6 reading stances on this model deferred. Their output keys start `summary, tone, strain_refs, ...` (4) or `summary, article_summary, priors_tested, ...` (2, the whole reading answer). The same model gave a valid stance on 5 of 5 non-reading turns. Example: `cb56bd61`, `5ef0a93b`.
- Bonsai-2 empty replies: content `""`, reasoning 23k-35k chars. `bc448682` reasoning: "There is tension: user message says required JSON shape exactly top-level keys summary ... But overall prompt says output only strict JSON that validates against ThoughtEventV1 ... Which schema?" It then counts letters until the budget runs out.
- Bonsai-2 `imperial`: 9 of its 59 non-empty stance outputs in 7 days, on reading and chat turns alike. `998f5800` reasoning: "Let's produce final now with correct spelling: i m p e r a t i v e" then writes `"imperial"`.
- Token budget: `llm_chat_budget ... verb=stance_react effective_max_tokens=8000` (cortex-exec log).

## Current architecture

A reading turn is a unified turn (`execute_unified_turn`, `reading_only=True`) started by `reading_turn_listener.py` from a durable brief. The brief's prompt went unchanged to both the stance pass (`orion-thought` -> cortex-exec `stance_react.j2`, rendered into `user_message`) and the motor. `stance_react.j2` already said "those schemas belong to the downstream task" (since 2026-09-26). That instruction did not hold for these two models.

## Architecture touched

- Hub reading prompt builders: the reader contract is fenced.
- Hub reading listener -> `execute_unified_turn`: new optional `stance_view_message`.
- `orion-thought` stance parsing: one key repair.

## Files changed

- `orion/world_pulse_read/reader_contract.py`: new. Fence markers, `fence_reader_contract`, `stance_view_of_reading_prompt`.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py`: Stage 1 contract fenced.
- `services/orion-hub/scripts/world_pulse_read_stage2.py`: Stage 2 contract fenced.
- `orion/world_pulse_read/assertions.py`: relationship-claim instructions fenced. Retained source text stays outside the fence.
- `services/orion-hub/scripts/reading_turn_listener.py`: passes the stance view.
- `orion/hub/turn_orchestrator.py`: `stance_view_message` kwarg. The stance request and `stance_inputs` take the view; the harness keeps the full message.
- `orion/thought/stance_react.py`: `imperial` -> `imperative` repair, only for real text.
- Tests: `orion/thought/tests/test_stance_react_reading_failures.py` + fixture of 5 real failing outputs, `orion/world_pulse_read/tests/test_reader_contract.py`, additions to hub reading tests and the orchestrator test.
- Eval: `services/orion-hub/evals/test_reading_stance_prompt_eval.py`, wired into `orion-reading-tests.yml`.
- `services/orion-hub/README.md`: one paragraph.

## Schema / bus / API changes

- Added: `execute_unified_turn(stance_view_message=...)` (Python kwarg, default None).
- Removed: none.
- Renamed: none.
- Behavior changed: reading prompts now contain `<<<READER OUTPUT CONTRACT>>>` fence lines; the stance pass of a reading turn does not see the fenced text.
- Compatibility notes: no schema, bus, or durable payload change. Briefs queued before deploy have no fences; their stance gets the full prompt, as before.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
orion/thought/tests/test_stance_react_reading_failures.py: 14 passed (10 fail with the repair removed)
orion/thought + orion/world_pulse_read: 313 passed
services/orion-thought/tests: 782 passed, 32 skipped
services/orion-hub/tests -k "reading or world_pulse or stage2 or claim or curiosity_recall": 421 passed, 87 skipped
services/orion-hub/tests/test_reading_turn_listener.py + test_turn_orchestrator_utterance_origin.py: 22 passed, 1 failed
  (test_execute_unified_turn_uses_mind_appraisal_text_for_stance_not_harness fails identically on main; pre-existing)
orion/evals/model_replay: 94 passed
```

## Evals run

```text
services/orion-hub/evals/test_reading_stance_prompt_eval.py: 3 passed
  stage1          reader_keys_before=4 after=0
  document_stage1 reader_keys_before=4 after=0
  stage2          reader_keys_before=8 after=0
services/orion-hub/evals/test_reading_handoff_eval.py + test_reading_document_eval.py: 12 passed
```

The eval checks the rendered prompt, not live model quality. Live check (opt-in, uses the GPU pool): `orion/thought/evals/stance_task_boundary.py` with a stance view as the task file. Not run.

## Docker/build/smoke checks

```text
Not run. No dependency, compose, or env change. Live path UNVERIFIED until deploy.
```

## Review findings fixed

A code-review subagent reviewed commit a23fd1722. No blockers. Fixed:

- Finding: the Stage 2 stance view still showed the Stage 1 handoff, a reader-shaped dict (`'what_i_learned': ..., 'candidate_priors': ...`).
  - Fix: the handoff stays (the stance must read it). It is now on one line labelled "Stage 1 handoff (input data from the earlier read, not an output shape)".
  - Evidence: the eval exempts only that labelled line. With the label removed, the eval finds 4 leaked keys (mutation check run by hand).
- Finding: the eval matched only double-quoted keys, so it could not see a dict-repr leak. Its Stage 2 case had no claim context.
  - Fix: the eval also matches single-quoted keys (the template escapes `'` as `\u0027`). It now builds a claim context, so the relationship-claims fence is rendered and checked.
  - Evidence: `test_reading_stance_prompt_eval.py` stage2 `reader_keys_before=9 after=0`.
- Finding: the reader's format orders ("return ONLY one fenced ```json block", "ONE JSON object", "say in summary") were outside the fence.
  - Fix: moved into the fenced contract. The retry note now says "say what could not be read".
  - Evidence: the eval asserts the view holds no "```json", "JSON object", or "Juniper-facing prose".
- Finding: a marker string in untrusted text made the view fail back to the full prompt with no record.
  - Fix: the handoff and retained excerpts are passed through `defuse_fence_markers`. Document text is not changed (the reader must get the exact snapshot bytes, which `_generate` checks). The listener logs `reading_stance_view_unavailable reason=damaged_fence` when fences exist but do not pair.
  - Evidence: `test_damaged_fence_falls_back_to_full_prompt_and_logs`, `test_defused_source_text_cannot_break_or_hide_behind_the_fence`.
- Finding (nit): the `harness_user_message` comment no longer matched the code. Fix: comment rewritten.
- Finding (nit): the regression-test docstring overstated which cases change. Fix: docstring says which are regressions and which are guards.

Not fixed:

- Finding (nit): mark the `imperial` repair in `disposition_reasons`. Not done: `evaluate_thought_disposition` replaces the reasons with `[]` on proceed, so the marker would be lost on exactly the repaired turns. The repair is visible in the WARNING log `stance_react_imperative_key_repaired`, and the raw text in `cognition_traces` keeps `imperial`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
```

Run from the clean primary `main` checkout after merge. Hub owns the prompt builders, the listener, and the turn orchestrator. Thought owns the stance parsing.

## Risks / concerns

- Severity: medium. Concern: the 6 + 3 live improvement is a prediction from the evidence, not a measurement. Mitigation: after deploy, re-run the failure-by-cause query on `substrate_durable_run_state` (workflow `reading.turn`) for 2 days and compare with the table above.
- Severity: low. Concern: the reader now sees two fence lines around its JSON shape. Mitigation: plain text lines; reader output parsing reads the model's JSON, not the prompt.
- Severity: low. Concern: a source document that contains the fence strings makes the view fail closed (the stance gets the full prompt, as before). Mitigation: no worse than today.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2966

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## Summary

Curiosity compaction (the weekly/monthly digest of Orion's Curiosity write-ups) had stopped saving any work. This patch gets it moving again in three ways.

- The model's answer is now shaped by a JSON schema that the llama.cpp worker enforces while it writes. A topic body cannot pass 1200 characters, every topic must cite at least one source, a topic can only cite sources from the current batch, and there are at most four topics (was six: six full-length topics could never fit the 6000-character pass budget, so such an answer could only be rejected).
- A rejected model answer is retried again. A new exception, `ModelOutputRejected`, marks "the model's answer broke the contract". The shared retry rule treats it as a bounded retry. A real code bug (any other `ValueError`, `TypeError`, `KeyError`, ...) still fails the run at once.
- The prompt asks for "about 120 to 150 words" per topic instead of "80 to 1200 characters". The model cannot count characters, so it wrote right up to the cap and often went past it. 1200 characters is still the hard limit.
- A body that fills the cap exactly and does not end a sentence was cut off by the schema mid-thought. It is rejected and re-sampled instead of being saved as a fragment.

## Outcome moved

Failure mode: from 2026-10-24 00:42Z, 9 compaction runs saved zero batches. PR #2899 made every `ValueError` (pydantic `ValidationError` included) fail at once. Before that, these answers were retried, and the retries let batches get saved. After this patch, an over-length or badly-referenced answer costs one attempt out of `DURABLE_RUNS_RETRY_MAX_ATTEMPTS`, not the whole run. Each batch that is accepted is saved before the next call, so retries move the cursor forward. The schema also stops most of these failures from happening at all.

## Current architecture

- `orion/curiosity/compaction.py`: `Topic` / `TopicPass` / `Summary` pydantic contracts. `record_answer` parses and validates one answer, and raised plain `ValueError` / `ValidationError` on rejection.
- `services/orion-durable-runs/app/curiosity_compaction_graph.py`: one `call_verb_text` per node under the GPU hold. Exceptions went to `failed_attempt`.
- `services/orion-durable-runs/app/admitted_graph.py`: `is_deterministic` classified every `ValueError` as "retry cannot help".
- `runner._call_verb_text` had no way to pass `structured_output_*` options.

## Architecture touched

- New shared exception: `orion/cognition/model_output.py` (`ModelOutputRejected(ValueError)`). It subclasses `ValueError`, so every existing `except ValueError` still catches it.
- Retry policy: `is_deterministic` returns False for `ModelOutputRejected`.
- `runner._call_verb_text(options=...)`: extra cortex options are merged in **before** the fixed keys. A caller cannot override `policy_dispatch_only`, thinking-off, the route, or the lease.
- Compaction: `output_schema(row)` (per phase), `structured_output_options(row)`, and `record_answer` raises `ModelOutputRejected`.

Other `failed_attempt` consumers were checked. None needed this change:
- `curiosity.investigate` `harness_turn`: a failed turn already raises `HarnessTurnFailed` (a `RuntimeError`, so it is transient).
- `self_study.reflect`: an unparseable answer becomes `llm_call_ok=False` state. It is not raised.
- `self_sense_eval`: `ask_one` records a failed answer and raises only `SelfSenseAskFailed` (a `RuntimeError`).
- `journal.compose` and `memory.episode_distill` do not use `failed_attempt`. They keep their own minimum-attempt policy.

## Files changed

- `orion/cognition/model_output.py`: new `ModelOutputRejected`.
- `orion/curiosity/compaction.py`: `output_schema`, `TOPIC_BODY_CHARS` / `PASS_BUDGET` / `MAX_TOPICS` (6 -> 4, derived from the budget) constants, prompt wording, `ModelOutputRejected` on every model-answer rejection, and the cut-off check.
- `services/orion-durable-runs/app/admitted_graph.py`: `is_deterministic` exemption and comment.
- `services/orion-durable-runs/app/curiosity_compaction_graph.py`: sends the schema options. Comment updated.
- `services/orion-durable-runs/app/runner.py`: `_call_verb_text(options=...)`.
- `services/orion-durable-runs/README.md`: retry policy and compaction schema documented.
- Tests: `services/orion-durable-runs/tests/test_curiosity_compaction_graph.py`, `test_admitted_graph.py`, `test_verb_text.py`, `services/orion-dream/tests/test_curiosity_compaction.py`.

## Schema / bus / API changes

- Added: the `options` kwarg on `_call_verb_text` (internal). The `structured_output_schema` payload for `curiosity_compactor_digest_v1`.
- Removed / Renamed: none.
- Behavior changed: model-answer rejections in compaction are retried (bounded). Answers are grammar-constrained. A topic pass now has at most 4 topics (was 6). "Topic reduction did not shrink" stays a plain `ValueError` (fails fast): it depends on every saved partial, and a retry only re-samples the last batch.
- Compatibility notes: no bus, channel, or registry change. `ModelOutputRejected` is a `ValueError`, so existing catchers are unaffected. The `curiosity_compaction` row format is unchanged.

## Env/config changes

- Added / Removed / Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed (no template change).
- Kill switch: `DURABLE_RUNS_RETRY_MAX_ATTEMPTS` still bounds these retries. Set it to `1` to fail on the first rejection.

## The JSON-grammar quality warning

I searched the repo for a warning that JSON grammars hurt writing quality. The one warning that exists is in `services/orion-cortex-exec/app/executor.py` (`_should_request_chat_reply_logprobs`). It says JSON-constrained decoding gives near-zero-entropy **logprobs**, so a constrained reply must not be used as a logprob/uncertainty signal. That is about measurement, not prose. Compaction does not request logprobs, so the warning does not apply directly. I still kept the grammar light on the prose:

- `body` is a free string, limited only by `minLength` / `maxLength`. There is no `pattern`, no enum, and no format.
- The structure that is constrained (refs enum, item counts) is metadata, not writing.
- The prompt targets about 800 characters, so the length cap should rarely bind. When it does bind mid-sentence, the answer is rejected and re-sampled rather than saved as a stump.

## Tests run

```text
cd services/orion-durable-runs && PYTHONPATH=<worktree> python -m pytest tests -q
531 passed, 107 skipped

PYTHONPATH=<worktree> python -m pytest -q services/orion-dream/tests/test_curiosity_compaction.py \
  services/orion-dream/tests/test_curiosity_compaction_postgres.py services/orion-dream/tests/test_dream_cycle_v2.py \
  services/orion-dream/evals/test_curiosity_compaction_eval.py
82 passed, 5 skipped
```

Mutation check: I removed the `ModelOutputRejected` exemption from `is_deterministic`. Three new tests then failed: the retry test, the bounded-attempts test, and the policy unit test.

New regression tests:
- an over-cap body is retried (attempt 1, `waiting_resource`, nothing saved), then the next sample saves the batch and the run completes;
- repeated rejections still stop at max attempts and are not tagged `deterministic:`;
- a real `ValueError` / `TypeError` / `KeyError` from the call still fails fast with `deterministic:`;
- the schema payload sent per phase (maxLength 1200 / 2400, refs minItems 1 + enum, maxItems 6, method `json_object_schema`);
- `_call_verb_text` forwards the options but cannot override thinking-off or `policy_dispatch_only`;
- `record_answer` raises `ModelOutputRejected` for parse, ref, and length failures; a body cut off at the cap is rejected, and a finished body at the cap is kept;
- the prompt wording.

## Evals run

```text
services/orion-dream/evals/test_curiosity_compaction_eval.py  (in the run above) -- passed
```

## Docker/build/smoke checks

```text
Live llama.cpp probe (quick lane, circe :8013, same llama.cpp server build), direct /v1/chat/completions
with response_format {"type":"json_object","schema": output_schema(...)}:
- maxLength lowered to 150, model told to write 600 words and cite "s7": every body <= 150 chars, and
  refs drawn only from the enum ["s0","s1"]. The grammar enforces both. One body ended mid-word
  ("...shape the") -- this is why the cut-off check exists.
- real schema (maxLength 1200): the grammar compiled and the answer came back in 15 s, body 1174 chars.
```

The full path through cortex-orch, cortex-exec, and the gateway on the agent lane is **UNVERIFIED** until deploy. Proof after deploy is a gateway log line with `structured_output_schema_name=CuriosityCompactionTopicPassV1` on a compaction call, plus `curiosity_compaction.state->'partials'` growing.

No docker build was run. The change is Python-only in an existing image, and there is no dependency change.

## Review findings fixed

The code review ran in a subagent against `git diff origin/main...HEAD`. It found no must-fix items.

- Finding: the schema allowed 6 x 1200-char topics (7200+), which `record_answer`'s 6000-char pass budget always rejects. A full answer could only burn an attempt.
  - Fix: `MAX_TOPICS = PASS_BUDGET // (TOPIC_BODY_CHARS + 200)` = 4, used by `TopicPass`, the schema, and the prompt.
  - Evidence: `test_a_full_pass_the_schema_allows_fits_the_pass_budget`.
- Finding: two comments in `admitted_graph.py` contradicted each other.
  - Fix: rewrote them. Plain `ValueError` = code/input bug; the #2899 data and the regression now sit under the `ModelOutputRejected` exception.
- Finding: "topic reduction did not shrink" was made retryable, but a retry only re-samples the last batch, so it would mostly burn attempts.
  - Fix: back to a plain `ValueError` (fails fast), with a comment saying why.
  - Evidence: `test_reduction_that_does_not_shrink_is_not_a_retryable_model_rejection`.
- Finding (nit): `options` could carry `llm_route` / `gpu_lease` when `_call_verb_text` did not set them.
  - Fix: `_VERB_TEXT_FIXED_OPTIONS` are filtered out of caller options.
  - Evidence: `test_verb_text.py`.
- Finding (nit): the cut-off check would wrongly reject bodies ending in curly quotes, an ellipsis, or markdown.
  - Fix: added those endings and documented the "e.g." blind spot.
  - Evidence: a new test for a curly-quote ending and for a trailing-space cut.
- Finding (nit): an empty batch would give an unsatisfiable empty enum.
  - Fix: explicit guard.
- Finding (nit): no test for the overall-phase cut-off.
  - Fix: `test_overall_body_cut_off_at_its_cap_is_rejected`.

Not changed: the reviewer noted that a bug inside a pydantic validator would also be wrapped as `ModelOutputRejected`. Retries stay bounded by max attempts, and both validators are small and tested, so this is accepted.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Run it from the clean primary main checkout after merge. orion-dream imports `orion/curiosity/compaction.py` only for `source_material` / scheduling, so its behavior does not change and it needs no restart.

## Risks / concerns

- Severity: low. Concern: a long `{0,1200}` grammar repetition could slow token sampling on the agent model. Mitigation: the probe answered in 15 s on the quick lane. Watch compaction call durations after deploy.
- Severity: low. Concern: the cut-off check can reject a finished body that happens to fill exactly 1200 characters and ends without punctuation. Mitigation: this is rare, and the cost is one bounded retry.
- Severity: low. Concern: "every input ref is cited somewhere" and the 6000-char pass budget cannot be expressed in the schema. Mitigation: `record_answer` still checks both, and they are now retried.

## PR link

(filled in after opening)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

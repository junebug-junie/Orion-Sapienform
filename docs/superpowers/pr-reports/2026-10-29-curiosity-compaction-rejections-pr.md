# fix(curiosity-compaction): the model can no longer skip a record; no stump pieces; retries are bounded

## Summary

Curiosity compaction rolls a week or a month of Curiosity write-ups into topic summaries. No run has finished since it started on 2026-10-14: 43 runs failed, 0 completed. This patch fixes the three causes.

- The model must now account for every source record. The answer starts with a map from each record (`s0`, `s1`, ...) to a topic. The output grammar makes every record a required key, so the model cannot leave one out. The checker is as strict as before.
- A long record is now split into equal pieces at a space between words. Before, a record was cut every 7000 characters, which left stump pieces such as a 32-character "elfDefinition 1, TurnOutcome 1.)". The model had nothing to say about such a stump.
- The prompt asks for shorter topics: about 80 to 110 words, not 120 to 150. At the real length of these words, 150 words is the full 1200-character limit, so the grammar cut topics off mid-sentence.
- Retries stop when they make no progress. A window is given up (`cancelled`) when three runs in a row fail at the same saved place because the model's answer was rejected there. The next window can then run. A run that saves a batch resets the count. A GPU deadline, an outage or a code bug does not count.

## Outcome moved

Failure mode: `curiosity.compaction` failed every run since 2026-10-14 (43 failed, 0 completed). The weekly 2026-10-05 window reached run r21 and the monthly 2026-09-01 window reached r20. Both windows block every later window of the same kind.

Evidence from the live rows (`curiosity_compaction`, read-only):

- **Weekly 2026-10-05, stuck at batch 1 of 5 since 2026-10-27 17:47.** Batch 1 is a reduction pass: 13 earlier topic summaries in, at most 4 topics out. The grammar already stopped invented refs (an enum per topic), but nothing forced all 13 refs to appear. Runs r17 to r20 all failed with "references omit or invent source records". Invention was impossible, so these were omissions.
- **Monthly 2026-09-01, at batch 144 of 182.** Run r20 saved batch 143 at 06:27:21 and failed at 06:28:23 on batch 144 with "references omit ...". Batch 144 holds 3 records: s0 is the 350-character tail of a record cut in the previous batch (it starts mid-word: "dAnswer with evidence..."), s1 is 7000 characters, and s2 is its 32-character tail.
- **Cut-off topics.** Of 384 accepted topic bodies, the median is 6.85 characters per word (90th percentile 7.41). Bodies ran up to 190 words when 150 were asked. 150 words x 7.4 = 1110 characters, at the 1200 limit. 20 accepted bodies were already at 1150 characters or more.
- **Splitter replay.** Re-splitting the live monthly input with the new splitter is lossless. It removes every split stump under 450 characters (32, 151, 167, 187, 272, 347, 350, 391, 445). It also needs 156 calls, not 182. The pieces under 1000 characters that remain are whole short records, not stumps.

The thinking-token trap (Qwen thinking eats `max_tokens`) does not apply here. `call_verb_text` turns thinking off, and only 1 of 43 failures was a `max_tokens` truncation (r11, 2026-10-23, before PR #2919).

## Current architecture

- `orion/curiosity/compaction.py`: `batches()` cut long bodies into 7000-character slices. `output_schema()` (V1) gave each topic a `refs` list limited to the batch's aliases. `record_answer()` rejected an answer whose refs did not equal the batch's alias set.
- `orion/curiosity/compaction_store.py`: `Session.prepare()` re-admitted a failed run under a new generation (`:rN`) on every sleep, with no limit.
- `services/orion-durable-runs/app/curiosity_compaction_graph.py`: one model call per node. Each rejected answer used one of `DURABLE_RUNS_RETRY_MAX_ATTEMPTS` (3) attempts, and the run failed after 3 rejections in a row.

## Architecture touched

- Answer contract for the `curiosity_compactor_digest_v1` topic pass: `CuriosityCompactionTopicPassV1` becomes `CuriosityCompactionTopicPassV2`.
- Window retry policy in `Session.prepare()` (dream service, sleep admission).
- No bus channel, registered schema, env key, or SQL change. The saved partial/topic format in `curiosity_compaction.state` is unchanged, so rows from before the patch resume as they are.

## Files changed

- `orion/curiosity/compaction.py`: V2 answer schema and `TopicPass`; `record_answer` maps records to topics and rejects a topic with no record; `_piece_len` for equal splits; shorter word targets; `MAX_STALLED_GENERATIONS` and `note_failed_generation`; the pass budget measures topics only (the map is not carried into the next pass).
- `orion/curiosity/compaction_store.py`: `prepare()` counts rejected generations at one cursor and gives the window up at 3. It saves the new generation and its count in one fenced UPDATE. It resumes the oldest pending window first, so an operator can re-open a given-up window.
- `services/orion-durable-runs/app/curiosity_compaction_graph.py`: on a rejected answer, saves a `rejected_at` mark in the row, so the next sleep can tell a model failure from an outage.
- `services/orion-dream/tests/test_curiosity_compaction.py`: tests updated to V2; new regression tests from the live shapes.
- `services/orion-dream/tests/test_curiosity_compaction_postgres.py`: new real-Postgres test for the retry bound.
- `services/orion-dream/evals/test_curiosity_compaction_eval.py`, `services/orion-durable-runs/tests/test_curiosity_compaction_graph.py`: answers in the V2 shape.
- `services/orion-dream/README.md`, `services/orion-durable-runs/README.md`: the new contract and the retry bound.

## Schema / bus / API changes

- Added: `record_topics` (alias -> topic index) in the topic-pass answer; `state.stall` (`{"cursor": [phase, batches, partials], "count": n}`) and `state.rejected_at` in the `curiosity_compaction` row; `ReductionDidNotShrink(ValueError)` (still fails a run at once, but counts as a model failure).
- Removed: per-topic `refs` in the model's answer. Saved topics still carry `sources`, as before.
- Renamed: schema name `CuriosityCompactionTopicPassV1` -> `CuriosityCompactionTopicPassV2`.
- Behavior changed: a window is set `cancelled` after 3 rejected generations in a row at one cursor. `prepare()` picks the oldest pending window before the newest row. Long records split into equal pieces at a word boundary (new windows only; a pending window keeps its saved batches).
- Compatibility notes: the schema is internal to this verb (model answer only). An answer in the V1 shape is rejected as a model-output error, which is a bounded retry.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=. pytest -q services/orion-dream/tests/test_curiosity_compaction.py \
  services/orion-dream/evals/test_curiosity_compaction_eval.py services/orion-dream/tests/test_dream_cycle_v2.py
CURIOSITY_COMPACTION_TEST_POSTGRES_URI=<disposable postgres:16 container> \
  pytest -q ...test_curiosity_compaction.py ...test_curiosity_compaction_postgres.py \
  ...test_dream_cycle_v2.py ...evals/test_curiosity_compaction_eval.py
  -> 99 passed
cd services/orion-durable-runs && PYTHONPATH=<root> pytest tests -q
  -> 547 passed, 110 skipped
python scripts/check_definition_drift.py --gate -> PASS
```

The new tests fail on the old code: 14 failed with the pre-patch `compaction.py`, including the stump split, the 13-record batch, the word target, the V2 schema, and the retry bound.

## Evals run

```text
pytest -q services/orion-dream/evals/test_curiosity_compaction_eval.py -> passed (offline chronology/provenance replay)
```

There is no live-model eval of whether Qwen fills the new map correctly. See UNVERIFIED below.

## Docker/build/smoke checks

```text
No image build: no dependency, Dockerfile or compose change.
llama.cpp grammar: checked against common/json-schema-to-grammar.cpp source (required properties are emitted in schema
order; integer enum values become literals). The converter was not run locally.
```

## Review findings fixed

- Finding: every failed run counted toward giving up, including a GPU-pool deadline (for example a cabinet-heat lockdown), a transport outage or a code bug. Three such sleeps would cancel a window the model never answered.
  - Fix: durable-runs saves `state.rejected_at` when the model's answer is rejected (`ModelOutputRejected` or `ReductionDidNotShrink`). `note_failed_generation` counts a generation only when that mark matches the cursor. An accepted answer clears the mark.
  - Evidence: `test_only_rejected_model_answers_count_toward_giving_a_window_up`; the Postgres test runs 5 unmarked failures and the count stays at 1; the graph tests check the mark is written on rejection and not on a code bug.
- Finding: the README's re-open step did not work, because `prepare()` only read the newest row.
  - Fix: `prepare()` reads the oldest pending row first.
  - Evidence: `test_given_up_window_can_be_reopened_after_the_next_window_exists` (real Postgres).
- Finding: the stall count was saved in a second commit after the generation UPDATE, so a crash between them lost one count.
  - Fix: one fenced UPDATE sets `run_id`, `generation` and `state`.
  - Evidence: the Postgres test reads the count back from the row.
- Finding (nit): `record_topics` accepted `0.0` or `true` as an index in lax mode.
  - Fix: `dict[str, StrictInt]`.
  - Evidence: `test_record_topics_index_must_be_an_integer`.
- Finding (nit): a stale comment about `refs`, and the prompt's budget sentence measured a different thing than the check. Both are fixed.
- Not fixed, by choice:
  - Saved batches keep their old split (see Risks).
  - Unicode-heavy text that needs the escape-length halving loop can still split unevenly or mid-word. The output stays lossless and has no tiny pieces. The live corpus did not hit this.
  - A record now credits exactly one topic. V1 let one record credit several topics, so topic provenance is narrower but never invented. A follow-up can make each value a list of topic indices.

## Restart required

Both services import `orion/curiosity`. From the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-dream up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: UNVERIFIED live. No run has used the V2 schema against the real model. The weekly window needs 3 more failures at batch 1 before it is given up, so a bad V2 result costs at most 3 more runs there.
  - Mitigation: check `substrate_durable_run_state` for `curiosity.compaction` after the next sleep, and `curiosity_compaction.state->'partials'` for movement.
- Severity: low
  - Concern: the monthly 2026-09-01 window keeps its saved batches, including the stump at batch 144. The grammar now forces the model to place the stump in a topic, so it cannot be skipped, but the stump stays in the input.
  - Mitigation: the splitter fix applies to every new window.
- Severity: low
  - Concern: a given-up window is not compacted unless an operator re-opens it (`status='pending'` and remove `state.stall`).
  - Mitigation: the give-up is logged (`curiosity_compaction_gave_up`) and visible at `GET /dreams/curiosity-compactions`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2944

🤖 Generated with [Claude Code](https://claude.com/claude-code)

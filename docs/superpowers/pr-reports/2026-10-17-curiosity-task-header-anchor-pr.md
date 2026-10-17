## Summary

- Self-inquiry prompts now open with a short task header. Code writes it, and it holds only trusted text: the built-in standing question, a question Juniper pinned, or the id of a question Orion wrote.
- The governor anchors that header in the system prompt for Orion's own turns. Compaction never removes the system prompt, so the task survives a compaction that writes no summary.
- The header is read only when it is the first byte of the request. Quoted material always comes later, so a fake header inside it is ignored.
- Resumed durable runs (retries) put their preamble after the header, not before it.
- An Orion-minted question can no longer overwrite one of Juniper's questions while the row still says `minted_by='juniper'`.

## Outcome moved

Since PR #2744 was deployed, 11 of 14 compacted curiosity turns lost their task, all with `task_anchor_chars=0`. That includes 5 of 5 self-inquiry turns. Self-inquiry turns now carry an anchor, so the question survives a compaction that writes no summary.

## Current architecture

`orion/harness/task_anchor.py::build_task_anchor` anchored only turns Juniper typed (`utterance_origin="juniper"`). Orion's own turns (`"orion"`) got none, because their prompts quote outside material and system-prompt text outranks everything else.

## Architecture touched

- `orion/harness/task_anchor.py`: new `render_task_header`, `extract_task_header` and `split_task_header`, and `HEADER_ORIGINS={"orion"}`. Origin `orion` anchors the header only.
- `orion/curiosity/self_inquiry_prompt.py`: new `self_inquiry_task_header`, prepended to the prompt.
- `services/orion-hub/scripts/curiosity_investigation.py`: `_prompt_for_attempt` keeps the header first (`_with_header_first`).
- `orion/curiosity/self_question_pool.py`: `UPSERT_MINT_SQL` updates only rows minted by Orion.

## Files changed

- `orion/harness/task_anchor.py`: the header contract.
- `orion/curiosity/self_inquiry_prompt.py`: header producer. It quotes question text only for the standing question or a Juniper question. Orion-minted questions get their id, and only if the id matches `[a-z0-9_.:-]{1,64}`.
- `orion/curiosity/self_question_pool.py`: conflict filter on Orion mints.
- `services/orion-hub/scripts/curiosity_investigation.py`: header ordering on drift and resume preambles.
- Tests: `orion/harness/tests/test_task_anchor.py`, `orion/curiosity/tests/test_self_inquiry_prompt_drawn_question.py`, `services/orion-hub/tests/test_curiosity_resume_prompt.py`.

## Schema / bus / API changes

- Added: none. The header is in-band text at the start of `user_message`.
- Behavior changed: Orion's turns whose request opens with the header get `--append-system-prompt`. An Orion mint whose `question_id` collides with a Juniper question now writes nothing. Before, it replaced her text.
- Compatibility notes: none.

## Env/config changes

None. The existing `HARNESS_FCC_TASK_ANCHOR_ENABLED` flag still turns anchoring off.

## Tests run

```text
pytest orion/harness/tests/test_task_anchor.py orion/curiosity: 216 passed, 20 skipped
hub: test_curiosity_resume_prompt, test_curiosity_fresh_priors_and_answers, test_curiosity_self_inquiry: 73 passed
hub -k "curiosity or self_question or mint": 536 passed
orion/harness + orion/curiosity full: 784 passed (before review fixes)
```

## Evals run

```text
No eval harness for anchoring. Live check after deploy: grep the governor log for
`fcc_task_anchor corr=` on self-inquiry turns. `fcc_compaction_degenerate ... task_anchor_chars=` should be nonzero for self-inquiry.
```

## Docker/build/smoke checks

```text
EXPLAIN of the new UPSERT_MINT_SQL against the live DB (plan only, no write):
  Conflict Resolution: UPDATE, Conflict Filter: (minted_by = 'orion')
Live question ids: 23 of 23 match the safe-id pattern (9 juniper, 14 orion).
```

## Review findings fixed

- Finding: an Orion-minted `question_id` has no charset check at mint time, so it could carry instructions into the anchor.
  - Fix: ids not matching `[a-z0-9_.:-]{1,64}` are left out of the header.
  - Evidence: `test_a_hostile_minted_question_id_never_reaches_the_anchor`. All 23 live ids pass the pattern.
- Finding: an Orion mint could overwrite a Juniper question's text while keeping `minted_by='juniper'`. That text would then be quoted as hers.
  - Fix: `ON CONFLICT ... DO UPDATE ... WHERE minted_by = 'orion'`.
  - Evidence: live `EXPLAIN` shows the conflict filter.
- Finding: on durable retries and drift preambles, text was put before the header, so the anchor silently disappeared on the long runs that compact.
  - Fix: `_with_header_first`.
  - Evidence: `test_retry_keeps_the_task_header_first_so_the_governor_still_anchors_it`.
- Not changed (nit): the header now differs per question at the very first byte of the user message. Prompt-cache sharing between self-inquiry runs is barely affected, because the system prompt and tool list come first.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium.
  - Concern: investigation turns (3 of 5 compacted turns lost their task) are not covered. They have no fixed task by design ("follow whatever you want"). Keeping their chosen thread would need a different mechanism, for example re-anchoring on the run's own first hop note.
  - Mitigation: none in this PR. `fcc_degenerate_compactions` keeps counting them.
- Severity: low.
  - Concern: self-sense eval is not covered on purpose. Changing its prompt would break comparisons between runs.
  - Mitigation: none needed.
- Severity: low.
  - Concern: the root cause, a summarizer that keeps doing the task instead of summarizing, is still unfixed.
  - Mitigation: anchors limit the damage for covered turns.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)

# Stance imperative scope: the message is the task, the imperative is guidance

Status: design, 2026-09-28. No code changed yet.

## Arsonist summary

A plain chat question ("What have you read about graphics cards lately, and what
did you actually learn from it?") ran 22+ minutes and 112 steps before it was
cancelled. Orion answered the real question in the first 6 tool calls (all
`orion-introspect` reading lookups). The next ~90 steps chased a side job that
nobody asked for: "ground it in Orion's recent rendering experience on
host:circe_gpu".

That side job came from stance. Stance saw a recent autonomy action ("render on
host:circe_gpu produced an image") next to the words "graphics cards" and wrote
it into the imperative. The harness then told the motor "Your imperative states
what this turn requires ... Execute your imperative", so the side job outranked
the question. The post-turn reflection grades the draft against the imperative
too, so skipping the side job would have been scored as misaligned.

Stance was told the autonomy signal is "advisory ... NOT task instruction"
(`stance_react.j2`). The prompt guard did not hold, and nothing downstream
catches it. The fix is to change who is in charge, not to add another warning.

Separately, the stance decision record for that turn carries invented
bookkeeping (`event_id=evt-9a2b3c4d5e6f7g8h9i0j`, `session_id=sess-orion-main-001`,
`created_at=12:00:00` on a 23:35 turn). The stance prompt asks the model for
these fields and the parser keeps whatever it writes.

## Current architecture

Evidence turn: `corr=f924c7b9-1c82-40d8-a6a2-5acb2edffbb3` (Hub `POST /api/chat`,
`mode=orion`, `no_write=true`).

1. **Stance writes the imperative.** `orion/cognition/prompts/stance_react.j2`
   defines it as "the efference copy: what Orion must DO this turn, not a summary
   of the user's message." The autonomy slice (`recent_actions`) and Mind coloring
   are in the same prompt, labeled advisory.
2. **Parser keeps model-written bookkeeping.**
   `orion/thought/stance_react.py::parse_stance_react_payload` uses
   `raw.setdefault("correlation_id"/"session_id", ...)`, so a model value wins.
   `event_id` and `created_at` are never stamped by code. The same function
   already strips `grounding_capsule` and `autonomy_slice` for exactly this reason.
3. **Harness makes the imperative the task.** `orion/harness/operator_brief.py`:
   `HARNESS_UNIFIED_OPERATOR_BRIEF` says "Your imperative states what this turn
   requires"; `harness_motor_instruction` ends every turn with "Execute your
   imperative." `orion/harness/prefix.py::compile_harness_prefix` renders
   `Imperative:` before `User message:`.
4. **Reflection grades against the imperative.**
   `orion/cognition/prompts/harness_finalize_reflect.j2`: "Compare draft_text
   against thought_event.imperative, tone, and strain_refs." `user_message` is
   present but not the yardstick.

Every unified-turn caller passes the real originating task as `user_message`:
Hub chat (`api_routes.py`, `websocket_handler.py`), reading
(`reading_turn_listener.py`), curiosity (`curiosity_investigation.py`, Orion's own
prompt), outreach (`endogenous_outreach.py`), collapse mirror. So "the message is
the task" holds for every turn type, including Orion-origin turns.

Reading-only turns: the last 10 stance decisions in `thought_decision` for
`orion_world_pulse_read*` sessions all paraphrase the reader's own instructions
("WebFetch the URL, emit only the JSON"), all `proceed`. Not the failure here;
out of scope.

## Missing questions

1. Does a `misaligned` reflection verdict trigger repair/retry or a tool
   recommendation that re-runs work? If yes, the reflection change also cuts
   cost, and the live eval should count those retries.
2. The ATTENTION FRAME path deliberately carries Orion's own question into the
   imperative ("do the task first, then the question may ride along"). Under the
   new hierarchy that question becomes guidance, which matches its stated intent.
   Confirm with one fixture that the question still reaches the prompt.
3. The motor's loop on reading its own Agent sub-task output file (~80 steps of
   `python3 -c` over `tool-results/*.txt`) is a second, independent bug. Not
   covered here.

## Proposed schema / API changes

No schema change in the first patch. The contract change is in who the harness
and reflection treat as the task.

**Patch 1 — stamp stance bookkeeping in code (small, deterministic).**
- `parse_stance_react_payload`: pop model-supplied `event_id`, `created_at`,
  `correlation_id`, `session_id`; set them from the request (`uuid4()`,
  `datetime.now(UTC)`, request correlation/session).
- `stance_react.j2`: drop the METADATA block that asks the model for them.

**Patch 2 — task hierarchy.**
- `HARNESS_UNIFIED_OPERATOR_BRIEF` / `harness_motor_instruction`: the originating
  message is the task. The imperative is stance's read on how to approach it.
  Work the imperative adds that the message did not ask for is optional: do it
  only if it is cheap and directly serves the answer. Replace "Execute your
  imperative." with an instruction to answer the message.
- `compile_harness_prefix`: render the originating message as the task, ahead of
  stance guidance, with the label change carried in the rendered text (one
  source of truth in `prefix.py`, not duplicated in the brief).
- `harness_finalize_reflect.j2`: judge the draft first against `user_message`
  (did it answer what was asked), then against imperative/tone. A draft that
  answers the message but skips an imperative extra is not `misaligned` for
  that reason alone.
- `stance_react.j2`: define the imperative as how to approach the task in
  `user_message`, and keep extras short.

**Deferred — schema split.** If the live eval after Patch 2 still shows stance
extras driving long side missions, add `ThoughtEventV1.optional_extras: list[str]`
(max 2) so extras are carried in a separate field the harness can render as
optional and budget. Not built now; one field on one schema, cheap to add later,
and only justified by eval evidence.

## Proposal-mode record

- **Capability change:** the motor treats the originating message as the task;
  stance shapes approach and tone but can no longer add mandatory work.
- **Data touched:** none written. Prompt text and one parser.
- **Privacy boundary:** unchanged.
- **Trace that proves it:** harness grammar steps per turn (`harness_grammar_step_published ... tool=`)
  and the Claude session log show whether off-question tool work happens;
  `thought_decision` rows show code-stamped `event_id`/`created_at`.
- **Dangerous failure:** a turn where stance correctly demanded world-contact the
  message implies (e.g. "why is Hub down?" → check logs) stops doing it. Guarded:
  the message itself asks for it, and the existing "use tools when the task needs
  verified facts" line stays.
- **Rollback:** revert the commit. No migrations, no env keys.

## Files likely to touch

- `orion/thought/stance_react.py`, `orion/thought/tests/` (Patch 1)
- `orion/cognition/prompts/stance_react.j2` (both)
- `orion/harness/operator_brief.py`, `orion/harness/prefix.py`
- `orion/cognition/prompts/harness_finalize_reflect.j2`
- `orion/harness/tests/test_harness_prefix.py`, `test_harness_runner.py`
  (they assert the current wording)
- `orion/harness/evals/` (new incident replay eval)

## Non-goals

- Removing stance from any turn type, including reading-only turns.
- Keyword or regex detection of "questions about reading" or any other topic.
- New env flags, budgets, or step caps.
- Fixing the Agent-output-file loop.
- Changing what stance sees (autonomy slice, Mind coloring stay as inputs).

## Acceptance checks

Patch 1:
- Parser test: a payload with model-written `event_id`, `session_id`,
  `created_at`, `correlation_id` comes out with the request's values and a fresh
  code-stamped id/time.
- Live: the next `thought_decision` row has a UUID `event_id` and a `created_at`
  within seconds of the turn.

Patch 2 (structural, replaying the incident):
- Fixture: the real `ThoughtEventV1` from `corr=f924c7b9...` (imperative with the
  circe_gpu side job, autonomy `recent_actions` render line) plus the real user
  message, through `compile_harness_prefix` + `harness_motor_instruction`. Assert
  the compiled prompt names the message as the task, marks the imperative as
  guidance, and no longer contains "Execute your imperative" or "Your imperative
  states what this turn requires".
- Same fixture through the reflect template render: `user_message` is the first
  yardstick.
- ATTENTION FRAME fixture: a selected `ask` question still reaches the prompt.
- Existing harness prefix/runner tests updated, not deleted.

Eval (live, before/after):
- Ask the incident question 3 times through Hub `POST /api/chat` (`mode=orion`,
  `no_write=true`). Baseline (2026-09-28): did not finish; 112+ steps, cancelled
  after 22 min. Pass: each run finishes on its own, the reply cites
  `orion-introspect` results, and there is no multi-step off-question hunt
  (count non-introspect tool steps from grammar logs). Report the raw step
  counts, not just pass/fail.

## Recommended next patch

Patch 1 (bookkeeping stamp) first: small, deterministic, and it makes
`thought_decision` trustworthy for measuring Patch 2. Then Patch 2 with the
incident replay fixture and the live before/after eval.

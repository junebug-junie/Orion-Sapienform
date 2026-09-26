# Keep the reading task out of stance's output contract

## Summary

Stage 2's full reading prompt and output schema were interpolated into the
stance system prompt twice, without an explicit task/assessment boundary.
A streaming reconstruction reproduced prolonged deliberation about whether to
emit ThoughtEventV1, WorldPulseReadStage2ResultV1, or a union of both. It then
started doing the downstream reading. Fix the prompt boundary, not the timeout.

## Current architecture

- Hub Stage 2 passes its reading task to execute_unified_turn.
- The orchestrator supplies it as both user_message and stance_inputs.user_message.
- Thought builds stance_react context; Cortex Exec renders stance_react.j2.
- Gateway sends JSON-mode requests to an admitted GPU worker. Stance's semantic
  output contract is ThoughtEventV1, not the embedded reading-result schema.
- No service entry points, settings, bus channels, schema registrations, Docker
  wiring, dependencies, or env keys change. No local env sync is needed.

## Runtime evidence

- Original failed Stage 2 correlation: d57a1f07-4c09-4ccc-aa23-2ce917334d6e.
- Gateway receipt 08:19:49.552 UTC; worker slot selection 08:19:49.735. This was
  not the historical minutes-long gateway thread starvation.
- Worker generated at about 30-33 tokens/sec, reaching 6819 tokens before the
  234.9-second gateway deadline. Its non-streaming response retained no partial
  output, so the exact original generation cannot be recovered from that trace.
- Reconstructed baseline 03f8056c-54f6-4cd0-8484-d581ff87a901 used the saved
  Stage 1 handoff, duplicated task inputs, minimal coalition context, and the
  normal gateway's agent route with unchanged thinking/8000-token settings.
  It reached 260 seconds with 30017 reasoning characters and zero final text.
  Captured output repeatedly confused the two schemas and attempted Stage 2.
- Initial prefix/dedup prototype 96b2f1c6-73e7-4392-9d92-f6e82dae26d0 completed
  in 245.67 seconds but overly restricted the downstream imperative. Rejected
  as acceptance evidence; the patch also quotes source strings and explicitly
  preserves downstream task execution.
- Quoted-source candidate 2d7fbe1c-826d-4c94-9ce8-50bfd1bc70c1 passed the model
  eval in 131.68 seconds: 3813 completion tokens, 1373 final characters, valid
  stance fields, correct evidence anchor, no downstream output fields.
- Final revision eval: pending at initial commit; result will be recorded here.

These are diagnostic reconstructions, not exact historical replays or full
pipeline completion. No reading seed, journal, or thought event was written by
the evals. Admission/accounting still runs through the normal gateway.

## Changes and checks

- Explicit assessment/task boundary; JSON-quoted task/context strings; suppress
  only identical duplicate user_message. Keep distinct additional context.
- Existing reading CI now runs shared thought tests and triggers on prompt edits.
- Opt-in streaming eval validates raw fields (no invented stance defaults),
  grounding, termination, and unexpected fields; reports imperative for review.
- 49 shared thought tests and 11 reading-handoff offline evals passed.
- Thought service tests/evals: 452 passed, 12 skipped, one pre-existing failure:
  test_mind_enrichment_defaults_off expects orion-mind while default is mind.
  Reproduced that failure unchanged on main; not modified in this patch.
- Candidate template rendered successfully using deployed Hub dependencies.
- Independent review: no remaining material findings after fixes below.

## Review findings fixed

- Finding: tolerant runtime parsing removed forbidden keys before eval checks.
  - Fix: validate raw model output without mutation; reject invented additions.
  - Evidence: regression cases for grounding_capsule, autonomy_slice, reader keys.
- Finding: tolerant parsing fabricated missing stance fields, masking bad output.
  - Fix: direct ThoughtEventV1 validation; supply only transport-owned metadata.
  - Evidence: missing-field regression tests; all 49 shared tests pass.

## Rollout and limits

Rebuild/deploy orion-cortex-exec from the merged revision using the normal
worktree-safe deployment wrapper. Prompt is baked into that image. Reverting
the template change rolls it back; no data migration or schedule change exists.
No production deployment was performed by this patch. After rollout, verify
Stage 2 and its actual journal landing on the saved paper. End-to-end reading
completion remains UNVERIFIED; this patch is not a claim that every reading
failure or every long generation is fixed.

## Summary

- A reply that names a `correction_id` no `record_correction` result of the same turn returned is rewritten: the id becomes `[unverified]` and the reply ends with "Nothing was recorded."
- The governor applies the gate on every path that puts text in front of a reader: the finalized reply, the degraded passthrough, the draft preview, and the draft fields of the run record.
- `TurnFindings` keeps `record_correction` results only (not capped by turn length); `HarnessMotorResult.correction_results` carries them.

## Outcome moved

Live incident 2026-10-15, turn `5fdf2faf` (111 steps): Orion was asked to record a correction. The tool was never called. The repair step wrote a `record_correction(...)` call, `correction_id: corr_9_218_reproduce` (real ids are UUIDs) and a ring-share number nothing supported. The brief already says never to claim a recording without a returned id; nothing enforced it. The table stayed at 0 rows.

## Current architecture

Finalize (reflect, then `orion_response_repair`) writes the reply with no tool access. `FINALIZE_LOOP_TOOL_ALLOWLIST` is `{look_at_camera}`, so a `record_correction` call there is dropped with `harness_finalize_tool_loop_unrecognized_tool`, and the text proceeds anyway.

## Architecture touched

`orion/harness/claim_gate.py` (new, pure), `orion/harness/cut_short.py` (`TurnFindings.correction_results`), `orion/harness/runner.py` (`HarnessMotorResult.correction_results`), `services/orion-harness-governor/app/bus_listener.py` (`_gate_claims`).

## Files changed

- `orion/harness/claim_gate.py`: the gate
- `orion/harness/cut_short.py`, `orion/harness/runner.py`: carry the real results
- `services/orion-harness-governor/app/bus_listener.py`: apply on every text path
- tests: `orion/harness/tests/test_claim_gate.py`, `services/orion-harness-governor/tests/test_unverified_correction_claim.py`

## Schema / bus / API changes

None on the wire. `HarnessMotorResult` is in-process only (checked: never dumped to the bus or artifact).

## Env/config changes

None.

## Tests run

```text
orion/harness            551 passed
orion-harness-governor    85 passed
```

## Evals run

No eval harness for the governor. The incident reply is a verbatim fixture in `test_claim_gate.py`.

## Docker/build/smoke checks

Not run (worktree). After deploy: ask Orion to record a correction; a tool-returned id passes, an invented one is rewritten.

## Review findings fixed

- Finding: degraded passthrough (`SubstrateAppraisalUnavailableError`) skipped the gate.
  - Fix: `_gate_claims` on every `final_text` and `draft_text`. Evidence: `test_the_degraded_path_without_substrate_appraisal_is_gated_too`
- Finding: the draft preview reached the Hub ungated.
  - Fix: gate before publish. Evidence: `test_the_draft_preview_shown_to_the_hub_is_gated_before_it_is_published`
- Finding: prose like "each correction id should be recorded" was mangled.
  - Fix: require `:`/`=` between label and id, and a digit, `_` or `-` in the id. Evidence: `test_ordinary_prose_about_correction_ids_is_left_alone`
- Finding: a 400-entry cap, a 4000-char cap and secret scrubbing could drop the one real result.
  - Fix: keep `record_correction` results only, unscrubbed (an id and a status), capped at 50. Evidence: `test_a_long_turn_cannot_push_the_real_result_out`
- Finding: substring match let an invented prefix of a real id pass; any tool's echo counted.
  - Fix: whole-token match, `record_correction` results only. Evidence: `test_an_invented_prefix_of_a_real_id_does_not_pass`

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build   # from clean primary main after merge
```

## Risks / concerns

- Low: the gate checks one structured claim only. A reply that says "I recorded it" with no id is not caught. A prose detector would need a word list, which is banned here.
- Low: `draft_text` on the run record is now gated too, so the stored draft can carry the notice when it held an invented id.
- Not fixed here: the first failed attempt lost its task to a compaction at the lane's 32k window after 36 Bash calls. Separate problem.

## PR link

(see PR)

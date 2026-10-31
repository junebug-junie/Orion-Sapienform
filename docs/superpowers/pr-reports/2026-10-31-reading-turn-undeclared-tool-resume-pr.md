## Summary

- A reading turn whose answer ends on a call to a tool it was never given (Bash, an MCP plugin tool) is now resumed once on the same Claude session, with a note: only WebFetch and WebSearch exist here, answer from what you read.
- The detector fires only when the answer ENDS on a `<tool_call>` block, so a good JSON answer that quotes tool-call syntax is never thrown away.
- No resume when less than min(stall window, 60 s) is left on the turn deadline; the original final is returned instead.
- `WEBFETCH_LIMIT_NOTE` now tells the model that PDFs WebFetch says it "saved" cannot be opened in a reading.

## Outcome moved

Live, 2026-10-30 23:38 → 2026-10-31 03:39 UTC (after circe chat moved to Strata): 193 harness sessions, 8 ended with a call left as text, **all 8 were readings**. 7 called Bash (5 of them to pdftotext a PDF WebFetch had saved), 1 called an MCP plugin tool (`ctx_batch_execute`). 7 of 8 failed with "Could not parse JSON object from LLM text". Non-reading turns (chat, Hub Agent, durable runs) have the full tool set: 0 of 185 hit this.

## Current architecture

Reading turns spawn `claude -p` with `--tools WebFetch,WebSearch`. Strata only turns a call into a real tool_use when that tool was declared, so an undeclared call comes back as the turn's final text and the reading consumer fails to parse it.

## Architecture touched

- `orion/harness/fcc_motor.py`: `READING_ONLY_TOOLS`, `undeclared_tool_call`, `UNDECLARED_TOOL_NOTE`, `RESUME_MIN_BUDGET_SEC`; `build_claude_argv(resume_session_id=)`; resume-once path in `run_fcc_turn` (spawn path, reading_only only).
- `orion/world_pulse_read/fetch_text.py`: note text.

## Files changed

- `orion/harness/fcc_motor.py`: detector + resume-once.
- `orion/harness/tests/test_fcc_reading_undeclared_tool.py`: 8 tests.
- `orion/world_pulse_read/fetch_text.py`: saved-PDF sentence.
- `.github/workflows/orion-fcc-warm-pool-tests.yml`: runs the new test file.

## Schema / bus / API changes

- Added: final-frame metadata `fcc_undeclared_tool_resume = {tool, session_id, held_text}` (only on resumed turns).
- Behavior changed: resumed final's `duration_ms` covers both runs; spawn/first-event timings are the resumed process's own.
- Compatibility notes: additive metadata; no bus/schema change.

## Env/config changes

None.

## Tests run

```text
orion/harness/tests/test_fcc_reading_undeclared_tool.py: 8 passed
orion/harness/tests orion/world_pulse_read orion/fcc: 943 passed, 1 skipped, 1 failed
  (test_no_undiscovered_root_callers_of_claude_permission_argv -- fails on main too, pre-existing)
Mutation (trailing-call guard disabled): 2 failed -> tests pin it
```

## Evals run

No eval harness for reading-turn tool discipline. Live check after deploy (below) stands in.

## Docker/build/smoke checks

None run; code-only change in the harness governor image. Live verify after deploy: `docker logs orion-athena-harness-governor 2>&1 | grep fcc_reading_undeclared_tool_resume`, and the next PDF reading's final metadata carries `fcc_undeclared_tool_resume`.

## Review findings fixed

- Finding: detector fired on valid JSON answers quoting `<tool_call><function=...>`; the good answer would be discarded.
  - Fix: fire only when the answer ends on `</tool_call>`, and read the name from the last block.
  - Evidence: `test_a_quoting_answer_is_not_resumed`, detector asserts.
- Finding: resume with seconds left turned a diagnosable final into a bare timeout.
  - Fix: require min(stall, 60 s) left; keep `held_text` (500 chars) in metadata.
  - Evidence: `test_no_resume_when_the_deadline_is_nearly_spent`.
- Finding: resumed final's start-up timings included the whole first run.
  - Fix: resumed drive gets its own `started`; `duration_ms` restamped as the total.
- Finding: transcript stat between MCP config render and drive could leak the config.
  - Fix: measure the transcript first, fall back to 0 on error.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull   # post-pull hook rebuilds the harness governor
```

## Risks / concerns

- Severity: low
- Concern: a reading whose genuine final text ends with a quoted undeclared `</tool_call>` block gets one resume.
- Mitigation: once only; the resumed answer is final; tool name and held text logged.

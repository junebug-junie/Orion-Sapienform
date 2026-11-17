## Summary

- Root cause of Orion's stance step failing with a reply of just `{`: PR #3306 (merged 2026-11-13 06:50Z) added a generic user turn, "Carry out the task in the system message.", to every LLM hop that had none. Under that sentence the stance model puts 7-17% of its next-token probability on ending its turn right after `{`.
- Fix: a hop that asks for JSON now gets a user turn that names the answer: "Respond now with the JSON object the system message asks for." Prose hops keep #3306's sentence, so the Qwen3.5 fix that #3306 shipped still holds.
- The gateway's corrective retry no longer echoes a stopped `{` back to the model, which copied it. A reply of 8 characters or fewer is retried as a fresh call.
- New operator eval: `services/orion-cortex-exec/evals/eval_json_task_turn_end_after_brace.py` measures P(end after `{`) for both wordings on a stored request.

## Outcome moved

Every stance failure defers a real turn with no answer. That includes Juniper's chat turns, for example "el Coco just had his butt stitches removed, yay!" on 11-13 and 11-15. Brace-only stance failures per day (cockpit_turn_sighting, stage `stance_decision`):

- 11-08 to 11-12: 0, 0, 0, 1, 0.
- 11-13 to 11-17: 6, 11, 15, 6, 7. The first one was at 09:42Z on 11-13, after #3306 merged.

## Current architecture

cortex-exec `_build_hop_messages` gave every system-only hop the same generic user turn (#3306). The gateway, since #3355, retries a malformed Strata JSON reply once. That retry shows the model its bad reply, and when the bad reply was `{`, the model copied it.

## Root cause evidence (investigation agent; key claims re-checked by hand)

- **Strata's own records show a 2-token reply.** Every `{` failure in Strata's request records (`/metrics?requests=all` on :8015) is `output_tokens: 2`, `finish: stop`: the `{` plus the model's own end-of-turn token. Nothing went to reasoning or to tool calls, and no stop sequence is sent.
- **The logprobs show the model wanting to stop.** On the exact failing request (taken from the bus mirror), the token after `{` is `<|im_end|>` with probability 0.343. On normal JSON prompts it is 0.001-0.013. One replay reproduced `'{'` outright.
- **A back-to-back comparison on the same lane shows the sentence is the trigger.** P(end after `{`), on two real prompts:
  - "Carry out the task in the system message." (since #3306): 0.146-0.158 and 0.073-0.166.
  - System message only (before #3306): 0.067-0.073 and 0.009-0.016.
  - "Respond now with the JSON object the system message asks for." (this fix): 0.012-0.029 and 0.0014-0.014.
- **Ruled out:**
  - The parser or stripping, and stop strings: the reply really is 2 tokens.
  - Prompt length: failing prompts range from 3.8k to 28k tokens.
  - The Strata image, unchanged since 10-27.
  - The lane mix, unchanged since before 11-05.
  - #3355: it came after the onset, and Strata's own 502 carried the same text before it.
  - #3420: my earlier "single-quoted Python dict" diagnosis was wrong. `json.loads('{')` raises the identical error.

## Architecture touched

- `orion-cortex-exec`: hop message shape.
- `orion-llm-gateway`: Strata corrective retry.

## Files changed

- `services/orion-cortex-exec/app/executor.py`: `SYSTEM_TASK_TURN`, `JSON_TASK_TURN`, `_wants_json_output()` (reads `response_format`, `return_json`, `structured_output_schema`/`_method`), and `_build_hop_messages(json_output=)`.
- `services/orion-cortex-exec/tests/test_system_only_llm_messages.py`: 12 cases. They cover the real `call_step_services` path, the prose default, and that an existing user turn is left alone. The new cases fail on the old executor.
- `services/orion-cortex-exec/evals/eval_json_task_turn_end_after_brace.py`: operator eval.
- `services/orion-cortex-exec/README.md`: hop message shape section.
- `services/orion-llm-gateway/app/strata_structured.py`: a stopped reply is retried fresh, not echoed.
- `services/orion-llm-gateway/tests/test_strata_structured.py`: tests for the fresh retry.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the user turn on JSON hops, and the shape of the gateway retry for stopped replies.
- Compatibility notes: prose hops are unchanged.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: not needed
- skipped keys requiring operator action: none

## Tests run

```text
cortex-exec tests/test_system_only_llm_messages.py   12 passed (new cases fail on the old executor)
cortex-exec metacog-capture CI steps                  174 + 65 passed (agent run)
orion-llm-gateway tests                               563 passed
```

## Evals run

```text
eval_json_task_turn_end_after_brace.py, live, prompt 2: generic 0.155 / 0.166, JSON turn 0.0014 / 0.0019
```

## Docker/build/smoke checks

```text
No compose change. orion-cortex-exec and orion-llm-gateway rebuild on the post-pull run.
```

## Review findings fixed

- Finding (should): JSON detection missed the `structured_output_schema` / `structured_output_method` route.
  - Fix: covered.
  - Evidence: tests.
- Findings (nits): the `return_json` test bypassed the real path; the eval did not check the first token; the eval could overwrite a real user turn; the eval's scope was not stated.
  - Fix: all fixed.

## Restart required

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

```bash
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
```

The post-pull rebuild covers both.

## Risks / concerns

- Severity: medium
  - Concern: the rates come from a shared live lane, so readings move with traffic (0.045-0.41 for the same request when interleaved).
  - Mitigation: only back-to-back readings were compared. The live check below decides.
- Live check, 24 h after deploy: `brace_only_failures` should fall to about 0 (from 6-15 a day).

  ```sql
  select date_trunc('day', ts)::date day, count(*) stances,
         sum((raw::text ilike '%line 1 column 2 (char 1)%')::int) brace_only_failures
  from cockpit_turn_sighting where stage='stance_decision' and ts > now() - interval '7 days' group by 1 order by 1;
  ```
- Issues: #3477 superseded (its logging would only have re-found the 2-token reply; filed separately as a gap below).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3620

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## Summary

- The turn's request is also passed to the `claude` CLI as `--append-system-prompt`, quoted with `json.dumps`. System-prompt text is never part of the history that compaction summarizes.
- Only turns Juniper typed (`utterance_origin == "juniper"`) are anchored. Reading-only, outreach, curiosity and other turns get none.
- New env flag `HARNESS_FCC_TASK_ANCHOR_ENABLED` (default on).
- Trace: log `fcc_task_anchor chars=N`; `task_anchor_chars` on the final frame metadata.

## Outcome moved

Live incident 2026-10-15, run `28709ff7` (213 steps, six compactions): compaction summary 1 kept the task; summary 2 was empty (156 chars); every later summary described only what was left in the window, so the agent spent 170 steps on a different topic and never did the task. Measured across 14 days of CLI transcripts on the governor volume: 253 of 708 compaction summaries (36%) are under 400 characters ("under 400" is a rough cut-off, not a measured rule). A smoke test on the production model reproduced the cause: `automatic compaction failed: summarization produced empty response`.

## Current architecture

The request lives only in the conversation history (the `-p` prompt). A bad summary erases it.

## Architecture touched

`orion/harness/task_anchor.py` (new, pure), `orion/harness/runner.py` (builds the anchor from `request.user_message`), `orion/harness/fcc_motor.py` (`run_fcc_turn(task_anchor=)`, `build_claude_argv`, metadata), governor `.env_example` and compose.

## Files changed

- `orion/harness/task_anchor.py`: the anchor block, trusted-origin gate, JSON quote, cap
- `orion/harness/runner.py`, `orion/harness/fcc_motor.py`: pass it through to argv
- `services/orion-harness-governor/.env_example`, `docker-compose.yml`: flag
- `orion/harness/tests/test_task_anchor.py`: 28 tests

## Schema / bus / API changes

None.

## Env/config changes

- Added keys: `HARNESS_FCC_TASK_ANCHOR_ENABLED` (governor, default `true`).
- `.env_example` updated: yes. Local `.env` synced with `scripts/sync_local_env_from_example.py`: yes. Skipped keys: none.

## Tests run

```text
orion/harness            579 passed
orion-harness-governor    85 passed
scripts/check_env_template_parity.py   PASS
```

## Evals run

Live smoke (governor container, production model, Read-only tool, throwaway files): a final instruction that existed only in the anchor was followed (`ANCHOR-OK PLUM-5512`), so the CLI honours the appended system prompt. A successful compaction could not be forced: the CLI reported `summarization produced empty response`. **Survival through a real compaction is UNVERIFIED.** It follows from system prompts being rebuilt every request; it is not observed.

## Docker/build/smoke checks

Not deployed. After deploy, check: a Juniper-typed turn that spawns logs `fcc_task_anchor chars=...`.

## Review findings fixed

- Finding: prompt injection — only `reading_only` was exempt, so outreach/curiosity briefs (which can quote third-party text) would be promoted into the system prompt.
  - Fix: anchor only `utterance_origin == "juniper"`. Evidence: `test_only_turns_juniper_typed_are_anchored`, `test_outreach_and_curiosity_turns_get_no_anchor`
- Finding: the quote did not hold — a `"` in the request closed it.
  - Fix: `json.dumps`; NUL bytes stripped. Evidence: `test_a_request_cannot_close_the_quote_or_forge_a_second_request_line`
- Finding: may quote the wrong thing for non-chat turns (a brief's head is a header).
  - Fix: those turns are excluded until they have a dedicated task header.
- Finding: `--resume` could assert a stale task.
  - Not changed: a durable retry sends the same request, so the anchor equals the first attempt's. `test_the_anchor_rides_a_resumed_turn_too` documents it.
- Finding: comment said chat replies are skipped. Fix: comments now say warm-pool processes are not covered and Juniper turns that run on the spawn path are.
- Finding: weak tests. Fix: added escape, origin, argv-order-with-permissions and runner cases.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build   # from clean primary main after merge
```

## Risks / concerns

- Medium: warm-pool chat processes are not covered (fixed argv). The main Juniper chat path may mostly run warm; a turn that falls back to spawn is anchored. Covering warm turns needs a per-turn system message over the stream input. Follow-up.
- Medium: durable curiosity runs (the long hunts) get no anchor until the kickoff has a dedicated task header holding no third-party text. Follow-up.
- Low: the anchor text is visible in the process list, same as the existing `-p` prompt.
- Open: 36% empty summaries is a model/lane problem. Putting the summarizer on a stronger model, or smaller tool results, would help more broadly.

## PR link

(see PR)

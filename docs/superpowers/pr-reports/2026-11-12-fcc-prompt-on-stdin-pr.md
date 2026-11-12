## Summary

- Orion's 11-11 day letter was never written. All three tries failed the same way: the motor passed the ~131,300-character prompt to `claude -p` as one command-line argument, and Linux refuses any single argument over 131,072 bytes (`[Errno 7] Argument list too long`). The CLI never started.
- The spawn step (`_SpawnTurnIO`) now moves a prompt over 100,000 bytes to stdin. `claude -p` with no positional prompt reads it from there. Prompts under the limit keep today's path.
- Any unified turn with a prompt this large had the same failure. The letter was simply the first to hit it.

## Outcome moved

Turns with a prompt over 128 KB failed 100% before the model saw them. They now run.

## Current architecture

`build_claude_argv` puts the prompt after `-p`. `_SpawnTurnIO.start` execs that argv. The warm pool already sends prompts on stdin (`--input-format stream-json`) and is unchanged.

## Architecture touched

`orion/harness/fcc_motor.py` only: `ARGV_PROMPT_MAX_BYTES`, `split_oversized_prompt`, `_SpawnTurnIO`.

## Files changed

- `orion/harness/fcc_motor.py`: move an oversized prompt from argv to stdin at spawn.
- `orion/harness/tests/test_fcc_prompt_on_stdin.py`: regression tests, including a real 140 KB spawn.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
pytest orion/harness/tests/test_fcc_prompt_on_stdin.py -q -W error::ResourceWarning   7 passed
pytest orion/harness/tests -q                               846 passed
```

## Evals run

No eval harness covers spawn transport. Live proof instead (below).

## Docker/build/smoke checks

```text
143,063-byte prompt as argv:  [Errno 7] Argument list too long: 'claude'
same prompt on stdin: claude -p --output-format stream-json --verbose ... < big.txt  ->  result: success 'PONG'
Live failure evidence: harness_turn_trace 174a5c2e-1a73-5bfa-9244-8883a9f48cc7
  grounding_status "[Errno 7] Argument list too long", step_count 0; motor_boot prompt_char_len 131249
```

## Review findings fixed

- Finding: the stdin write blocked inside `start()`. A CLI that hung before reading its prompt would block forever, before the readline loop's stall timer and deadline could run.
  - Fix: the prompt is written by a background task. `start()` returns at once, the existing timers govern the turn, and `close()` cancels a writer that is still blocked.
  - Evidence: `test_start_returns_when_the_cli_never_reads_stdin` (child sleeps without reading; start returns, writer done after kill+close).
- Finding (test gap): no test for a CLI that exits without reading stdin.
  - Fix: `test_cli_exiting_without_reading_stdin_is_not_an_error` (BrokenPipe swallowed, exit code 3 reported).
- Nit: tests passed `env={}`. Now `dict(os.environ)`.
- Nit: comment now says the limit is per argument, not ARG_MAX.
- Not changed: a prompt that itself starts with `--` stays in argv. The CLI already misreads such a prompt, so this is not new.

## Restart required

Post-pull rebuild picks up the orion/ package change (harness governor, hub).

## Risks / concerns

- Severity: low. Concern: the CLI exits before reading stdin. Mitigation: broken pipe is caught; the CLI's own exit code and stderr report the failure as before.
- Severity: medium. Concern: the 11-11 letter run is closed as `not_written`; Prime does not retry it. It needs one manual re-order after deploy.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

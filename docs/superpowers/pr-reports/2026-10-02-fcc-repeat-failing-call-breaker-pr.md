## Summary

- Orion's harness turns now stop Orion from repeating a tool call that keeps failing the same way. After 3 identical failures in one turn, the next identical call is refused with a message: this exact call has failed N times with error X, it will not work, change approach or answer with what you have.
- Built as a harness-owned PreToolUse hook (a check Claude Code runs before every tool call), passed to `claude -p` with `--settings`, so it covers every tool: MCP tools, Bash, built-ins. It does not depend on the Context Mode plugin or on `--setting-sources`.
- State is the turn's own Claude Code session transcript, so it is per turn by construction and survives auto-compaction (the transcript file keeps the pre-compaction calls).
- A call that ever succeeded in the turn is never blocked; a different input is a different call; a successful file edit resets the counts (edit, then retry the same tests, is a fair retry). Bash's `description` label is ignored when comparing calls, because a live smoke showed the model numbering it per attempt ("attempt 1", "attempt 2") on identical commands.
- Configurable with `HARNESS_FCC_REPEAT_FAILURE_THRESHOLD` (default 3, `0` disables). When it fires, the governor logs `fcc_repeat_failure_breaker_fired corr=<id>`.

## Outcome moved

Urgent run a153451fe423 (2026-10-01, harness corr 6ddea5f1-a59d-5c31-bca1-bb06b6fd3bb2) made 19 identical `mcp__firecrawl__firecrawl_scrape` calls to `http://host.docker.internal:8080/api/cabinet/sensors/latest`, all failing with the same DNS error, about 10 of a 15-minute budget, and went back to the same call after auto-compaction. Replaying that transcript through the breaker lets 3 of those calls through and blocks the other 16. In a live container smoke that reproduced the same call, the model stopped retrying after the block and reported what it could not reach.

## Current architecture

`orion/harness/fcc_motor.py:run_fcc_turn` spawned `claude -p` with no harness-owned tool policy at all. The only PreToolUse hooks were the Context Mode plugin's (it redirects curl and similar). Repeated tool failures showed up only after the fact in step frames (`_extract_tool_result_errors`). Nothing stopped a loop while the turn was running.

## Architecture touched

- New: `orion/fcc/repeat_failure_breaker.py`, a stdlib-only hook script and pure decision functions.
- `orion/harness/fcc_motor.py`: installs the hook via `--settings` on every turn, including reading-only turns, and logs when a blocked result comes through the stream.
- Governor config: new env key (`.env_example`, `settings.py` mirror, README).

## Files changed

- `orion/fcc/repeat_failure_breaker.py`: the breaker (call key, transcript replay, decision, block message, `--settings` JSON, hook `main`).
- `orion/harness/fcc_motor.py`: `repeat_failure_threshold()`, `repeat_failure_breaker_argv()`, `_log_repeat_failure_blocks()`, wiring in `run_fcc_turn`.
- `orion/fcc/tests/test_repeat_failure_breaker.py` and `orion/fcc/tests/fixtures/fcc_repeat_failure_a153451fe423.jsonl`: replay of the real incident (tool blocks only, trimmed results) plus unit rules.
- `orion/harness/tests/test_fcc_motor_repeat_failure_breaker.py`: argv wiring, env threshold, disable, and the fire log line with the corr id.
- `services/orion-harness-governor/.env_example`, `app/settings.py`, `README.md`: new key and documentation.

## Schema / bus / API changes

- Added: none. No bus or grammar event. The trace is a log line plus the blocked tool_result, which already rides the normal step frames (marker `[orion-repeat-failure-breaker]`).
- Removed / Renamed: none.
- Behavior changed: identical tool calls that already failed 3 times in a turn are refused.
- Compatibility notes: none. The hook fails open: any error inside it lets the call through.

## Env/config changes

- Added keys: `HARNESS_FCC_REPEAT_FAILURE_THRESHOLD=3` (governor).
- Removed keys / Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (key present in `services/orion-harness-governor/.env` in the primary checkout).
- skipped keys requiring operator action: none.

## Tests run

```text
python -m pytest orion/fcc/tests/test_repeat_failure_breaker.py orion/harness/tests/test_fcc_motor_repeat_failure_breaker.py -q  -> all pass
python -m pytest orion/harness/tests orion/fcc/tests -q -> all pass except test_claude_spawn.py::test_no_undiscovered_root_callers_of_claude_permission_argv, which fails identically on main (orion-room-companion caller, unrelated)
python scripts/check_env_template_parity.py -> PASS
```

## Evals run

```text
The incident replay (test_replay_cuts_nineteen_identical_failures_to_three) is the eval: real transcript in, 3 allowed / 16 blocked out. No separate eval harness was added.
```

## Docker/build/smoke checks

Live smokes inside the running `orion-athena-harness-governor` (Claude Code 2.1.286, bypassPermissions, `--setting-sources user,local`, local `harness` model). The breaker script was copied to the container's /tmp for the smoke only and removed afterwards. Nothing was deployed.

```text
Bash, threshold 2, first attempt: NOT blocked. The model labeled each identical command "(attempt N)" in Bash's description field.
  -> fixed by ignoring Bash `description` in the call key; regression test added.
Bash, threshold 2, after the fix: calls 1-2 ran and failed; calls 3-4 were refused:
  "PreToolUse:Bash hook error: [...]: [orion-repeat-failure-breaker] Blocked: this exact Bash call (same input) has already failed 2 times in this turn ..."
MCP, the incident call itself (firecrawl_scrape of host.docker.internal sensors/latest), threshold 2:
  calls 1-2 failed with DNS; call 3 was refused; the model then stopped and said the breaker blocked it.
```

## Review findings fixed

Review ran in a subagent. It found no must-fix issues.

- Finding (should-fix): an edit-then-retry loop on a coding turn (tests fail, Edit, the same tests fail again) would be blocked for the rest of the turn even though the code changed each time.
  - Fix: a successful Edit, Write, MultiEdit or NotebookEdit resets all failure counts. A failed edit or an unrelated successful read does not reset them.
  - Evidence: `test_successful_edit_resets_failure_counts`. The incident replay still gives 3 allowed and 16 blocked.
- Finding (nit): a bad `--threshold` argument would make argparse exit with code 2, which is the same code that blocks a call, so every tool call would be blocked.
  - Fix: `parse_args` is now inside the fail-open path.
  - Evidence: fail-open test extended.
- Finding (nit): the hook parsed every line of a large transcript on every tool call.
  - Fix: lines that do not mention `tool_use` or `tool_result` are skipped before parsing.
- Finding (nit): setting `HARNESS_FCC_REPEAT_FAILURE_THRESHOLD=` to an empty value would fail the governor's settings validation at boot, while the motor treats empty as "use the default".
  - Fix: the `settings.py` mirror field is now a string.
- Finding (nit): two things are out of scope and were not said anywhere: subagent calls (written to separate transcripts) and MCP tools that report failure only in text.
  - Fix: both are now stated in the module docstring.
- Not changed (nit): if the stream ever replays a blocked result, the fire log line could be duplicated. It is trace-only, so this is low risk.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: low. Concern: each tool call now starts one short Python process that re-reads the session transcript (tens of ms on normal transcripts). Mitigation: stdlib-only, fail-open; set `HARNESS_FCC_REPEAT_FAILURE_THRESHOLD=0` to turn it off.
- Severity: low. Concern: parallel identical calls in the same model message all go through, because none of them has a result yet. Only later rounds are blocked.
- Severity: low. Concern: Hub's own agent-claude bridge (outside the harness) does not get this hook. Out of scope here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2467

🤖 Generated with [Claude Code](https://claude.com/claude-code)

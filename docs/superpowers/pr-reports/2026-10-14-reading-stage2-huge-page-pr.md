## Summary

- A reading turn can no longer WebFetch the same page twice. WebFetch only ever reads a page's first ~100k characters, so a repeat (or a request for a "slice") returns the same text and costs another multi-minute summarizer pass on the single agent-lane slot. GitHub raw spellings of one file count as one URL. Reading-only turns only (`orion/fcc/repeat_failure_breaker.py`, `--refetch-limit`).
- After half of a reading turn's budget, new tool calls are refused and the model is told to answer with what it has (`--tool-window-fraction`). A tool already running is not stopped.
- Hub's reading listener now tells the harness the turn's real limit (`inference_timeout_sec` + reply deadline, the keys curiosity's held turns already use). Before, the motor ran on `HARNESS_FCC_TIMEOUT_SEC=7200` inside a 3500 s durable attempt, so the motor's own deadline, the sandbox clock and the tool window all pointed past the real cutoff.
- Stage 1 and Stage 2 prompts state the WebFetch limit in one line. A Stage 2 retry whose previous attempt timed out is told not to fetch at all. Before, attempts 2 and 3 got byte-identical prompts.
- Regression fixture of the real turn plus a replay eval.

## Outcome moved

Seed `reading:c31227dc-...` Stage 2 (2026-10-14, turn 98c6a963) spent 54 min and ended with `draft_len=0`. Replaying that turn's real tool calls through the new rules: 4 of 8 calls are refused (two repeat fetches, the `refs/heads` spelling of the same file, and one late fetch). The last tool result the model waits for lands at 1728 s of a 3241 s budget, leaving more than 20 min to write the JSON. Before, nothing was refused and the turn was cut at 3251 s with nothing written.

## Current architecture

Root cause, from the live records:
- Stage 1 read the GitHub *blob* page. Its retained text (`reading_document_snapshot`, 5,544 chars) is page chrome: "zero lines of C source code". The prompt was not too big: the stored Stage 2 prompt is 20,879 chars, and #2581's retained text is already capped at 16k chars in the prompt.
- Stage 2 then fetched the 222 KB raw file itself. It asked for "characters 100000 to 200000" twice, fetched the `.h` twice, and fetched the `refs/heads` spelling of the `.c`. Each fetch handed the first ~100k characters (a 42.2k-token prompt, the same size every time in the circe :8015 llama.cpp log) to the single slot. Waits ran 5 to 19 min.
- The ~154 `tool=none` steps every 30 s are Claude Code `tool_progress` heartbeats while those fetches waited. They are not empty model turns. They also keep feeding the 420 s stream-stall watchdog, so the only thing that stopped the turn was the 3500 s attempt timeout.
- Thinking: the lane's decodes ran 1k to 14k tokens at 20 to 30 tok/s. That makes each wait longer but is not what made the turn fail. Not changed here.

## Architecture touched

- orion-harness-governor: the FCC PreToolUse hook (`orion/fcc/repeat_failure_breaker.py`) and its argv (`orion/harness/fcc_motor.py`).
- orion-hub: `reading_turn_listener.py` (turn budget), `world_pulse_read_stage2.py` (prompt note, retry note, prior-error read), `world_pulse_read_pipeline.py` (Stage 1 note).
- No schema, bus, or API changes. `HarnessRunRequestV1.inference_timeout_sec` / `reply_budget_sec` already exist.

## Files changed

- `orion/fcc/repeat_failure_breaker.py`: refetch and tool-window rules, `canonical_fetch_url`, CLI flags.
- `orion/harness/fcc_motor.py`: passes the reading rules on `reading_only` turns, with env knobs.
- `services/orion-hub/scripts/reading_turn_listener.py`: `stamp_reading_turn_budget`.
- `services/orion-hub/scripts/world_pulse_read_stage2.py`: `retry_note`, `_prior_error`, prompt note.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py`, `orion/world_pulse_read/fetch_text.py`: shared `WEBFETCH_LIMIT_NOTE`.
- `services/orion-harness-governor/{.env_example,app/settings.py,README.md}`: the two new keys.
- Tests: `orion/fcc/tests/test_repeat_failure_breaker.py` (+fixture), `orion/harness/tests/test_fcc_motor_repeat_failure_breaker.py`, `services/orion-hub/tests/test_reading_turn_listener.py`, `services/orion-hub/tests/test_world_pulse_read_stage2.py`.
- Eval: `services/orion-hub/evals/run_reading_huge_page_replay.py`, `test_reading_huge_page_eval.py`, fixture `reading_huge_page_c31227dc_2026-10-14.json`.
- `.github/workflows/orion-reading-tests.yml`: runs the breaker tests and the new eval.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: reading-only FCC turns refuse repeat WebFetches and late tool calls. Reading turns now carry their real budget to the harness.
- Compatibility notes: the hook flags are only passed by the new governor. Deploy order does not matter.

## Env/config changes

- Added keys: `HARNESS_FCC_READING_REFETCH_LIMIT=1`, `HARNESS_FCC_READING_TOOL_WINDOW_FRACTION=0.5` (orion-harness-governor).
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. Both keys are present in the primary checkout's `services/orion-harness-governor/.env`.
- Skipped keys requiring operator action: none.

## Tests run

```text
reading CI set (hub reading + world_pulse_read + orion/fcc + orion/harness tests): 1339 passed, 76 skipped, 1 failed (before review fixes)
after review fixes: breaker + harness + hub reading/stage2/curiosity/orchestrator + eval: 994 passed, 28 skipped; curiosity held-turn finalize 12 passed
  failure = orion/fcc/tests/test_claude_spawn.py::test_no_undiscovered_root_callers_of_claude_permission_argv, also fails on main (untouched files)
orion-harness-governor tests: 77 passed
```

## Evals run

```text
services/orion-hub/evals reading evals incl. test_reading_huge_page_eval.py: 40 passed
run_reading_huge_page_replay.py:
  prompt: stored 20,879 chars (attempts 2 and 3 identical); note +376 chars; retry note +259 chars
  before (7200 s budget, no rules): blocked [] ; turn cut at 3251 s, no draft
  after  (3241 s budget, refetch 1, window 0.5): blocked [t20,t21 refetch; t99,t100 window]; last result 1728 s
```

## Docker/build/smoke checks

```text
Not built or deployed. The live path is UNVERIFIED until the next reading turn runs on the new governor and hub.
Expected trace: governor log `fcc_repeat_failure_breaker_fired corr=<reading turn> detail=...already fetched...` or `...tool window...`
```

## Review findings fixed

- Finding: the 60 s reserve was spent before stance/recall. After a slow stance (249 s live), the motor got the whole reply budget, which left finalize/reply 0 s.
  - Fix: listener stamps `harness_finalize_reserve_sec`; `_held_turn_budgets` re-applies it after measuring the time already used. Curiosity's held turns don't set it and are unchanged.
  - Evidence: `test_reading_turn_budget_is_the_briefs_wall_not_the_harness_default` asserts `reply - inference == 60` at 249 s, and that no-reserve callers are unchanged; `test_curiosity_held_turn_finalize.py` 12 passed.
- Finding: the block message and prompt note said a refetch "returns the same text", but WebFetch returns a summarizer's answer to each prompt.
  - Fix: reworded to "reads the same characters; ask everything you need in one prompt". The README states the trade-off (a differently worded second question is refused too).
- Finding: the no-fetch retry note matched any `TimeoutError`.
  - Fix: narrowed to `turn_exception:TimeoutError` (durable-runs' attempt timeout, the live failure), the listener's and motor's own timeout codes, and `workflow_deadline`. Dropped the unproven "while still fetching". Test pins that `reading_binding_unavailable:TimeoutError` gets no retry note.
- Finding: `canonical_fetch_url` dropped the port and kept http/https as separate keys.
  - Fix: key on netloc (port kept); http is folded into https. Tests added.
- Finding: two same-URL fetches in one assistant message both run.
  - Fix: documented, not fixed. Counting pending calls risks blocking every first fetch, because the current call may already be in the transcript.
- Not changed (nits): the hardcoded 60 s reserve (no tuning need yet); no backoff after the window closes; no end-to-end env assertion across listener -> runner -> subprocess (each hop is tested separately).

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-harness-governor up -d --build && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: should. Concern: the tool window is a fixed share of the budget. A source that really needs a late fetch now goes without it. Mitigation: set the fraction to 0 to turn it off, or raise it. On the last 14 days of reading turns, the latest tool call in a successful turn started at about 25 min, inside the 27 min window.
- Severity: should. Concern: Stage 1 keeps "reading" a GitHub blob URL as page chrome. That is the upstream reason Stage 2 went fetching. Mitigation: follow-up, not in this PR.
- Severity: nit. Concern: WebFetch summarizer passes still think at length on the 27B lane. Mitigation: out of scope; repeat passes are what got cut.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2672

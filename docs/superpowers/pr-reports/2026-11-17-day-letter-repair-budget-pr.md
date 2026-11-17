# fix(harness): size the response-repair budget from the draft

Fixes #3607. Refs #3439.

## Summary

The day letter's first attempt failed on 3 of the last 6 days. The cause was not the letter. A check step after the letter (the "repair pass") rewrites the whole reply, and it had a fixed limit of 3,072 tokens. A letter is about 5,400 to 6,900 tokens. The rewrite was cut off every time, so the letter step failed and the retry spent about 1 hour of GPU time.

- The repair limit now grows with the draft. A short reply keeps the old limit of 3,072 tokens. A letter gets room for the whole rewrite.
- On a turn with a reply deadline (held curiosity runs), the limit also stays within what the time left can produce. It never goes below 3,072.
- If a repair is still cut off, and the judge accepted the draft's content, Orion sends the draft. The trace says so: the repair reason ends with `_kept_draft:truncated`.
- If the judge rejected the draft (misaligned, uncertain, an unsupported claim, or the judge itself failed), a cut-off repair still fails the turn, as before. The retry path still runs.

## Outcome moved

- Before: every letter repair stopped at 3,072 tokens and failed (4 of 4 letter repairs, 11-16 and 11-17). Each failure cost about 1 h of GPU time.
- After: a 24,118-character letter (the longest live first attempt) gets a 10,562-token limit. Its real size is 6,827 tokens.

## What I found first (live, read-only)

**When repair runs on a letter.** The 5b judge sends a draft to repair when it says `misaligned`, `uncertain`, or `strain_unresolved` (open strain remains) (`needs_response_repair`, `orion/harness/finalize.py`). For the 6 letter turns that reached the judge (14 days): 4 went to repair, and all 4 failed. 3 of the 4 were `aligned` with `strain_unresolved=true`. 1 was `misaligned` with an unsupported claim (9061adc5). The 2 letters that passed were `aligned` with no open strain.

**How often repair runs elsewhere.** 30 days of `harness_turn_trace`: 524 repair runs. 244 of them (47%) were `aligned` + `strain_unresolved` only. Only 8 of 524 had a draft longer than 6,144 characters, so the new limit changes almost nothing for chat turns. `llm_call_usage`, 14 days: 332 repair calls, mean 546 completion tokens. 5 hit the 3,072 limit: the 4 letters and 1 long chat turn (ef880ff1, 14,297 chars, 11-13).

**Other long-form workflows that can reach repair.** `session_roundup:*` (drafts up to 10.4k chars; one repair ran at 8.4k chars), `curiosity_investigation` (up to 7.1k), `world_pulse_read` (up to 13k, none triggered in 30 days), long chat turns. `world_pulse_read_stage2` and other reading-only turns skip prose repair (structured output is preserved).

**Thinking mode.** The repair call already sends `enable_thinking: false`. The gateway forwards it to Strata (`OPENAI_COMPAT_BACKENDS` includes `strata`). All 3 truncated letter repairs on 11-17 logged `reasoning_len=0`, so all 3,072 tokens were answer text. Thinking was not the cause; the limit was.

**Lane and window.** The letter holds a GPU lease, so the repair runs on the hold (`resolve_finalize_llm_lane`). Live: route `agent`, model `qwen3.8-flash-next-iq3_s` on Strata, 131,072-token window. Letter repair prompts were 38,465 to 42,125 tokens.

**Token size of a letter.** Measured with a Qwen tokenizer on the 6 live letter drafts: 5,408 to 6,905 tokens, 3.07 to 3.55 characters per token. Hex ids like `[arc:46e924094d7a7c32]` and markdown make letters denser than plain prose.

## Current architecture

`run_harness_finalize_chain` runs the 5b judge, then `run_orion_response_repair` when the verdict needs it. The repair call asked for `max_tokens=3072` (`RESPONSE_REPAIR_MAX_TOKENS`). `extract_response_repair_text` refuses a reply cut at `finish_reason=length`, which became `HarnessFinalizeFailedError` and a failed turn.

## Architecture touched

- `orion/harness/finalize.py` only. No service, schema, bus, or env change.

## Files changed

- `orion/harness/finalize.py`: `response_repair_max_tokens(draft, seconds_left=)` sizes the limit; the repair call passes the turn's reply deadline; `ResponseRepairTruncatedError` names the cut-off case; `repair_may_keep_draft_on_truncation(reflection)` decides when the draft may ship; `run_orion_response_repair` uses both.
- `orion/harness/tests/test_response_repair_budget.py`: regression tests.
- `docs/superpowers/pr-reports/2026-11-17-day-letter-repair-budget-pr.md`: this report.

## How the limit is set, and why

`limit = ceil(ceil(chars / 3.0) x 1.25) + 512`, never below 3,072 and never above 12,288.

- 3.0 characters per token is below the lowest live letter ratio (3.07), so the estimate is high, not low.
- 1.25 gives room for a rewrite that is a little longer than the draft. 512 is a fixed margin.
- 3,072 stays the floor. Any draft under about 6,100 characters gets exactly 3,072, as today.
- 12,288 is the ceiling. It covers a draft of about 28,000 characters (the longest live letter is 24,118). Two limits set it:
  - The lane's window. With the largest live letter repair prompt (42,125 tokens), the total stays far inside the smallest window the repair can run on (131,072). The finalize code does not know the lane's window at call time, so the ceiling is chosen against the smallest one.
  - The repair's timeout (540 s). A timeout fails the turn outright. At a slow 35 tokens per second plus 60 s to read the prompt, 12,288 tokens take about 410 s. Live repairs on 11-17 ran at about 115 tokens per second. A test reads both timeouts from the verb file and the governor `.env_example` and fails if the ceiling outgrows them.
- With a reply deadline: `limit = min(limit, (seconds_left - 60) x 35)`, never below 3,072. Before this PR every repair asked for 3,072, so a held turn is never worse off than before.

## Why the keep-draft path exists

The limit can still fail: a draft over about 28,000 characters, a turn whose reply deadline clamps the limit, or a rewrite much longer than the draft. In that case:

- If the judge said `aligned`, the judge did not fail, and there is no unsupported claim, the judge accepted the draft's content. The repair was asked only to ease open strain. Sending the draft is safe. The trace records it (`response_repair_reason` = `strain_unresolved_kept_draft:truncated`), using the same field the existing `no_change` and `scaffold_echo` keep-draft paths use.
- In every other case the turn fails, as before. Sending a draft the judge rejected would be a silent fallback.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the repair call's `max_tokens` grows with the draft (floor 3,072). A new `response_repair_reason` suffix value, `_kept_draft:truncated`.
- Compatibility notes: `ResponseRepairTruncatedError` subclasses `ValueError`, so existing `except ValueError` handlers still catch it.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/harness/tests/test_response_repair_budget.py      17 passed
  (with the old 3072 limit patched back in: the 3 letter tests fail with
   "orion_response_repair reply truncated at max_tokens (finish_reason=length)",
   the exact live failure)
pytest orion/harness                                             960 passed
pytest orion/cognition/tests/test_cortex_payload_extract.py      11 passed
pytest services/orion-harness-governor/tests                     106 passed, 1 failed
  (test_fcc_warm_pool_wiring: fails the same way on main -- PermissionError
   reading ~/.fcc/.env on this host; unrelated)
```

## Evals run

```text
orion/harness/evals ran inside `pytest orion/harness` (960 passed).
No eval measures repair length on long drafts. The live check below is the proof.
```

## Docker/build/smoke checks

```text
No image or compose change. Not run.
```

## Review findings fixed

A code-review subagent reviewed commit 83202a4b3. It found nothing that had to be fixed. It found two should-fix items and five small items. All are fixed.

- Finding (should-fix): the 16,384 ceiling was not tied to the 540 s repair timeout. A timeout fails the turn, and does not take the keep-draft path.
  - Fix: the ceiling is now 12,288. `test_a_ceiling_size_repair_finishes_inside_the_repair_timeout` reads the verb timeout and `RESPONSE_REPAIR_TIMEOUT_SEC` and checks that ceiling / 35 tok/s + 60 s stays 60 s under both.
  - Evidence: the test passes (about 411 s against 540 s).
- Finding (should-fix): on held turns with a reply deadline, a longer repair can run out the deadline. The deadline path sends the draft whatever the verdict, so a rejected draft could ship.
  - Fix: `run_orion_response_repair` takes `deadline_monotonic`, and both repair calls (the main one and the re-check's second one) pass it. The limit is clamped to what the time left can produce, never below 3,072 (the old limit).
  - Evidence: `test_a_reply_budget_clamps_the_repair_but_never_below_3072`, `test_the_chain_passes_the_reply_budget_to_the_repair`.
- Finding (nit): `if kept_draft: pass / elif` was hard to read.
  - Fix: the `no_change` and `scaffold_echo` checks now sit under `if kept_draft is None:`.
  - Evidence: 960 harness tests pass, including the existing `no_change` and `scaffold_echo` tests.
- Finding (nit): a stale comment said "3072 starves".
  - Fix: the comment now names the sized budget, Strata, and the keep-draft case.
  - Evidence: diff.
- Finding (nit): the judge-failed guard was not tested.
  - Fix: `test_only_a_judge_approved_draft_may_be_kept_after_a_cut_off_repair` covers aligned + judge-failed, misaligned, uncertain, and an unsupported claim.
  - Evidence: the test passes.
- Finding (nit): nothing pinned the re-check path when the second repair is cut off.
  - Fix: `test_a_cut_off_second_repair_in_the_recheck_keeps_the_first_repair`. The turn sends the first repair, and the reason is `unsupported_claim_unchecked`.
  - Evidence: the test passes.
- Finding (nit): this report was not committed.
  - Fix: committed with the branch.
  - Evidence: `git log`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

The finalize code runs in the harness governor. Deploy from the clean primary `main` checkout after merge.

## Live check after deploy

1. After the next day letter (about 07:00Z), run:
   `docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select correlation_id, created_at, outcome_molecule->>'failure_reason', outcome_molecule->>'response_repair_reason' from harness_turn_trace where workflow='orion_day_letter' and created_at > '<deploy time>' order by created_at"`
   Expect: no row with `truncated at max_tokens`.
2. When a letter goes to repair, check the call: `select correlation_id, completion_tokens from llm_call_usage where verb='orion_response_repair' and created_at > '<deploy time>' order by created_at`. Expect: a letter repair may use more than 3,072 tokens, and stops before its limit.
3. #3607 closes after 7 letters in a row with no `no_final_frame` first attempt.

## Risks / concerns

- Severity: low. Concern: a letter repair now takes longer (up to about 10,600 tokens at about 80 to 115 tokens per second, so about 1.5 to 2 minutes more). Mitigation: the ceiling is pinned below the 540 s repair timeout by a test, and letters run without a reply deadline.
- Severity: low. Concern: about half of all repair runs (244 of 524 in 30 days) rewrite drafts the judge called aligned, only because strain was open. That is existing behavior and out of scope here. Mitigation: filed as a design ticket (#3618).
- Issues opened: #3618. Issues closed: #3607 (on merge, after the live check).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3625

🤖 Generated with [Claude Code](https://claude.com/claude-code)

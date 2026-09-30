# Compactors: full previous day, full-text journal body, GitHub compactor reliability

## Summary

- The GitHub and chat compactors now summarize the **whole previous Denver calendar day** on their scheduled run (every merged PR, every chat turn). Before, GitHub read a rolling 24h window from one page of results and kept at most 32 PRs, and chat kept only the newest 30 turns.
- The journal body is stored **in full**. The 8000-char (GitHub) and 4000-char (chat) trims are gone, because the text will be embedded verbatim in the daily HTML letter. Only the memory-card summary and title stay capped.
- A busy day is split into chunks sized to fit the model's context. Each chunk is digested, then one merge call joins them. If the merge fails, would overflow the context, or drops PRs/turns the chunks covered, the chunk digests are joined as-is and the run is marked `concatenated`.
- GitHub reliability fixes:
  - explicit `agent` route with one retry;
  - an empty completion is retried instead of failing the day;
  - JSON with raw newlines/tabs inside strings now parses;
  - digest calls get a 16k-token completion budget.
- The actions scheduler used to give up after 420s on a pass that could legitimately run longer, then retry it. Scheduled compactor passes now get a 3600s wait.

## Outcome moved

- **GitHub coverage:** was up to 32 PRs from one API page, with each PR body cut to 1500 chars. Now it is every PR merged in the day (the API is paginated), with bodies up to 30k chars. Live bodies (2026-09-26..29) have a median of about 9-10k chars and a max of 38.6k.
- **Chat coverage:** was the newest 30 turns. Now it is every turn in the day (ceiling 5000; live volume is 5-14 per day).
- **Failure modes addressed**, from `workflow_schedules.json` in `orion-athena-actions`:

  | Failure seen live | How often | Fix |
  |---|---|---|
  | `invalid_json:Expecting value` (empty completion) | about 7 | retried |
  | `invalid_json:Invalid control character` | 2 | tolerant parse |
  | `structured_output_rejected` (JSON cut off at the token limit) | about 9 | 16k max_tokens, smaller per-call input, retry |
  | actions RPC timeout at 420s | — | 3600s for compactor passes |

- **New run evidence** in workflow metadata (both passes): `total_count`, `covered_count`, `input_truncated`, `digest_chunk_count`, `digest_merge_mode`, `digest_merge_skipped_reason`, `digest_llm_route`, `digest_attempts`, `journal_body_chars`. GitHub also records `window_mode/start/end`, `github_pages_fetched`, `github_page_cap_hit`, `fetch_window_unconfirmed`, and `truncated_pr_numbers`.

## Current architecture

- **GitHub fetch:** `github_compactor_pass` (cortex-orch) calls `skills.repo.github_recent_prs.v1` (cortex-exec). That skill made a single `pulls?per_page=100` request filtered to `now - lookback_days`.
- **GitHub digest:** `trim_github_compactor_input` capped the input at 32 PRs × 1500 chars. The digest call had no explicit route, no retry, and strict `json.loads`. `fit_digest_within_budget` trimmed `journal_body` to 8000.
- **Chat fetch:** `chat_history_compactor_pass` asked `skills.chat.discussion_window.v1` for `max_turns=30`. The SQL was `ORDER BY ASC LIMIT 500`, which kept the *oldest* 500 rows and then the newest 30 of those.
- **Chat digest:** the route loop was `("chat","quick")`, which used Juniper's reserved Hub lane (allow-listed in the poacher gate). The journal body was trimmed to 4000.
- **Actions:** scheduled workflow dispatch waited `ACTIONS_EXEC_TIMEOUT_SECONDS=420`.

## Architecture touched

- `orion/cognition/compactor/` (new shared seams):
  - `constants.py`: routes and budgets
  - `calendar_day.py`: the one definition of "yesterday, Denver"
  - `chunking.py`: budget measured the way the prompt renders it
  - `digest.py`: tolerant parse, empty-completion token
- GitHub/chat compactor packages: input builders now chunk instead of trimming, plus merge-input and concatenation helpers. Added a GitHub window resolver.
- cortex-orch `workflow_runtime.py`: one shared `_run_compactor_digest` (single call, or map-reduce) and `_call_compactor_digest_with_retry` used by both passes.
- cortex-exec `verb_adapters.py`: the GitHub skill paginates, accepts `window_start_utc`/`window_end_utc`, and de-duplicates PRs across pages.
- `orion/discussion_window/sql_fetch.py`: newest-first row limit scaled to `max_turns`.
- orion-actions: the 3600s wait applies only to compactor workflows (other scheduled workflows keep 420s); the claim TTL is raised to match.

## Files changed

- `orion/cognition/compactor/{constants,calendar_day,chunking,digest}.py`, `README.md`: shared budgets, day window, chunking, parse.
- `orion/cognition/github_compactor/{constants,digest,window}.py`: no count cap, 30k body safety cap, chunked inputs, merge/concat, calendar-day window.
- `orion/cognition/chat_history_compactor/{constants,digest,window}.py`: all turns, larger per-turn safety caps, chunked inputs, merge/concat.
- `orion/cognition/prompts/*_compactor_digest_v1.j2`: chunk and merge modes; journal body has no length limit.
- `orion/cognition/verbs/chat_history_compactor_digest_v1.yaml`: timeout 90s → 600s. The GitHub verb yaml changed comments only.
- `orion/schemas/discussion_window.py`: `max_turns` `le` 200 → 5000.
- `orion/discussion_window/sql_fetch.py`: `ORDER BY DESC LIMIT :row_limit`, then reversed.
- `services/orion-cortex-orch/app/workflow_runtime.py`: map-reduce, retry, window, metadata.
- `services/orion-cortex-exec/app/verb_adapters.py`: pagination, window, dedupe, `body_truncated`.
- `services/orion-actions/app/{main,settings}.py`, `.env_example`, `README.md`: compactor dispatch timeout, claim TTL.
- `scripts/check_chat_route_poachers.py`: removed the now-stale chat-compactor allow entry.
- `scripts/analysis/measure_pr_lifecycle.py`, `orion/structural_mass/tests/test_pr_lifecycle.py`: inline the retired `MAX_DIGEST_INPUT_PRS=32` as the historical reference.
- Tests: `services/orion-cortex-orch/tests/test_compactor_full_window.py` (new), `services/orion-actions/tests/test_workflow_dispatch_timeout.py` (new), `orion/cognition/github_compactor/tests/test_window.py` (new), plus updated compactor, discussion-window, exec skill, and orch lane tests and the chat digest eval.

## Schema / bus / API changes

- **Added:**
  - GitHub skill args `window_start_utc` / `window_end_utc`.
  - GitHub skill result fields `window_mode`, `window_start_utc`, `window_end_utc`, `pages_fetched`, `page_cap_hit`, and per-item `body_truncated`. These live in the skill's `final_text` JSON, not a registered bus schema.
  - Workflow metadata fields listed above.
- **Removed:** `MAX_DIGEST_INPUT_PRS`, `DIGEST_INPUT_BODY_MAX_CHARS`, both `JOURNAL_BODY_MAX_CHARS`, `DEFAULT_MAX_TURNS` (chat compactor), `trim_github_compactor_input`, `trim_chat_history_compactor_input`.
- **Behavior changed:**
  - `DiscussionWindowRequestV1.max_turns` accepts up to 5000 (this is an `extra="forbid"` model).
  - A scheduled GitHub run covers the previous Denver day, and its journal date label is that day instead of today's UTC date.
  - An on-demand GitHub run is unchanged (rolling).
- **Compatibility / deploy order:**
  - **Deploy cortex-exec before cortex-orch.** An old exec rejects `max_turns=5000` (`le=200`), which would fail every chat compactor run until exec catches up.
  - New orch + old exec on GitHub: the orch widens `lookback_days` to 2 in day mode and re-filters to the day, but the old exec does not paginate. The run is flagged `fetch_window_unconfirmed` / `input_truncated=true`.

## Env/config changes

- Added keys: `ACTIONS_WORKFLOW_DISPATCH_TIMEOUT_SECONDS=3600` (orion-actions).
- Removed / renamed keys: none.
- `.env_example` updated: yes (`services/orion-actions/.env_example`).
- Local `.env` synced: yes. `python scripts/sync_local_env_from_example.py --all-keys orion-actions` was needed, because the default run skips keys outside its sync prefixes. Verified: `/mnt/scripts/Orion-Sapienform/services/orion-actions/.env:170 ACTIONS_WORKFLOW_DISPATCH_TIMEOUT_SECONDS=3600`.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/cognition orion/discussion_window orion/structural_mass services/orion-cortex-orch/evals \
  services/orion-cortex-orch/tests/test_compactor_full_window.py services/orion-cortex-orch/tests/test_workflow_lane.py \
  tests/test_github_compactor_memory_cards.py tests/test_indexed_compactor_memory_cards.py tests/test_chat_workflow_registry.py \
  tests/scripts/test_sync_local_env_from_example.py scripts/tests/test_check_env_template_parity.py      -> 265 passed
(cd services/orion-cortex-exec) pytest tests/test_skill_verbs.py tests/test_router_final_text_assembly.py -> 79 passed
(cd services/orion-actions) pytest tests/                                                                -> 201 passed
Full services/orion-cortex-orch/tests run together: 35 failures, identical list on origin/main 781f01c12
(test-order pollution; test_workflow_lane.py is 65/65 alone) -- not introduced here.
```

Key new tests:

- 40-PR fixture fully covered via map-reduce, with the journal body untrimmed (48k chars).
- 120-turn chat fixture fully covered, with the journal body untrimmed.
- Pagination: 3 pages, stopping at the window start, with a duplicate across pages, filtered on `merged_at`.
- Empty completion retried; control-character JSON parses; all attempts failing raises and writes no journal.
- Merge fails, overflows the budget, or drops refs → concatenated.
- Pass budget runs out mid-map → fails cleanly.
- Old exec in day mode → lookback widened to 2 and flagged.
- Page cap hit → `input_truncated`.
- Denver day bounds, including a 25h daylight-saving day.
- The chunk budget matches what the template actually renders.
- The actions timeout exceeds the worst case.

## Evals run

```text
pytest services/orion-cortex-orch/evals/test_chat_history_compactor_digest_eval.py -> passed (in the 265 above)
```

Eval gap: digest *quality* (does the merged narrative faithfully cover every PR/turn) still needs an LLM-in-the-loop eval. The deterministic ref-union check after merge is the only runtime guard.

## Docker/build/smoke checks

```text
Not run: this task is explicitly no-deploy. Static gates (CI orion-static-gates subset):
check_chat_route_poachers PASS, check_metric_lineage --gate PASS, check_definition_drift --gate PASS
(no metric definition changed -> no re-lock), check_env_template_parity PASS, git diff --check clean.
```

## Review findings fixed

The code review ran in a subagent against `git diff origin/main...HEAD`.

- **Finding (MUST):** new orch + old exec in day mode would silently miss yesterday's first ~6h and still report full coverage.
  - Fix: day mode sends `lookback_days = ceil((now - day_start)/1d)` (2 at 06:10). A missing `window_mode=window` echo sets `fetch_window_unconfirmed` and `input_truncated`.
  - Evidence: `test_github_day_mode_widens_lookback_and_flags_old_exec`.
- **Finding:** the chunk budget undercounted the rendered prompt about 2x, because `tojson` uses `ensure_ascii` and escapes `< > & '`.
  - Fix: `json_char_len` now measures `indent=2`, `ensure_ascii` JSON plus the escape cost.
  - Evidence: `test_chunk_budget_measures_the_rendered_prompt`. On the fixture's prose the old measure reads 30112 chars against 58116 by the new one.
- **Finding:** the merge input had no size limit.
  - Fix: if it exceeds the budget, skip the merge call and concatenate (`merge_input_over_budget`).
  - Evidence: `test_github_merge_input_over_budget_skips_merge_call`.
- **Finding:** a merge that dropped PRs/turns was still reported as full coverage.
  - Fix: after a merge, check its `pr_refs`/`turn_refs` against the union of the chunks' refs. On a miss, concatenate (`merge_dropped_refs:N`).
  - Evidence: `test_github_merge_that_drops_refs_falls_back_to_concatenation`.
- **Finding:** a 3600s wait on every scheduled workflow could stall the serial scheduler for an hour, and the claim TTL (300s) could reap and re-run a pass still in flight after a restart.
  - Fix: the long wait applies only to `github_compactor_pass` and `chat_history_compactor_pass`; the claim TTL is now the dispatch timeout + 60s.
  - Evidence: `test_workflow_dispatch_timeout.py`.
- **Finding:** deploy order (forbid schema) must be explicit.
  - Fix: documented above and in the restart commands.
- **Finding:** test gaps (page cap, dedupe, old exec, budget exhaustion).
  - Fix: tests added, listed above.
- **Not fixed (NIT, recorded):**
  - The SQL row ceiling of 10k is not flagged if hit. It is unreachable at live volume, and flagging it would need a new field on a forbid result model.
  - The orch may stop waiting on a last call while exec's 600s step keeps running.

## Restart required

Deploy order matters: exec before orch.

```bash
cd /mnt/scripts/Orion-Sapienform-compactor-full-window
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
```

(Or deploy from main after merge, same order.)

## Risks / concerns

- **Severity: medium.** Live behavior is **UNVERIFIED** until the next scheduled runs (chat around 06:00, GitHub 06:10 Denver) after deploy. Check `workflow_schedules.json` for completed runs, then the metadata: `covered_count == total_count`, `input_truncated=false`, `digest_merge_mode`, `journal_body_chars`.
- **Severity: medium.** Digest calls now go to the `agent` route, which queues (backlog) behind durable runs. A long agent backlog could exhaust the 3000s pass budget, and the pass then fails and is retried by the scheduler.
- **Severity: low.** The scheduler loop is serial, so a compactor pass can delay the next due job by up to 3600s (it was 420s).
- **Severity: low.** The exec GitHub fetch uses blocking `urlopen` inside an async verb (pre-existing). Pagination adds a few more blocking calls per run.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2422

🤖 Generated with [Claude Code](https://claude.com/claude-code)

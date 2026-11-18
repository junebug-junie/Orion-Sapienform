## Summary

- When Prime takes on a promise, the carry turn's prompt now ends with one line: for a long multi-step dig or independent parallel pieces, call the Agent tool with `subagent_type "Explore"` (read-only search) or `"general-purpose"` (one multi-step piece). Only the take-on prompt carries it (`orion/prime/intention.py` `CARRY_SUBAGENT_HINT`).
- New dig command `python -P -m orion.dig take-ons [<days>]`: each take-on, its attempts, tool calls, and sub-agent calls by type (`orion/dig/take_ons.py`).
- The line does NOT name `orion-repo-agent`: verified live that Orion's turns cannot see it (see "Risks").
- Fixes #3639.

## Outcome moved

Sub-agents become a stated option in take-on turns, and there is now a number for how often Orion uses them per take-on. Before: no prompt line, no count. Baseline from Orion's own transcripts (last 30 days): 13,709 turns, 9 used any sub-agent (0.07%).

## Current architecture

- Take-on: Prime's `take_on` move -> Hub `/api/intention/run` -> `build_carry_request` -> durable run `intention.carry` (one harness step) -> Hub `execute_unified_turn` via the curiosity turn RPC -> governor `claude -p` in `/mnt/orion-fcc/repo`.
- Prompt: `carry_prompt()` = `prime_concern:<id>` + the promise body. Nothing about sub-agents.
- Tool-call traces: `cockpit_turn_sighting` `motor_hop` rows exist only when a step queue is registered (the live chat WebSocket path). Durable turns pass the relay but no queue, so carry turns write no `motor_hop` rows (checked live on the 11-13 carry turn `0cf659ad`: stages ingress..finalize, no motor_hop). `harness_turn_trace.run_artifact` holds no tool names. The governor's `claude` transcripts (`/root/.claude/projects/-mnt-orion-fcc-repo/<session>.jsonl`) hold every call.

## Architecture touched

- `orion/prime/intention.py` (prompt text, used by Hub at deploy).
- `orion/dig` (read-only CLI that Orion runs from a turn's Bash inside the governor).
- No bus, schema, env, or service-config change.

## Files changed

- `orion/prime/intention.py`: `CARRY_SUBAGENT_TYPES`, `CARRY_SUBAGENT_HINT`; `carry_prompt()` appends the hint after the task.
- `orion/prime/tests/test_prime_intention.py`: the hint is one line, last, names only the visible types, keeps the concern ref first and the only one; only `intention.py` uses the hint (no general chat prompt).
- `orion/dig/take_ons.py`: SQL over `durable_run_workflow`, transcript discovery from the resume key, tool/sub-agent count, render.
- `orion/dig/__main__.py`: `cmd_take_ons`, dispatch, help line (help stays under its 4,000-char cap: 3,993).
- `orion/dig/tests/test_take_ons.py`: resume key/session id pinned to the live 11-13 transcript id; Agent + Task counted by type; tool_use ids deduped; generation gaps probed; missing transcript reads UNREAD; days validated and clamped.
- `orion/dig/README.md`: `take-ons` row.
- `services/orion-hub/static/catalog/orion-abilities.json`: `dig-take-ons` entry (CI: every dig command needs a catalog entry, `test_orion_abilities_catalog.py`).

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the carry turn's prompt gains one trailing line.
- Compatibility notes: `orion/prime/handback.py` still finds exactly one `prime_concern:` ref (tested).

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed)
- skipped keys requiring operator action: none

## Metric gate (sub-agent dispatches per take-on)

1. **Provenance.** Each count is a `tool_use` block named `Agent` (or the older `Task`) in the carry turn's own `claude` transcript. The file is found by name: Hub sets `resume_key=f"{run_id}:{source_tag}"` (`services/orion-hub/scripts/curiosity_investigation.py`), durable-runs sets `source_tag=f"intention_carry:{concern_id}"` (`services/orion-durable-runs/app/harness_smoke_graph.py` `_intention_step`), and `orion/harness/session_resume.py` `session_id_for(key, gen)` is the session id the governor gives `claude`. Take-ons come from `durable_run_workflow` (`workflow='intention.carry'`), readable by `orion_readonly` (checked).
2. **Independence.** No other metric reads sub-agent use. It is a raw count, not derived from tool-call totals (those are printed next to it for scale, not combined).
3. **Theory anchor.** Direct observation, not inference: the count is the number of times the turn called the tool the prompt line names. It measures uptake of the line, nothing more.
4. **Live-data sanity.** Run inside the governor against the real transcripts (this branch's module, 2026-11-17):
   `1 take-on(s) in the last 90 day(s); transcripts read for 1; 0 started a sub-agent (0 sub-agent call(s))`
   `2026-11-13 19:47Z intention-d8db9d7b1bba4cf7 completed: 2 attempt(s), 2 tool call(s), sub-agents: none`
   One take-on has ever run, so today the count reads 0 of 1. That is a true zero (the transcript was read; 2 Bash calls found), not a dead reader. The rest state (0) is reachable by construction and is the expected state for most take-ons. A missing transcript prints UNREAD, never 0. The number will stay near-empty until more take-ons run; it cannot say anything about the line's effect yet.
5. **Existing mechanism.** `dig turn-steps` reads `cockpit_turn_sighting` `motor_hop` rows, which carry turns do not write. `harness_turn_trace.run_artifact` has no tool names. `orion/dev_economics/claude_code_ingest.py` parses Juniper's transcripts, not the governor's volume. Reusing `session_resume.session_id_for`/`find_transcript` instead of a new store.
6. **Reversibility.** Read-only CLI and one prompt line. No store, schema or default. Removing either is a one-file revert.

## Tests run

```text
PYTHONPATH=. pytest -q orion/prime/tests orion/dig/tests            -> 527 passed
services/orion-hub: pytest -q tests/test_intention_order.py         -> 11 passed
services/orion-durable-runs: pytest -q tests/test_intention_carry.py -> 6 passed
services/orion-hub: pytest -q tests -k 'catalog or abilities' -> 27 passed, 3 skipped
CI first run: 15 pass, 1 fail (Hub full tests: missing catalog entry for take-ons) -> fixed
```

## Evals run

```text
python -m orion.prime.evals.run_intention_take_on_replay -> "ok": true (prompt now carries the line)
```

No dig eval harness covers `take-ons`; the live run above is the smoke.

## Docker/build/smoke checks

```text
docker exec -i orion-athena-harness-governor python3 - < (take_ons.py + driver)
-> 1 take-on(s) ... 0 started a sub-agent; intention-d8db9d7b1bba4cf7: 2 attempt(s), 2 tool call(s), sub-agents: none
```

## Review findings fixed

Review ran in a subagent (by hand, against `HEAD` vs `origin/main`). No blockers.

- Finding: a fixed 8-generation probe could undercount a take-on with more generations, silently.
  - Fix: read the last generation from the resume registry (`session_resume._registry_path`/`_read`); without it, probe until 4 misses in a row, cap 64, and print `(+ generations past 64 not read)` on a cap hit.
  - Evidence: `test_probing_follows_the_registry_generation_and_says_when_capped`; live run read registry generation 1 -> 2 attempts.
- Finding: the time was printed with a `Z` regardless of the psql session's zone.
  - Fix: SQL formats `created_at AT TIME ZONE 'UTC'`.
  - Evidence: `test_sql_prints_utc_minutes_and_non_ascii_days_are_refused`.
- Finding: `"²".isdigit()` is True, so `int()` crashed outside `DigError`.
  - Fix: `days.isascii() and days.isdigit()`.
  - Evidence: same test.
- Finding: older CLIs log a sub-agent's own calls into the parent file (`isSidechain`), which would inflate the count.
  - Fix: sidechain events are skipped.
  - Evidence: `test_count_tools_counts_agent_and_task_by_type_and_dedupes_ids` has a sidechain line that is not counted.
- Finding: dedupe test comment was wrong; the "only carry prompt" test grepped the constant name, not the text.
  - Fix: comment reworded; the test now greps a distinctive phrase of the hint text.
  - Evidence: 527 passed.
- Not fixed (nit, not live): the carry prompt is ~300 chars longer, so a source prompt within 300 chars of the 20,000-char `max_length` would fail validation. The only adapter (`turn_promise`) caps at ~3,000. Left alone to stay out of `orion/schemas/intention.py`, which #3642 is changing.

## Restart required

Hub builds the carry prompt, so Hub needs a rebuild after merge (the post-pull rebuild does this from the clean primary main checkout). `dig take-ons` is live for Orion when the FCC sandbox syncs to main (automatic after the post-pull rebuild; check `git -C /mnt/orion-fcc/repo log -1 --oneline`).

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: the task asked to name `orion-repo-agent`. Orion's turns cannot call it. The governor runs `claude --setting-sources user,local` (`HARNESS_FCC_SETTING_SOURCES`, `orion/fcc/claude_spawn.py`), so the project's `.claude/agents/` is not loaded. The Agent tool's own list in recent transcripts is `claude, Explore, general-purpose, Plan, statusline-setup`. Naming `orion-repo-agent` would point Orion at a call that fails. Its body is also a coding-agent contract for this repo (worktrees, PRs), which FCC turns drop on purpose.
  - Mitigation: the line names the two types that exist. A test refuses `orion-repo-agent` in the hint. Making a repo agent visible needs a governor change (`--agents` or a user-scope agent file) and is out of scope here.
- Severity: low
  - Concern: sub-agents in a carry turn share the same lane and the turn's time budget (#3354 timeout work is in flight in #3642).
  - Mitigation: sub-agents spend the run's GPU lease and `timeout_sec` like any other step. The line says "for a long multi-step dig or independent pieces" only; the count will show whether it is used at all.
- Severity: low
  - Concern: `dig take-ons` only reads transcripts where they are (the governor). From any other container it prints UNREAD.
  - Mitigation: UNREAD is explicit; Orion's Bash runs in the governor.
- Issues: opened #3639 (gap), #3646 (verify-live). Closed: #3639 by this PR.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3647

🤖 Generated with [Claude Code](https://claude.com/claude-code)

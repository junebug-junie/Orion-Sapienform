# fix(actions): nightly metacog report fits its prompt and stops retrying all night (redesign Stage 0B)

## Summary

- Orion's nightly self-reflection report (Daily Metacog) had failed every night since 2026-09-03. Its prompt was a few hundred characters over the limit cortex-exec allows, and the largest part of it was the list of skills Orion can pick an experiment from. That list now uses one short line per skill instead of a JSON blob, and its size comes from a budget computed from the template and the largest memory digest recall can produce. It fits by construction; the limit was not raised.
- When the report fails, the scheduler now tries at most 3 times that night (10 minutes, then 30 minutes apart) instead of every 45 seconds until midnight (230–280 runs a night). After the third failure it records the failure once, with the real error, and stops until the next day. It is not marked as completed.
- The real error now reaches orion-actions. Before, a failed cortex-exec step arrived only as `cortex_exec_missing_final_text`; now it carries the step name and error text (for example `step=draft_daily_metacog error=... daily_metacog_prompt_over_limit chars=8467 limit=8192`).
- Daily Pulse gets the same retry cap. It keeps its existing JSON skills list, because it has no prompt limit and works today.

## Outcome moved

- Prompt size for the night that failed (958-char memory digest): **8,477 → 3,772 chars** (limit 8,192). The live failure logged 8,467; the 10-char gap is the window timestamp format in my reproduction.
- Worst case (recall's maximum 1,280-char digest): **8,799 → 4,094 chars**.
- Skills list: **6,126 → 1,421 chars**, now covering the 15 read-only skills. The 6 host-changing skills were never selectable, because the daily selector rejects them.
- Headroom: the list budget is 5,218 chars. With 2× today's skills every id still fits with its description. With 10× it falls back to bare ids, then to "(+N more not listed)", and still stays under the limit.
- Failed runs per night: about 245 → at most 3, plus one warning record.

## Current architecture

- orion-actions builds the daily context. For both pulse and metacog it passed `skills_catalog_compact = build_compact_skill_catalog()`, which is JSON for all 21 skills with descriptions up to 200 chars (`services/orion-actions/app/main.py`, module level).
- cortex-exec renders `orion/cognition/prompts/daily_metacog_prompt.j2`. `_enforce_daily_metacog_prompt_budget` (`services/orion-cortex-exec/app/executor.py:1509`) then fails the step when the prompt is over `CORTEX_DAILY_METACOG_PROMPT_MAX_CHARS=8192`. The step error is stored in `cognition_traces.steps[].error` (confirmed live below).
- The scheduler loop runs every 45s. `should_run_daily` stays true until local midnight unless the done-today cursor is set, and only a success sets it, so a deterministic failure re-ran all evening.

## Architecture touched

- `orion/cognition` (shared): a new budget module and a bounded catalog builder.
- `orion-actions`: per-action skills context, a retry gate for the two daily jobs, error detail from failed plans, and a warning notification on give-up.
- No changes to cortex-exec code, the template, the env, the schema, or the bus.

## Files changed

- `orion/cognition/daily_metacog_budget.py` (new): computes the skills-list budget. It takes the prompt limit (8,192; a test pins it to cortex-exec's setting default), subtracts the rendered template length with the recall profile's `render_char_budget` (1,280, read from `orion/recall/profiles/journal.daily.metacog.grounded.v1.yaml`), and subtracts a 256-char margin. Then it builds the metacog list.
- `orion/cognition/skills_manifest.py`: adds `build_bounded_skill_catalog`, which writes `skill_id: purpose` lines (purpose ≤ 60 chars, read-only skills only). When that is too long it falls back to ids only, then to as many ids as fit plus "(+N more not listed)". It returns the number of ids actually listed.
- `services/orion-actions/app/main.py`:
  - metacog gets the bounded list and pulse keeps the JSON (`_daily_skills_catalog_context`);
  - `_plan_failure_detail` adds the failing step's error to `cortex_exec_missing_final_text`;
  - `_execute_daily` records why it failed, and a dedupe skip does not count as an attempt;
  - pulse and metacog scheduling goes through `_scheduled_daily_tick`;
  - `daily_failure_notify_request` builds the warning notification;
  - a give-up persisted before a restart is reloaded at startup.
- `services/orion-actions/app/daily_retry_gate.py` (new): `DailyAttemptGate` (max 3 tries per scheduled local date, with backoff) and `run_gated_daily_tick`, which the scheduler loop and the tests share.
- `services/orion-actions/README.md`: documents the retry cap and the catalog.
- Tests: `orion/cognition/tests/test_daily_metacog_budget.py`, `services/orion-actions/tests/test_daily_retry_gate.py`.

## Schema / bus / API changes

- Added: none to any schema or channel. There is one new notification `event_kind`, `orion.daily.failed` (severity `warning`), sent through the existing NotifyClient. `event_kind` is free text in notify, and `orion.workflow.failed` already uses the same pattern. There is one new audit status value, `gave_up`, on the existing `orion:actions:audit` event.
- Removed / Renamed: none.
- Behavior changed:
  - metacog's `skills_catalog_compact` is now newline-separated `skill_id: purpose` lines instead of JSON;
  - `skills_catalog_count` is the number of ids listed (15 today, not 21);
  - the scheduled pulse and metacog runs stop after 3 failures per scheduled local date.
- Compatibility notes: a new cursor-store key, `<job>.gave_up` → date, sits in the same JSON file. It is ignored by every other reader, because they look up exact job keys.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
Baseline (before patch, same worktree):
  pytest services/orion-actions/tests -q                                   -> 221 passed
  pytest orion/cognition/tests/test_daily_metacog_prompt.py test_recall_profiles.py -> 13 passed
  pytest (daily_metacog_prompt, recall_profiles, recall_render_lane_separation,
          schemas/test_self_experiments, tests/test_image_prune_skill)     -> 42 passed

After:
  pytest services/orion-actions/tests -q                                   -> 229 passed (+8 new)
  pytest (above 5 files + test_daily_metacog_budget.py)                    -> 50 passed (+8 new)
  pytest services/orion-self-experiments/tests/test_experiment_registry.py -> 17 passed
```

New tests:

- `test_old_json_catalog_with_max_digest_was_over_limit` pins the bug.
- `test_current_manifest_with_max_digest_fits_and_passes_guard` uses the current manifest, a 1,280-char digest, cortex-exec's real `_render_prompt` and `_append_memory_digest`, and the real guard.
- `test_grown_manifest_still_fits[2|10]`.
- `test_simple_render_matches_cortex_exec_jinja_render`.
- `test_settings_default_matches_shared_budget_constant` and `test_digest_ceiling_is_read_from_recall_profile`.
- `test_every_read_only_skill_id_is_selectable_today`.
- `test_deterministic_failure_runs_three_times_a_night_not_250` replays two nights of 45s ticks through the real `should_run_daily` and `SchedulerCursorStore`. It gets 3 runs per night, the backoff holds, there is one give-up per night naming `daily_metacog_prompt_over_limit`, and the job is never marked completed.
- Also: transient failure then success, a dedupe skip that does not spend an attempt, a give-up that survives a restart, pulse getting the same cap, recovery of the plan error detail, and the warning notification shape.

Static gates: every `run:` step in `.github/workflows/orion-static-gates.yml` ran locally, including `check_definition_drift.py --gate`. All 24 steps exited 0.

## Evals run

```text
None. orion-actions has no eval harness for daily reports. The deterministic tests above
cover the budget and the retry behavior. Whether the report is any good once it lands
is not evaluated here (follow-up).
```

## Docker/build/smoke checks

```text
Not run, by instruction (no deploy, restart, or job trigger).
Live read-only evidence that the step error is durable and queryable:
  cognition_traces, verb=daily_metacog_v1, step draft_daily_metacog, 2026-09-30 06:00:06 UTC:
  "LLMGatewayService: daily_metacog_prompt_over_limit chars=8467 limit=8192
   section_sizes={'memory_digest': 958, 'skills_catalog_compact': 6126} estimated_prompt_tokens=2117"
  cognition_traces rows for daily_metacog_v1 per UTC day: 09-27 247, 09-28 250, 09-29 234, 09-30 246.
```

## Review findings fixed

- No review subagent was run, by instruction from the dispatching agent.

## Restart required

Only orion-actions changed at runtime. cortex-exec is unchanged: it reads the template from `orion/` and does not use the new module.

```bash
scripts/safe_docker_build.sh orion-actions up -d --build
```

## Risks / concerns

- Severity: high (for the acceptance check, not for this code)
  - Concern: Daily Metacog and Daily Pulse are **paused live**. `ACTIONS_DAILY_METACOG_ENABLED=false` and `ACTIONS_DAILY_PULSE_ENABLED=false` in the running container, the local `.env`, and `.env_example` since 2026-09-30, with the reason "no real consumer". So after deploy nothing runs, and Stage 0's "one nightly report lands" check cannot pass until Juniper re-enables metacog. I did not flip it; that is her call. **UNVERIFIED: a real report landing.**
- Severity: low
  - Concern: the shared constant `DAILY_METACOG_PROMPT_MAX_CHARS=8192` mirrors cortex-exec's env default. If an operator lowers `CORTEX_DAILY_METACOG_PROMPT_MAX_CHARS`, the list budget does not follow it.
  - Mitigation: a test pins the two defaults together, there is a 256-char margin, and the guard still fails loudly rather than truncating.
- Severity: low
  - Concern: the retry cap counts every failure, including transient ones like an LLM timeout, so a night with three transient failures gives up.
  - Mitigation: the 10 and 30 minute backoff spreads the tries over about 40 minutes, and the give-up is visible as a warning notification.
- Severity: low
  - Concern: the model now sees only the 15 read-only skills, at most 60 chars each. It loses the longer descriptions (for example render_scene's 819 chars). The id set it can validly pick is unchanged.
- Follow-up (spec item 4, optional, not done): passing the report's day window to recall is not trivial. The recall profile selects by `sql_since_minutes: 1440` relative to now, not by the report window, so it needs a recall-side change.

## PR link

(filled on open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

# fix(actions): nightly metacog report fits its prompt and stops retrying all night (redesign Stage 0B)

## Summary

- Orion's nightly self-reflection report (Daily Metacog) had failed every night since 2026-09-03. Its prompt was a few hundred characters over the limit cortex-exec allows, and the largest part of it was the list of skills Orion can pick an experiment from. That list now uses one short line per skill instead of a JSON blob, and its size comes from a budget computed from the template and the largest memory digest recall can produce. It fits by construction; the limit was not raised.
- A failed report no longer retries every 45 seconds until midnight (230–280 runs a night). How often it retries depends on why it failed:
  - A failure that will repeat (prompt too long, unreadable model output) gets 3 tries, 10 and then 30 minutes apart.
  - A failure that may clear (timeout, GPU busy) gets one try an hour until the day ends.
  - An unrecognized failure gets one try an hour, at most 6.
- After the last try it records the failure once, with the real error, and stops for that date. It is never marked as completed. The attempt counts survive a restart.
- The real error now reaches orion-actions. Before, a failed cortex-exec step arrived only as `cortex_exec_missing_final_text`; now it carries the step name and error text.
- Two skills that change the world were labelled "read-only", so the nightly reports could pick them as harmless experiments. One starts Docker stacks; the other spends GPU making an image. Both are now labelled correctly, in both copies of the skill classifier. Daily pulse had in fact picked the Docker one as a "read-only probe" on 2026-08-30.
- Daily Pulse gets the same retry policy. It keeps its existing JSON skills list, because it has no prompt limit and works today.

## Outcome moved

- Prompt size for the night that failed (958-char memory digest): **8,477 → 3,572 chars** (limit 8,192). The live failure logged 8,467; the 10-char gap is the window timestamp format in my reproduction.
- Worst case (recall's maximum 1,280-char digest): **8,799 → 3,894 chars**.
- Skills list: **6,126 → 1,221 chars**. It covers the 13 skills that are really read-only. The 8 that change something (host-mutating, notify, compose bring-up, render_scene) are left out, because the daily selector rejects them anyway.
- Headroom: the list budget is 5,218 chars. With 2× today's skills every id still fits with its description. With 10× it falls back to bare ids, then to "(+N more not listed)", and still stays under the limit.
- Failed runs per night when the failure is deterministic: about 245 → at most 3, plus one warning record.

## Current architecture

- orion-actions builds the daily context. For both pulse and metacog it passed `skills_catalog_compact = build_compact_skill_catalog()`, which is JSON for all 21 skills with descriptions up to 200 chars.
- cortex-exec renders `orion/cognition/prompts/daily_metacog_prompt.j2`. `_enforce_daily_metacog_prompt_budget` (`services/orion-cortex-exec/app/executor.py:1509`) then fails the step when the prompt is over `CORTEX_DAILY_METACOG_PROMPT_MAX_CHARS=8192`. The step error is stored in `cognition_traces.steps[].error` (confirmed live below).
- The scheduler loop runs every 45s. `should_run_daily` stays true until local midnight unless the done-today cursor is set, and only a success sets it.
- Skill risk labels come from two copies of one classifier: `orion/cognition/skills_manifest.py` (used by orion-actions and self-experiments) and `services/orion-cortex-exec/app/actions_skill_registry.py` (used by cortex-exec's capability bridge and agent traces). Anything not on a list fell through to "read-only".

## Architecture touched

- `orion/cognition` (shared): a new budget module, a bounded catalog builder, and corrected labels for two skills.
- `orion-actions`:
  - per-action skills context;
  - `_execute_daily` returns its outcome;
  - a failure-class retry gate with persisted state;
  - error detail from failed plans;
  - a warning notification on give-up.
- `orion-cortex-exec`: the same two label corrections in its copy of the classifier. No dispatch or selection change (see Risks).
- No env, schema, bus, or template changes.

## Files changed

- `orion/cognition/daily_metacog_budget.py` (new): computes the skills-list budget. It takes the prompt limit (8,192; a test pins it to cortex-exec's settings field default), subtracts the rendered template length with the recall profile's `render_char_budget` (1,280, read from `orion/recall/profiles/journal.daily.metacog.grounded.v1.yaml`), and subtracts a 256-char margin.
- `orion/cognition/skills_manifest.py`:
  - `build_bounded_skill_catalog`: `skill_id: purpose` lines with purposes of at most 60 chars, read-only skills only, falling back to ids only and then "(+N more)";
  - `STATE_CHANGING_SKILL_MARKERS` (`compose_service_bringup` → `state_change`) and `ACTUATING_SKILL_MARKERS` (`notify`, `render_scene` → `benign_actuation`).
- `services/orion-cortex-exec/app/actions_skill_registry.py`: the same two lists, so the two classifiers agree. A test now enforces that across every real skill.
- `services/orion-actions/app/main.py`:
  - metacog gets the bounded list and pulse keeps the JSON;
  - `_plan_failure_detail` adds the step error to the failure message;
  - `_execute_daily` now returns a `DailyAttemptResult`. There is no shared failure dict: the only non-attempt exit is the dedupe skip, and every other path ends at one `return outcome` that starts as an attempted failure;
  - `_scheduled_daily_tick` (exposed as `app.state.scheduled_daily_tick` for tests);
  - `daily_failure_notify_request`;
  - the per-tick INFO log fires only when the outcome changes (DEBUG otherwise).
- `services/orion-actions/app/daily_retry_gate.py` (new): `classify_daily_failure`, `DailyAttemptGate` (per-class counts, backoff, persisted to `daily_attempts.json` next to the scheduler cursors), and `run_gated_daily_tick`.
- `services/orion-actions/app/scheduler_cursor_store.py`: a read-only `path` property, used to place the attempt-state file.
- `services/orion-actions/README.md`: documents the retry policy, the persisted state, the `ACTIONS_DAILY_RUN_ON_STARTUP` interaction, and the catalog.
- Tests:
  - `orion/cognition/tests/test_daily_metacog_budget.py`
  - `services/orion-actions/tests/test_daily_retry_gate.py`
  - `services/orion-self-experiments/tests/test_experiment_registry.py`

## Schema / bus / API changes

- Added:
  - A new notification `event_kind`, `orion.daily.failed` (severity `warning`), sent through the existing NotifyClient. `event_kind` is free text, as `orion.workflow.failed` already is.
  - A new audit status value, `gave_up`, carrying `failure_class`, on the existing `orion:actions:audit` event.
- Removed / Renamed: none.
- Behavior changed:
  - metacog's `skills_catalog_compact` is newline-separated `skill_id: purpose` lines, and `skills_catalog_count` is the number listed (13);
  - scheduled pulse and metacog retry by failure class;
  - `skills.docker.compose_service_bringup.v1` is `read_only=False, risk_class=state_change` and `skills.imagination.render_scene.v1` is `read_only=False, risk_class=benign_actuation`, in both classifiers.
- Compatibility notes: a new state file `daily_attempts.json` sits in the same directory as the scheduler cursors. The cursor JSON itself is unchanged.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
Baseline (main @ f44dc1127, before this PR):
  pytest services/orion-actions/tests -q                                    -> 221 passed
  pytest (daily_metacog_prompt, recall_profiles, recall_render_lane_separation,
          schemas/test_self_experiments, tests/test_image_prune_skill)      -> 42 passed
Baseline (branch head before the review round):
  pytest services/orion-actions/tests -q                                    -> 229 passed
  pytest services/orion-self-experiments/tests -q                           -> 20 passed
  pytest services/orion-cortex-exec/tests/test_skill_verbs.py -q            -> 44 passed

After the review round:
  pytest services/orion-actions/tests -q                                    -> 249 passed
  pytest services/orion-self-experiments/tests -q                           -> 22 passed
  pytest services/orion-cortex-exec/tests/test_skill_verbs.py -q            -> 44 passed
  pytest (the 5 files above + test_daily_metacog_budget.py
          + test_verb_activation_docker_compose_bringup + test_express_outward_action) -> 77 passed
  pytest cortex-exec capability/skill-args/notify/compose/phase8 tests      -> 34 passed
  pytest tests/test_agent_trace_summary, test_curiosity_atlas,
         test_execution_dispatch_builder, test_policy_decision_builder       -> 50 passed
Mutation check: making _execute_daily's exception path return attempted=False
  fails 3 tests (both real-path tests and the structural test).
```

Static gates: every `run:` step in `.github/workflows/orion-static-gates.yml` ran locally, including `check_definition_drift.py --gate`. All exited 0.

## Evals run

```text
None. orion-actions has no eval harness for daily reports. The deterministic tests above
cover the budget and the retry behavior. Whether the report is any good once it lands
is not evaluated here (follow-up).
```

## Docker/build/smoke checks

```text
Not run, by instruction (no deploy, restart, or job trigger).
Live read-only evidence:
  cognition_traces, verb=daily_metacog_v1, step draft_daily_metacog, 2026-09-30 06:00:06 UTC:
  "LLMGatewayService: daily_metacog_prompt_over_limit chars=8467 limit=8192
   section_sizes={'memory_digest': 958, 'skills_catalog_compact': 6126} estimated_prompt_tokens=2117"
  cognition_traces rows for daily_metacog_v1 per UTC day: 09-27 247, 09-28 250, 09-29 234, 09-30 246.
  Step-error kinds, all verbs, last 30 days (basis for the failure classes):
    daily_metacog_prompt_over_limit 3784, RPC timeout (LLMGatewayService) 2038,
    gateway_capacity_rejected:* 152, gpu_pool_unavailable:* 301, timeout:caller_budget_exhausted 72,
    RPC timeout (RecallService) 41, resource_lease_rejected 15, gpu_pool_recalled 3.
  self-experiments store (orion-athena-orion-self-experiments): one skill_probe for
    skills.docker.compose_service_bringup.v1 from daily pulse on 2026-08-30, status validated.
```

## Review findings fixed

- Finding 1: the failure reason passed through a shared dict that `_execute_daily` cleared before taking the dedupe lock, so a concurrent manual run could erase or inject a scheduled run's failure.
  - Fix: `_execute_daily` returns a `DailyAttemptResult`. The dict is gone.
  - Evidence: `test_every_execute_daily_exit_is_an_attempted_failure_unless_proven_otherwise` asserts there is no `daily_last_failure` in the function.
- Finding 2: the glue between the scheduler and `_execute_daily` was untested, because the tests re-implemented the loop.
  - Fix: two tests drive the real lifespan → `_scheduled_daily_tick` → `_execute_daily` → `_run_plan` path, with only the cortex RPC stubbed. A structural AST test pins the exits: there are exactly two returns, the only `attempted=False` is under `deduper.try_acquire`, `outcome` starts as an attempted failure, and only one path sets `completed=True`.
  - Evidence:
    - `test_real_path_deterministic_failure_gives_up_after_three`: outcomes go failed, backoff, failed, backoff, gave_up, gave_up; there are 3 RPC calls, one `gave_up` audit, and one `orion.daily.failed` notify.
    - `test_real_path_transient_failure_is_retried_hourly`.
    - The mutation check above.
- Finding 3: a transient outage cost the whole day.
  - Fix: `classify_daily_failure`, built from the real error strings above, sets the policy: deterministic gets 3 tries; transient retries hourly until the date ends and is then recorded on the next date's first tick; unknown retries hourly, at most 6.
  - Evidence: `test_classification_uses_real_error_strings` (12 cases), `test_transient_failure_keeps_hourly_retry_until_the_date_ends` (4 tries at 20:15, 21:15, 22:15 and 23:15, then one give-up), `test_transient_outage_then_recovery_completes_same_night`, and `test_unknown_failures_are_capped_at_six`.
  - Tradeoff: a transient failure costs up to 4 tries a night instead of 3, and a sustained outage still loses that night's report. "Could not parse JSON" is treated as deterministic even though sampling can vary, because the run already retries once internally. An unrecognized error string gets up to 6 tries a day instead of 3.
- Finding 4: compose_service_bringup and render_scene were labelled read-only.
  - Fix: compose_service_bringup becomes `state_change` (it runs `docker compose build` and `up -d`). render_scene becomes `benign_actuation`: it spends GPU on circe and saves a new image, so it isn't an observation, but it deletes nothing on the host. Both are non-read-only and non-idempotent, in both classifier copies. compose_service_bringup was deliberately not put in `HOST_MUTATING_SKILL_MARKERS`, because that would move its family and change cortex-exec's capability selection (see Risks).
  - What each consumer does with the labels:
    - **orion-actions daily selector** (pulse focus and metacog tomorrow-experiment): rejects non-read-only ids. Both are now rejected, which is intended.
    - **orion-self-experiments** `normalize_create_request`: `skill_probe` rejects non-read-only unless `allow_non_read_only`, and proposal-only types are still allowed. Both are now rejected as probes, which is intended, since probes are `mutation_policy=forbidden`.
    - **cortex-exec capability bridge**: copies `risk_class`, `confirmation_required`, `execute_opt_in` and `observational` into decision metadata. Nothing gates on them. `requires_confirmation` and `requires_execute_opt_in` are unchanged (False) for both. Family is unchanged, so selection is unchanged.
    - **agent_trace normalizer** (`_bound_effect_kind`): `state_change` now reports compose bring-up as `side_effect` (correct). render_scene stays `external_io`.
    - **execution-dispatch** (render_scene's real outward-action path): routes by `cortex_verb` and its own dispatch policy, and never reads these labels.
    - **Hub direct compose dispatch**: calls the verb directly. Its gate is cortex-exec's `SKILLS_ALLOW_DOCKER_COMPOSE_BRINGUP`.
  - Evidence:
    - `test_world_changing_skills_are_not_labelled_read_only`, `test_world_changing_skills_are_not_offered_to_metacog`, `test_shared_manifest_and_cortex_exec_registry_classify_every_skill_identically`;
    - `test_daily_selectors_reject_world_changing_skills` (4 cases) and `test_world_changing_skills_rejected_as_read_only_probes` (2 cases, real manifest);
    - the existing `test_skill_verbs.py` (44) and `test_image_prune_skill.py` still pass.
- Finding 5 (nit): the per-tick INFO log fired every 45s.
  - Fix: it logs at INFO only when the (date, outcome) pair changes, and at DEBUG otherwise.
  - Evidence: code in `_scheduled_daily_tick`. There is no dedicated test.
- Finding 6 (nit): attempt counts lived only in memory, and the `ACTIONS_DAILY_RUN_ON_STARTUP` interaction was undocumented.
  - Fix: counts, last failure time and gave_up are persisted in `daily_attempts.json`, using wall-clock timestamps. The README documents that startup tries spend the same per-date budget.
  - Evidence: `test_attempt_counts_and_give_up_survive_restart` (2 failures, restart, exactly 1 more try, give up; restart again, 0 tries).
- Finding 7 (nit): the 8,192 parity test read the env-loaded value.
  - Fix: it now compares against `Settings.model_fields["daily_metacog_prompt_max_chars"].default`.
  - Evidence: `test_settings_default_matches_shared_budget_constant`.

## Restart required

Both services changed at runtime: orion-actions (retries, catalog, labels via the shared manifest) and cortex-exec (labels in its classifier copy; metadata and traces only). orion-self-experiments also reads the shared manifest at request time, so it picks up the new labels on restart.

```bash
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-self-experiments up -d --build
```

## Risks / concerns

- Severity: high (for the acceptance check, not for this code)
  - Concern: Daily Metacog and Daily Pulse are **paused live**. `ACTIONS_DAILY_METACOG_ENABLED=false` and `ACTIONS_DAILY_PULSE_ENABLED=false` in the running container, the local `.env`, and `.env_example` since 2026-09-30. After deploy nothing runs, and Stage 0's "one nightly report lands" check cannot pass until Juniper re-enables metacog. I did not flip it. **UNVERIFIED: a real report landing.**
- Severity: high (pre-existing, found during finding 4, not fixed here)
  - Concern: cortex-exec's capability bridge resolves `assess_runtime_state` (declared `side_effect_level: none`, `preferred_skill_families: [system_inspection]`) to `skills.docker.compose_service_bringup.v1`. That happens only because it is alphabetically first in that family. This is the same accident the builder_prune comment describes. `SKILLS_ALLOW_DOCKER_COMPOSE_BRINGUP=true` in the live cortex-exec container, so a bound `assess_runtime_state` call could reach a compose bring-up. I checked this offline against the registry. Live: in 60 days of `cognition_traces`, `assess_runtime_state` appears only as text inside 3 `self_study.reflect` traces, none mentioning compose bring-up, so the path has not fired recently. It is latent, not active.
  - Mitigation: left unchanged on purpose. Moving compose bring-up out of the family would make `render_scene` (which spends GPU) the pick instead. The right fix is a dispatch decision (pin `assess_runtime_state` to a read-only skill, or prefer read-only candidates). It needs its own proposal.
- Severity: low
  - Concern: the shared constant `DAILY_METACOG_PROMPT_MAX_CHARS=8192` mirrors cortex-exec's settings default. If an operator lowers `CORTEX_DAILY_METACOG_PROMPT_MAX_CHARS`, the list budget does not follow it.
  - Mitigation: a test pins the field defaults together, there is a 256-char margin, and the guard still fails loudly rather than truncating.
- Severity: low
  - Concern: classification is by substring. A new error wording falls to "unknown", which is hourly and capped at 6.
- Severity: low
  - Concern: the model sees only 13 read-only skills, at most 60 chars each.
- Follow-up (spec item 4, optional, not done): passing the report's day window to recall needs a recall-side change. The profile selects by `sql_since_minutes: 1440` relative to now.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2454

🤖 Generated with [Claude Code](https://claude.com/claude-code)

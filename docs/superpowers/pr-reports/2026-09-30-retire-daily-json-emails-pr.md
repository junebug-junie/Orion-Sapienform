# Retire the raw-JSON daily emails and the dead World Pulse email path

## Summary

- Orion stops emailing the two raw-JSON reports, "Orion — Daily Pulse" (08:30) and "Orion — Daily Metacog" (20:15). Both are still generated every day and still show up in-app. The 08:30 slot is left open for the upcoming "Orion's Day" HTML letter, which ships in a separate PR.
- How: `ACTIONS_DAILY_EMAIL_ENABLED` now defaults to `false` in `settings.py`, `.env_example`, the local `.env`, and the README. `main.py` is not touched.
- World Pulse's old direct-email path is deleted outright: the `publish-email` route, `publish_email.py`, `wiring.py`, `render_email_digest`, the `WORLD_PULSE_EMAIL_*` and `NOTIFY_*` settings and env keys, and the `EmailWorldPulseRenderV1` schema and its registry entry.
- Regression tests cover both halves.

- Daily Pulse / Daily Metacog **generation paused** (Juniper, 2026-09-30): `ACTIONS_DAILY_PULSE_ENABLED` and `ACTIONS_DAILY_METACOG_ENABLED` default `false` in `settings.py` and `.env_example`; local `services/orion-actions/.env` set to `false` by hand (sync does not overwrite existing values). Evidence the consumer is thin: orion-self-experiments logged 6 `self_experiment_created type=skill_probe source=daily_pulse_v1` in 8 days, none from metacog, no completion events. Reversible by flipping either flag. Test: `test_daily_pulse_and_metacog_generation_paused_by_default`.

## Outcome moved

Two daily emails of pretty-printed JSON no longer reach Juniper's inbox. Everything that depends on those reports keeps working:
- orion-self-experiments still gets its experiment sources (`focus_skill_id`, `tomorrow_experiment_skill_id`, `experiment_registry.py:218-258`).
- The Hub notification and the async chat message still arrive.

World Pulse now has one way to reach email instead of two: the Journal Pass (`trigger_kind=world_pulse_digest` in orion-actions), which this PR does not change.

## Current architecture

- orion-actions `_daily_notify_request` added `channels_requested=["email"]` only when `ACTIONS_DAILY_EMAIL_ENABLED` was true. The default was true.
- orion-notify's `/notify` (`email_delivery.should_send_email`) emails in only two cases: `channels_requested` contains `"email"`, or severity is `error`/`critical`. The daily reports are `info`, so `channels_requested=None` means no email. The `/chat/message` endpoint never sends email.
- `ACTIONS_DAILY_EMAIL_ENABLED` has exactly one consumer, the daily pulse/metacog block (`main.py:1489-1491`).
- `ACTIONS_PRESERVE_GENERIC_NOTIFY_ENABLED` is shared with workflow schedule attention alerts (`main.py:993`), so this PR leaves it alone.
- orion-world-pulse had a `POST /api/world-pulse/runs/{run_id}/publish-email` route. It was disabled by default, and nothing in the repo called it.

## Architecture touched

- orion-actions: config default only.
- orion-world-pulse: route, service, settings, and env removed.
- `orion/schemas`: one model and its registry entry removed.

## Files changed

- `services/orion-actions/app/settings.py`: default changed to `False`, with a comment.
- `services/orion-actions/.env_example`: value set to `false`, with a comment.
- `services/orion-actions/README.md`: says the daily emails are retired and what the flag does and does not gate.
- `services/orion-actions/tests/test_async_notify_producers.py`: regression test.
- `services/orion-world-pulse/app/routers/publish.py`: `publish-email` route removed. The hub publish route stays.
- `services/orion-world-pulse/app/services/publish_email.py`: deleted.
- `services/orion-world-pulse/app/wiring.py`: deleted. Its only caller was the email route.
- `services/orion-world-pulse/app/services/renderers.py`: `render_email_digest` removed.
- `services/orion-world-pulse/app/settings.py`, `.env_example`: `WORLD_PULSE_EMAIL_ENABLED`, `WORLD_PULSE_EMAIL_DRY_RUN`, `WORLD_PULSE_NOTIFY_URL`, `NOTIFY_URL`, and `NOTIFY_API_TOKEN` removed.
- `services/orion-world-pulse/tests/test_safety_defaults.py`, `test_publish_paths.py`: email tests removed, plus a regression test that the route and settings stay gone.
- `scripts/world_pulse_integration_smoke.py`: email-preview check removed.
- `orion/schemas/world_pulse.py`, `orion/schemas/registry.py`: `EmailWorldPulseRenderV1` removed.
- `docs/world_pulse_dev.md`: email references removed, and a note added saying where news email comes from now.

## Schema / bus / API changes

- Added: none.
- Removed: `EmailWorldPulseRenderV1` (no producer or consumer outside world-pulse) and the HTTP route `POST /api/world-pulse/runs/{run_id}/publish-email`.
- Renamed: none.
- Behavior changed: daily pulse/metacog `NotificationRequest.channels_requested` is `None` by default instead of `["email"]`.
- Compatibility notes: `WorldPulseRunV1.email_status` and the digest `email_status` fields are kept on purpose. Those models use `extra="forbid"`, and payloads that were already stored include the field.

## Env/config changes

- Added keys: none.
- Removed keys (orion-world-pulse): `WORLD_PULSE_EMAIL_ENABLED`, `WORLD_PULSE_EMAIL_DRY_RUN`, `WORLD_PULSE_NOTIFY_URL`, `NOTIFY_URL`, `NOTIFY_API_TOKEN`.
- Changed default (orion-actions): `ACTIONS_DAILY_EMAIL_ENABLED` `true` -> `false`.
- `.env_example` updated: yes, both services.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: I ran it, but the script does not overwrite existing values or delete keys, so I applied these by hand in `/mnt/scripts/Orion-Sapienform/services/<svc>/.env`:
  - orion-actions: `ACTIONS_DAILY_EMAIL_ENABLED=true` changed to `false`.
  - orion-world-pulse: deleted `WORLD_PULSE_EMAIL_ENABLED=false`, `WORLD_PULSE_EMAIL_DRY_RUN=true`, the `# Optional Notify endpoint...` comment, `WORLD_PULSE_NOTIFY_URL`, `NOTIFY_URL`, and the empty `NOTIFY_API_TOKEN=`.
  - Both edits are harmless against the code currently deployed: the flag only suppresses the two emails, and the removed world-pulse keys already matched the old defaults.
- Skipped keys requiring operator action: none. Until this merges, `check_env_template_parity` warns (without blocking) that the world-pulse `.env` is "missing" the 5 removed keys. That is because the primary checkout still has main's `.env_example`.

## Tests run

```text
cd services/orion-actions && PYTHONPATH=<wt>:. pytest tests -q            -> 202 passed
pytest services/orion-world-pulse/tests (from repo root)                  -> 82 passed
  (run from the service dir, 2 pipeline fixture tests fail with source_registry_error because
   sources.yaml is looked up relative to the working directory; they fail the same way on origin/main)
scripts/world_pulse_integration_smoke.py --dry-run --approved-sources     -> PASS (reviewer run)
pytest tests/test_agent_trace_schema_registry.py                         -> 2 passed
git diff --check                                                          -> clean
check_env_template_parity                                                 -> PASS
check_metric_lineage --gate / check_definition_drift --gate               -> PASS (no metric touched; no re-lock)
check_journal_dispatch_registry, check_service_hostname_refs,
check_async_routes_not_blocking, check_system_health_producers,
check_scripts_dir_no_stdlib_shadow                                        -> OK
check_daily_schedule_collisions (report-only)                             -> Daily Journal <-> Daily Pulse 0m apart (pre-existing, see Risks)
```

## Evals run

```text
No eval harness exists for this seam. The change removes a delivery channel; it adds no cognition behavior to evaluate.
```

## Docker/build/smoke checks

```text
Not run. The task said not to deploy. The live effect is UNVERIFIED until the restart below.
```

## Review findings fixed

The code review ran in a subagent against `origin/main...HEAD`. It found no MUST issues.

- Finding: the PR report was not committed yet.
  - Fix: it is committed in this branch.
  - Evidence: this file.
- Finding: changing the default does not reach a host whose `.env` already has `ACTIONS_DAILY_EMAIL_ENABLED=true`.
  - Fix: the local primary `.env` is set to `false`. The per-host check is listed under Restart required.
  - Evidence: `services/orion-actions/.env:84`.
- Finding (NIT): `WorldPulseRunV1.email_status` now reads "pending" forever and looks like a real in-flight state.
  - Fix: added a comment saying it is retired and always "pending".
  - Evidence: `orion/schemas/world_pulse.py`.
- Finding (NIT): a dormant notify rule could start emailing the daily chat copies later.
  - Fix: recorded under Risks.
- Finding (NIT): the second half of the regression test cannot fail on its own.
  - Not changed. The default-revert case is still caught by the `Settings(_env_file=None)` assertion and the `.env_example` assertion, and the `main.py:1489` call site is unchanged.
- Verified by the reviewer:
  - No live references remain to any removed symbol, key, or route.
  - orion-notify does not email an info request with `channels_requested=None` on any path (`/notify`, policy, digest, escalation).
  - `ACTIONS_DAILY_EMAIL_ENABLED` gates nothing besides the two daily emails.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-actions up -d --build
./scripts/safe_docker_build.sh orion-world-pulse up -d --build
```

Other hosts running orion-actions: check `grep ACTIONS_DAILY_EMAIL_ENABLED services/orion-actions/.env` and set it to `false` if it says `true`, because the sync script does not overwrite existing values.

Live proof after restart: the next 08:30/20:15 notify rows for `orion.daily.pulse` / `orion.daily.metacog` should show `email_status=skipped`, and the Hub notification and chat message should still appear.

## Risks / concerns

- Severity: low
  - Concern: the daily journal trigger reuses `ACTIONS_DAILY_PULSE_HOUR_LOCAL`/`_MINUTE_LOCAL` (`main.py:2237`), so it fires at the same minute as Daily Pulse.
  - Mitigation: not fixed here, to keep `main.py` untouched. It is noted in the README and `check_daily_schedule_collisions` already reports it. It becomes more relevant once "Orion's Day" takes 08:30.
- Severity: low
  - Concern: anyone who sets `ACTIONS_DAILY_EMAIL_ENABLED=true` gets the JSON emails back.
  - Mitigation: this is a deliberate reversible switch, and the README and `.env_example` both say it is retired.

- Severity: low
  - Concern: the orion-notify rule `chat_message_default` (`services/orion-notify/app/policy/rules.yaml`) has `escalation_channels: ["email"]` with a 60-minute read-receipt deadline. Nothing emails an unread chat message today (`CHAT_MESSAGE_ESCALATION_EVENT_KIND` is defined but never used). If someone wires up message escalation later, the daily pulse/metacog chat copies would start emailing again.
  - Mitigation: none needed now. Anyone adding chat-message escalation should exclude the `daily` tags.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2429

🤖 Generated with [Claude Code](https://claude.com/claude-code)

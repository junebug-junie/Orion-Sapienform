## Summary

- orion-actions no longer writes the metacog digest journal. It stopped listening on `orion:equilibrium:metacog:trigger`, and the handler, `build_metacog_trigger`, and `ACTIONS_JOURNAL_METACOG_RECALL_PROFILE` are deleted.
- The "Orion — Journal Pass" emails from that journal stop.
- `daily_summary` and `notify_summary` journals still get written but no longer email. They had stayed out of the inbox only because metacog won the shared one-email-a-day slot every day.
- The world pulse digest email is unchanged.

## Outcome moved

Juniper got up to 6 "Journal Pass" emails a day (35 in 14 days) that retold telemetry with no grounding, for example: "pressure is low, exit code -9, I should review the logs". Behind them, orion-actions made about 140 LLM calls a day (1,945 in 14 days, 103k entries total) to write that prose. Both stop.

## Current architecture

equilibrium publishes `MetacogTriggerV1`. orion-actions `_handle_journal_metacog` composed a journal through cortex, sql-writer persisted it, and `_dispatch_journal_notifications` emailed it under `JOURNAL_DISPATCH_REGISTRY["metacog_digest"]`. The shared daily cap leaked: some days sent 3 to 6 emails.

## Architecture touched

- orion-actions: subscription, handler, settings.
- `orion/journaler`: trigger builder and dispatch registry.
- `orion/bus/channels.yaml`: consumer list.

## Files changed

- `services/orion-actions/app/main.py`: remove the handler, the gate, the dispatch branch, and the recall-profile branch.
- `services/orion-actions/app/settings.py`, `.env_example`: drop the channel and `ACTIONS_JOURNAL_METACOG_RECALL_PROFILE`.
- `orion/journaler/worker.py`, `__init__.py`: remove `build_metacog_trigger`.
- `orion/journaler/dispatch_registry.py`: metacog_digest, daily_summary, and notify_summary now have email off. The metacog row stays because old rows and the schema Literal still carry the kind, and `check_journal_dispatch_registry.py` needs a row.
- `orion/bus/channels.yaml`: orion-actions is no longer a consumer.
- `config/metrics/metric_definitions.lock.json`: re-locked for the consumer change.
- `services/orion-actions/tests/test_journal_actions.py`: the cap tests now use kinds that still email. New tests cover the silenced kinds and the removed subscription.
- `services/orion-actions/tests/test_metacog_self_signal_episode_journal.py`: deleted. It tested the removed gate.
- `services/orion-actions/README.md`: documents the removal.

## Schema / bus / API changes

- Removed: the orion-actions consumer of `orion:equilibrium:metacog:trigger`. The producer is unchanged. cortex-orch, sql-writer, and the hub notifier still consume it.
- Compatibility notes: the `metacog_digest` Literal is kept for old rows. Old `source_kind='metacog'` journal rows stay, and the personality graph keeps the 747 evidence rows it already has from them.

## Env/config changes

- Removed keys: `ACTIONS_JOURNAL_METACOG_RECALL_PROFILE`.
- Changed: `ACTIONS_SUBSCRIBE_CHANNELS` no longer lists `orion:equilibrium:metacog:trigger`.
- `.env_example` updated: yes.
- Local `.env`: the sync script does not overwrite diverged values, so both lines were edited by hand in `services/orion-actions/.env`.

## Tests run

```text
services/orion-actions/tests: 257 passed
orion/journaler: 4 passed
services/orion-cortex-exec/tests/test_executor_journal_pageindex_skip.py: 4 passed
scripts/check_journal_dispatch_registry.py: OK (14 kinds)
scripts/check_env_template_parity.py: PASS
tests/test_journaler_worker.py: 1 failure, also fails on main (unrelated parse-error test)
```

## Evals run

```text
None. This removes a producer; there is no quality behavior to evaluate.
```

## Docker/build/smoke checks

```text
Not run. Post-merge live check: no new journal_entries with source_kind='metacog'
after the orion-actions redeploy, and no orion.journal.notify rows with
trigger_kind=metacog_digest.
```

## Review findings fixed

- Finding: the definition-drift CI gate failed because the consumer list changed without re-locking.
  - Fix: ran `scripts/check_definition_drift.py --update` and committed the lock.
  - Evidence: `definition drift gate: PASS`.
- Finding: `channels.yaml` still had a comment calling `_handle_journal_metacog` live.
  - Fix: removed the comment.
- Nits fixed: the `_journal_email_daily_cap_key` docstring, the long line in the registry comment, and the docstring of the double-email test.

## Restart required

```bash
scripts/safe_docker_build.sh orion-actions up -d --build
```

## Risks / concerns

- Severity: medium.
  - Concern: the personality graph loses its largest journal feed (41% of evidence so far came from metacog narration).
  - Mitigation: that evidence was telemetry retelling. The other lanes (reverie, reading, self-study, chat, dreams) keep running.
- Severity: low.
  - Concern: the shared daily cap leak still exists for the kinds that still email (manual, autonomy_episode, collapse_response).
  - Mitigation: those kinds are rare or dormant. Not fixed here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2896

🤖 Generated with [Claude Code](https://claude.com/claude-code)

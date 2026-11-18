## Summary

Orion asked a fourth round of questions on two session threads (PRs #3561, #3598). This PR fixes the 2 real bugs and makes 3 small report changes. The design questions went to #3531 and #3595.

- **Bug A.** Promise acts could open from turns that were not Juniper's own browser turns. Hub opens acts only for a turn that got the pre-turn promise view, and any `utterance_origin="juniper"` turn got that view. That includes every `/api/chat` call. On 11-15, two coding-session notes were posted that way into Juniper's own session and saved as `hub_orion` rows. Now the view, and so act-opening, also needs a Hub browser surface (`surface_context`). This is the same rule the report's guard counts by. Session round-ups and Prime's `intention.carry` were already outside the rule, because they run as `utterance_origin="orion"`.
- **Bug B.** Several ways for a turn to end after the view read still logged no `promise_ledger` line: the governor returned no run (a timeout or an error reply), a harness error, a raise or cancel anywhere after the view read, and a failed background publish. Each now logs one line: `outcome=skipped` with `reason=no_run|harness_error|interrupted`, or `outcome=failed`.
- **Report.** The report prints the baseline and the keep-rate ceiling on both bases: 23/289 and 9/23, and the clean 21/287 and 9/21. Red is still judged on the frozen basis.
- **Report.** The report prints how many same-session pairs within 72 h the duplicate rule compares, beside the number of candidates.
- **Spec.** A round-3 amendment. It counts the abstentions (2 of the 23) and states that the ceiling already treats them as not chat-showable.

## Outcome moved

- A note posted to `/api/chat` into Juniper's session can no longer open a promise act graded against her. The `memory-scoreboard-eval` turns no longer read the view, so they stop adding `promise_view` lines with no matching `promise_ledger` line.
- Every turn that reads the view logs one `promise_ledger` line on every exit path we know of. Before the 17:58Z restart on 11-17, there were 29 view lines and 1 ledger line. All 28 silent turns were `memory-scoreboard-eval` `/api/chat` turns with `no_write`, which is the class #3619 fixed. One of them (a6dd7120) was still running when Hub restarted. After the restart, all 4 view reads ended with a ledger line.

## Current architecture

`execute_unified_turn` (`orion/hub/turn_orchestrator.py`) read the promise view when `utterance_origin == "juniper"` and both ledger flags were on. `_publish_promise_acts` opens acts only when that view exists. The browser-surface rule (`JUNIPER_INGRESS_PREDICATE`) existed only in SQL, in the report's guard. The skip line from #3619 covered only the early returns inside `_publish_unified_turn_chat_history`: `no_write`, an empty reply, and no bus.

## Architecture touched

- Hub turn path: one gate function and one skip-log function, plus 3 new skip reasons. No bus, schema or env change.
- `orion/regulation/juniper_turns.py`: `is_browser_surface_turn(payload)`, the payload form of `JUNIPER_INGRESS_PREDICATE`.
- Report: new JSON fields `guard.baseline_clean`, `guard.baseline_clean_rate`, `guard.ratio_clean`, `ledger.keep_rate_ceiling.clean_*` and `duplicate_window_pairs`.

## Files changed

- `orion/hub/turn_orchestrator.py`: adds `_promise_ledger_applies`, the gate for the view and therefore for act-opening. Adds `_log_promise_ledger_skip` for the `no_run` and `harness_error` paths; the `no_write` / empty-reply / no-bus path now uses it too. `execute_unified_turn` is now a thin wrapper around `_execute_unified_turn_body`. It keeps per-turn state in a ContextVar and logs `reason=interrupted` for a viewed turn that raises or is cancelled before its line. A failed background publish logs `outcome=failed`.
- `orion/regulation/juniper_turns.py`: adds `is_browser_surface_turn`.
- `services/orion-hub/tests/test_promise_ledger_turn_gate.py`: new tests that run `execute_unified_turn` end to end. They cover which turns read the view, and the 3 new skip lines.
- `services/orion-hub/tests/test_mind_tier_by_caller.py`: the call-site table names `_execute_unified_turn_body`, where the stance request is now built.
- `scripts/report_promise_ledger.py`: adds `REPLAY_BASELINE_CLEAN`, `KEEP_RATE_CEILING_CLEAN` and `duplicate_window_pairs`, plus the text lines that print them.
- `tests/test_report_promise_ledger.py`: tests for the clean basis and the window-pair count, including the live float-up / won't-float pair.
- `docs/superpowers/specs/2026-11-11-intention-ledger-design.md`: the round-3 amendments.

## Schema / bus / API changes

- Added: none on the bus. New report JSON fields only.
- Removed: none
- Renamed: none
- Behavior changed: `/api/chat` turns and other turns without `surface_context` no longer read the promise view, and their replies open no promise acts. Viewed turns log a ledger line on every exit: `no_run`, `harness_error`, `interrupted`, or `outcome=failed`.
- Compatibility notes: on the live data, every hub_orion Juniper prompt row in the last 30 days that came from a browser carried `surface_context` (181 of 181). The only 2 rows without it are the 11-15 coding-session notes.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
services/orion-hub: test_promise_ledger_turn_gate, test_promise_ledger_hub, test_turn_orchestrator_utterance_origin,
  test_turn_orchestrator_ws_frames, test_turn_orchestrator_cockpit_hops, test_unified_turn_records_user_turn,
  test_handle_chat_request_http_fallback_tts, test_handle_chat_request_orion_mode_degraded,
  test_handle_chat_request_orion_mode_continuity, test_curiosity_investigation, test_curiosity_urgent_turn,
  test_world_pulse_read_pipeline, test_turn_orchestrator_stance_recent_conversation,
  test_turn_orchestrator_role_teach_disclosure, test_curiosity_self_inquiry -> 423 passed
orion/harness/tests/test_promise_gate.py orion/harness/evals/test_promise_ledger_replay_eval.py
  orion/grammar/tests/test_promise_acts.py orion/self_monitor/tests/test_self_monitor_promise_shadow.py
  tests/test_report_promise_ledger.py orion/regulation -> 105 passed
tests/test_no_personal_terms.py tests/test_no_private_eval_artifacts.py -> 10 passed
static: check_definition_drift --gate, check_metric_lineage --gate, check_scripts_dir_no_stdlib_shadow,
  check_env_key_single_source, check_chat_route_poachers, check_async_routes_not_blocking, check_personal_terms,
  check_integration_seam, git diff --check -> all pass
mutation: the browser test removed from the gate -> 2 gate cases fail; the no_run skip line removed -> its case fails;
  the wrapper narrowed to `except Exception` -> the CancelledError case fails; the "viewed" mark removed -> 3 interrupted cases fail
```

## Evals run

```text
No finder eval was rerun. The finder is unchanged.
Replay data (read-only, /tmp/promise-ledger-replay-v3): the 289 judged turns were rebuilt exactly (179/86/24).
This rebuild finds 24 firm items, not 23 (the dev reruns differ by file set). The frozen 23 stands.
Duplicate rule on those 24 items: 44 same-session pairs within 72 h, 0 candidates. The float-up / won't-float
pair is not flagged: its longest shared run is 11 characters, and act and done_when differ.
Live (read-only): no Juniper turn since the finder went live stored any promise item (7 turns, 6 judged).
```

## Docker/build/smoke checks

```text
No build in this worktree. The change reaches the Hub at its next post-pull rebuild.
Live report (read-only): scripts/report_promise_ledger.py --days 28
  -> 0 acts; ceiling 0.39 (9/23; clean 0.43 9/21); guard 0 firm over 6 judged, ratio 0.0
     [clean baseline 0.0732]; not armed; duplicate candidates 0 of 0 same-session pairs within 72 h
```

## Review findings fixed

A code review ran in a subagent. It found no blockers. It confirmed that no browser send path drops `surface_context`: typed WS, spoken, draft-first, and the HTTP fallback all keep it.

- Finding (should): a raise or cancel between the view read and the governor call ended silent. Examples: request validation, the cockpit frame delivery, a browser disconnect.
  - Fix: `execute_unified_turn` wraps the whole turn. It logs `reason=interrupted` for any viewed turn that raises or is cancelled before its line.
  - Evidence: `test_a_raise_between_the_view_and_the_governor_still_logs_one_line`.
- Finding (should): a failed background publish logged only `promise_acts_failed`. A cancel during the history publish ended silent.
  - Fix: the background publish now logs `promise_ledger ... outcome=failed reason=<type>`. The wrapper covers the cancel.
  - Evidence: `test_a_failed_background_publish_logs_a_failed_ledger_line`.
- Finding (should): the interrupted test used only `RuntimeError`, so narrowing the handler to `except Exception` would still pass.
  - Fix: the test is parametrized with `asyncio.CancelledError`.
  - Evidence: mutation to `except Exception` -> the CancelledError case fails.
- Finding (nit): `run is None` also covers an error reply, not only a timeout.
  - Fix: the reason is now `no_run`.
- Finding (nit): a background task could leak between tests, and the `no_write` /api/chat class was not tested.
  - Fix: an autouse async fixture drains `_PROMISE_ACT_TASKS`. Added the `({"no_write": True}, "juniper")` case.
- Finding (nit): `ratio_clean` ignores a passed `baseline`.
  - Fix: the docstring says so. It is a fixed side number, not a second guard.
- Finding (nit, by design): any `/api/chat` client can send a `surface_context`.
  - Fix: no change. It matches `JUNIPER_INGRESS_PREDICATE`, so the guard and the gate cannot disagree.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run this from the clean primary `main` checkout after merge. A post-pull rebuild does the same.

## Risks / concerns

- Severity: medium
  - Concern: a real Juniper browser turn that arrives without `surface_context` (an older cached client, or a new surface) would read no view and open no act. It would fail quietly, toward fewer acts.
  - Mitigation: 181 of 181 browser prompt rows in the last 30 days carried it. The report's guard already used this same rule, so the guard and the ledger now agree on which turns count.
- Severity: low
  - Concern: the clean ceiling 9/21 comes from one reader's hand count, not a blind one.
  - Mitigation: the report says so in its `source` field. #3531 has the blind-pass plan.
- Issues: none opened or closed. Comments added on #3531 (blind pass, definition, kappa, report timing) and #3595 (abstentions, carry delivery after a void close, the two records, the baseline while Prime runs).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3641

🤖 Generated with [Claude Code](https://claude.com/claude-code)

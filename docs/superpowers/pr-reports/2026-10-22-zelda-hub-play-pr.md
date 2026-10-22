# Zelda play, Lane B (Hub): planner v2, solo leisure runner, play panel

Spec: [Orion plays The Legend of Zelda](../specs/2026-10-22-orion-plays-zelda-design.md) (PR #2846),
sections D and E, the Hub view, and acceptance checks 4 and 5.

**Merge order: this PR depends on Lane A (#2851, `feat/zelda-retro-core`) merging first.** The Hub reads
Lane A's Retro contracts: GoalV2, the `zelda-1-ram-v1` observation, `GET /sessions/{id}/atlas`, and
the `leisure_id` sponsor. Before Lane A is deployed, the Zelda paths here stay idle. Nova co-play
is unchanged. The shapes were checked against Lane A's PR #2851 (`feat/zelda-retro-core` at
`f20bb00d2`).

## Summary

- **Planner v2** (`scripts/retro_planner.py`, `plan_zelda`). Retro offers a decision. The Hub
  answers each decision once, with at most one request in flight. The prompt holds a text map of
  the room, the atlas neighbourhood and the last 5 verdicts. The planner uses the local GPU
  (`metacog_background`) only. A heat shed returns `gpu_pool_unavailable` with
  `details.reason` and `retry_after_sec`, and retries go through the shared `ShedBackoff`. A bad
  answer or an unknown observation produces no goal. There is no fallback goal and no fallback
  planner.
- **Leisure runner** (`scripts/retro_leisure.py`). It is a server-side sponsor that holds a
  `leisure_id` grant. Lane C's dispatch consumer calls `should_start(config)`, then
  `run_leisure_session(config, act_id=...)`. Every doer gate has its own named reason in the
  journal. The gates are re-checked every 30 s. Caps: one start per 3 h, 90 min per rolling
  24 h, 30 min per session. The session ends after 3 consecutive sheds. The runner loads the
  newest save, or starts fresh, and saves at the end.
- **Human safety.** A human acquire or takeover preempts leisure, through Lane A's arbiter. The
  runner never stops or saves over a game when a human may be in control. It only releases its
  own port.
- **Play panel** (`/retro`). It shows the atlas grid (known rooms, current room, used exits, caves,
  deaths), the goal with its reason and prediction, and the last 5 verdicts. During a heat shed
  it reads "Orion is waiting: cabinet hot". A leisure session is shown live when the page opens,
  by watching only (media socket). Takeover still works.
- **Shared presence store** (`scripts/retro_presence.py`, Redis). Browser traffic reaches
  hub-retro-gateway, and the runner may run in hub-app. This store is how the two processes agree
  on "human present" and on the leisure status and ledger.
- **Session-end data** (`scripts/retro_session_end.py`, `RetroSessionEnded`). This is the data
  contract for Lane C's summary and memory card. It has no default writer and no no-op hook.

## Follow-up for Lane C (#2854)

- `RetroSessionEnded` adds `rooms_new`, `deaths` and `controller_frames`. They are read from
  Retro's `zelda.atlas.session` (`retro_session_end.atlas_session_fields`). A value Retro did
  not report stays None, never 0. `leisure_id` is now Optional.
- `controller_frames` comes from Retro's `frames_by_controller` (`{human, script, leisure, none}`,
  Lane A 954909f7a). A frame counts as human when its tag is `human`.
- Co-play records report the change during the handoff, not Retro's session-wide totals. If Retro
  reset its record (a save load), the field is None.
- Second review of this follow-up (subagent): no blockers. Fixed:
  - the real key name, so counts were previously always None;
  - per-handoff deltas;
  - shed/rejection counts keyed by `(sid, port, epoch)`;
  - a watcher crash is now logged instead of dying silently.
- Co-play session end: the relay watches every browser cognition handoff on a Zelda session
  (`watch_coplay`, 2 s polls) until Retro ends it. The causes are stop, takeover, expiry,
  disconnect, or the session ending. It then calls `app.state.retro_session_end_sink(ended)`,
  which may be sync or async.
  - The record has `leisure_id=None` and `controller='co_play'`.
  - `human_may_be_in_control` is True unless Retro's counts show zero human frames.
  - When no sink is registered, nothing is written.
  - A sink error is logged and never touches play.
- Leisure end records also carry the atlas fields. Human frames in the record count as "a human
  may be in control", so the runner does not save over or stop that game.
- The planner's room map now uses Retro's `attempt.observation.terrain` (#/./x/C), with Link,
  enemies and drops drawn on top. The terrain is not repeated in the observation JSON.

## Outcome moved

Before: Retro play could only run with a browser sponsoring it, one Nova goal every 5 s.

After: the Hub can plan Zelda goals once per room or event, with checkable predictions. It can
show Orion's own map and verdicts. It can also run a solo session that is capped, preemptible and
journaled whenever the allocator admits `play` and the cabinet is cool.

## Current architecture

- `retro_routes.py` held a same-origin relay with a 5-second Nova planner. `retro_planner.py`
  held `choose`, a Nova call with five goals.
- Nothing ran without a browser connection. Hub had no record of human presence that both
  processes could see.

## Architecture touched

- Hub-only. No bus channel, schema registry, or `orion/schemas` change.
- New Redis keys, documented in `retro_presence.py` and the runbook:
  - `orion:retro:presence:{human_at,browser:<id>}`
  - `orion:retro:leisure:{status,ledger,journal,lock}`
- These are Hub-internal state. They are not bus contracts.
- Lane C owns `play` in `orion/schemas/grammar.py` and the `retro_leisure_session` dispatch route.
  This PR touches neither.

## Files changed

- `services/orion-hub/scripts/retro_planner.py`: planner v2 (GoalV2 models, room map, atlas
  neighbourhood, history, validation that mirrors Retro's own checks, shed and backoff, single
  flight). Nova `choose` is unchanged.
- `services/orion-hub/scripts/retro_leisure.py`: the pure gate `leisure_block_reason`, plus
  `should_start`, `run_leisure_session`, `RetroClient` and `read_regulation`.
- `services/orion-hub/scripts/retro_presence.py`: Redis presence, status, ledger, journal and
  lock. Reads fail closed. Lock refresh and release are atomic Lua compare steps.
- `services/orion-hub/scripts/retro_session_end.py`: `RetroSessionEnded`.
- `services/orion-hub/scripts/retro_routes.py`:
  - Zelda `/plan` path, and the `/atlas` GET proxy.
  - `GET /api/retro/leisure`, which reports a runner that stopped updating as `stale`.
  - A human mark on every proxied POST and on input-socket open, and a browser heartbeat per
    relay socket.
- `services/orion-hub/templates/retro.html`, `static/js/retro-panel.js` (asset `v=8`): the play
  panel, watch mode, the co-play v2 planner with shed wait, and client-side verdict history.
- `services/orion-hub/app/settings.py`, `.env_example`: `RETRO_LEISURE_*`.
- `services/orion-hub/tests/retro/test_{planner_v2,leisure,play_panel_static}.py`, plus one
  intended route-set update in `test_boundary.py`.
- `services/orion-hub/evals/retro_play_panel.{py,mjs}`: a browser eval (real template, asset and
  router; fixture Retro).
- `.github/workflows/retro-preflight.yml`: path filters for the new files.
- `docs/runbooks/retro-browser.md`, `services/orion-hub/README.md`: operator docs.

## Schema / bus / API changes

- Added (Hub HTTP):
  - `GET /api/retro/leisure`.
  - `GET /api/retro/sessions/{id}/atlas`, proxied.
  - The Zelda branch of `POST /api/retro/sessions/{id}/plan`. On a shed it returns `503
    {detail:"gpu_pool_unavailable", reason, retry_after_sec}` with a `Retry-After` header.
- Added (Python, for Lane C):
  - `should_start(config, deps=None) -> str | None`
  - `run_leisure_session(config, *, act_id=None, deps=None) -> LeisureOutcome(started, refusal, ended: RetroSessionEnded | None)`
- Removed / renamed: none.
- Behavior changed: every proxied Retro POST now writes a best-effort "human action" mark (1 s
  bound, never blocks play).
- Compatibility: Nova co-play is unchanged. The Zelda branch is chosen only when
  `cognition.policy_id == "zelda-navigator-v1"`.

## Env/config changes

- Added keys: `RETRO_LEISURE_ENABLED=true`, `RETRO_LEISURE_GAME_IDS=the-legend-of-zelda`,
  `RETRO_LEISURE_SESSION_MAX_SEC=1800`, `RETRO_LEISURE_DAILY_MAX_SEC=5400`,
  `RETRO_LEISURE_MIN_GAP_SEC=10800`. They ship on. There is no port key: Lane A's Zelda motor
  reads controller 1 only, so leisure always uses port 0.
- Removed / renamed: none.
- `.env_example` updated: yes (`services/orion-hub/.env_example`).
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: run. The five keys were
  also appended to the live main checkout's `services/orion-hub/.env`. An earlier
  `RETRO_LEISURE_PORT` line was removed from that file again.
- Skipped keys requiring operator action: none.

## Tests run

```text
python -m pytest services/orion-hub/tests/retro -q                     89 passed (py3.12)
uv run --python 3.11 (CI's native-boundary deps) pytest .../tests/retro  89 passed
python -m pytest services/orion-hub/tests/test_regulation_must_not_readers.py \
  services/orion-hub/tests/test_regulation_arousal_readers.py -q        77 passed
python -m pytest tests/test_no_personal_terms.py -q                     7 passed
python scripts/check_settings_defaults.py orion-hub --example-drift     OK
python scripts/check_env_template_parity.py                             PASS
python scripts/check_env_key_single_source.py                           OK
node --check static/js/retro-panel.js; node --check evals/retro_play_panel.mjs   OK
git diff --check                                                        clean
Mutation: making _finish stop the session in every mode -> 4 leisure tests fail.
```

## Evals run

```text
python services/orion-hub/evals/retro_play_panel.py --output <scratch>/ppanel2   PASS
  The rendered /retro loads retro-panel.js?v=8. Leisure is visible on open (media socket only,
  no input socket, no control, no human mark). The atlas shows 3 known rooms, the current room
  119, N/E exits, a cave and x2 deaths. Goal, rationale, prediction and 5 verdicts render; the
  page keeps them itself, because Retro shows only last_finished. The shed shows "Orion is
  waiting: cabinet hot". Take this controller -> input socket + control acquire + human mark.
```

The existing native browser eval (`evals/retro_browser.py`) needs the retro container and Pad Lab.
It was not re-run, because the human play path is unchanged apart from the watch guard.

## Docker/build/smoke checks

```text
Not run. No deploy (instructed). No compose or Dockerfile change. hub-app and hub-retro-gateway
already carry ORION_BUS_URL and redis[hiredis].
```

## Review findings fixed

The review ran in a subagent (orion-repo-agent, review-only), against the branch diff vs
origin/main.

- Finding (blocker): when the policy disappeared and the human mark was late, or Redis failed,
  the runner ended with `policy_ended_by_retro` and stopped the human's game.
  - Fix: three end modes. When the policy is gone, the runner never stops the game. When a
    human acted since the start, or presence cannot be read, the runner neither saves nor stops.
  - Evidence: `test_policy_gone_with_unreadable_presence_fails_safe` and
    `test_policy_gone_without_a_human_is_not_called_preemption`, which asserts no `/stop`. The
    mutation check above makes them fail.
- Finding: transport errors and cancellation escaped `_loop` without `_finish`. That left the
  policy driving, with no end-save, no journal row and no lock release.
  - Fix: per-poll retry (5 polls, then `retro_unreachable`). `_finish` runs under
    `asyncio.shield` on error and on `CancelledError`, which is then re-raised.
  - Evidence: `test_retro_transport_errors_retry_then_close_out` and
    `test_cancellation_still_releases_the_port_and_journals`.
- Finding: `human_present` (for example, the human pressed Pause) skipped the policy stop, so the
  port stayed held.
  - Fix: `RELEASE` mode posts the scripted stop and leaves the game alone.
  - Evidence: `test_human_action_without_takeover_releases_the_port_but_keeps_their_game`.
- Finding: the page's `watch()` could bump the transport epoch under a human's own `connect()`.
  - Fix: a guard on `connecting`, `wanted` and `healPending`.
  - Evidence: the browser eval takeover still passes.
- Finding: a stale-decision refusal from Retro counted as a planner failure.
  - Fix: `STALE_GOAL_REFUSALS` are logged as `goal_stale` and not counted.
  - Evidence: `test_stale_decision_refusal_is_play_not_failure`.
- Finding: the daily cap was checked only every 30 s.
  - Fix: the cap is checked on every poll.
  - Evidence: `test_daily_cap_is_enforced_every_poll`.
- Finding: the lock GET and EXPIRE/DEL were not one atomic step.
  - Fix: Lua compare-and-expire and compare-and-delete.
- Finding: a crashed runner's status read as `playing` for up to 1 h.
  - Fix: the route reports `stale` after 15 s with no update.
  - Evidence: a route test.
- Finding: an end-save on a preempted game can push out the human's oldest save.
  - Fix: the save is skipped when a human may be in control, with
    `end_save_error=skipped:human_may_be_in_control`.
- Second review (subagent, review-only, target = fix commit 90558aa04): no blockers. Fixed:
  - Retro drops a leisure policy after 60 s with no goal, so a long heat shed ends before the
    third shed. The game is then left paused with nobody in it, and that blocked every later
    start. Fix: the end reason is `shed_wait_policy_lost`. When no human acted, the runner closes
    the game, but only in the orphan state (paused, no script). Evidence:
    `test_long_shed_lets_retro_drop_the_policy_and_the_orphan_game_is_closed`.
  - A cancel during start or the ledger write skipped `_finish`. Evidence:
    `test_cancellation_during_start_still_closes_out`.
  - A non-200 status read (5xx, 401) counted as "game ended". It is now a transport failure
    (5 polls), unless Retro says `unknown_session`. Evidence:
    `test_non_200_status_is_a_transport_problem_not_a_finished_game`.
  - Retro's `observation_unknown` on a goal (Link not controllable during a long call) now counts
    as stale, not as a failure. `fight` with no enemy is rejected Hub-side (`planner_no_enemies`).
  - The script epoch is read from the scripted reply. The start loop resumes a paused session.
    `lineage_id` refreshes on every atlas read.
  - Fakes now match real Retro (60 s drop, neighbourhood atlas, off-grid sides absent).
- CI (Python 3.11): `asyncio.wait_for` can swallow a cancel. The loop now honours
  `Task.cancelling()`. It was reproduced and verified with `uv run --python 3.11`.
- Alignment to Lane A's real contract, found while the review ran:
  - Field names: `attempt` instead of `decision`, and `last_finished` instead of a history list.
  - The atlas endpoint returns a neighbourhood view.
  - Leisure acquires first (`control`), then starts the script with `expected_epoch`.
  - A prediction has a value only for `reach_room`.
  - The atlas field is `rooms_with_unvisited_sides`. GoalV2 uses port 0 only.
  - Cave items are not observed, so the prompt says that `collect` needs a tile the planner gives.
  - Death recovery is native in Lane A. The planner never answers while `observation.menu` is set
    or the mode is `death`/`menu`, and it makes no model call then. An outcome of `death` is only
    the reason for the next decision.
    Evidence: `test_unknown_observation_never_calls_the_model[menu/death]` and
    `test_death_is_only_a_trigger`.
  - The Hub uses Retro's tile convention.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build hub-app hub-retro-gateway
```

Run this from the clean primary main checkout, after Lane A's PR and this PR are merged and pulled.

## Risks / concerns

- Severity: medium.
  - Concern: acceptance check 4 (a real ≥10-minute local-GPU session: sword, 8 new rooms,
    verdicts, BLOCKED while hot) is UNVERIFIED. It needs Lane A deployed and the cabinet below
    the `cabinet_hot` clear line.
  - Mitigation: the shed path is tested deterministically.
- Severity: medium.
  - Concern: the live preemption half of check 5 ("within one frame") belongs to Lane A's
    arbiter. It is UNVERIFIED from the Hub side. The browser eval only shows that the takeover
    sends the acquire.
- Severity: low.
  - Concern: a long heat shed ends a leisure session after about 60 s (`shed_wait_policy_lost`).
    This is Retro's sponsor-loss rule, not "3 consecutive sheds". The spec's 3-shed end applies
    only while shed windows are shorter than Retro's 60 s.
- Severity: low.
  - Concern: the leisure path is idle until Lane C wires the dispatch consumer to `should_start`
    and `run_leisure_session`.
  - Concern: the leisure gate refuses while arousal is `unknown` (fail closed). A stale regulation
    producer therefore blocks leisure.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2850

🤖 Generated with [Claude Code](https://claude.com/claude-code)

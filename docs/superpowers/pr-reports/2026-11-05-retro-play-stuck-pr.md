# fix(hub/retro): Orion can play Zelda alone again

## Summary

Orion could not play Zelda on their own because of two bugs. The first bug caused the second.

- **Bug 1: every solo session quit after one tenth of a second.** Retro gives Orion the controller and answers at once. But Retro's session snapshot (the record Hub reads to check that Orion still holds the controller) is refreshed only about every 0.1 s. Hub's first check came 20-90 ms after the grant, so it read the older snapshot. That snapshot had no controller for Orion yet. Hub decided "Retro took the controller away" (`policy_ended_by_retro`) and quit with 0 decisions. Fix: Hub now treats a missing controller as lost only when it stays missing for 2 s with no human present. A human action still ends the session at once.
- **Bug 2: the game left behind blocked every later start for 6 days.** When Hub quit, Retro kept the controller for 60 s with nobody giving it goals. Then Retro dropped it and paused the game ("no controller"). Hub's start gate refuses to start while any game is active, to protect Juniper's games, so this paused game refused every start from 10-30 to now. Fix: the gate now recognises a game that is clearly Orion's own leftover. The runner stops that game and starts fresh from the newest save.
- A leftover counts as Orion's own only when all of these are true:
  - Retro shows it paused with no script.
  - Orion's leisure ledger row for that session says the run ended because the controller was gone (`policy_ended_by_retro` or `shed_wait_policy_lost`), or the row is still open long after the lock TTL and no runner holds the lock (Hub was killed mid-run).
  - Retro's port-0 frame counts show 0 human frames, 0 browser co-play frames, and more than 0 leisure frames.
- A run that ended because a human acted (`human_present`, for example Juniper pressed Pause) is never an orphan. Anything else still refuses.
- Before the stop, the runner checks again for a browser or a recent human. It reads the game again. It saves the game first when that run had placed goals. Then it journals `orphan_stopped` with the frame counts and the save result.
- Ledger rows now record `end_reason`, `mode` and `decisions`, so the next run can tell how the last one ended.

## Outcome moved

- Solo leisure sessions now last past the first poll and reach the planner. Before: 4 of 4 runs from 10-23 to 10-30 ended in 0.7-1.0 s with 0 decisions.
- From now on, a game an Orion run leaves behind no longer blocks the start timer.
- The game stuck today (f0ef1446) needs a one-time stop by an operator (command below). Its ledger row was written before this patch and does not record how its run ended. On purpose, the code does not guess for old rows.

## Live evidence (root cause)

- `orion:retro:leisure:journal`: each of the 4 runs went `started` and then `ended policy_ended_by_retro mode=policy_gone` 62-93 ms later. For example, f0ef1446 has `started` at .676 and `ended` at .769.
- Retro's own record for the same run (`~/.local/share/orion/retro/service/evidence/f0ef1446-....jsonl`): `policy_started` at 02:10:19.659, two `decision_offered` rows, and `policy_stopped reason=leisure_sponsor_lost` at 02:11:19.696, 60 s later. Then `transition paused reason=no_controller`. So Retro kept the controller for a full minute after Hub had already left.
- Live `GET /health/ready` today: session f0ef1446, `state=paused`, frame 3616. That is the same frame as the no-controller pause, so nobody touched it in 6 days. `frames_by_controller={human:0, script:0, leisure:3610, none:6}`.
- `own_orphan()` run on that live session and the live ledger returns `f0ef1446-1d67-4a91-96b6-d5db11f4769f`.
- `grammar_acts` rows with actor `hub.retro_leisure`: 4 play acts, all closed at 0 with `no_planner_decisions` 0.07-0.26 s after they opened.
- Retro code: `services/orion-retro/app/worker.py` sends the status snapshot at most every 0.1 s. The scripted-start reply is sent at once. `app/session.py` `monitor()` updates `session['scripted']` only from status messages. `app/scripted.py` `tick()` stops an idle leisure policy after 2 unanswered offers (`leisure_sponsor_lost`).
- Ruled out: the planner. Co-play used the same `retro_planner.plan_zelda` on 10-30 and made 21 decisions in one handoff. The dojo is not involved: the leisure runner does not use it. Heat is not involved: the runs were admitted past the heat gate, and `cabinet_hot` would be journaled under that name.

## Current architecture

Hub's leisure timer (`scripts/retro_leisure_self_start.py`, 300 s) calls `should_start()` and then `run_leisure_session()` (`scripts/retro_leisure.py`). The runner starts a Retro session, acquires a leisure lease, and hands it to a cognition script. Then it polls `GET /sessions/{sid}` every 0.25 s and plans a goal for each decision Retro offers.

## Architecture touched

Hub only (`services/orion-hub`). No change to Retro, bus, schema, or env.

## Files changed

- `services/orion-hub/scripts/retro_leisure.py`: `POLICY_GONE_GRACE_SEC`, the grace in `_loop`, `own_orphan()`/`_orphan_row()`, `LeisureGateInputs.orphan_session_id`, `_stop_orphan()`, the new ledger row fields, and the module doc.
- `services/orion-hub/scripts/retro_presence.py`: `PresenceStore.lock_free()` (a read of the leisure lock).
- `services/orion-hub/tests/retro/test_leisure.py`: FakeRetro can now serve a stale snapshot after the grant, as real Retro does. Adds regression and orphan tests.
- `docs/superpowers/pr-reports/2026-11-05-retro-play-stuck-pr.md`: this report.

## Schema / bus / API changes

- Added: none. A new journal event `orphan_stopped` goes in the existing Redis list `orion:retro:leisure:journal`.
- Removed / Renamed: none.
- Behavior changed: the gate no longer refuses for an active session that is Orion's own orphan. The runner may `POST /sessions/{sid}/stop` that game.
- Compatibility notes: none.

## Env/config changes

- Added keys / Removed keys / Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-hub && pytest tests/retro tests/test_retro_leisure_self_start.py tests/test_retro_leisure_self_start_health.py tests/test_physiology_shadow_bid.py -q
281 passed

python -P services/orion-hub/evals/retro_leisure_self_start_replay.py
fixture replay: 9/9 checks; 6 starts, 576 ticks

Against the old retro_leisure.py, the new tests fail (they reproduce the bugs):
FAILED test_a_snapshot_older_than_the_grant_is_not_a_lost_policy   ('policy_ended_by_retro' == 'session_cap')
FAILED test_orions_own_paused_leftover_is_stopped_and_play_starts  ('retro_session_active' is None)
FAILED test_a_policy_that_stays_missing_still_ends_after_the_grace
```

## Evals run

```text
No eval harness for solo leisure. The live check after deploy (below) is the eval: a leisure
session that lasts more than 2 s and has decisions_planned >= 1 in the journal and in retro_play_sessions.
```

## Docker/build/smoke checks

```text
Not run. Hub-only Python change, no dependency or compose change. UNVERIFIED live until deploy.
After deploy:
  redis-cli -u redis://100.92.216.81:6379/0 lrange orion:retro:leisure:journal 0 5
  expect: orphan_stopped for f0ef1446, then started, then goal_accepted rows.
```

## Review findings fixed

The review ran in a subagent. It found no must-fix bugs. It confirmed that the orphan path cannot stop a game a human is playing or co-playing: a held lease counts as `human` frames, and co-play counts as `script` frames. It confirmed with mutations that both regression tests pin the fix.

- Finding: a game Juniper paused on purpose (Pause needs no lease, so it has 0 human frames) would count as an orphan.
  - Fix: ledger rows now store `end_reason`/`mode`. Only a policy-gone ending counts. `human_present`/release endings and old rows with no reason refuse.
  - Evidence: parametrized cases in `test_anything_short_of_a_clear_orphan_still_refuses_and_touches_nothing`.
- Finding: stopping an orphan threw away unsaved play.
  - Fix: save before the stop when that run placed goals. Skip the save when it placed none, so a title screen never becomes the newest save.
  - Evidence: `test_an_orphan_whose_run_played_goals_is_saved_before_the_stop` (fails with the save removed).
- Finding: a Hub restart mid-run leaves an open ledger row, so the same 6-day block could come back.
  - Fix: an open row older than the session cap plus 300 s, with no lock held, counts as closed.
  - Evidence: `test_a_runner_killed_mid_run_leaves_an_orphan_once_its_lock_is_gone`.
- Finding: presence was not checked again between the gate and the stop.
  - Fix: `_stop_orphan` reads browsers and human presence again, and refuses if either changed.
  - Evidence: `test_presence_is_rechecked_just_before_an_orphan_is_stopped`.
- Finding (CI, retro-memory job): the self-start replay passes a store without `lock_free()`, and the gate crashed.
  - Fix: an unreadable lock means "not an orphan", so the start stays refused (fails closed).
  - Evidence: `test_a_store_without_the_lock_read_fails_closed`; replay 9/9.
- Nits fixed:
  - The grace test now has a lower bound.
  - New tests: a human during the grace still preempts at once; a policy seen again restarts the grace (fails with the reset removed); a failed orphan stop starts nothing and leaves the ledger as it was.
  - The docstring now says the frame counts cover port 0 only.

## Operator action (one time, for the game stuck since 10-30)

Safe because:
- Retro shows 0 human and 0 co-play frames in it.
- Its frame (3616) has not moved in 6 days.
- Orion's journal shows the run ended `policy_ended_by_retro` with 0 goals placed, so nothing unsaved is lost.
- The save it loaded (d052cb45) is still on disk.

```bash
curl -s --unix-socket ~/.local/share/orion/retro/service/ipc/retro.sock -H "Authorization: Bearer $(cat ~/.local/share/orion/retro/service/hub-auth/hub.token)" -H 'Content-Type: application/json' -X POST -d '{}' http://retro/sessions/f0ef1446-1d67-4a91-96b6-d5db11f4769f/stop
```

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run it from the clean primary main checkout after merge. The post-pull rebuild does the same thing.

## Risks / concerns

- Severity: low. Concern: the runner can now stop a paused game. Mitigation: it stops only a game with 0 human and 0 co-play frames that Orion's ledger shows Orion started and finished. Every presence gate still runs first, and the runner reads the game again just before it stops it. Tests cover each case where it must refuse.
- Severity: low. Concern: a real policy loss is now noticed up to 2 s later. Mitigation: Retro itself waits 60 s before it drops an idle leisure policy, and a human action still ends the session at once.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3142

🤖 Generated with [Claude Code](https://claude.com/claude-code)

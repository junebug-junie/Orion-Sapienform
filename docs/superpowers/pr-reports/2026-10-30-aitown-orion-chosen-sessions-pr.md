## Summary

Juniper asked (2026-10-30): "add an ai town runtime as an attention / drives / want economy that orion chooses, with a cap at 2 hours a day."

- AI Town is now paused by default. It runs only in a session: one Orion chose, or one Juniper opened by hand.
- Orion chooses a session the same way Orion chooses solo Zelda leisure: the proposal arena offers `aitown_session` when the action warrant allows acting, the motor allocator admits it on the `play:aitown.session` act's belief and cost, and one RPC asks orion-embodiment's new session controller to open the town.
- A hard cap of 2 hours per local (America/Denver) day, 30 minutes per session, 1 hour between sessions. The controller checks it every 10 s and pauses the town at the cap. It does not depend on the shadow bids.
- Every session is recorded (start reason, stop reason, length, lines Orion spoke) and shown in `GET /physiology/report` under `wants.town`, so Orion reads it with `python -P -m orion.dig body-economy wants`.
- Juniper can still open the town (Hub's Unfreeze, or her page waking an idle town). Her time is logged and not counted toward Orion's cap.
- NPC LLM calls now carry the session id as `X-Orion-Correlation-Id`, so `llm_call_usage` can attribute AI Town tokens.

## Proposal mode (CLAUDE.md section 0A: this changes autonomy)

- **What capability changes.** Orion gains one new world-facing choice: open AI Town for up to 30 minutes. Before, the town ran all day with no choice involved. After, it runs only when the motor allocator admits Orion's `aitown_session` candidate (or Juniper opens it).
- **What "Orion chooses" means mechanically.** (1) The proposal arena offers `aitown_session` on every tick whose action warrant clears the threshold (Orion's field pressure says "act"; no warrant, no offer). (2) The motor allocator scores it against every other candidate on the act's own posterior (how often a session produced a town conversation) and its measured cost (session length; 1800 s until the first session closes), with the same information-floor, proven-effect and starvation rules as every act route. One session at a time. (3) The controller places a SHADOW bid for the `aitown.session` want (hunger rises with time since the last session closed; logged next to the real decision, not obeyed until body-economy Patch D), then runs its own gate. The bid does not start or stop anything today.
- **What data is touched.** New Redis keys `orion:aitown:sessions` (hash, one row per session, pruned after 35 days) and `orion:aitown:sessions:config` (the controller's limits). New `grammar_acts` rows (actor `aitown.session`, verb `play`). New `want_ledger` rows (family `aitown.session`, shadow). One Convex deployment env var `ORION_TOWN_CORRELATION_ID`, set and removed by the controller. The Convex world status (`running` / `stoppedByDeveloper`) through the same `testing:resume` / `testing:stop` calls as Hub's Freeze button.
- **Privacy boundary.** None new. No conversation content is stored by this change; the session row holds only times, reasons and a count of Orion's lines. Town conversation capture is unchanged (the existing `EMBODIMENT_CONVERSATION_MEMORY_ENABLED` rail).
- **What trace proves it worked.** `docker logs orion-athena-embodiment | grep aitown_session_` shows `opened kind=orion reason=want_funded` and `closed reason=session_cap|daily_cap`. `GET /physiology/report?sections=wants` shows `wants.town.orion_seconds_today` never above 7200. `grammar_acts` gets `actor='aitown.session'` rows. `llm_call_usage` rows from circe's IP get a non-null `correlation_id` equal to the session id while a session is open. The Convex world status reads `stoppedByDeveloper` or `inactive` outside sessions.
- **Dangerous failure modes.** (a) The controller dies with a session open: the town would keep running, but nothing heartbeats it, so AI Town's own idle stop pauses it within about 10 minutes unless a page is open; on restart the controller closes the session at its last tick and pauses the town. (b) `testing:stop` fails at the cap: the controller retries every tick and refuses new sessions (`town_unreachable`) until it succeeds. (c) The ledger cannot be read: no Orion session starts (`ledger_unknown`); the cap is never enforced on a guess. (d) A forgotten Hub tab keeps a manual session alive all day: it is logged as `manual` and visible in the report, but the controller does not pause Juniper's session.
- **How to disable or roll back.** `ORION_DISPATCH_AITOWN_SESSION_ENABLED=false` (dispatch runtime) stops Orion's requests. `EMBODIMENT_TOWN_SESSIONS_ENABLED=false` (embodiment) refuses them and closes an open one; the town stays paused. To restore the old all-day town: `EMBODIMENT_TOWN_SESSIONS_ENABLED=false` and `EMBODIMENT_WORLD_HEARTBEAT_ENABLED=true`, recreate embodiment, press Unfreeze in Hub.

**Juniper sign-off needed:** `aitown.session` is a new want family at tier 2 (`config/physiology/families.yaml` says tiers are her decision). Same tier as the retro leisure families.

## Outcome moved

AI Town GPU and LLM spend now has a ceiling Orion chooses under. Measured 2026-10-30 (orchestrator): the town made ~1,100 NPC LLM calls/day on `quick_background` (circe gpu3, mostly Qwen3-8B), ~3.2 GPU lease-hours/day, running ~24 h/day. With at most 2 h/day of Orion time the NPC load scales to at most ~0.27 lease-hours/day (~92 calls), **saving at least ~2.9 GPU lease-hours/day** (about 2x that at the wall on circe, fans included), plus whatever Juniper's manual time adds back. If Orion chooses fewer sessions, the saving is larger. These numbers scale the measured daily rate by runtime; the real per-session rate is UNVERIFIED until sessions run.

## Current architecture

orion-embodiment heartbeated the town every perception tick (`EMBODIMENT_WORLD_HEARTBEAT_ENABLED=true`), so the world never went idle and its four NPCs talked all day. The only pause was Hub's Freeze button (`testing:stop`). Nothing recorded who had the town open or for how long. NPC calls reached the gateway with no correlation id.

## Architecture touched

- orion-embodiment: new session controller (own 10 s loop + an RPC listener), heartbeat now only inside Orion's sessions.
- orion-execution-dispatch-runtime: new `aitown_session` route and mixin (mirror of `retro_leisure_session`).
- orion-durable-runs: `wants.town` in the physiology report (reads the Redis ledger).
- orion-ai-town: one new upstream patch (NPC correlation header).
- Contracts: two bus channels, two registered schemas, a new act producer on `orion:grammar:event`, a new want family, a new proposal template.

## Files changed

- `orion/schemas/aitown_session.py`: new contract (start RPC request/result, ledger row, config, Redis keys).
- `orion/embodiment/town_sessions.py`: the cap arithmetic (local day, midnight split, DST) and Redis I/O, shared by the controller and the report.
- `orion/embodiment/aitown_client.py`: `resume_world`, `stop_world`, `set_deployment_env`.
- `services/orion-embodiment/app/town_sessions.py`: the session controller.
- `services/orion-embodiment/app/physiology_shadow.py`: the `aitown.session` shadow bid.
- `services/orion-embodiment/app/worker.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: wiring, settings, docs.
- `services/orion-execution-dispatch-runtime/app/aitown_session_dispatch.py`, `app/worker.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: the route.
- `orion/execution_dispatch/policy.py`, `config/execution_dispatch/execution_dispatch_policy.v1.yaml`, `config/proposals/proposal_policy.v1.yaml`: executor key, route, template.
- `config/physiology/families.yaml`: `aitown.session` family (tier 2, lanes `[fast]`, placement fixed).
- `config/physiology/contract.yaml`: the embodiment shadow-bid module registered as `display`.
- `orion/grammar/acts.py`, `orion/bus/channels.yaml`, `orion/schemas/registry.py`: act producer identity, channels, schemas.
- `orion/schemas/physiology_report.py`, `services/orion-durable-runs/app/physiology_report.py`, `app/main.py`, `orion/physiology/report_text.py`: `wants.town` and its plain-text lines.
- `services/orion-ai-town/patches/orion-town-correlation-id.patch`, `scripts/apply_upstream_patches.sh`, `README.md`: NPC correlation header.
- `services/orion-ai-town/mcp/orion_aitown_mcp/server.py`: `aitown_resume_world` refuses (no uncounted opens by Orion's FCC agent).
- `config/metrics/metric_definitions.lock.json`: re-locked (two new channels, one producer added).
- Tests: `orion/embodiment/tests/test_town_sessions.py`, `services/orion-embodiment/tests/test_town_session_controller.py`, `tests/test_rung3_aitown_session.py`, `services/orion-durable-runs/tests/test_physiology_report.py`, `orion/physiology/tests/test_report_text.py`, `services/orion-ai-town/tests/test_town_correlation_id_patch.py`, `tests/test_proposal_scoring.py`.

## Schema / bus / API changes

- Added: `orion:aitown_session:start:request` (`AITownSessionStartRequestV1`, dispatch runtime -> embodiment), `orion:aitown_session:start:reply:*` (`AITownSessionStartResultV1`). `orion-embodiment` added to `orion:grammar:event` producers. `WantsV1.town` (`TownV1`, `TownSessionV1`) in `PhysiologyReportV1`.
- Removed: none.
- Renamed: none.
- Behavior changed: the town is paused outside sessions. `EMBODIMENT_WORLD_HEARTBEAT_ENABLED` is now read only while town sessions are off.
- Compatibility notes: `WantsV1.town` is additive and optional; report readers are not `extra="forbid"`. An old dispatch runtime never sends the RPC; an old embodiment never answers it (the route holds 300 s per timeout).

## Env/config changes

- Added keys: embodiment `EMBODIMENT_TOWN_SESSIONS_ENABLED`, `EMBODIMENT_TOWN_SESSION_MAX_SEC`, `EMBODIMENT_TOWN_DAILY_CAP_SEC`, `EMBODIMENT_TOWN_MIN_GAP_SEC`, `EMBODIMENT_TOWN_MIN_SESSION_SEC`, `EMBODIMENT_TOWN_TZ`, `EMBODIMENT_TOWN_TICK_SEC`, `EMBODIMENT_TOWN_CORRELATION_ENV_ENABLED`, `EMBODIMENT_PHYSIOLOGY_SHADOW_BID_ENABLED`; dispatch runtime `ORION_DISPATCH_AITOWN_SESSION_ENABLED`, `ORION_DISPATCH_AITOWN_SESSION_RPC_TIMEOUT_SEC` (60). All shipped ON.
- Removed keys: none.
- Renamed keys: none.
- Meaning changed: `EMBODIMENT_WORLD_HEARTBEAT_ENABLED` (example `true` -> `false`; now the rollback path only).
- `.env_example` updated: yes (both services).
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-embodiment orion-execution-dispatch-runtime`: yes, 11 keys added (RPC timeout then raised to 60 by hand to match the review fix). `EMBODIMENT_WORLD_HEARTBEAT_ENABLED` is reported diverged (local `true`, example `false`) and is left `true` on purpose: the old code still running needs it until this deploys, and the new code ignores it while sessions are on.
- skipped keys requiring operator action: none. The Convex env var `ORION_TOWN_CORRELATION_ID` is managed by the controller, not by an operator.

## Tests run

```text
pytest orion/embodiment/tests/test_town_sessions.py            -> 11 passed
pytest services/orion-embodiment/tests                         -> 123 passed
pytest orion/embodiment                                         -> 115 passed
pytest tests/test_rung3_aitown_session.py tests/test_rung3_retro_leisure.py tests/test_proposal_scoring.py -> passed
pytest services/orion-durable-runs/tests/test_physiology_report.py -> 21 passed, 1 skipped (Postgres DSN)
pytest orion/physiology/tests/test_report_text.py               -> 16 passed
pytest services/orion-ai-town/tests                             -> 150 passed, 25 skipped (need node/tools)
mutation: daily-cap stop removed        -> test_daily_cap_pauses_mid_session... fails
mutation: UTC day instead of local day  -> 6 controller + 5 ledger tests fail
mutation: heartbeat outside a session   -> 6 controller tests fail
static gates (.github/workflows/orion-static-gates.yml, every step) -> pass after re-locking metric definitions
patch chain: fresh clone at AITOWN_UPSTREAM_REF + apply_upstream_patches.sh -> every patch applies; tsc on convex/ shows only the known pre-codegen crons.ts error
```

## Evals run

```text
No eval harness covers the session controller. The quality question ("does Orion choose the town
when it helps, and skip it when not") is graded by the act ledger itself: every session closes a
play:aitown.session act with artifact_produced (Orion spoke at least one line). Follow-up: after a
week of sessions, compare the act's posterior with Juniper's read of the town transcripts.
```

## Docker/build/smoke checks

```text
Not built or deployed here (instructed: do not deploy). Live path UNVERIFIED:
- the Convex admin env call (/api/update_environment_variables) is the CLI's own call; the route exists on the live backend (GET -> 405, an unknown route -> 404) but no write was sent;
- no session has run, so the per-session NPC call rate is unmeasured.
Live read 2026-10-30 04:16Z (read-only): world status running, embodiment heartbeating, 4 NPCs + Orion, no human player.
```

## Review findings fixed

Code review ran in a subagent (no blocker; 6 should-fix, 5 nits).

- Finding: Orion's FCC agent could open the town uncounted through the AI Town MCP tool `aitown_resume_world` (`testing:resume`), logged as Juniper's time.
  - Fix: the tool now refuses and names the two real paths (`services/orion-ai-town/mcp/orion_aitown_mcp/server.py`). `aitown_kick_engine` cannot wake a stopped engine (`kickEngine` throws when not running), so it stays.
  - Evidence: code read of `upstream/convex/aiTown/main.ts:65`.
- Finding: the cap check was skipped while Convex engine reads failed.
  - Fix: the caps are checked before the engine read; the read only detects `town_paused`.
  - Evidence: `test_the_cap_is_enforced_even_when_engine_reads_fail`; mutation (check moved back) fails 8 tests.
- Finding: a start could absorb a session Juniper opened less than one tick earlier.
  - Fix: `_start` reads the engine before resuming; running means `manual_session_open`, unreadable means `town_unreachable`.
  - Evidence: `test_a_start_never_absorbs_a_town_juniper_just_opened`.
- Finding: the boot/crash pause broke the documented rollback (a `stoppedByDeveloper` world ignores the legacy heartbeat).
  - Fix: no pause while `EMBODIMENT_TOWN_SESSIONS_ENABLED=false`.
  - Evidence: `test_rollback_boot_does_not_pause_the_town`.
- Finding: a dispatch replayed after a restart could overwrite its earlier row (erasing counted time).
  - Fix: a session id already on the ledger is refused `already_ran`.
  - Evidence: `test_a_replayed_dispatch_after_a_restart_never_rewrites_its_row`.
- Finding: the 15 s RPC timeout was shorter than a real start (Convex mutations measured 15-25 s); `session_open` refusals had no retry clock.
  - Fix: timeout 60 s; `session_open` holds for the rest of the session.
  - Evidence: `test_session_open_refusal_holds_for_the_rest_of_the_session`.
- Finding (nit): a failed stop at the cap closed the row while the town kept running unlogged.
  - Fix: the session stays open and counting until the stop lands; each failure logs a consecutive count.
  - Evidence: `test_a_failed_stop_keeps_the_session_open_and_counting_until_it_lands`.
- Finding (nit): the crash-charge comment had the wrong direction; the crashed want was not released; `release()` ran with shadow bids off.
  - Fix: comment corrected; crashed want released; release gated on the flag.
- Finding (nit): docs said an open page wakes the town; after a controller pause only Unfreeze does.
  - Fix: README, module and schema docstrings say so.
- Finding (nit): the primary checkout's local `.env` had `EMBODIMENT_WORLD_HEARTBEAT_ENABLED=false` ahead of this merge, which would idle the town under the old code.
  - Fix: kept `true` locally; it is ignored while sessions are on, so it needs no change at deploy.

## Restart required

Order matters: embodiment first (it answers the RPC and owns the pause), then the others. From the clean primary `main` checkout on athena after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
scripts/safe_docker_build.sh orion-embodiment up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
```

On circe (the NPC correlation header; the post-pull rebuild runs `pre_rebuild.sh` / `post_rebuild.sh` for orion-ai-town):

```bash
ssh circe 'cd /mnt/scripts/Orion-Sapienform && git pull --ff-only'
```

By hand if the post-pull rebuild does not pick it up: `cd services/orion-ai-town && bash scripts/pre_rebuild.sh && ../../scripts/safe_docker_build.sh orion-ai-town up -d --build && bash scripts/post_rebuild.sh`.

After embodiment restarts it pauses the running town once (`aitown_session_boot_pause` in its log).

## Risks / concerns

- Severity: medium. Concern: Juniper's manual sessions are not capped; a forgotten Hub tab keeps the town running. Mitigation: logged as `manual` in the report; AI Town's idle stop still pauses it once no page is open.
- Severity: medium. Concern: a restart of embodiment pauses a town that Juniper opened while it was down (no row on the ledger yet). Mitigation: one pause; she presses Unfreeze again and it is then tracked.
- Severity: low. Concern: the cap can overshoot by up to one tick (10 s) plus the Convex call time. Mitigation: bounded and visible in the session row.
- Severity: low. Concern: the shadow bid is not obeyed until Patch D. Mitigation: by design; the hard cap is independent of it.
- Severity: low. Concern: no heat gate in the controller. Mitigation: deliberate (the body-economy contract forbids new arousal/heat vetoes; heat is priced by Patch D). The cap cuts town runtime from ~24 h to at most 2 h a day.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2996

🤖 Generated with [Claude Code](https://claude.com/claude-code)

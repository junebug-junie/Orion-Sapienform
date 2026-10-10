# fix(gpu-pool): alert when a lane controller can't act on its config

## Summary

- When the GPU lane controller keeps refusing the pool for a reason a retry can't fix, the pool now marks that seat **degraded** after two refusals in a row. At the pool's 10-minute retry interval, that is about 10 minutes after the first refusal. Two examples of such reasons: the controller can't read its own config, or the two hosts disagree about the config.
- The degraded seat is shown in three places. `/health` gets `degraded: [seat]` and `actuation.controller.<seat>` (why, since when, how many refusals, and a plain fix such as "rebuild the controller on circe"). The pool state payload carries the same thing on the seat's card. The Hub GPU pool panel shows it as **CONTROLLER BROKEN**.
- When a seat turns degraded, Juniper gets one Hub Pending Attention card (severity error), sent through orion-notify. If nobody acks it within 60 minutes, notify escalates it to email. When the seat recovers, one info card is sent that needs no ack.
- The two refusals must also be at least 60 s apart, so one bad read during a `git pull` can't trip it. A succeeded `status` reply only clears problems that a status call actually checks.
- Refusals that a retry does fix (`busy`, `deadline_passed`, `upstream_not_idle:*`, `stale_generation`) neither trip the alarm nor clear it.
- The pool keeps retrying on its normal cooldown. The first retry the rebuilt controller accepts clears the alarm.

## Outcome moved

The incident: from 2026-10-09 04:07 to 2026-10-10 05:51 UTC, the controller refused every agent-gpu2 load with `config_unloadable:ValidationError`. The live `gpu_pool_events` table has 155 `actuate_refused` rows for it. The seat granted nothing, the agent backlog waited 3–12 h, and nobody was told. Before this patch it produced 0 alerts.

Replayed through the real runtime (`evals/run_controller_stale_eval.py`, 136 refusals):
- The first alert lands **601 s** after the first refusal.
- Exactly one "degraded" alert is sent for the whole outage.
- The "recovered" alert is sent in the same tick that the rebuilt controller answers.
- Replaying 136 retryable refusals produces **0** alerts.

## Metric quality gate (the "two refusals in a row" signal)

1. **Provenance:** the signal comes from the controller's own refusal replies.
   - `config_unloadable:<Exc>` and `fence_state_unreadable:<Exc>`: `services/orion-gpu-lane-controller/app/actuator_bus.py` (`_admit`, `_status`).
   - `launch_digest_mismatch`, `profile_not_allowed`, `unknown_role`, `role_not_on_this_actuator`, `cards_mismatch`, `no_launch_block:*`, `not_a_swap_seat`: `app/pool_fence.py` `resolve`.
   - `invalid_request:*`: `actuator_bus.py` (request schema rejected).
   - The pool consumes them in `on_actuate_result` and `_on_reconcile`.
2. **Independence:** this is a new reading of an existing event (`actuate_refused`), not a second sensor. It doesn't feed any model or aggregate. It only drives health, display and the alert.
3. **Theory:** the reasons are definitional. Each one is raised before the controller touches a container, and it comes from the controller's own config or code, not from the GPU's state. So the same request gets the same refusal until someone changes the config or the image. Retrying cannot fix it.
4. **Live data:** `gpu_pool_events` (30-day retention) holds 155 `actuate_refused` rows, all `config_unloadable:ValidationError`, at a ~10-minute cadence from 10-09 04:07 to 10-10 05:51. Agent-gpu2 was `swapped` at 06:01 after the rebuild. No other refusal reason appears, so history shows no false-positive source. The rest state, with no refusals, is the normal state: the tracker is empty and `degraded: []`.
5. **Existing mechanism:** I checked four places.
   - `scripts/gpu_pool_actuator_probe.py` detects this, but only when run by hand.
   - The mesh guardian's stability checks (#2472) watch crash loops, Redis and FalkorDB, not the pool.
   - The pool's `/health` had no actuation-health field.
   - The Hub GPU leases modal (#2570) shows leases, not controller refusals.
   
   This patch reuses the notify attention-card path that the guardian and the health monitors already use.
6. **Reversibility:**
   - No schema, registry or channel changes.
   - The state payload addition rides in the existing free-form `actuation` dict and is never persisted.
   - `GPU_POOL_CONTROLLER_ALERT_ENABLED=false` turns off the cards.
   - Removing the tracker is a small revert.

## Current architecture

The pool sent `GpuActuateV1` to the circe controller. A `refused` result set the card idle with a 600 s cooldown, emitted `actuate_refused`, and logged a warning. Nothing counted repeated refusals, and `/health` stayed `ok: true`. The controller reads YAML from its bind-mounted checkout but runs the code in its image, so a `git pull` without a rebuild breaks it silently.

## Architecture touched

- orion-gpu-pool runtime: tracks controller health per seat, fires the alert outside the lease lock, adds the snapshot overlay.
- `/health`.
- orion-notify `/attention/request`: called as a client. Notify itself is unchanged.
- Hub GPU pool panel: one rendered line.

## Files changed

- `services/orion-gpu-pool/app/controller_health.py`: new. Pure tracker: classifies refusal reasons, applies the two-in-a-row threshold, writes the plain advice text.
- `services/orion-gpu-pool/app/controller_alert.py`: new. Builds the `ChatAttentionRequest` and sends it with an httpx POST to notify. Delivery failures are logged at ERROR and never raised.
- `services/orion-gpu-pool/app/runtime.py`:
  - feeds the tracker from action results, in-flight status replies, and boot/resume reconcile replies;
  - fires the alert as a fire-and-forget task;
  - adds `controller_degraded` to the card's `actuation` in snapshots.
- `services/orion-gpu-pool/app/main.py`: wires the alert; adds `/health` `degraded` and `actuation.controller`.
- `services/orion-gpu-pool/app/settings.py`, `.env_example`, `docker-compose.yml`: three keys.
- `services/orion-gpu-pool/README.md`: a "When the controller can't act" section.
- `services/orion-gpu-pool/tests/test_controller_health.py`: 9 regression tests.
- `services/orion-gpu-pool/evals/run_controller_stale_eval.py`: incident replay with hard targets.
- `services/orion-hub/static/js/gpu_pool.js`, `services/orion-hub/tests/test_gpu_pool_panel_browser_smoke.py`: CONTROLLER BROKEN line and its browser assertion.
- `.github/workflows/orion-gpu-pool-tests.yml`: runs the replay eval; triggers on `orion/schemas/notify.py`.

## Schema / bus / API changes

- Added: none to schemas or channels. `/health` gains `degraded` (a list) and `actuation.controller` (a dict). In the state payload, the card's `actuation` dict may carry `controller_degraded` (the field is already `dict[str, Any]`).
- Removed / renamed: none.
- Behavior changed: one notify attention card per degraded or recovered transition.
- Compatibility: old consumers of `GpuPoolStateV1` (extra=forbid) see no new fields. Only the free dict changes.

## Env/config changes

- Added keys: `GPU_POOL_CONTROLLER_ALERT_ENABLED=true`, `NOTIFY_BASE_URL=http://orion-athena-notify:7140`, `NOTIFY_API_TOKEN=` (empty, which matches notify's live `API_TOKEN`).
- `.env_example` updated: yes.
- Local `.env` synced with `python3 scripts/sync_local_env_from_example.py orion-gpu-pool --all-keys`: yes, all 3 keys added to the primary checkout's `services/orion-gpu-pool/.env`. `POSTGRES_URI` was reported as diverged; that is intentional and was left alone.
- Skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-gpu-pool && pytest tests -q            -> 169 passed, 15 skipped (incl. 14 new)
cd services/orion-hub && pytest tests/test_gpu_pool_panel_browser_smoke.py -q -> 3 passed (local chromium)
mutation: runtime action-result wiring removed  -> 5 new tests fail
mutation: status answers clear every kind       -> digest-mismatch runtime test fails
mutation: Hub JS reverted to main               -> fault smoke fails
mutation: Hub `if (a)` instead of `if (a && a.action)` -> emergency-stop smoke fails
python3 scripts/check_env_template_parity.py orion-gpu-pool -> PASS
git diff --check -> clean
```

## Evals run

```text
python services/orion-gpu-pool/evals/run_controller_stale_eval.py
PASS  A refusals replayed: 136
PASS  A first alert within budget: 601.0 s (budget 1260.0 s)
PASS  A one degraded alert: ['degraded', 'recovered']
PASS  A one recovered alert, same tick: 0.0
PASS  A degraded visible every cycle after the 2nd refusal: 135
PASS  B refusals replayed: 136
PASS  B zero alerts on retryable refusals: []
```

## Docker/build/smoke checks

```text
docker build -t orion-gpu-pool-check:ctrl-alert -f services/orion-gpu-pool/Dockerfile .  -> ok
  (separate tag on purpose: a compose build from a worktree would overwrite production's image tag; image removed after)
docker run ... python -c "import app.main"  -> import ok, controller_alert_enabled=True, notify_base_url=http://orion-athena-notify:7140
docker compose ... -f services/orion-gpu-pool/docker-compose.yml config -> the 3 keys resolve
docker exec orion-athena-gpu-pool: urlopen http://orion-athena-notify:7140/health -> 200 (pool can reach notify on app-net)
```

Live card delivery: UNVERIFIED. I did not post a test card into Juniper's live attention queue. It gets verified on the first real degradation, or by hand after deploy.

## Review findings fixed

Reviewed in a subagent. It found no must-fix issues. All 8 findings were fixed:

- Finding (should-fix): a successful `status` reply cleared a seat that was degraded for a config mismatch. The controller's status path resolves with `digest=None` and skips the digest, profile and launch-block checks, so this caused flapping "recovered" then "degraded" cards.
  - Fix: `on_answered(seat, via_status=True)` clears only `config_unreadable` and `request_rejected`. Mismatch kinds clear only when a load or unload is admitted.
  - Evidence: `test_a_status_answer_cannot_clear_a_config_mismatch` and `test_reconcile_status_does_not_clear_a_digest_mismatch_through_the_runtime`. The second fails when the fix is reverted (mutation-checked).
- Finding (should-fix): the Hub panel showed a fake "in flight" line when a card had `controller_degraded` but no action record.
  - Fix: `if (a && a.action)`.
  - Evidence: the emergency-stop smoke now carries an actuation dict with only `controller_degraded` and asserts there is no "in flight" line. It fails against the old JS (mutation-checked).
- Finding (should-fix): `fence_state_unwritable:*` was not classified.
  - Fix: new `state_unwritable` kind, with advice to check the controller's mount, disk and permissions.
  - Evidence: classify test.
- Finding (nit): a boot reconcile refusal and the first load refusal can arrive seconds apart, inside the same mid-pull window.
  - Fix: degrading now also needs the refusals to be at least 60 s apart.
  - Evidence: `test_two_refusals_seconds_apart_do_not_degrade_until_the_span_floor`. The replay eval still gives the first alert at 601 s.
- Finding (nit): the `request_rejected` advice blamed a stale controller, but `invalid_request:deadline_at_naive` is a pool-side bug.
  - Fix: the advice now names both possibilities.
- Finding (nit): alert tasks were not drained on shutdown.
  - Fix: `PoolRuntime.drain_alerts()` (waits up to 5 s, then cancels), called from the lifespan shutdown.
  - Evidence: `test_drain_alerts_waits_for_an_in_flight_post_then_cancels_the_rest`.
- Finding (nit): the tracker is memory-only and that wasn't documented.
  - Fix: documented in the module docstring, the README and Risks.
- Finding (nit): no test that a replayed old result cannot clear the seat.
  - Fix: `test_a_stale_replayed_result_does_not_clear_the_seat`.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-gpu-pool/.env -f services/orion-gpu-pool/docker-compose.yml up -d --build gpu-pool
```

Hub: no restart. `services/orion-hub/static` is bind-mounted from the primary checkout, so the `git pull` above ships the panel line. Hard-refresh the GPU pool page if it's cached.

## Risks / concerns

- Severity: low. Concern: a pool restart forgets the count, so a controller that is still broken re-alerts about 10 min after the restart. Mitigation: that duplicate is on purpose. The problem is still real.
- Severity: low. Concern: the pool keeps sending one refused request every 10 min while degraded, as it did before. Mitigation: each one is cheap, and those retries are how recovery gets detected.
- Severity: medium. Concern: this detects the stale controller; it does not prevent it. The root cause is that the controller reads config from its live checkout while running code from its image. Mitigation (follow-up): have the controller compare its own image's commit against the checkout, or validate its config at boot, and refuse to start.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)

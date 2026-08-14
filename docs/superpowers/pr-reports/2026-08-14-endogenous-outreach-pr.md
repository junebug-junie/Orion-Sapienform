# Endogenous outreach — Orion opens a conversation Juniper didn't start

Branch: `feat/endogenous-outreach`

## Summary

- Adds `services/orion-hub/scripts/endogenous_outreach.py`: a gated background loop that lets Orion send an unsolicited chat message.
- The **trigger is a deliberate stub** (randomized timer) until autonomy supplies a real endogenous signal. Everything downstream of the trigger is real.
- Reuses three delivery rails that already existed — no new bus channel, no new schema, no new service.
- Grounds every message in live substrate signals + real chat history; a tick with no grounding is skipped rather than filled with placeholder text.
- Off by default, with seven safety gates and a debug status/trigger surface.
- Review found 8 issues, 2 of them CRITICAL and both falsifying the patch's central safety claim. All fixed, all now covered by tests.

## Outcome moved

Orion could not previously initiate contact at all — every message was a response to something Juniper typed. It can now open a conversation on its own, grounded in its own substrate state, without a human starting the exchange.

## Current architecture (before this patch)

Hub chat was strictly request/response. A message existed only as a reply to an inbound websocket frame.

What already existed and got reused rather than rebuilt:

- `notification_cache.py:50` subscribes to `NOTIFY_IN_APP_CHANNEL` and fans into every connected socket's outbound queue (`websocket_handler.py:731`), drained to the client by `drain_queue` (`websocket_handler.py:460`).
- `app.js:10250` already renders any payload carrying `llm_response` as an Orion chat bubble.
- `chat_history.py:280` publishes `chat.history.message.v1` to the bus for `orion-sql-writer`.
- `bus_synaptic_trigger_notifier.py` is the precedent for a Hub-side producer of `HubNotificationEvent`.
- `curiosity_hint.py` already reads `substrate_endogenous_curiosity_candidates`.

So the transport for "Orion speaks unprompted" was fully present; what was missing was a producer and the gates around it.

## Architecture touched

`orion-hub` only. One new module, five wiring edits, no cross-service contract change.

Pipeline:

```text
tick -> gates -> grounding read -> quick/metacog cortex call -> re-gate -> 3 delivery rails
```

## Files changed

- `services/orion-hub/scripts/endogenous_outreach.py`: new — loop, pure gate functions, grounding reads, delivery.
- `services/orion-hub/scripts/main.py`: construct/start on startup, stop on shutdown.
- `services/orion-hub/scripts/websocket_handler.py`: register/unregister connection, record `session_id`, and set/clear the per-connection busy flag.
- `services/orion-hub/scripts/api_routes.py`: `GET/POST /api/debug/endogenous-outreach/{status,trigger}`.
- `services/orion-hub/scripts/curiosity_hint.py`: `_fetch_fresh_candidates` gains a keyword-only `max_age_sec` (default unchanged) so outreach can widen the freshness window without duplicating the query.
- `services/orion-hub/static/js/app.js`: early-return `orion_outreach` branch; toast suppression in `addNotification`.
- `services/orion-hub/app/settings.py`, `.env_example`: 11 new keys.
- `scripts/sync_local_env_from_example.py`: `HUB_ENDOGENOUS_OUTREACH_` added to `SYNC_PREFIXES` — without this the sync script silently ignores the new keys.
- `services/orion-hub/tests/test_endogenous_outreach.py`: new, 55 tests.
- `services/orion-hub/README.md`: new section 4.1.

## Schema / bus / API changes

- Added: none. Reuses `chat.history.message.v1` and `notify.in_app.v1` (`HubNotificationEvent`).
- Removed / renamed: none.
- Behavior changed: `orion:notify:in_app` gains a new `event_kind` value, `hub.endogenous_outreach.v1`, with `notification_type=endogenous_outreach`. `orion/bus/channels.yaml:1699` already lists `orion-hub` as a producer on that channel, so no catalog change is needed.
- New websocket frame kind `orion_outreach` (server→client only).
- Compatibility: additive. Consumers that do not know the new `event_kind` see an ordinary info-severity notification.

## Env/config changes

- Added keys (11): `HUB_ENDOGENOUS_OUTREACH_ENABLED`, `_TICK_SEC`, `_PROBABILITY`, `_MIN_COOLDOWN_SEC`, `_DAILY_CAP`, `_QUIET_START_HOUR`, `_QUIET_END_HOUR`, `_TZ`, `_LLM_ROUTE`, `_TIMEOUT_SEC`, `_FALLBACK_SESSION_ID`
- Removed / renamed: none
- `.env_example` updated: yes
- Local `.env` synced with `python3 scripts/sync_local_env_from_example.py`: **yes** — all 11 keys verified present in `/mnt/scripts/Orion-Sapienform/services/orion-hub/.env`
- Skipped keys requiring operator action: none

**Operator action worth taking:** `HUB_ENDOGENOUS_OUTREACH_TZ` defaults to `UTC`. Hub's compose and Dockerfile set no `TZ`, so leaving it at `UTC` means the 23→08 quiet window is 23:00–08:00 **UTC**. Set it to the real operating zone before enabling.

## Tests run

```text
pytest tests/test_endogenous_outreach.py -q
  -> 55 passed

pytest tests/test_endogenous_outreach.py tests/test_hub_presence.py tests/test_agent_repl_bridge.py -q
  -> 69 passed, 1 warning
     (the latter two are the existing callers of the curiosity_hint signature change)
```

Two bugs were found by tests before review, not after:

- whitespace-only generation was reaching delivery, because `maybe_outreach` trusted `_generate` to have stripped;
- `_publish_notification` was monkeypatched out of the success-path test, so `HubNotificationEvent`/`BaseEnvelope` construction had never actually been validated (`message_id` is UUID-typed, carried around as `str`).

## Evals run

```text
None. services/orion-hub has no evals/ directory.
```

Gap acknowledged, not papered over. The meaningful eval here — "is the message Orion produces worth interrupting for?" — needs real generations to judge and cannot be written before the feature has run live. Follow-up below.

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml config           -> OK
python3 scripts/check_service_env_compose_parity.py orion-hub -> OK
python3 scripts/check_single_consumer_channels.py            -> OK (live bus)
python3 scripts/check_scripts_dir_no_stdlib_shadow.py        -> OK
node --check services/orion-hub/static/js/app.js            -> OK
```

**No live deploy smoke was run, deliberately.** `services/orion-hub/docker-compose.yml:26-30` mounts `../..:/repo` plus `./templates` and `./static`. Bringing Hub up from this worktree would swap the entire running Hub onto this branch — the exact collision class documented in AGENTS.md §8. Runtime status of the delivery path is therefore **UNVERIFIED** until merge + restart.

What *was* verified against live data, in-process, without touching the running Hub: the grounding reads. Running the real `_fetch_curiosity_summaries` / `_fetch_recent_turns` / `build_outreach_prompt` against live Postgres produced a genuinely grounded prompt:

```text
Live signals from your own substrate right now:
- concept-dense area with no ontology_branch nodes
- chat repair pressure level=0.56

Your chat presence: idle; last turn with Juniper was 41 minutes ago.
```

Both sources confirmed non-degenerate: `substrate_endogenous_curiosity_candidates` had 1444 rows with `max(generated_at)` minutes old; `chat_history_log` had 1668 rows, freshest the same day.

## Review findings fixed

Review ran in a subagent against commit `a8202eb5e`. Eight findings, all confirmed by re-reading the cited code before fixing.

- **Finding (CRITICAL): the turn-in-flight gate was blind to 3 of the UI's 4 chat modes.**
  - Fix: added a per-connection `busy` flag set on every inbound message regardless of mode, cleared at the top of the receive loop — the one point every `continue` path passes through. `_turn_in_flight()` now ORs it with `active_turn`.
  - Evidence: `active_turn["correlation_id"]` is assigned at exactly `websocket_handler.py:993` (unified-`orion`) and `:1277` (`agent-claude`); the UI selector (`index.html:276-279` → `app.js:9111-9116`) offers Orion/Quick/Story/Agent, of which Quick and Story map to `brain` and Agent to `agent`, none of which touch it. `test_busy_connection_blocks_even_without_active_turn`.

- **Finding (CRITICAL): TOCTOU — gate checked once, never re-checked across a call bounded by a 60s default timeout.**
  - Fix: re-gate immediately before `_deliver`; drop with reason `<gate>_after_generation`.
  - Evidence: `test_turn_starting_during_generation_drops_the_outreach` marks the connection busy from inside the stubbed generate and asserts nothing is queued. The pre-existing `test_turn_in_flight_blocks_even_when_forced` could not catch this — it set state before the call.

- **Finding (HIGH): `force=True` exempted the `disabled` gate, on an unauthenticated endpoint.**
  - Fix: carve-out removed. `force` now skips the random roll and nothing else.
  - Evidence: `api_routes.py` registers on a bare `APIRouter()` with no auth dependency; `start()` assigns `_bus`/`_cortex_client` before the enabled check, so the object is live even when disabled. `test_force_cannot_override_the_disabled_flag`.

- **Finding (HIGH): `options["no_write"]` was inert on this path.**
  - Fix: set `tool_execution_policy="none"`, `action_execution_policy="none"`, `no_write_active=True` directly.
  - Evidence: `cortex_request_builder.py:419-422` is the only translator and reads `no_write` as a **top-level payload key**, not an option — and this module calls `cortex_client` directly, bypassing that builder entirely. The executor reads `no_write_active` (`orion-cortex-exec/app/supervisor.py:648`). `test_generation_pins_execution_policies_and_disables_recall`.

- **Finding (HIGH): `options["use_recall"]` has no reader; every outreach fired full default recall.**
  - Fix: pass the typed `recall={"enabled": False}`.
  - Evidence: `orion-cortex-gateway/app/bus_client.py:388-394` builds `RecallDirective` from `req.recall` and falls back to `RecallDirective()` (`enabled=True`) when it is `None`. `test_recall_directive_accepts_the_payload_we_send` pins the coercion.

- **Finding (MEDIUM): quiet hours and the daily cap ran on the container's process timezone while documented as local time.**
  - Fix: `HUB_ENDOGENOUS_OUTREACH_TZ`, resolved via `zoneinfo` with a logged UTC fallback on a bad zone.
  - Evidence: no `TZ` in compose or Dockerfile; host is `Etc/UTC`. `test_quiet_hours_use_configured_zone_not_the_container_clock` uses 18:00 UTC = 13:00 Chicago against a 17→23 window and would fail under the old naive-local behavior; `test_daily_cap_resets_on_the_configured_zones_date_boundary` covers the rollover.

- **Finding (MEDIUM): no re-entrancy guard between the background tick and the debug endpoint.**
  - Fix: `asyncio.Lock`; a concurrent call returns `already_sending`.
  - Evidence: counters are only bumped after `_deliver`, so two overlapping passes each saw a clean cooldown. `test_concurrent_ticks_cannot_both_send`.

- **Finding (MEDIUM): every outreach reached an open browser twice — bubble and toast.**
  - Fix: `addNotification` returns before `showToast` for `notification_type === 'endogenous_outreach'`; the list entry is kept.
  - Evidence: `notification_cache.py:82-86` fans into the same `tts_q` that `websocket_handler.py:730-731` registers, and `showToast` was unconditional.

Two LOW findings were doc corrections rather than code changes, and are now stated honestly in the module docstring and README: chat-history persistence does **not** restore on reload (Hub has no conversation-restore fetch at all; `/api/chat/messages` has zero callers in `app.js`), and the outreach frame's piggybacked biometrics snapshot is dropped by the early return — harmless, since `biometrics_heartbeat` re-pushes every 5s.

The review also cleared, with evidence, six areas I had specifically asked it to attack: connection lifecycle/leaks, frontend branch placement and `appendMessage` tolerance, gate arithmetic, env parity across all three surfaces, the `curiosity_hint` signature change, and the `from .main import ...` function-scoped import pattern.

## Restart required

```bash
# after merge, from the primary checkout
sudo docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build hub-app
```

Then, to actually turn it on (it ships off):

```bash
# 1. set the real zone first, or quiet hours are UTC hours
#    services/orion-hub/.env: HUB_ENDOGENOUS_OUTREACH_TZ=<IANA zone>
# 2. HUB_ENDOGENOUS_OUTREACH_ENABLED=true
# 3. restart, then confirm the gates read as expected:
curl -fsS http://localhost:8080/api/debug/endogenous-outreach/status | jq
# 4. force one tick past the random roll (all safety gates still apply):
curl -fsS -XPOST http://localhost:8080/api/debug/endogenous-outreach/trigger | jq
```

## Risks / concerns

- **Severity: medium** — Concern: the live delivery path is `UNVERIFIED`. Compose mounts the checkout root, so a worktree deploy would swap the running Hub onto this branch. Mitigation: ships off by default; the `trigger` endpoint makes the first real firing a deliberate, observable operator action rather than a surprise.
- **Severity: medium** — Concern: no eval harness for outreach quality. Nothing currently distinguishes a message worth interrupting for from a plausible-sounding one. The `PASS` affordance and the grounding requirement are structural guards, not quality measurement. Mitigation/follow-up below.
- **Severity: low** — Concern: rail 1 is in-process, so it assumes Hub runs a single uvicorn worker. True today (`Dockerfile` CMD has no `--workers`) and documented in both the module docstring and README, but it is an implicit coupling that a future scaling change would silently break.
- **Severity: low** — Concern: `HUB_ENDOGENOUS_OUTREACH_TZ` defaults to `UTC`, which is almost certainly not the intended quiet window. Mitigation: called out in `.env_example`, `settings.py`, README, and the restart steps above.

Follow-ups worth filing:

1. Outreach quality eval — collect real generations, score whether each was worth interrupting for, and measure the `PASS` rate. Cannot be written before the feature has run live.
2. Conversation restore on page load, which would make rail 2 do what it looks like it does.
3. Replace `_should_roll()` with a real endogenous trigger when autonomy provides one. The module is shaped so this is the only function that needs to change.

## PR link

<to be filled after `gh pr create`>

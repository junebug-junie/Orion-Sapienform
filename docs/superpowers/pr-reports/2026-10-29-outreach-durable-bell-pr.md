# Orion's messages stay in the tray until Juniper dismisses them

## Summary

- Orion's own messages (outreach) now get a saved record in orion-notify. They stay in the Hub tray until Juniper dismisses them. A Hub restart no longer erases them.
- The tray pins open Orion messages at the top, with a small "Orion N" badge in the existing header. There is no new panel.
- Each Orion row has two buttons. Reply opens that conversation in the chat window, shows the message there, and puts the cursor in the input. Dismiss removes the row, and the removal is saved.
- Alerts that turn themselves off no longer bury Orion's messages. If an alert kind has a matching `<kind>_cleared` notice, the tray hides the all-clear and every alert it resolved. A condition that is still on shows once.

## Bell sketch (written before the frontend change)

```
NOTIFICATIONS  [Orion 2]           Notify settings  [Dismiss all]  [All v]
+-- notificationList -----------------------------------------------------+
| Orion  10:42                                         [Reply] [Dismiss]  |
|   I noticed the cabinet ran hot again while you were out...             |
| Orion  08:15                                         [Reply] [Dismiss]  |
|   Something from the reading earlier stuck with me...                   |
| ----------------------------------------------------------------------  |
| (normal notification cards, as today)                                   |
| 966 self-cleared alerts hidden                                          |
+-------------------------------------------------------------------------+
```

The real render from the browser smoke matches this sketch: two green Orion rows with Reply and Dismiss, a divider, then the normal cards.

## Outcome moved

- Before: on 2026-10-28, Orion sent 24 outreach messages. orion-notify saved 0 of them. Hub kept them only in a 200-item memory list. A restart wiped that list, and alert noise pushed messages out of it.
- After: each outreach message writes one `notify_requests` row. The row uses the same `message_id` as the `chat_message` row. The row is "open" until a `dismissed` receipt arrives.
- Noise: in the last 3 days, orion-notify saved 1,046 notifications. 966 of them were the four dispatch alert kinds and their all-clears. With the self-clearing rule on the same 1,046 rows, the tray shows 80. The alert that is still on (`execution_dispatch_all_refused`) shows once. The other 79 are attention items (they have their own panel), journal notes, letters, digests, and SQL-writer backlog notices. Nothing else is hidden.

## Current architecture

- `services/orion-hub/scripts/endogenous_outreach.py::_publish_notification` sent a `HubNotificationEvent` straight to the `orion:notify:in_app` bus channel. It never called orion-notify.
- Hub's `NotificationCache` (in memory, max 200) is the only reader of that channel. `/api/notifications` reads that cache.
- orion-notify already had a saved message lifecycle: `POST /chat/message` writes a `notify_requests` row with `message_*` columns, and `POST /chat/message/{id}/receipt` records seen/opened/dismissed. The day letter and journal paths use it. Outreach did not.
- The dispatch runtime sends `execution_dispatch_all_refused` and `execution_dispatch_theater_tripwire`, each followed later by a `*_cleared` notice. These two pairs flap hundreds of times a day.

## Architecture touched

- Hub outreach to orion-notify: outreach calls `/chat/message` through the shared `NotifyClient`. If orion-notify refuses the call or cannot be reached, Hub falls back to the old direct publish. The live tray still gets the message in that case, but the message is not saved.
- orion-notify: a caller can set the type of the live in-app event, so outreach keeps the `endogenous_outreach` type. orion-notify also forwards a new `tag` filter to sql-writer.
- sql-writer `/api/notify-read/chat/messages`: adds `status=open` (not dismissed) and `tag` (JSONB `@>`).
- Hub `/api/chat/messages` proxy: forwards `tag`.
- Hub tray: a new pure module `static/js/orion-message-tray.js` holds the logic. `app.js` renders the Orion rows and applies the rule.

## Files changed

- `orion/schemas/notify.py`: adds optional `ChatMessageNotification.notification_type`.
- `orion/notify/client.py`: `chat_message()` accepts `message_id` and `notification_type`.
- `services/orion-notify/app/main.py`: the in-app event uses the caller's type when one is set. `/chat/messages` forwards `tag`.
- `services/orion-sql-writer/app/api_notify.py`: adds the `status=open` and `tag` filters.
- `services/orion-hub/scripts/endogenous_outreach.py`: only `_publish_notification` (durable path plus fallback) and the new `_persist_outreach_via_notify`, plus a `notify_client` class attribute. `is_pass_response`, `offer_message`, `_deliver`, and reading spacing are not touched.
- `services/orion-hub/scripts/main.py`: gives outreach a `NotifyClient` when `NOTIFY_BASE_URL` is set.
- `services/orion-hub/scripts/api_routes.py`: the `/api/chat/messages` proxy forwards `tag`.
- `services/orion-hub/static/js/orion-message-tray.js` (new): the merge logic for Orion messages and the self-clearing rule.
- `services/orion-hub/static/js/app.js`: Orion rows, badge, Reply, Dismiss, the rule, and a 200-item hydrate.
- `services/orion-hub/templates/index.html`: the badge in the existing header, and the module script tag before `app.js`.
- Tests: `static/js/orion-message-tray.test.js`, `tests/test_outreach_durable_notify.py`, `tests/test_orion_message_tray_wiring.py`, `tests/test_hub_ui_layout_pass.py` (dismiss-all scope), `services/orion-notify/tests/test_chat_message_notification_type.py`, `services/orion-sql-writer/tests/test_notify_chat_messages_open_filter.py`.
- Eval: `services/orion-hub/evals/orion_message_tray_browser.py` (real Chromium smoke).

## Schema / bus / API changes

- Added: `ChatMessageNotification.notification_type: Optional[str] = None`. It is additive, and an older orion-notify ignores it.
- Added: query params `status=open` and `tag=<tag>` on `GET /chat/messages` (sql-writer, orion-notify, and the Hub proxy).
- Removed / renamed: none.
- Behavior changed: the live in-app event for outreach now comes from orion-notify. Its `event_kind` is `orion.chat.message` (was `hub.endogenous_outreach.v1`), and its `notification_type` is still `endogenous_outreach`. No code reads the old kind. The only match was one test assert, and that test still covers the fallback path.
- Bus channels: unchanged. `orion:notify:in_app` already lists both orion-notify and orion-hub as producers.
- Compatibility: if Hub deploys before orion-notify, the in-app event arrives as `chat_message`, which would toast once. The tray still treats it as an Orion message, because it also matches on the `endogenous_outreach` tag. If Hub deploys before sql-writer, `status=open` and `tag` are ignored, and the tray would list every chat message (96 old "actions" rows). Deploy sql-writer and orion-notify first.

## Self-clearing rule: why a rule and not a list

A producer that sends `<kind>_cleared` has declared that `<kind>` is a condition with a start and an end. An alert for a condition that already ended asks nothing of Juniper. The rule works on any producer that follows this convention. No list of names needs upkeep. Kinds without a `_cleared` sibling are never touched. If the all-clear has already dropped out of the cache window, the alert stays visible, because the tray cannot prove that the condition ended. The raw alerts are still in `notify_requests` and in `/api/notifications`. The tray shows a one-line count ("N self-cleared alerts hidden") so the hiding is visible.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- Skipped keys requiring operator action: none.

## Tests run

```text
node --test (all 26 Hub *.test.js files, as CI runs them)          381 passed
  incl. static/js/orion-message-tray.test.js                         13 passed
pytest services/orion-notify/tests                                  68 passed
pytest services/orion-sql-writer/tests/test_notify_chat_messages_open_filter.py   3 passed
pytest services/orion-hub/tests/test_outreach_durable_notify.py \
       test_orion_message_tray_wiring.py test_hub_ui_layout_pass.py \
       test_endogenous_outreach.py                                  280 passed
pytest services/orion-hub/tests (whole dir, local)   4355 passed, 33 failed
  -> the same 33 also fail on a clean origin/main checkout on this host
     (checked by running exactly those 33 there: 33 failed). Not from this patch.
pytest services/orion-sql-writer/tests (whole dir, local)   14 failed
  -> the same 14 fail with the new test file excluded. Not from this patch;
     CI runs an explicit file list, not the whole directory.
python scripts/check_env_template_parity.py                         PASS
git diff --check                                                    clean
```

## Evals run

```text
python services/orion-hub/evals/orion_message_tray_browser.py   (real Chromium, real index.html + app.js)
PASS two open Orion messages pinned
PASS newest Orion message is first
PASS badge reads 'Orion 2'
PASS flapping raise/clear pair folded away
PASS ordinary notification still shown
PASS self-cleared note shown
PASS Reply switched the chat session
PASS Reply put the message in the chat window
PASS Reply focused the chat input
PASS Reply saved opened + dismissed receipts
PASS badge dropped to 'Orion 1'
PASS Dismiss saved a dismissed receipt
PASS badge hidden at zero
PASS after reload, dismissed messages stay gone
PASS after reload, badge still hidden
PASS no tray page errors ([])
```

Mutation check: with the reload fix removed, the two "after reload" checks fail. The fix was then restored.

Live-data check of the self-clearing rule (read-only, the last 3 days of `notify_requests`): 1,046 rows. After the rule, 80 rows are visible and 966 are hidden. All 966 hidden rows are dispatch alert kinds.

Live SQL check (read-only): the compiled `status=open` query runs on the live database. It returns 96 open "actions" chat messages and 0 outreach rows, which is expected before deploy. The `tag` filter returns 0 rows, so the 96 old actions messages do not leak into the tray.

## Docker/build/smoke checks

```text
No docker build or deploy was run (task rule: no deploys).
Live endpoint reachability (read-only): orion-notify /health ok; /chat/messages returns rows.
UNVERIFIED: the post-deploy live path (a real outreach writes a notify_requests row with
tags @> ["endogenous_outreach"], and a tray Dismiss writes a 'dismissed' notify_receipts row).
Check after deploy:
  docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select message_id, created_at from notify_requests where tags @> '[\"endogenous_outreach\"]' order by created_at desc limit 5"
```

## Review findings fixed

A code-review subagent reviewed commit 2ef8bc325.

- Finding (blocker): a dismissed message came back after a page reload. Hub's memory cache still held the live in-app event.
  - Fix: `closedByDurable()`. If orion-notify published a message but no longer lists it as open, the message stays closed. Messages from the fallback direct publish were never saved, so the saved list does not judge them.
  - Evidence: the browser eval checks "after reload, dismissed messages stay gone" (it fails when the fix is removed), plus a node test.
- Finding: with the wrong deploy order, an older orion-notify drops the `tag` filter. The tray then fills with every chat message.
  - Fix: saved rows must carry the outreach tag. The deploy order is written down in this report.
  - Evidence: node test "durable rows without the outreach tag are ignored".
- Finding: server timestamps without a timezone were read as browser-local time. That broke the sort order and the clear-after-raise comparison.
  - Fix: `parseTime()` adds `Z`, the same rule as `formatHubLocalTime`.
  - Evidence: two node tests.
- Finding: Reply switched the session while the old thread was still on screen.
  - Fix: when the message is not on screen and the thread changes, a one-line System note says so. When the message is already on screen, the session does not change.
- Finding (nit): orion-notify `/chat/message` returned 200 without a bus, so Hub skipped its fallback.
  - Fix: it now returns 503.
  - Evidence: an orion-notify test.
- Finding (nit): the prefix rule could fold a critical alert.
  - Fix: critical items are never folded. The "`_cleared` means one global condition" contract is written in the code.
  - Evidence: a node test.
- Finding (nit): the receipt dedupe returned `true` while the first request was still in flight.
  - Fix: a second caller now gets the same pending result.
- Finding: Hub `/api/chat/messages` returned `[]` on failure, which reads as "nothing open".
  - Fix: it now returns 502.
  - Evidence: a wiring test.
- Noted, no change: outreach now gets an automatic `seen` receipt on arrival (it is `orion.chat.message` now). This is harmless, because "open" ignores `seen`.

## Restart required

Order matters: sql-writer and orion-notify go first, then Hub. Each command runs from the clean primary `main` checkout after merge.

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-notify up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium.
  - Concern: deploy order. If Hub goes first, an old sql-writer ignores `status=open`, so dismissed outreach can still show.
  - Mitigation: restart sql-writer, then orion-notify, then Hub (commands above). Before sql-writer is updated, the tag check keeps non-outreach rows out.
- Severity: low.
  - Concern: when Reply changes thread, Hub cannot load that thread's older history.
  - Mitigation: a System line in the chat says the turns above belong to another thread.
- Severity: low.
  - Concern: the self-clearing rule relies on the `<kind>_cleared` naming convention.
  - Mitigation: critical items are never folded. The hidden count shows in the tray. Raw items stay in `/api/notifications` and `notify_requests`.
- Severity: low.
  - Concern: if orion-notify is down when Orion speaks, that one message is live-only. It is not saved, and it disappears when Hub's cache drops it.
  - Mitigation: the `endogenous_outreach_durable_notify_failed` log line records each case.
- Collision: this PR does not touch `is_pass_response`, `reading_finding_outreach.py`, `offer_message`, or `_deliver`. `git merge-tree` against `origin/fix/outreach-pass-leak-and-morning-burst` (#2942) is clean.

## PR link

(added after the PR opened)

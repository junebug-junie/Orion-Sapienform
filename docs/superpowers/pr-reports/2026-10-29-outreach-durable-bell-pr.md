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

(filled in below)

## Evals run

(filled in below)

## Docker/build/smoke checks

(filled in below)

## Review findings fixed

(filled in below)

## Restart required

Order matters: sql-writer and orion-notify go first, then Hub. Each command runs from the clean primary `main` checkout after merge.

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-notify up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

(filled in below)

## PR link

(filled in below)

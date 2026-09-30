## Summary

- orion-notify can now send HTML email: a new optional `body_html` goes out as the rich half of a `multipart/alternative` message, with `body_text`/`body_md` as the plain-text fallback.
- Images can be embedded inline: an attachment with a `content_id` becomes an inline part next to the HTML (`multipart/related`), referenced as `<img src="cid:...">`. Needed because Gmail cannot reach tailnet-hosted hub image URLs.
- `/notify` now tells the caller whether the email actually went out (`email_status`: sent / failed / skipped / deferred, plus `detail` with the reason when not sent). `status` stays `"queued"` for wire compatibility.
- `content_id` is validated at the boundary (safe charset, unique per request) so a bad id is a 422, not a mid-send failure or broken header.
- Plain-text-only requests produce exactly the same MIME structure as before.

## Outcome moved

Unblocks a daily "Orion's Day" letter as untruncated HTML+CSS with inline visual-reverie images. Callers can now distinguish a sent email from a failed one from the HTTP response instead of querying `notify_requests.status`.

## Current architecture

`EmailTransport.send` (`orion/notify/transport.py`) built a single `text/plain` message via `set_content(body_text or body_md)` and added every attachment with `add_attachment`. `NotificationRequest` had no HTML field, `NotificationAttachment` no content id. `/notify` sent synchronously but always returned `status="queued"`; the real outcome existed only in the persisted `notify_requests.status`.

## Architecture touched

- Shared schema `orion/schemas/notify.py` (additive optional fields; default pydantic `extra="ignore"`, so old clients/servers interoperate in both directions).
- `orion/notify/transport.py` MIME assembly.
- `services/orion-notify/app/main.py` `/notify` response.
- `NotificationRequest` is HTTP-only (not a bus payload). `NotificationRecord` (persisted via sql-writer) and `HubNotificationEvent` (in-app) are built field-by-field and do not carry `body_html`; no sql-writer or hub change needed.

## Files changed

- `orion/schemas/notify.py`: `NotificationRequest.body_html`, `NotificationAttachment.content_id` (+validator), unique-cid model validator, `NotificationAccepted.email_status`.
- `orion/notify/transport.py`: HTML alternative + inline CID related parts; plain path unchanged.
- `orion/notify/client.py`: `timeout` typed `float` (constructor override already existed).
- `services/orion-notify/app/main.py`: return `email_status`/`detail` from the real `EmailOutcome`.
- `services/orion-notify/README.md`: HTML + inline images + `email_status` section.
- `services/orion-notify/tests/test_notify_email_delivery.py`: MIME structure, CID resolution, SMTP line-length, ordering, validation tests.
- `services/orion-notify/tests/test_delivery_status.py`: `email_status` response tests; body_html-not-on-any-bus-channel test.

## Schema / bus / API changes

- Added: `NotificationRequest.body_html`, `NotificationAttachment.content_id`, `NotificationAccepted.email_status` (all optional).
- Removed: none. Renamed: none.
- Behavior changed: `/notify` response `detail` is now set (reason) when the email was not sent; `ok` unchanged (still true). All existing `.detail` readers only read it when `ok` is false.
- Compatibility notes: registry entries unchanged (same class names). No bus channel changes. No consumer-first deploy ordering required.

## Env/config changes

- Added/removed/renamed keys: none.
- `.env_example` updated: no. local `.env` sync: not needed. Skipped keys: none.

## Tests run

```text
pytest services/orion-notify/tests -q                          62 passed
pytest tests/test_agent_trace_schema_registry.py              2 passed
pytest services/orion-sql-writer/tests/test_fallback_watch.py 40 passed
pytest services/orion-thought/tests/test_store.py             57 passed
pytest services/orion-thought/tests/test_resonance_monitor.py 17 passed
pytest services/orion-actions/tests/test_journal_actions.py   14 passed
pytest services/orion-hub/tests/test_urgent_report.py         51 passed
pytest tests/test_disk_threshold_watchdog.py                  40 passed
static gates: check_async_routes_not_blocking, check_metric_lineage --gate, check_definition_drift --gate,
  check_service_hostname_refs, check_control_surface_store_parity, check_chat_route_poachers: all PASS
Mutation check: making the in-app event carry body_html fails test_body_html_is_not_published_on_any_bus_channel.
```

## Evals run

```text
None. orion-notify has no eval harness; the change is deterministic MIME assembly fully covered by gate tests.
Follow-up: a render eval (send to a sink mailbox, confirm Gmail renders inline images) once the letter producer exists.
```

## Docker/build/smoke checks

```text
Not run: no dependency/compose/env change, and the task forbids deploy and real email. UNVERIFIED live.
```

## Review findings fixed

- Finding: `content_id` accepted values producing invalid Content-ID headers; CR/LF failed the whole letter at send time.
  - Fix: pydantic validator (strip `cid:`/`<>`, require `[A-Za-z0-9._@+-]{1,200}`) + unique-per-request check -> 422.
  - Evidence: `test_invalid_content_id_is_rejected_at_the_boundary`, `test_duplicate_content_ids_are_rejected`, `test_content_id_is_normalized`.
- Finding: body_html-not-persisted test could never fail (checked a model without the field).
  - Fix: assert on serialized envelopes actually handed to the bus on both persistence and in-app channels.
  - Evidence: mutation (in-app event carrying body_html) makes it fail.
- Finding (nit): long-HTML test did not check wire line length.
  - Fix: parametrized ASCII/non-ASCII/no-space single-line bodies, assert every wire line <= 998 octets.
- Finding (nit): regular-before-inline attachment order untested.
  - Fix: `test_regular_attachment_before_inline_one_still_nests_correctly`.
- Finding (nit): bare cids lack `@domain` (RFC 2392). README now recommends `name@orion`.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform-notify-html-email   # or main after merge
scripts/safe_docker_build.sh orion-notify up -d --build
```

Only orion-notify needs a rebuild. Producers of the letter pick up the shared schema on their own next rebuild; no ordering constraint (all fields optional, extra ignored).

## Risks / concerns

- Severity: medium (pre-existing). Concern: callers treat `ok=True` as "email sent" (`services/orion-actions/app/main.py` `journal_notify_email_sent`, `services/orion-world-pulse/app/services/publish_email.py`, `services/orion-notify-digest/app/main.py`). Mitigation: follow-up to switch them to `email_status == "sent"`.
- Severity: low (pre-existing). Concern: SMTP send runs synchronously inside the async `/notify` handler; large inline-image letters lengthen it. Mitigation: callers raise `NotifyClient(timeout=...)`; follow-up `asyncio.to_thread`.
- Severity: low. Concern: Gmail rendering not verified live. Mitigation: MIME structure pinned by tests; first real letter is the smoke.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2418

🤖 Generated with [Claude Code](https://claude.com/claude-code)

# Record what actually happened to a notification

## Summary

- `notify_requests.status` has been a column that lies: **10,900 rows since 2026-07-24, 100% `pending`** — including emails Juniper confirmed receiving.
- The outcome was already in hand and thrown away: `maybe_send_email()` logged succeeded/failed and returned `None`.
- It now returns an `EmailOutcome`, and the two email-bearing endpoints persist the real result.
- Separately: this service's own logging **never existed**. Root logger, no handlers, every `[NOTIFY]` line dropped.
- Deliberately did **not** wire `notify_attempts`.

## Outcome moved

"Was this notification delivered?" becomes an answerable question. It was not one before — the only reason we knew yesterday's email arrived is that Juniper said so.

## Current architecture

`orion-notify` decides and attempts email, then publishes a `NotificationRecord` on the bus; `orion-sql-writer` inserts it. `status` was set at insert from a hardcoded literal at three call sites (`main.py:257`, `:306`, `:381`) and never updated by anything, anywhere.

## Files changed

- `services/orion-notify/app/email_delivery.py`: `maybe_send_email()` returns `EmailOutcome(status, reason)` on all five paths instead of `None`.
- `services/orion-notify/app/main.py`: `_request_status()` mapping; both email-bearing endpoints persist the real outcome; `/chat/message` records `no_email`; `logging.basicConfig`.
- `services/orion-notify/.env_example` + `docker-compose.yml`: `NOTIFY_LOG_LEVEL`.
- `services/orion-notify/tests/test_delivery_status.py`: new, 22 tests.
- `services/orion-notify/app/attention_escalation.py`: the third call site now captures its outcome.
- `orion/notify/transport.py`: partial recipient refusal now raises.
- `services/orion-notify/README.md`: the status vocabulary and its cutover.

## The status vocabulary

| value | meaning |
|---|---|
| `sent` | **SMTP accepted the message.** Not proof it reached an inbox — nothing downstream of SMTP reports back here. |
| `failed` | the send raised; the error class and message are in the log line |
| `no_email` | no email was attempted (policy declined, no transport, or an endpoint that never emails) |
| `pending` | nothing attempted and nothing known — now means what it says, rather than being the only value the column ever held |

`sent` is deliberately not called `delivered`. Overclaiming here would be the same defect in a new coat.

## Blast radius — why changing the value is safe

- Nothing anywhere filters or selects on the stored `notify_requests.status` column.
- The attention view **derives** its own status: `api_notify.py::_attention_to_schema` computes `"pending" if row.attention_require_ack and row.attention_acked_at is None else "acked"`. So `/attention?status=pending` — which `health_monitor._has_open_alert` depends on for #1944's alert dedup — reads `attention_acked_at`, not this column, and is untouched.
- The one remaining literal `"pending"` in `main.py` is that attention record, correctly left alone.

## Why `notify_attempts` is not wired

It has **zero code references anywhere in the repo** outside its own table definition in `orion-notify-digest/app/db_models.py`, and 0 rows. A table nothing writes and nothing reads is the keyword cathedral the contract bans. One honest terminal status buys the thing that was actually missing; an attempts table buys nothing until something reads it.

## The logging bug

Verified live in the running container, before the fix:

```text
effective level: WARNING
root handlers: []
propagate: True
--- does an info line reach stdout? ---
(nothing above this line = dropped)
```

uvicorn configures its own loggers, which is why `docker logs` showed 203 access lines in 24h and **zero** application lines while 230 notifications were created. Every `[NOTIFY]` breadcrumb this service has ever written went nowhere — a large part of why nobody noticed delivery accounting did not exist.

`force=True` because uvicorn may already have installed handlers by import time; without it `basicConfig` is a silent no-op.

## Schema / bus / API changes

- Added: none structurally. `NotificationRecord.status` is an unconstrained `str` and now carries a wider vocabulary (`sent`/`failed`/`no_email` in addition to `pending`).
- Compatibility: additive in values only. No consumer reads the column (see blast radius).

## Env/config changes

- Added: `NOTIFY_LOG_LEVEL` (default `INFO`)
- `.env_example` updated: yes; also added to compose's explicit `environment:` list rather than relying on `env_file:` alone
- local `.env` synced: by hand (the sync script reads `.env_example` from the *primary* checkout, so a worktree-added key is invisible to it), verified at line 41

## Tests run

```text
pytest services/orion-notify/tests -q  -> 39 passed
```

### Mutation tests (real files, each anchor verified to match once)

| mutation | result |
|---|---|
| a successful send reports `skipped` | CAUGHT |
| a **failed** send reports `sent` | CAUGHT |
| no-transport returns `None` again (the original bug) | CAUGHT |
| `failed` maps to `sent` | CAUGHT |
| `skipped` maps back to `pending` | CAUGHT |
| one call site hardcodes `pending` again | CAUGHT |
| logging not configured (the live bug) | CAUGHT |
| `basicConfig` without `force=True` | CAUGHT |

## Docker/build/smoke checks

Logging fix, inside the running `orion-athena-notify` container:

```text
2026-08-30 02:53:07,063 INFO orion-notify [NOTIFY] email_send_succeeded notification_id=SMOKE
```

Pre-fix the same emit produced nothing. Not deployed.

## Evals run

```text
None. services/orion-notify has no evals/ directory.
```

## Review findings fixed

Review found seven defects. Two were severe, and both were reproduced before fixing.

- **The third email call site was missed, and it made the column lie in a NEW way.**
  `attention_escalation.py:130` still discarded the outcome — and it is the site that sends the real escalation email. Worse, it interacts: `/attention/request` calls `maybe_send_email(..., immediate_critical_only=True)`, so `severity="error"` skipped → my code stamped `no_email`. But escalation emails **exactly** `severity == "error"` past the ack deadline. Live: **37 of 46 error attentions escalated; every other severity escalated zero times.** So the patch would have stamped `no_email` on the only class that reliably emails.
  Fixed with a new `deferred` outcome (→ `pending`, honest: an email may still follow) and by capturing the outcome at the escalation site.
  **Did not** take the review's suggestion to make `sent` count only successful sends: `test_escalation_marks_before_send_even_if_smtp_fails` deliberately asserts `count == 1` with a raising transport, because the attention is marked escalated before the send and must not be retried. That is existing intent, so the outcome goes to the log line instead.

- **The entire headline change could be reverted with all 31 tests green.** My guard was a source-text grep counting characters, so `status=_request_status(email_outcome) and "pending"` — a full revert at both endpoints — walked straight through it. Replaced with tests that drive the real handlers and assert `status` on the published `NotificationRecord`. Verified: that mutation now fails 4 tests.

- **`_request_status` failed silently back to the known-bad value.** Now typed and logs `unmapped_email_outcome` at WARNING — visible in the logs this same commit made work.

- **`skipped` collapsed three different facts and threw the reason away.** `drop_reason` is a free, already-persisted column. Now carries policy decline vs unconfigured transport vs deferral. Also fixed `EmailOutcome.reason: str` → `Optional[str]` (`should_send_email` returns `(False, None)`).

- **My "nothing filters on the stored column" claim was wrong.** `orion-notify-digest/app/digest.py:124,135` filter `status == "throttled"` / `"deduped"`. Behaviour is unaffected — neither value was ever written and neither is written here — but the justification was false and is corrected in both the code comment and the README.

- **Partial recipient refusal reported as `sent`.** `smtplib.send_message` raises only when *every* recipient is refused; on partial refusal it returns the refused addresses and `transport.py` discarded them. Now raises. `NOTIFY_EMAIL_TO` is comma-separated, so this is supported config, latent only because it currently holds one address. `EmailTransport` is used solely by orion-notify, so the change is contained.

- **Two vacuous assertions removed.** `"None" not in str(signature.return_annotation)` passes for `Optional[EmailOutcome]` and, under `from __future__ import annotations`, only ever saw the string `'EmailOutcome'`.

- **README added** documenting the vocabulary, the pre/post cutover ambiguity, the digest consumer, and the known escalation gap (AGENTS §6/§16).

### Mutations after the fixes — 9/9 caught

| mutation | result |
|---|---|
| **full revert of the headline change** | CAUGHT (4 tests) |
| silent fallback (no warning) | CAUGHT |
| `drop_reason` discarded | CAUGHT |
| `deferred` maps to `no_email` | CAUGHT |
| `immediate_critical_only` skipped, not deferred | CAUGHT |
| a failed send reports `sent` | CAUGHT (5 tests) |
| no-transport returns `None` again | CAUGHT |
| third call site discards the outcome again | CAUGHT |
| partial recipient refusal ignored again | CAUGHT |

### Claims the review confirmed held up

Blast radius on paging dedup (verified independently and live); control flow around `email_outcome` (no `UnboundLocalError` reachable); `basicConfig(force=True)` not stomping uvicorn; no secret or PII at INFO; env parity including compose.

## Restart required

```bash
scripts/safe_docker_build.sh orion-notify up -d --build
```

Only new notifications get a real status; the 10,900 existing rows stay `pending` and are not backfilled — there is no record anywhere of what happened to them, so any backfill would be invention.

## Risks / concerns

- **Severity: low.** `basicConfig(force=True)` at import replaces root handlers. uvicorn's own named loggers are unaffected, but access-log formatting may shift.
- **Severity: low.** INFO now actually reaches `docker logs`. Existing `[NOTIFY]` lines interpolate `notification_id`, `event_kind` and a reason — no bodies or tokens — but this is worth a look during review since `body_text` exists on the same payload.

## PR link

<!-- filled after gh pr create -->

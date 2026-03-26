# Workflow Schedule Production Hardening (v1)

## Scope
This note captures the v1 production-hardening decisions for durable workflow scheduling across Actions, Cortex, and Hub.

## Attention notification philosophy
- Notifications are transition-based and derived from backend analytics truth.
- Emission happens in Actions after each scheduler iteration from durable schedule records.
- Conditions are bounded to: `failing`, prolonged `overdue`, and persistent `degraded` (repeated recent failures).
- Recovery emits once when an active condition clears.
- Reminder notifications are cooldown-gated and deduped with stable keys.

## Structured error codes
Schedule management responses include machine-readable `error_code` plus human-readable `message`.

Current codes:
- `invalid_management_payload`
- `ambiguous_selection`
- `schedule_not_found`
- `already_cancelled`
- `already_paused`
- `unsupported_transition`
- `missing_patch`
- `invalid_patch`
- `schedule_policy_missing`
- `schedule_revision_conflict`

UI and chat surfaces should branch on `error_code`, not message text matching.

### Lightweight counters (v1 hardening)
Actions now exposes narrow in-process counters at decision points:
- `workflow_schedule_attention_entered_total`
- `workflow_schedule_attention_reminder_total`
- `workflow_schedule_attention_recovered_total`
- `workflow_schedule_error_total|error_code=<code>`

Counters are incremented where the backend actually decides transitions or emits structured management errors.
They are intentionally internal-only for v1 and are meant to be wired into a future exporter without changing schedule logic.

## Integration proof coverage
- Actions→Notify attention publishing is now integration-tested across real schedule/store transitions.
- Coverage asserts:
  - exact dedupe key shapes for failing/overdue/recovered
  - transition payload fields (`transition`, `condition`, `state`, short schedule id)
  - no-spam behavior for unchanged loops under reminder cooldown

## Concurrency and idempotency guarantees
- Store operations are serialized with an in-process lock (`RLock`) and persisted atomically via temp-file replace.
- Claim marks schedule state before dispatch, preventing duplicate claims after restart for one-shot schedules.
- Recurring dispatch failure restores `next_run_at` to the claimed slot for retry instead of skipping silently.
- Guarantee scope is **single Actions process** sharing one schedule store path.

## Storage substrate rationale for v1
Current JSON store is acceptable for v1 under explicit constraints:
- low-to-moderate schedule volume
- single writer process
- operational preference for inspectable local state
- simple backup via file copy / host volume snapshot

### Migration triggers
Move to transactional DB storage when any of the following is true:
- multi-instance Actions active/active deployment needed
- schedule volume/history cardinality materially increases
- stronger cross-process claim atomicity is required
- operational recovery workflows demand queryable audit at scale

Likely next substrate: PostgreSQL with optimistic revision checks and `SELECT … FOR UPDATE SKIP LOCKED` claim semantics.

## Known limitations
- Cross-process locking is not provided; active/active writes to the same file are unsupported.
- Reminder notifications are bounded by scheduler cadence and configured cooldown, not cron-precise windows.
- Browser smoke remains focused to the critical operator lane, not a full UI matrix.
- CI now guarantees that focused lane via `schedule-browser-smoke` GitHub Actions workflow (Playwright Chromium).
- Local equivalent command: `pytest -q services/orion-hub/tests/test_schedule_panel_browser_smoke.py` (or `make -C services/orion-hub test-smoke-browser`).

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
- Browser smoke still validates a focused critical lane rather than full UI matrix.

# World-Pulse Stage 2 + Backfill + Status (thin follow-on)

**Status:** approved-by-operator-request (Juniper: “do all those things”)  
**Parent:** `docs/superpowers/specs/2026-09-06-world-pulse-concept-read-pipeline-design.md`  
**Date:** 2026-09-07

## Arsonist summary

Stage 1 is live. This patch adds (1) Wallet B Stage 2 sibling loop that chews Stage 1 handoffs without touching Curiosity Atlas, (2) an operator backfill enqueue script, (3) a richer Hub status JSON surface. Sibling isolation stays — no merge into `curiosity_investigation.py`.

## Current architecture

- Stage 1 Hub loop + Wallet A + Postgres `world_pulse_read_seed` + Concept Atlas materialize + journal.
- Schedule JSON is minimal (`enabled`, `done_today`, keys).
- Stage 1 handoff is in-memory only today — Stage 2 cannot restart from it after Hub bounce.

## Missing questions (locked for this patch)

- Stage 2 is sibling-only (same pattern as Stage 1); may reuse pure helpers, not curiosity Redis keys.
- Wallet B default daily cap: **6**. Keys: `orion:wp_read:wallet_b:last_at`, `orion:wp_read:wallet_b:count:`.
- Tool round-trip ceiling per `trace_id`: **5** (Stage 2→Stage 1 re-entries count toward this).
- Persist Stage 1 handoff JSON on the seed row when Stage 1 completes (`handoff_json` jsonb + `handoff_at`).
- Stage 2 claims seeds with `status='done'` and `handoff_json IS NOT NULL` and `stage2_status` pending.

## Proposed schema / API changes

### Postgres (migration)

`world_pulse_read_seed` add:

- `handoff_json jsonb null`
- `handoff_at timestamptz null`
- `stage2_status text not null default 'pending'` check in (`pending`,`claimed`,`done`,`failed`,`skipped`)
- `stage2_claimed_at`, `stage2_completed_at`, `stage2_error` nullable
- `stage2_trace_id text null` (may equal Stage 1 `trace_id` or a Stage-2 child id; store Stage 1 `trace_id` on handoff and journal both)

Index: `(stage2_status, priority, handoff_at)` for Stage 2 claim.

### Redis Wallet B

Mirror Wallet A helpers in `orion/world_pulse_read/wallet_b.py` (duplicate pure gate helpers — same Juniper choice as Wallet A).

### Hub loop

`services/orion-hub/scripts/world_pulse_read_stage2.py` — `WorldPulseReadStage2Pipeline`:

1. Gate Wallet B (force skips schedule gates).
2. Claim next Stage-2-ready seed (has handoff).
3. Debit Wallet B before FCC.
4. FCC seeded pass: prompt includes handoff JSON; instruct priors/hops; may request Stage 1 re-read by returning `need_stage1_urls` — if present and ceiling allows, enqueue/claim those URLs via Stage 1 path **debiting Wallet A** (call Stage 1 `_stage1_read` helper or publish a one-shot), increment round-trip counter.
5. Journal Stage 2 summary; mark `stage2_status=done|failed`.
6. Hard stop when round-trips ≥ 5 or Wallet A/B blocked.

### Stage 1 change

On success, `mark_seed_done` also stores `handoff_json`.

### Backfill

`services/orion-hub/scripts/world_pulse_read_backfill.py` CLI:

- Scan digests (limit / all), `enqueue_seeds` idempotent.
- Flags: `--findings-only`, `--limit-digests N`, `--dry-run`.
- No FCC — only enqueue.

### Status

`GET /world-pulse-read/api/status` returns:

- Wallet A + B done_today / caps / keys
- Queue: pending/claimed/done/failed counts; stage2 pending/claimed/done/failed
- last Stage 1 / Stage 2 timestamps
- enabled flags

Keep existing `/api/schedule` for Wallet A compatibility.

### Env

- `HUB_WORLD_PULSE_READ_STAGE2_ENABLED` default true (alongside Stage 1)
- `HUB_WORLD_PULSE_READ_WALLET_B_DAILY_CAP=6`
- Stage 2 tick/cooldown/window/timeout/session/route keys (mirror Stage 1 naming with `_STAGE2_` / `_WALLET_B_`)
- `HUB_WORLD_PULSE_READ_STAGE2_MAX_ROUND_TRIPS=5`

## Files likely to touch

- `orion/world_pulse_read/queue.py`, `wallet_b.py`
- `services/orion-sql-db/manual_migration_world_pulse_read_stage2_v1.sql`
- `services/orion-hub/scripts/world_pulse_read_pipeline.py` (persist handoff)
- `services/orion-hub/scripts/world_pulse_read_stage2.py` (new)
- `services/orion-hub/scripts/world_pulse_read_backfill.py` (new)
- `services/orion-hub/scripts/world_pulse_read_routes.py`, `main.py`, `settings.py`, `.env_example`
- Tests under `services/orion-hub/tests/` + root if needed

## Non-goals

- Merging with Curiosity Atlas wallet
- Rich HTML Hub tab (JSON status only this patch)
- Auto-running backfill on Hub start
- Changing Stage 1 daytime window defaults in `.env_example` (live operator may use `-1` overnight)

## Acceptance checks

1. Stage 1 done row has non-null `handoff_json`.
2. Stage 2 debit increments Wallet B only; Curiosity Atlas count unchanged.
3. Round-trip counter stops further Stage 1 re-entry at 5.
4. Backfill dry-run prints counts; real run inserts 0 on second pass (idempotent).
5. `/api/status` shows both wallets + queue counts.
6. Focused pytest green.

## Recommended next patch after this

Richer Hub UI tab; live eval of Stage 2 prior quality; optional findings-only operator mode in Hub UI.

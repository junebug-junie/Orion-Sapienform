# Heartbeat Ignition: Retention + First Workspace Consumers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the always-on substrate heartbeat safe to run (bounded retention for the field→attention→self_state chain), turn the merged-but-dark rung-3/4 loops on, and give the Global Workspace broadcast its first two consumers — present focus (attention broadcast → chat beliefs) and remembered past (episode summaries → chat beliefs).

**Architecture:** Four seams, all riding existing patterns. (1) Retention: the three pacemaker services (`orion-field-digester`, `orion-attention-runtime`, `orion-self-state-runtime`) each gain a batched, guard-railed prune loop cloned from `services/orion-substrate-runtime/app/receipt_pruner.py` — their tables (`substrate_field_state`, `substrate_attention_frames`, `substrate_self_state`, `self_state_predictions`, `identity_snapshots`) are INSERT-only today and grow ~43k rows/day/table once the 2s idle tick is enabled. (2) Ignition: an operator runbook applies the two rung-3/4 migrations and flips flags in dependency-safe order. (3) Broadcast consumer: the felt-state ctx reader (`services/orion-cortex-exec/app/substrate_felt_state_reader.py`) is already a generic multi-lane reader — add one `LaneSpec` for `substrate_attention_broadcast_projection` plus a new ctx adapter and 13th `ProducerEntryV1`. (4) Episodic readback: same pattern with a per-lane freshness override (episodes summarize 15-minute windows; the global 120s gate would always reject them) plus a 14th producer.

**Tech Stack:** Python 3.12, SQLAlchemy Core + psycopg2, pydantic v2, asyncio poll-loop workers, the substrate relational unification layer (`CognitiveUnificationLayer` / `ProducerRegistryV1`), pytest.

## Global Constraints

- **Fork from `origin/main`, not the current checkout.** The working checkout sits on `feat/agent-repl-lane`, which does NOT contain the rungs-3-4-5 merge (`390cd659`); the attention-broadcast/episodes code this plan consumes exists only on `main`.
- **Work in an isolated git worktree** (`git worktree add <scratch>/wt-heartbeat-ignition origin/main -b feat/heartbeat-ignition-v1`). A concurrent automation process drives git in the main checkout (documented in `docs/plans/substrate/CONTINUATION.md`) and has moved branches mid-session.
- Service tests run via `./scripts/test_service.sh <service> <test-path> -v` from the worktree root; `orion/` package tests run via `/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest <path> -v` with cwd = the worktree.
- Adapters degrade to `None` (never raise) on absent/unparseable input — repo-wide adapter contract.
- Retention defaults **on** (72h) because it is protective, not behavioral; every prune keeps the newest row per table so `load_latest_*` readers can never observe an empty table. Setting a retention value to `0` disables that pruner.
- Belief-node caps: attention adapter emits exactly 1 node with ≤8 attended ids; episodes adapter emits exactly 1 node — same bound discipline as the evidence-id caps.
- Episodes stay proposal-marked (`status='proposal'` in node metadata) and are never promoted to accepted truth by this plan (Knowledge Forge rule).
- Two pre-existing failures in `services/orion-substrate-runtime/tests/` (`test_quarantine_truth`, `test_worker_independent_reducers`) also fail on `main`; ignore them.
- No GitHub auth in this checkout: push via SSH, keep the PR body in-repo.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `services/orion-field-digester/app/store.py` | Modify | `prune_field_state()` + exported `PRUNE_FIELD_STATE_SQL` |
| `services/orion-field-digester/app/worker.py` | Modify | `_prune_tick`/`_prune_loop`, registered in `start()` |
| `services/orion-field-digester/app/settings.py` | Modify | `FIELD_STATE_RETENTION_HOURS`, `FIELD_STATE_PRUNE_INTERVAL_SEC` |
| `services/orion-field-digester/.env_example` | Modify | Document the two new vars |
| `services/orion-field-digester/tests/test_worker_prune.py` | Create | Task 1 tests |
| `services/orion-attention-runtime/app/{store,worker,settings}.py`, `.env_example` | Modify | Same pattern for `substrate_attention_frames` |
| `services/orion-attention-runtime/tests/test_worker_prune.py` | Create | Task 2 tests |
| `services/orion-self-state-runtime/app/{store,worker,settings}.py`, `.env_example` | Modify | Same pattern for `substrate_self_state`, `self_state_predictions`, `identity_snapshots` |
| `services/orion-self-state-runtime/tests/test_worker_prune.py` | Create | Task 3 tests |
| `services/orion-cortex-exec/app/substrate_felt_state_reader.py` | Modify | Per-lane `max_age_sec` + 2 new `LaneSpec`s |
| `services/orion-cortex-exec/tests/test_felt_state_reader_new_lanes.py` | Create | Task 4 tests |
| `orion/substrate/relational/adapters/attention_ctx.py` | Create | `map_attention_broadcast_ctx_to_substrate` |
| `orion/substrate/relational/tests/test_attention_ctx_adapter.py` | Create | Task 5 tests |
| `orion/substrate/relational/adapters/episodes_ctx.py` | Create | `map_episode_ctx_to_substrate` |
| `orion/substrate/relational/tests/test_episodes_ctx_adapter.py` | Create | Task 6 tests |
| `orion/cognition/projection_builder.py` | Modify | Register `attention` (Task 5) and `episodes` (Task 6) producers |
| `docs/plans/substrate/PR_heartbeat_ignition_v1.md` | Create | PR body (Task 7) |

## Non-goals

- No change to `orion/substrate/attention_broadcast.py`, `episodic_consolidation.py`, or any rung-3/4/5 engine logic — this plan adds retention and *readers*, not new substrate machinery.
- No `_project_attention_from_beliefs` stance projector in `chat_stance.py` (v1 lets the belief nodes flow through the unified set; explicit hazard/topic steering is a follow-up).
- No rung-5 (`ORION_ENDOGENOUS_CURIOSITY_ENABLED`) enablement — stays off pending operator sign-off.
- No pruning of `substrate_episode_summaries` (rung 4 already ships `prune_episode_summaries`, 14-day retention) or `substrate_reduction_receipts` (owned by `receipt_pruner.py`).
- No multi-episode readback (k>1) — v1 surfaces the single latest episode.
- No VACUUM in the new pruners — hourly small batches rely on autovacuum (tuned after the TOAST-OOM incident).

---

## Verified findings (read before implementing — confirmed against code, not assumed)

1. All three pacemaker workers share an identical loop shape: `start()` registers `asyncio.create_task(self._poll_loop(), ...)`; `_poll_loop` runs `asyncio.to_thread(self._tick)` then `asyncio.wait_for(self._stop.wait(), timeout=<interval>)` catching `TimeoutError`/`CancelledError` (`field-digester/app/worker.py:28-48`, `attention-runtime/app/worker.py:23-43`, `self-state-runtime/app/worker.py:49-71`). The prune loop clones this shape.
2. `substrate_field_state` has columns `tick_id (PK), generated_at, field_json, created_at` and its only reader is `ORDER BY generated_at DESC LIMIT 1` (`field-digester/app/store.py:129-149`); `substrate_attention_frames` has `frame_id (PK), source_field_tick_id, source_field_generated_at, generated_at, policy_id, frame_json, created_at`; `substrate_self_state` has `self_state_id (PK), source_field_tick_id, source_attention_frame_id, generated_at, policy_id, self_state_json, created_at`. `self_state_predictions` (`prediction_id (PK), source_self_state_id, generated_at, prediction_json, created_at`) is written **every tick** (`self-state-runtime/app/worker.py:142`); `identity_snapshots` every 10th tick (`_IDENTITY_SNAPSHOT_EVERY_N = 10`).
3. The felt-state reader's `_LANES` is a tuple of frozen `LaneSpec(ctx_key, table, payload_col, ts_col, projection_id)`; `projection_id=None` → `ORDER BY ts_col DESC LIMIT 1`; freshness is a single global `_max_age_sec` (default 120) applied in `hydrate()` (`substrate_felt_state_reader.py:120-151`). `_fetch_lane` is the only DB touchpoint and is what tests override.
4. On `main`, the broadcast writer persists a single row keyed `BROADCAST_PROJECTION_ID = "substrate.attention.broadcast.v1"` into `substrate_attention_broadcast_projection` (`projection_id PK, generated_at, projection_json, created_at`). The projection payload fields are exactly: `generated_at`, `frame`, `selected_action_type`, `selected_open_loop_id`, `selected_description`, `attended_node_ids` (`orion/substrate/attention_broadcast.py::broadcast_projection_from_frame`).
5. On `main`, `EpisodeSummaryV1` (`orion/core/schemas/substrate_episodes.py`) is flat (no nested models): `episode_id`, `status='proposal'`, `window_start/end`, `window_seconds`, `receipt_refs`, `receipt_count_total`, `receipt_count_capped`, `organ_counts`, `reducer_counts`, accepted/rejected/merged/noop counts, `state_delta_count`, `projection_update_count`, `warning_count`, `sample_warnings`, `notes`, `created_at`. Table `substrate_episode_summaries` has `episode_id PK, status, window_start, window_end, episode_json, created_at` with an index on `window_end desc`.
6. Adapter conventions (from `self_state_ctx.py` / `biometrics_ctx.py`): module-level `_TIER_RANK` (4 = snapshot_ephemeral), `_make_prov()` returning `SubstrateProvenanceV1(authority="local_inferred", ...)`, `_coerce()` accepting model/dict/JSON-string, `map_*_ctx_to_substrate(ctx) -> SubstrateGraphRecordV1 | None`, nodes are `ConceptNodeV1(anchor_scope="orion", subject_ref="entity:orion", ...)` with `make_temporal(observed_at=now)`.
7. The registry (`orion/cognition/projection_builder.py::build_projection_unification_registry`, lines 65-172) has 12 producers; ctx-sourced lanes use `trust_tier=SNAPSHOT_EPHEMERAL, pull_on_cold=False`. New adapters are imported directly (`from orion.substrate.relational.adapters.<mod> import <fn>`, lines 43-46 pattern) — the rung-2 adapters are *not* re-exported through `relational/__init__.py`, so don't add exports there.
8. Settings singletons: every service caches `_settings` module-level; tests reset with `settings_mod._settings = None` after `monkeypatch.setenv` (pattern in `field-digester/tests/test_worker.py`).

---

## Task 1: Field-digester retention (`substrate_field_state`)

**Files:**
- Modify: `services/orion-field-digester/app/store.py`
- Modify: `services/orion-field-digester/app/worker.py`
- Modify: `services/orion-field-digester/app/settings.py`
- Modify: `services/orion-field-digester/.env_example`
- Test: `services/orion-field-digester/tests/test_worker_prune.py`

**Interfaces:**
- Produces: `FieldDigesterStore.prune_field_state(*, retention_hours: float, batch_size: int = 5000) -> int` (rows deleted); module constant `PRUNE_FIELD_STATE_SQL: str`; settings fields `field_state_retention_hours: float` (alias `FIELD_STATE_RETENTION_HOURS`, default `72.0`, `<=0` disables) and `field_state_prune_interval_sec: float` (alias `FIELD_STATE_PRUNE_INTERVAL_SEC`, default `3600.0`); worker methods `_prune_tick()` / `_prune_loop()`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

```python
# services/orion-field-digester/tests/test_worker_prune.py
from __future__ import annotations

from unittest.mock import MagicMock

from app.store import PRUNE_FIELD_STATE_SQL
from app.worker import FieldDigesterWorker


def _make_worker(monkeypatch, *, retention_hours: str) -> FieldDigesterWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("FIELD_STATE_RETENTION_HOURS", retention_hours)
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = FieldDigesterWorker.__new__(FieldDigesterWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    return worker


def test_prune_tick_calls_store_with_configured_retention(monkeypatch):
    worker = _make_worker(monkeypatch, retention_hours="48.0")
    worker._store.prune_field_state.return_value = 3

    worker._prune_tick()

    worker._store.prune_field_state.assert_called_once_with(retention_hours=48.0)


def test_prune_tick_disabled_when_retention_zero(monkeypatch):
    worker = _make_worker(monkeypatch, retention_hours="0")

    worker._prune_tick()

    worker._store.prune_field_state.assert_not_called()


def test_prune_sql_is_batched_and_guards_latest_tick():
    # The newest row must never be deletable, so load_latest_field() can never
    # observe an empty table even if the writer is paused (idle tick flag off).
    assert "DELETE FROM substrate_field_state" in PRUNE_FIELD_STATE_SQL
    assert "LIMIT :batch_size" in PRUNE_FIELD_STATE_SQL
    assert "tick_id <>" in PRUNE_FIELD_STATE_SQL
    assert "ORDER BY generated_at DESC" in PRUNE_FIELD_STATE_SQL
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test_service.sh orion-field-digester services/orion-field-digester/tests/test_worker_prune.py -v`
Expected: FAIL (`ImportError: cannot import name 'PRUNE_FIELD_STATE_SQL'`)

- [ ] **Step 3: Add settings fields**

In `services/orion-field-digester/app/settings.py`, after `enable_idle_tick` (line 25):

```python
    field_state_retention_hours: float = Field(72.0, alias="FIELD_STATE_RETENTION_HOURS")
    field_state_prune_interval_sec: float = Field(3600.0, alias="FIELD_STATE_PRUNE_INTERVAL_SEC")
```

Add to `services/orion-field-digester/.env_example`:

```bash
# Bounded retention for substrate_field_state (the idle tick writes ~43k
# rows/day at 2s cadence). Newest row is always kept. 0 disables pruning.
FIELD_STATE_RETENTION_HOURS=72.0
FIELD_STATE_PRUNE_INTERVAL_SEC=3600.0
```

- [ ] **Step 4: Add the prune SQL + store method**

In `services/orion-field-digester/app/store.py`, add `timedelta` to the datetime import (line 5: `from datetime import datetime, timedelta, timezone`), then add after `FIELD_DIGEST_CURSOR_NAME` (line 17):

```python
# Batched, guard-railed prune: never deletes the newest row (by generated_at,
# the same ordering load_latest_field uses), so readers can never observe an
# empty table even when the writer is paused.
PRUNE_FIELD_STATE_SQL = """
DELETE FROM substrate_field_state
WHERE ctid IN (
    SELECT ctid
    FROM substrate_field_state
    WHERE created_at < :cutoff
      AND tick_id <> (
          SELECT tick_id FROM substrate_field_state
          ORDER BY generated_at DESC LIMIT 1
      )
    ORDER BY created_at ASC
    LIMIT :batch_size
)
"""
```

And add this method to `FieldDigesterStore` (after `save_field`, line 173):

```python
    def prune_field_state(self, *, retention_hours: float, batch_size: int = 5000) -> int:
        if retention_hours <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        total_deleted = 0
        while True:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text(PRUNE_FIELD_STATE_SQL),
                    {"cutoff": cutoff, "batch_size": batch_size},
                )
            deleted = result.rowcount or 0
            total_deleted += deleted
            if deleted < batch_size:
                break
        return total_deleted
```

- [ ] **Step 5: Add the prune loop to the worker**

In `services/orion-field-digester/app/worker.py`, change `start()` (line 28) and add the two methods after `_poll_loop` (line 48):

```python
    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="field-digester-poll")
        asyncio.create_task(self._prune_loop(), name="field-digester-prune")
```

```python
    def _prune_tick(self) -> None:
        retention = float(self._settings.field_state_retention_hours)
        if retention <= 0:
            return
        deleted = self._store.prune_field_state(retention_hours=retention)
        if deleted:
            logger.info(
                "field_state_pruned deleted=%d retention_hours=%.1f", deleted, retention
            )

    async def _prune_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._prune_tick)
            except Exception:
                logger.exception("field_state_prune_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.field_state_prune_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `./scripts/test_service.sh orion-field-digester services/orion-field-digester/tests/test_worker_prune.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Run the full field-digester suite for regressions**

Run: `./scripts/test_service.sh orion-field-digester services/orion-field-digester/tests/ -v`
Expected: PASS (including the pre-existing `test_worker.py` idle-tick tests)

- [ ] **Step 8: Commit**

```bash
git add services/orion-field-digester/app/store.py services/orion-field-digester/app/worker.py services/orion-field-digester/app/settings.py services/orion-field-digester/.env_example services/orion-field-digester/tests/test_worker_prune.py
git commit -m "feat(field-digester): bounded retention for substrate_field_state"
```

---

## Task 2: Attention-runtime retention (`substrate_attention_frames`)

**Files:**
- Modify: `services/orion-attention-runtime/app/store.py`
- Modify: `services/orion-attention-runtime/app/worker.py`
- Modify: `services/orion-attention-runtime/app/settings.py`
- Modify: `services/orion-attention-runtime/.env_example`
- Test: `services/orion-attention-runtime/tests/test_worker_prune.py`

**Interfaces:**
- Produces: `AttentionRuntimeStore.prune_attention_frames(*, retention_hours: float, batch_size: int = 5000) -> int`; constant `PRUNE_ATTENTION_FRAMES_SQL: str`; settings `attention_frame_retention_hours` (alias `ATTENTION_FRAME_RETENTION_HOURS`, default `72.0`) and `attention_frame_prune_interval_sec` (alias `ATTENTION_FRAME_PRUNE_INTERVAL_SEC`, default `3600.0`).
- Consumes: nothing from other tasks (same pattern as Task 1, restated in full — do not cross-reference).

- [ ] **Step 1: Write the failing tests**

```python
# services/orion-attention-runtime/tests/test_worker_prune.py
from __future__ import annotations

from unittest.mock import MagicMock

from app.store import PRUNE_ATTENTION_FRAMES_SQL
from app.worker import AttentionRuntimeWorker


def _make_worker(monkeypatch, *, retention_hours: str) -> AttentionRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ATTENTION_FRAME_RETENTION_HOURS", retention_hours)
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = AttentionRuntimeWorker.__new__(AttentionRuntimeWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    return worker


def test_prune_tick_calls_store_with_configured_retention(monkeypatch):
    worker = _make_worker(monkeypatch, retention_hours="48.0")
    worker._store.prune_attention_frames.return_value = 3

    worker._prune_tick()

    worker._store.prune_attention_frames.assert_called_once_with(retention_hours=48.0)


def test_prune_tick_disabled_when_retention_zero(monkeypatch):
    worker = _make_worker(monkeypatch, retention_hours="0")

    worker._prune_tick()

    worker._store.prune_attention_frames.assert_not_called()


def test_prune_sql_is_batched_and_guards_latest_frame():
    assert "DELETE FROM substrate_attention_frames" in PRUNE_ATTENTION_FRAMES_SQL
    assert "LIMIT :batch_size" in PRUNE_ATTENTION_FRAMES_SQL
    assert "frame_id <>" in PRUNE_ATTENTION_FRAMES_SQL
    assert "ORDER BY generated_at DESC" in PRUNE_ATTENTION_FRAMES_SQL
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test_service.sh orion-attention-runtime services/orion-attention-runtime/tests/test_worker_prune.py -v`
Expected: FAIL (`ImportError: cannot import name 'PRUNE_ATTENTION_FRAMES_SQL'`)

- [ ] **Step 3: Add settings fields**

In `services/orion-attention-runtime/app/settings.py`, after `enable_attention_runtime` (line 18):

```python
    attention_frame_retention_hours: float = Field(72.0, alias="ATTENTION_FRAME_RETENTION_HOURS")
    attention_frame_prune_interval_sec: float = Field(3600.0, alias="ATTENTION_FRAME_PRUNE_INTERVAL_SEC")
```

Add to `services/orion-attention-runtime/.env_example`:

```bash
# Bounded retention for substrate_attention_frames (~43k rows/day once the
# field-digester idle tick is on). Newest row is always kept. 0 disables.
ATTENTION_FRAME_RETENTION_HOURS=72.0
ATTENTION_FRAME_PRUNE_INTERVAL_SEC=3600.0
```

- [ ] **Step 4: Add the prune SQL + store method**

In `services/orion-attention-runtime/app/store.py`: ensure the datetime import includes `timedelta` (`from datetime import datetime, timedelta, timezone`), add at module level (below the imports):

```python
# Batched, guard-railed prune: never deletes the newest frame (by generated_at,
# matching load_latest_attention_frame's ordering).
PRUNE_ATTENTION_FRAMES_SQL = """
DELETE FROM substrate_attention_frames
WHERE ctid IN (
    SELECT ctid
    FROM substrate_attention_frames
    WHERE created_at < :cutoff
      AND frame_id <> (
          SELECT frame_id FROM substrate_attention_frames
          ORDER BY generated_at DESC LIMIT 1
      )
    ORDER BY created_at ASC
    LIMIT :batch_size
)
"""
```

And add to `AttentionRuntimeStore` (after `save_attention_frame`):

```python
    def prune_attention_frames(self, *, retention_hours: float, batch_size: int = 5000) -> int:
        if retention_hours <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        total_deleted = 0
        while True:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text(PRUNE_ATTENTION_FRAMES_SQL),
                    {"cutoff": cutoff, "batch_size": batch_size},
                )
            deleted = result.rowcount or 0
            total_deleted += deleted
            if deleted < batch_size:
                break
        return total_deleted
```

- [ ] **Step 5: Add the prune loop to the worker**

In `services/orion-attention-runtime/app/worker.py`, change `start()` (line 23) and add after `_poll_loop` (line 43):

```python
    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="attention-runtime-poll")
        asyncio.create_task(self._prune_loop(), name="attention-runtime-prune")
```

```python
    def _prune_tick(self) -> None:
        retention = float(self._settings.attention_frame_retention_hours)
        if retention <= 0:
            return
        deleted = self._store.prune_attention_frames(retention_hours=retention)
        if deleted:
            logger.info(
                "attention_frames_pruned deleted=%d retention_hours=%.1f", deleted, retention
            )

    async def _prune_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._prune_tick)
            except Exception:
                logger.exception("attention_frame_prune_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.attention_frame_prune_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
```

- [ ] **Step 6: Run tests to verify they pass, then the full service suite**

Run: `./scripts/test_service.sh orion-attention-runtime services/orion-attention-runtime/tests/ -v`
Expected: PASS (3 new tests + pre-existing suite)

- [ ] **Step 7: Commit**

```bash
git add services/orion-attention-runtime/app/store.py services/orion-attention-runtime/app/worker.py services/orion-attention-runtime/app/settings.py services/orion-attention-runtime/.env_example services/orion-attention-runtime/tests/test_worker_prune.py
git commit -m "feat(attention-runtime): bounded retention for substrate_attention_frames"
```

---

## Task 3: Self-state-runtime retention (3 tables)

**Files:**
- Modify: `services/orion-self-state-runtime/app/store.py`
- Modify: `services/orion-self-state-runtime/app/worker.py`
- Modify: `services/orion-self-state-runtime/app/settings.py`
- Modify: `services/orion-self-state-runtime/.env_example`
- Test: `services/orion-self-state-runtime/tests/test_worker_prune.py`

**Interfaces:**
- Produces: `SelfStateRuntimeStore.prune_history(*, retention_hours: float, batch_size: int = 5000) -> int` (total rows deleted across the three tables); constant `PRUNE_HISTORY_SQL: dict[str, str]` (table name → SQL); settings `self_state_retention_hours` (alias `SELF_STATE_RETENTION_HOURS`, default `72.0`) and `self_state_prune_interval_sec` (alias `SELF_STATE_PRUNE_INTERVAL_SEC`, default `3600.0`).
- Consumes: nothing from other tasks.

**Why three tables:** `substrate_self_state` and `self_state_predictions` are written every tick (`worker.py:142` calls `save_self_state_prediction` unconditionally); `identity_snapshots` every 10th tick (`_IDENTITY_SNAPSHOT_EVERY_N = 10`, `worker.py:29`). All three are INSERT-only today.

- [ ] **Step 1: Write the failing tests**

```python
# services/orion-self-state-runtime/tests/test_worker_prune.py
from __future__ import annotations

from unittest.mock import MagicMock

from app.store import PRUNE_HISTORY_SQL
from app.worker import SelfStateRuntimeWorker


def _make_worker(monkeypatch, *, retention_hours: str) -> SelfStateRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("SELF_STATE_RETENTION_HOURS", retention_hours)
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = SelfStateRuntimeWorker.__new__(SelfStateRuntimeWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    return worker


def test_prune_tick_calls_store_with_configured_retention(monkeypatch):
    worker = _make_worker(monkeypatch, retention_hours="48.0")
    worker._store.prune_history.return_value = 5

    worker._prune_tick()

    worker._store.prune_history.assert_called_once_with(retention_hours=48.0)


def test_prune_tick_disabled_when_retention_zero(monkeypatch):
    worker = _make_worker(monkeypatch, retention_hours="0")

    worker._prune_tick()

    worker._store.prune_history.assert_not_called()


def test_prune_sql_covers_all_three_tables_with_guards():
    assert set(PRUNE_HISTORY_SQL) == {
        "substrate_self_state",
        "self_state_predictions",
        "identity_snapshots",
    }
    guards = {
        "substrate_self_state": "self_state_id <>",
        "self_state_predictions": "prediction_id <>",
        "identity_snapshots": "snapshot_id <>",
    }
    for table, sql in PRUNE_HISTORY_SQL.items():
        assert f"DELETE FROM {table}" in sql
        assert "LIMIT :batch_size" in sql
        assert guards[table] in sql
        assert "ORDER BY generated_at DESC" in sql
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test_service.sh orion-self-state-runtime services/orion-self-state-runtime/tests/test_worker_prune.py -v`
Expected: FAIL (`ImportError: cannot import name 'PRUNE_HISTORY_SQL'`)

- [ ] **Step 3: Add settings fields**

In `services/orion-self-state-runtime/app/settings.py`, after `self_state_max_previous_age_sec`:

```python
    self_state_retention_hours: float = Field(72.0, alias="SELF_STATE_RETENTION_HOURS")
    self_state_prune_interval_sec: float = Field(3600.0, alias="SELF_STATE_PRUNE_INTERVAL_SEC")
```

Add to `services/orion-self-state-runtime/.env_example`:

```bash
# Bounded retention for substrate_self_state, self_state_predictions and
# identity_snapshots (per-tick / per-10-ticks writers). Newest row of each
# table is always kept. 0 disables.
SELF_STATE_RETENTION_HOURS=72.0
SELF_STATE_PRUNE_INTERVAL_SEC=3600.0
```

- [ ] **Step 4: Add the prune SQL dict + store method**

In `services/orion-self-state-runtime/app/store.py`: ensure the datetime import includes `timedelta`, add at module level:

```python
def _prune_sql(table: str, pk: str) -> str:
    # Batched, guard-railed prune: never deletes the newest row (by
    # generated_at, matching the latest-row readers' ordering).
    return f"""
DELETE FROM {table}
WHERE ctid IN (
    SELECT ctid
    FROM {table}
    WHERE created_at < :cutoff
      AND {pk} <> (
          SELECT {pk} FROM {table}
          ORDER BY generated_at DESC LIMIT 1
      )
    ORDER BY created_at ASC
    LIMIT :batch_size
)
"""


# Trusted module constants (NOT user input) — table/pk names are literals.
PRUNE_HISTORY_SQL: dict[str, str] = {
    "substrate_self_state": _prune_sql("substrate_self_state", "self_state_id"),
    "self_state_predictions": _prune_sql("self_state_predictions", "prediction_id"),
    "identity_snapshots": _prune_sql("identity_snapshots", "snapshot_id"),
}
```

And add to `SelfStateRuntimeStore`:

```python
    def prune_history(self, *, retention_hours: float, batch_size: int = 5000) -> int:
        if retention_hours <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        total_deleted = 0
        for sql in PRUNE_HISTORY_SQL.values():
            while True:
                with self._engine.begin() as conn:
                    result = conn.execute(
                        text(sql), {"cutoff": cutoff, "batch_size": batch_size}
                    )
                deleted = result.rowcount or 0
                total_deleted += deleted
                if deleted < batch_size:
                    break
        return total_deleted
```

- [ ] **Step 5: Add the prune loop to the worker**

In `services/orion-self-state-runtime/app/worker.py`, change `start()` (line 49) and add after `_poll_loop` (line 71):

```python
    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="self-state-runtime-poll")
        asyncio.create_task(self._prune_loop(), name="self-state-runtime-prune")
```

```python
    def _prune_tick(self) -> None:
        retention = float(self._settings.self_state_retention_hours)
        if retention <= 0:
            return
        deleted = self._store.prune_history(retention_hours=retention)
        if deleted:
            logger.info(
                "self_state_history_pruned deleted=%d retention_hours=%.1f",
                deleted,
                retention,
            )

    async def _prune_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._prune_tick)
            except Exception:
                logger.exception("self_state_prune_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.self_state_prune_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
```

- [ ] **Step 6: Run tests to verify they pass, then the full service suite**

Run: `./scripts/test_service.sh orion-self-state-runtime services/orion-self-state-runtime/tests/ -v`
Expected: PASS (3 new tests + pre-existing suite)

- [ ] **Step 7: Commit**

```bash
git add services/orion-self-state-runtime/app/store.py services/orion-self-state-runtime/app/worker.py services/orion-self-state-runtime/app/settings.py services/orion-self-state-runtime/.env_example services/orion-self-state-runtime/tests/test_worker_prune.py
git commit -m "feat(self-state-runtime): bounded retention for self-state history tables"
```

---

## Task 4: Felt-state reader — per-lane freshness + two new lanes

**Files:**
- Modify: `services/orion-cortex-exec/app/substrate_felt_state_reader.py`
- Test: `services/orion-cortex-exec/tests/test_felt_state_reader_new_lanes.py`

**Interfaces:**
- Produces: `LaneSpec` gains optional field `max_age_sec: int | None = None` (None → use the reader's global max age). Two new lanes hydrate ctx keys:
  - `ctx["attention_broadcast"]` — dict payload of `AttentionBroadcastProjectionV1` (keys: `generated_at`, `frame`, `selected_action_type`, `selected_open_loop_id`, `selected_description`, `attended_node_ids`), from single-row table `substrate_attention_broadcast_projection` where `projection_id = "substrate.attention.broadcast.v1"`, ts col `generated_at`, default freshness (120s vs the 30s broadcast cadence).
  - `ctx["episode_summary"]` — dict payload of `EpisodeSummaryV1`, latest row of `substrate_episode_summaries` by `created_at`, with `max_age_sec=1800` (episodes are 15-minute windows consolidated every 300s; the global 120s gate would always reject them).
- Consumes: nothing from other tasks (Tasks 5/6 consume these ctx keys).

- [ ] **Step 1: Write the failing tests**

```python
# services/orion-cortex-exec/tests/test_felt_state_reader_new_lanes.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.substrate_felt_state_reader import _LANES, SubstrateFeltStateReader


def _lane(ctx_key):
    return next(lane for lane in _LANES if lane.ctx_key == ctx_key)


def _reader() -> SubstrateFeltStateReader:
    return SubstrateFeltStateReader(
        enabled=True,
        database_url="postgresql://unused/unused",
        max_age_sec=120,
    )


def test_attention_broadcast_lane_registered():
    lane = _lane("attention_broadcast")
    assert lane.table == "substrate_attention_broadcast_projection"
    assert lane.payload_col == "projection_json"
    assert lane.ts_col == "generated_at"
    assert lane.projection_id == "substrate.attention.broadcast.v1"
    assert lane.max_age_sec is None  # global 120s gate applies


def test_episode_lane_registered_with_extended_max_age():
    lane = _lane("episode_summary")
    assert lane.table == "substrate_episode_summaries"
    assert lane.payload_col == "episode_json"
    assert lane.ts_col == "created_at"
    assert lane.projection_id is None
    assert lane.max_age_sec == 1800


def test_hydrate_injects_episode_older_than_global_gate(monkeypatch):
    reader = _reader()

    def fake_fetch(lane):
        if lane.ctx_key == "episode_summary":
            # 15 minutes old: stale under the global 120s gate, fresh under
            # the lane's 1800s override.
            return (
                {"episode_id": "ep1", "status": "proposal"},
                datetime.now(timezone.utc) - timedelta(seconds=900),
            )
        return None

    monkeypatch.setattr(reader, "_fetch_lane", fake_fetch)
    ctx: dict = {}
    reader.hydrate(ctx)
    assert ctx["episode_summary"] == {"episode_id": "ep1", "status": "proposal"}


def test_hydrate_rejects_episode_older_than_lane_max_age(monkeypatch):
    reader = _reader()

    def fake_fetch(lane):
        if lane.ctx_key == "episode_summary":
            return (
                {"episode_id": "ep_old", "status": "proposal"},
                datetime.now(timezone.utc) - timedelta(seconds=3600),
            )
        return None

    monkeypatch.setattr(reader, "_fetch_lane", fake_fetch)
    ctx: dict = {}
    reader.hydrate(ctx)
    assert "episode_summary" not in ctx


def test_hydrate_rejects_stale_attention_broadcast(monkeypatch):
    reader = _reader()

    def fake_fetch(lane):
        if lane.ctx_key == "attention_broadcast":
            return (
                {"selected_action_type": "focus"},
                datetime.now(timezone.utc) - timedelta(seconds=600),
            )
        return None

    monkeypatch.setattr(reader, "_fetch_lane", fake_fetch)
    ctx: dict = {}
    reader.hydrate(ctx)
    assert "attention_broadcast" not in ctx
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_felt_state_reader_new_lanes.py -v`
Expected: FAIL (`StopIteration` in `_lane` — the lanes don't exist; `LaneSpec` has no `max_age_sec`)

- [ ] **Step 3: Extend `LaneSpec` and `_LANES`**

In `services/orion-cortex-exec/app/substrate_felt_state_reader.py`, add the field to the dataclass (line 20-26):

```python
@dataclass(frozen=True)
class LaneSpec:
    ctx_key: str
    table: str
    payload_col: str
    ts_col: str
    projection_id: str | None
    # Per-lane freshness override; None → the reader's global max_age_sec.
    # Episodes roll up 15-minute windows, so the global 120s gate would
    # unconditionally reject them.
    max_age_sec: int | None = None
```

Append two entries to `_LANES` (after the `active_node_pressure_projection` entry, line 53-59):

```python
    LaneSpec(
        ctx_key="attention_broadcast",
        table="substrate_attention_broadcast_projection",
        payload_col="projection_json",
        ts_col="generated_at",
        projection_id="substrate.attention.broadcast.v1",
    ),
    LaneSpec(
        ctx_key="episode_summary",
        table="substrate_episode_summaries",
        payload_col="episode_json",
        ts_col="created_at",
        projection_id=None,
        max_age_sec=1800,
    ),
```

- [ ] **Step 4: Honor the per-lane max age in `hydrate()`**

In `SubstrateFeltStateReader.hydrate()` (lines 120-151), replace the two uses of `self._max_age_sec` with a per-lane value. At the top of the `for lane in _LANES:` body (right after the `try:`), add:

```python
                max_age = lane.max_age_sec if lane.max_age_sec is not None else self._max_age_sec
```

Then change the cache-TTL check from `if (time.monotonic() - fetched_at) <= self._max_age_sec:` to:

```python
                    if (time.monotonic() - fetched_at) <= max_age:
```

and the staleness check from `if age > self._max_age_sec:` to:

```python
                if age > max_age:
```

- [ ] **Step 5: Run tests to verify they pass, then the pre-existing reader/stance tests**

Run: `./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_felt_state_reader_new_lanes.py -v`
Expected: PASS (5 tests)

Run: `./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/ -k "felt_state or stance" -v`
Expected: PASS (no regressions in existing reader/stance suites)

- [ ] **Step 6: Commit**

```bash
git add services/orion-cortex-exec/app/substrate_felt_state_reader.py services/orion-cortex-exec/tests/test_felt_state_reader_new_lanes.py
git commit -m "feat(cortex-exec): hydrate attention broadcast + latest episode into felt-state ctx"
```

---

## Task 5: Attention adapter + 13th producer (`attending:*` beliefs)

**Files:**
- Create: `orion/substrate/relational/adapters/attention_ctx.py`
- Modify: `orion/cognition/projection_builder.py`
- Test: `orion/substrate/relational/tests/test_attention_ctx_adapter.py`

**Interfaces:**
- Consumes: `ctx["attention_broadcast"]` — dict (or JSON string) with keys `selected_action_type: str`, `selected_open_loop_id: str | None`, `selected_description: str | None`, `attended_node_ids: list[str]` (produced by Task 4; also present in `AttentionBroadcastProjectionV1` on main).
- Produces: `map_attention_broadcast_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None` emitting exactly one `ConceptNodeV1` labeled `attending:current_focus` (anchor `orion`), plus a `ProducerEntryV1(producer_id="attention", ...)` in the registry.

- [ ] **Step 1: Write the failing tests**

```python
# orion/substrate/relational/tests/test_attention_ctx_adapter.py
from __future__ import annotations

import json

from orion.substrate.relational.adapters.attention_ctx import (
    map_attention_broadcast_ctx_to_substrate,
)


def _payload(**overrides):
    payload = {
        "selected_action_type": "invoke",
        "selected_open_loop_id": "loop_1",
        "selected_description": "sustained execution prediction error",
        "attended_node_ids": [f"node:{i}" for i in range(12)],
    }
    payload.update(overrides)
    return payload


def test_maps_broadcast_to_single_attending_node():
    record = map_attention_broadcast_ctx_to_substrate({"attention_broadcast": _payload()})
    assert record is not None
    assert record.anchor_scope == "orion"
    assert len(record.nodes) == 1
    node = record.nodes[0]
    assert node.label == "attending:current_focus"
    assert node.metadata["selected_action_type"] == "invoke"
    assert node.metadata["selected_description"] == "sustained execution prediction error"
    # Hard cap: at most 8 attended node ids in metadata.
    assert node.metadata["attended_node_ids"] == [f"node:{i}" for i in range(8)]


def test_accepts_json_string_payload():
    record = map_attention_broadcast_ctx_to_substrate(
        {"attention_broadcast": json.dumps(_payload())}
    )
    assert record is not None
    assert record.nodes[0].metadata["selected_action_type"] == "invoke"


def test_returns_none_when_nothing_attended():
    record = map_attention_broadcast_ctx_to_substrate(
        {"attention_broadcast": _payload(selected_action_type="none", attended_node_ids=[])}
    )
    assert record is None


def test_returns_none_on_missing_or_garbage_ctx():
    assert map_attention_broadcast_ctx_to_substrate({}) is None
    assert map_attention_broadcast_ctx_to_substrate({"attention_broadcast": "not json"}) is None
    assert map_attention_broadcast_ctx_to_substrate(None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/relational/tests/test_attention_ctx_adapter.py -v`
(cwd = the worktree root)
Expected: FAIL (`ModuleNotFoundError: orion.substrate.relational.adapters.attention_ctx`)

- [ ] **Step 3: Implement the adapter**

```python
# orion/substrate/relational/adapters/attention_ctx.py
"""Attention-broadcast adapter — folds the workspace winner into beliefs.

GWT consumer rung: maps the rung-3 continuous-broadcast projection
(``AttentionBroadcastProjectionV1`` — the coalition that won the substrate
workspace competition) into a single belief node, so the unified belief set
contains a belief about *what Orion is currently attending to*. This is the
broadcast's first audience: without a consumer, the workspace winner is a
log row, not a broadcast.

ctx-sourced, pure (no network, no DB): reads ``ctx['attention_broadcast']``
as a dict or JSON string (the raw projection_json hydrated by the felt-state
reader), and degrades to ``None`` when absent, unparseable, or when nothing
is attended — never raises.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.attention_ctx")

_TIER_RANK = 4  # snapshot_ephemeral: derived workspace state, refreshed every tick
_ATTENDED_NODE_CAP = 8


def _make_prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="attention_broadcast",
        source_channel="substrate.attention.broadcast",
        producer="attention_broadcast_adapter",
        tier_rank=_TIER_RANK,
    )


def _coerce(raw: Any) -> dict[str, Any] | None:
    try:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
    except Exception as exc:
        logger.debug("attention_broadcast_adapter_parse_failed error=%s", exc)
    return None


def map_attention_broadcast_ctx_to_substrate(
    ctx: dict[str, Any],
) -> SubstrateGraphRecordV1 | None:
    """Map ``ctx['attention_broadcast']`` → one ``attending:current_focus`` node."""
    ctx = ctx if isinstance(ctx, dict) else {}
    payload = _coerce(ctx.get("attention_broadcast"))
    if payload is None:
        return None

    selected_action_type = str(payload.get("selected_action_type") or "none")
    attended_raw = payload.get("attended_node_ids") or []
    attended = [str(x) for x in attended_raw if x][:_ATTENDED_NODE_CAP]
    if selected_action_type == "none" and not attended:
        # Workspace competition produced no winner — nothing is attended,
        # so there is no belief to assert.
        return None

    description = payload.get("selected_description")
    node = ConceptNodeV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="attending:current_focus",
        temporal=make_temporal(observed_at=datetime.now(timezone.utc)),
        provenance=_make_prov(),
        signals=SubstrateSignalBundleV1(confidence=0.6, salience=0.6),
        metadata={
            "source_kind": "attention_broadcast",
            "selected_action_type": selected_action_type,
            "selected_open_loop_id": payload.get("selected_open_loop_id"),
            "selected_description": str(description) if description else None,
            "attended_node_ids": attended,
        },
    )
    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=[node])
```

- [ ] **Step 4: Register the producer**

In `orion/cognition/projection_builder.py`, add the import after line 46 (`map_biometrics_ctx_to_substrate`):

```python
from orion.substrate.relational.adapters.attention_ctx import (
    map_attention_broadcast_ctx_to_substrate,
)
```

And add the entry in `build_projection_unification_registry()`, after the `transport` producer (line 147):

```python
            # Workspace-broadcast lane (rung-3 consumer): what Orion is
            # currently attending to, from the continuous substrate broadcast.
            # ctx-sourced; no cold fan-out (read from ctx).
            ProducerEntryV1(
                producer_id="attention",
                trust_tier=SNAPSHOT_EPHEMERAL,
                anchor_scopes=("orion",),
                freshness_ttl_sec=120,
                pull_on_cold=False,
                adapter_fn=map_attention_broadcast_ctx_to_substrate,
            ),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/relational/tests/test_attention_ctx_adapter.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Check for registry-shape tests and update expected producer lists**

Run: `grep -rn "producer_id" orion/substrate/relational/tests orion/cognition --include=*.py | grep -i "test\|expected"` then run:
`/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/relational/tests orion/cognition -v`
If any test asserts the registry's producer-id list/count (the rung-2 work added a 12-producer assertion), add `"attention"` to its expected list. Expected end state: PASS.

- [ ] **Step 7: Commit**

```bash
git add orion/substrate/relational/adapters/attention_ctx.py orion/substrate/relational/tests/test_attention_ctx_adapter.py orion/cognition/projection_builder.py
git commit -m "feat(substrate): attention-broadcast adapter — the workspace winner enters unified beliefs"
```

---

## Task 6: Episodes adapter + 14th producer (`episode:*` beliefs)

**Files:**
- Create: `orion/substrate/relational/adapters/episodes_ctx.py`
- Modify: `orion/cognition/projection_builder.py`
- Test: `orion/substrate/relational/tests/test_episodes_ctx_adapter.py`

**Interfaces:**
- Consumes: `ctx["episode_summary"]` — dict (or JSON string) of `EpisodeSummaryV1` (produced by Task 4). Relevant fields: `episode_id`, `status`, `window_start`, `window_end`, `receipt_count_total`, `organ_counts`, `warning_count`, `notes`.
- Produces: `map_episode_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None` emitting exactly one `ConceptNodeV1` labeled `episode:latest`, proposal-marked in metadata; plus `ProducerEntryV1(producer_id="episodes", ...)`.

- [ ] **Step 1: Write the failing tests**

```python
# orion/substrate/relational/tests/test_episodes_ctx_adapter.py
from __future__ import annotations

import json
from datetime import datetime, timezone

from orion.substrate.relational.adapters.episodes_ctx import map_episode_ctx_to_substrate


def _episode(**overrides):
    payload = {
        "schema_version": "substrate.episode_summary.v1",
        "episode_id": "ep_abc123",
        "status": "proposal",
        "window_start": "2026-07-02T10:00:00+00:00",
        "window_end": "2026-07-02T10:15:00+00:00",
        "window_seconds": 900,
        "receipt_refs": ["r1", "r2"],
        "receipt_count_total": 42,
        "organ_counts": {"biometrics_pressure": 30, "execution": 8, "transport": 4},
        "warning_count": 1,
        "notes": ["transport backlog spike"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(overrides)
    return payload


def test_maps_episode_to_single_proposal_marked_node():
    record = map_episode_ctx_to_substrate({"episode_summary": _episode()})
    assert record is not None
    assert record.anchor_scope == "orion"
    assert len(record.nodes) == 1
    node = record.nodes[0]
    assert node.label == "episode:latest"
    assert node.metadata["status"] == "proposal"
    assert node.metadata["episode_id"] == "ep_abc123"
    assert node.metadata["receipt_count_total"] == 42
    assert node.metadata["window_end"] == "2026-07-02T10:15:00+00:00"


def test_accepts_json_string_payload():
    record = map_episode_ctx_to_substrate({"episode_summary": json.dumps(_episode())})
    assert record is not None
    assert record.nodes[0].metadata["episode_id"] == "ep_abc123"


def test_salience_scales_with_receipt_count_and_is_clamped():
    quiet = map_episode_ctx_to_substrate(
        {"episode_summary": _episode(receipt_count_total=0)}
    )
    busy = map_episode_ctx_to_substrate(
        {"episode_summary": _episode(receipt_count_total=500)}
    )
    assert quiet is not None and busy is not None
    assert quiet.nodes[0].signals.salience < busy.nodes[0].signals.salience
    assert busy.nodes[0].signals.salience <= 1.0


def test_returns_none_on_missing_or_garbage_ctx():
    assert map_episode_ctx_to_substrate({}) is None
    assert map_episode_ctx_to_substrate({"episode_summary": "not json"}) is None
    assert map_episode_ctx_to_substrate(None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/relational/tests/test_episodes_ctx_adapter.py -v`
Expected: FAIL (`ModuleNotFoundError: orion.substrate.relational.adapters.episodes_ctx`)

- [ ] **Step 3: Implement the adapter**

```python
# orion/substrate/relational/adapters/episodes_ctx.py
"""Episodic-continuity adapter — the remembered past enters beliefs.

Rung-4 consumer: maps the latest ``EpisodeSummaryV1`` (a proposal-marked
rollup of one time-window of reduction receipts — "what happened to me")
into a single belief node so the present self can read its own recent
history. The node stays proposal-marked: episodes inform stance, they never
silently become accepted truth (Knowledge Forge rule).

ctx-sourced, pure (no network, no DB): reads ``ctx['episode_summary']`` as an
``EpisodeSummaryV1``, dict, or JSON string, and degrades to ``None`` when
absent or unparseable — never raises.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.core.schemas.substrate_episodes import EPISODE_RECEIPT_CAP, EpisodeSummaryV1
from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.episodes_ctx")

_TIER_RANK = 4  # snapshot_ephemeral: derived autobiographical rollup, proposal-marked
_ORGAN_COUNT_CAP = 5
_NOTE_CAP = 4


def _make_prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="episode_summary",
        source_channel="substrate.episodes",
        producer="episodes_adapter",
        tier_rank=_TIER_RANK,
    )


def _coerce(raw: Any) -> EpisodeSummaryV1 | None:
    try:
        if isinstance(raw, EpisodeSummaryV1):
            return raw
        if isinstance(raw, str) and raw.strip():
            return EpisodeSummaryV1.model_validate_json(raw)
        if isinstance(raw, dict):
            return EpisodeSummaryV1.model_validate(raw)
    except Exception as exc:
        logger.debug("episodes_adapter_parse_failed error=%s", exc)
    return None


def map_episode_ctx_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Map ``ctx['episode_summary']`` → one proposal-marked ``episode:latest`` node."""
    ctx = ctx if isinstance(ctx, dict) else {}
    episode = _coerce(ctx.get("episode_summary"))
    if episode is None:
        return None

    top_organs = dict(
        sorted(episode.organ_counts.items(), key=lambda kv: kv[1], reverse=True)[
            :_ORGAN_COUNT_CAP
        ]
    )
    # Busier windows are more salient; EPISODE_RECEIPT_CAP receipts saturate.
    salience = max(0.0, min(1.0, episode.receipt_count_total / float(EPISODE_RECEIPT_CAP)))
    node = ConceptNodeV1(
        anchor_scope="orion",
        subject_ref="entity:orion",
        label="episode:latest",
        temporal=make_temporal(observed_at=datetime.now(timezone.utc)),
        provenance=_make_prov(),
        signals=SubstrateSignalBundleV1(confidence=0.5, salience=salience),
        metadata={
            "source_kind": "episode_summary",
            "status": episode.status,  # stays 'proposal' — never accepted truth
            "episode_id": episode.episode_id,
            "window_start": episode.window_start.isoformat(),
            "window_end": episode.window_end.isoformat(),
            "receipt_count_total": episode.receipt_count_total,
            "organ_counts": top_organs,
            "warning_count": episode.warning_count,
            "notes": list(episode.notes[:_NOTE_CAP]),
        },
    )
    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=[node])
```

- [ ] **Step 4: Register the producer**

In `orion/cognition/projection_builder.py`, add the import next to the attention adapter import (Task 5):

```python
from orion.substrate.relational.adapters.episodes_ctx import map_episode_ctx_to_substrate
```

And add the entry immediately after the `attention` producer added in Task 5:

```python
            # Episodic lane (rung-4 consumer): the latest proposal-marked
            # episode summary — "what happened to me" — enters beliefs.
            # ctx-sourced; longer TTL (episodes are 15-minute windows).
            ProducerEntryV1(
                producer_id="episodes",
                trust_tier=SNAPSHOT_EPHEMERAL,
                anchor_scopes=("orion",),
                freshness_ttl_sec=1800,
                pull_on_cold=False,
                adapter_fn=map_episode_ctx_to_substrate,
            ),
```

- [ ] **Step 5: Run tests to verify they pass, plus the relational + cognition suites**

Run: `/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/relational/tests/test_episodes_ctx_adapter.py orion/substrate/relational/tests orion/cognition -v`
If a registry-shape test asserts the producer list, add `"episodes"` to its expected list (as done for `"attention"` in Task 5).
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add orion/substrate/relational/adapters/episodes_ctx.py orion/substrate/relational/tests/test_episodes_ctx_adapter.py orion/cognition/projection_builder.py
git commit -m "feat(substrate): episodes adapter — latest proposal-marked episode enters unified beliefs"
```

---

## Task 7: Ignition runbook + PR body (operator steps, no code)

**Files:**
- Create: `docs/plans/substrate/PR_heartbeat_ignition_v1.md` (PR body — no GitHub auth in this checkout, PR is created from this file)

**Interfaces:**
- Consumes: everything above, deployed.

- [ ] **Step 1: Write the PR body**

Create `docs/plans/substrate/PR_heartbeat_ignition_v1.md` summarizing Tasks 1–6 (branch `feat/heartbeat-ignition-v1` → `main`), including the test status and the runbook below, following the format of `docs/plans/substrate/PR_substrate_rungs_3_4_5.md`.

- [ ] **Step 2: Commit and push**

```bash
git add docs/plans/substrate/PR_heartbeat_ignition_v1.md
git commit -m "docs: PR body + ignition runbook for heartbeat-ignition-v1"
git push origin feat/heartbeat-ignition-v1
```

- [ ] **Step 3: Operator runbook (live stack, after merge + deploy — record outcomes in the PR)**

0. **Resolve the DB-identity question first:** confirm the substrate-runtime `POSTGRES_URI` and cortex-exec `SUBSTRATE_FELT_STATE_DATABASE_URL` point at the same database:
   `psql "$POSTGRES_URI" -c "\dt substrate_*"` — `substrate_attention_broadcast_projection` and `substrate_episode_summaries` must be visible from the DSN cortex-exec reads. If they are in different databases, the felt-state lanes will be silently empty; stop and reconcile before flipping any flag.
1. Apply migrations:
   `psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_episodes_v1.sql`
   `psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_broadcast_v1.sql`
2. Measure before enabling the 2s heartbeat: `psql "$POSTGRES_URI" -c "SELECT pg_size_pretty(pg_total_relation_size('substrate_field_state')), (SELECT count(*) FROM substrate_field_state)"` — record the baseline; re-check after 24h of idle tick and confirm growth is bounded by the 72h retention.
3. `SUBSTRATE_EPISODIC_TICK_ENABLED=true` on orion-substrate-runtime; restart; after ≥15 min confirm one row: `psql "$POSTGRES_URI" -c "SELECT episode_id, status, window_end FROM substrate_episode_summaries ORDER BY window_end DESC LIMIT 1"`.
4. Deploy Tasks 1–3 (retention); confirm `field_state_pruned` / `attention_frames_pruned` / `self_state_history_pruned` log lines appear within one prune interval on a table with >72h-old rows.
5. `FIELD_DIGESTER_IDLE_TICK_ENABLED=true` on orion-field-digester; restart; stop all chat/biometrics activity >30s and confirm `substrate_field_state.tick_id`, `substrate_attention_frames`, and `substrate_self_state` all continue advancing ~every 2s (heartbeat-pacemaker-v1 acceptance check).
6. Confirm `SUBSTRATE_STORE_BACKEND=sparql` is configured and Fuseki reachable on orion-substrate-runtime, then set `SUBSTRATE_DYNAMICS_TICK_ENABLED=true` and `ORION_ATTENTION_BROADCAST_ENABLED=true`; restart; confirm `substrate_dynamics_tick_completed` log lines and a fresh row: `psql "$POSTGRES_URI" -c "SELECT generated_at, projection_json->>'selected_action_type' FROM substrate_attention_broadcast_projection"`.
7. End-to-end belief check: send one Hub chat turn and confirm (cortex-exec logs) the unified belief set now contains `attending:current_focus` and `episode:latest` nodes for the `orion` anchor.
8. `ORION_ENDOGENOUS_CURIOSITY_ENABLED` **stays false** — rung 5 is explicitly out of scope pending operator sign-off.

---

## Self-Review (spec coverage)

| Spec requirement (brainstorm options 1–4) | Task |
|---|---|
| Retention for `substrate_field_state`, newest row guarded | 1 |
| Retention for `substrate_attention_frames` | 2 |
| Retention for `substrate_self_state` + `self_state_predictions` + `identity_snapshots` | 3 |
| Retention default-on (protective), `0` disables, batched deletes | 1–3 |
| Migrations applied + flags flipped in dependency-safe order | 7 |
| Row-size/growth measured before committing to 2s cadence | 7 (step 2) |
| DB-identity ambiguity (substrate-runtime URI vs felt-state DSN) resolved before flags | 7 (step 0) |
| Broadcast lane hydrated into ctx (`LaneSpec`, single-row projection id) | 4 |
| Episode lane hydrated with per-lane freshness (1800s vs global 120s) | 4 |
| `attending:*` belief node, ≤8 attended ids, fail-open adapter | 5 |
| 13th/14th producers registered (`attention`, `episodes`), ctx-sourced, SNAPSHOT_EPHEMERAL | 5, 6 |
| Episodes stay proposal-marked, never accepted truth | 6 |
| Rung 5 stays off | 7 (step 8), Non-goals |

**Known v1 limits (documented, not tasks):** the episodes lane surfaces only the latest episode (k=1); the attention belief node carries fixed confidence/salience (0.6) rather than deriving them from the frame's signal strengths; no `chat_stance.py` projector folds `attending:*` into explicit hazards/topics yet — the nodes flow through the unified belief set only. All three are deliberate thin-slice choices with obvious follow-ups.

**Placeholder scan:** none. Every code step contains complete code; the only conditional instruction (registry-shape test update in Tasks 5/6) names the exact change to make (`add "attention"` / `add "episodes"` to the expected producer list) and the exact command that reveals whether it's needed.

**Type consistency check:** `prune_field_state` / `prune_attention_frames` / `prune_history` all use keyword-only `retention_hours: float, batch_size: int = 5000` and return `int`; both adapters are `(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None`; ctx keys `attention_broadcast` / `episode_summary` match between Task 4 lanes and Task 5/6 adapters; `LaneSpec.max_age_sec` name matches between the dataclass change and the tests.

# Heartbeat Ignition: Retention + First Workspace Consumers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the always-on substrate heartbeat safe to run (bounded retention for the field→attention→self_state chain), turn the merged-but-dark rung-3/4 loops on, and give the Global Workspace broadcast its first two consumers — present focus (attention broadcast → chat beliefs) and remembered past (episode summaries → chat beliefs).

**Architecture:** Four seams, all riding existing patterns. (1) Retention: the three pacemaker services (`orion-field-digester`, `orion-attention-runtime`, `orion-self-state-runtime`) each gain a batched, guard-railed prune loop cloned from `services/orion-substrate-runtime/app/receipt_pruner.py` — their tables (`substrate_field_state`, `substrate_attention_frames`, `substrate_self_state`, `self_state_predictions`, `identity_snapshots`) are INSERT-only today and grow ~43k rows/day/table once the 2s idle tick is enabled. (2) Ignition: an operator runbook applies the two rung-3/4 migrations and flips flags in dependency-safe order. (3) Broadcast consumer: the felt-state ctx reader (`services/orion-cortex-exec/app/substrate_felt_state_reader.py`) is already a generic multi-lane reader — add one `LaneSpec` for `substrate_attention_broadcast_projection` plus a new ctx adapter and 13th `ProducerEntryV1`. (4) Episodic readback: same pattern with a per-lane freshness override (episodes summarize 15-minute windows; the global 120s gate would always reject them) plus a 14th producer.

**Tech Stack:** Python 3.12, SQLAlchemy Core + psycopg2, pydantic v2, asyncio poll-loop workers, the substrate relational unification layer (`CognitiveUnificationLayer` / `ProducerRegistryV1`), pytest.

## Global Constraints

- **Fork from `origin/main`, not the current checkout.** The working checkout sits on `feat/agent-repl-lane`, which does NOT contain the rungs-3-4-5 merge (`390cd659`, main-only); the attention-broadcast/episodes code this plan consumes exists only on `main`.
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
| `services/orion-attention-runtime/tests/test_worker_prune.py` | Create | Task 2 tests (+ new conftest.py — no tests dir exists) |
| `services/orion-self-state-runtime/app/{store,worker,settings}.py`, `.env_example` | Modify | Same pattern for `substrate_self_state`, `self_state_predictions`, `identity_snapshots` |
| `services/orion-self-state-runtime/tests/test_worker_prune.py` | Create | Task 3 tests (+ new conftest.py) |
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

1. All three pacemaker workers share an identical loop shape: `start()` registers `asyncio.create_task(self._poll_loop(), ...)`; `_poll_loop` runs `asyncio.to_thread(self._tick)` then `asyncio.wait_for(self._stop.wait(), timeout=<interval>)` catching `TimeoutError`/`CancelledError`. The prune loop clones this shape.
2. `substrate_field_state` has columns `tick_id (PK), generated_at, field_json, created_at` and its only reader is `ORDER BY generated_at DESC LIMIT 1`; `substrate_attention_frames` has `frame_id (PK), source_field_tick_id, source_field_generated_at, generated_at, policy_id, frame_json, created_at`; `substrate_self_state` has `self_state_id (PK), source_field_tick_id, source_attention_frame_id, generated_at, policy_id, self_state_json, created_at`. `self_state_predictions` (`prediction_id (PK), source_self_state_id, generated_at, prediction_json, created_at`) is written **every tick**; `identity_snapshots` every 10th tick (`_IDENTITY_SNAPSHOT_EVERY_N = 10`).
3. The felt-state reader's `_LANES` is a tuple of frozen `LaneSpec(ctx_key, table, payload_col, ts_col, projection_id)`; `projection_id=None` → `ORDER BY ts_col DESC LIMIT 1`; freshness is a single global `_max_age_sec` (default 120) applied in `hydrate()`. `_fetch_lane` is the only DB touchpoint and is what tests override.
4. On `main`, the broadcast writer persists a single row keyed `BROADCAST_PROJECTION_ID = "substrate.attention.broadcast.v1"` into `substrate_attention_broadcast_projection` (`projection_id PK, generated_at, projection_json, created_at`). The projection payload fields are exactly: `generated_at`, `frame`, `selected_action_type`, `selected_open_loop_id`, `selected_description`, `attended_node_ids` (`orion/substrate/attention_broadcast.py::broadcast_projection_from_frame`).
5. On `main`, `EpisodeSummaryV1` (`orion/core/schemas/substrate_episodes.py`) is flat (no nested models): `episode_id`, `status='proposal'`, `window_start/end`, `window_seconds`, `receipt_refs`, `receipt_count_total`, `receipt_count_capped`, `organ_counts`, `reducer_counts`, accepted/rejected/merged/noop counts, `state_delta_count`, `projection_update_count`, `warning_count`, `sample_warnings`, `notes`, `created_at`. Table `substrate_episode_summaries` has `episode_id PK, status, window_start, window_end, episode_json, created_at` with an index on `window_end desc`.
6. Adapter conventions (from `self_state_ctx.py` / `biometrics_ctx.py`): module-level `_TIER_RANK` (4 = snapshot_ephemeral), `_make_prov()` returning `SubstrateProvenanceV1(authority="local_inferred", ...)`, `_coerce()` accepting model/dict/JSON-string, `map_*_ctx_to_substrate(ctx) -> SubstrateGraphRecordV1 | None`, nodes are `ConceptNodeV1(anchor_scope="orion", subject_ref="entity:orion", ...)` with `make_temporal(observed_at=now)`.
7. The registry (`orion/cognition/projection_builder.py::build_projection_unification_registry`) has 12 producers; ctx-sourced lanes use `trust_tier=SNAPSHOT_EPHEMERAL, pull_on_cold=False`. New adapters are imported directly (`from orion.substrate.relational.adapters.<mod> import <fn>`) — the rung-2 adapters are *not* re-exported through `relational/__init__.py`, so don't add exports there. A registry-shape test (`orion/substrate/relational/tests/test_reducer_lane_adapters.py::test_registry_registers_three_reducer_lanes`) asserts the producer count and must be updated 12 → 14.
8. Settings singletons: every service caches `_settings` module-level; tests reset with `settings_mod._settings = None` after `monkeypatch.setenv`.

---

## Task summaries (executed 2026-07-02; see git history on this branch for the full diffs)

- **Task 1 — field-digester retention** (`a326eecd`): `PRUNE_FIELD_STATE_SQL` (ctid-batched delete, newest-`tick_id`-by-`generated_at` guard), `prune_field_state(*, retention_hours, batch_size=5000) -> int`, `_prune_tick`/`_prune_loop` registered in `start()`, settings + `.env_example`, 3 tests in `test_worker_prune.py`. Suite: 11 passed.
- **Task 2 — attention-runtime retention** (`cf03d32b`): same pattern for `substrate_attention_frames` (`frame_id` guard); new `tests/` dir with conftest mirroring field-digester's. Suite: 3 passed.
- **Task 3 — self-state retention** (`665d37ef`): `_prune_sql(table, pk)` factory + `PRUNE_HISTORY_SQL` dict over `substrate_self_state`/`self_state_predictions`/`identity_snapshots`, `prune_history()` looping batches per table; new `tests/` dir. Suite: 3 passed.
- **Task 4 — felt-state reader lanes** (`5dd01280`): `LaneSpec.max_age_sec: int | None = None`; `attention_broadcast` lane (projection id `substrate.attention.broadcast.v1`, global gate) and `episode_summary` lane (`created_at` DESC, `max_age_sec=1800`); `hydrate()` uses per-lane `max_age` for cache TTL and staleness. 5 new + 6 pre-existing tests passed.
- **Task 5 — attention adapter** (`7a4cc484`): `map_attention_broadcast_ctx_to_substrate` → one `attending:current_focus` node, attended ids capped at 8, `None` when nothing attended. 4 tests.
- **Task 6 — episodes adapter** (`83eab9ea`): `map_episode_ctx_to_substrate` → one proposal-marked `episode:latest` node, organ counts top-5, notes ≤4, salience = `receipt_count_total/EPISODE_RECEIPT_CAP` clamped. 4 tests.
- **Registry wiring** (`2f1a20e8`): `attention` (ttl 120) + `episodes` (ttl 1800) producers, both `SNAPSHOT_EPHEMERAL`/ctx-sourced; registry-shape test updated 12 → 14. Relational + cognition suites: 81 passed.
- **Task 7 — PR body + runbook**: `docs/plans/substrate/PR_heartbeat_ignition_v1.md` (includes the 8-step operator ignition runbook: DB-identity check, migrations, growth baseline, episodic tick → retention deploy → idle tick → sparql-gated dynamics tick + broadcast, end-to-end belief check; rung 5 stays off).

**Known v1 limits (documented, not tasks):** the episodes lane surfaces only the latest episode (k=1); the attention belief node carries fixed confidence/salience (0.6) rather than deriving them from the frame's signal strengths; no `chat_stance.py` projector folds `attending:*` into explicit hazards/topics yet — the nodes flow through the unified belief set only. All three are deliberate thin-slice choices with obvious follow-ups.

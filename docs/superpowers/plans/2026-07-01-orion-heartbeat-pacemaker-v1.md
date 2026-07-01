# Orion Heartbeat Pacemaker v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the field → attention → self_state chain advance on a continuous cadence instead of only when new substrate reduction receipts arrive, so `SelfStateV1`/φ exists as an ongoing process rather than only when something pokes it. Then make chat stance actually read the resulting φ signal, and (lower priority) add a single whole-self surprise scalar. Separately, close the rung-1 gap left open by PR #766 (`feat/substrate-rung1-bridge-rung2-lanes`, merged) by giving the graph-substrate's own `SubstrateDynamicsEngine` a periodic caller, so the prediction-error nodes that PR already writes actually seed pressure instead of sitting inert.

**Architecture:** The mesh has *two* substrates that both need a pacemaker, and this plan closes the gap in both. (1) A three-stage field/attention/self-model reduction chain: `orion-field-digester` (receipts → `FieldStateV1` tick) → `orion-attention-runtime` (field tick → `AttentionFrameV1`) → `orion-self-state-runtime` (attention frame → `SelfStateV1`, broadcast on `substrate.self_state.v1`). Attention-runtime and self-state-runtime already free-run off of whatever the upstream tick/frame id is — they do not check "did new receipts arrive," they check "has this tick/frame already been processed." The single chokepoint is `FieldDigesterWorker._tick()`, which returns immediately when `fetch_new_receipts()` is empty and therefore never mints a new `tick_id` during quiet periods. This plan removes that chokepoint (Task 1 / Phase 1), confirms the two downstream services free-run once it is removed (Task 2 / Phase 2 prerequisite check, no code change), wires a real consumer of the now-continuous φ broadcast into `chat_stance.py` (Task 3 / Phase 2), and optionally adds a whole-self surprise aggregate (Task 4 / Phase 3). (2) A separate graph-substrate dynamics engine (`orion/substrate/dynamics.py::SubstrateDynamicsEngine`) that computes activation decay, pressure propagation (including seeding pressure from `metadata['prediction_error']`), and dormancy transitions over the durable substrate graph (`orion/substrate/graphdb_store.py`) — but which nothing calls on a schedule against that graph anywhere in the mesh today. PR #766 wired the write side of this (execution/transport prediction-error nodes land on durable nodes) and documented the read side as an explicit known gap in its PR body. Task 5 closes that gap with the same pattern used in Task 1: a bounded, fail-open, flag-gated poll loop.

**This plan supersedes the tensor-network approach.** `docs/research/2026-05-01-orion-heartbeat-research-charter.md` and its companion `docs/superpowers/specs/2026-05-01-orion-heartbeat-engineering-spec.md` describe a dedicated `orion-heartbeat` service built on a `quimb` matrix-product-state substrate (N=24, χ=4), with a four-hypothesis measurement harness and ablation research program. That substrate was never the deployed reality: the substrate that actually exists and runs today is the graph/field-dynamics engine (`orion/substrate/dynamics.py`, `orion-field-digester`, `orion-attention-runtime`, `orion-self-state-runtime`). This plan does not build the tensor-network service, does not add a `quimb` dependency, and does not stand up an H1–H4 measurement harness. It takes the existing, already-running field→attention→self_state chain and removes the one gap that keeps it from being continuous. Treat the charter/spec as historical prior-art context for *why a pacemaker matters conceptually*, not as a design to implement.

**Tech Stack:** Python 3.12, SQLAlchemy Core + psycopg2 (`FieldDigesterStore`), pydantic v2 (`FieldStateV1`, `SelfStateV1`), asyncio poll-loop workers, Orion bus (`OrionBusAsync`, `publish_with_reconnect`), the substrate relational unification layer (`CognitiveUnificationLayer`, `ProducerRegistryV1`) already wired into `orion-cortex-exec`.

**Design spec:** No new companion spec doc — the mechanism is small enough (one worker's idle-tick branch, one new stance projector, one optional schema field) that architecture rationale is carried inline in this header and in each task's context notes. `docs/research/2026-05-01-orion-heartbeat-research-charter.md` and `docs/superpowers/specs/2026-05-01-orion-heartbeat-engineering-spec.md` remain as historical context only (see supersession note above).

**Parent plan:** None. This is a standalone fix to an existing, already-deployed three-service chain.

**Worktree:** Implement in an isolated worktree (`using-superpowers:using-git-worktrees`) before touching main — this plan touches a live poll-loop worker (`orion-field-digester`) and a hot chat-turn path (`chat_stance.py`).

---

## Verified findings (read before implementing)

These are load-bearing facts confirmed by reading the actual code, not assumptions — implementers should not need to re-derive them:

1. **The chokepoint is exactly one guard clause.** `services/orion-field-digester/app/worker.py::FieldDigesterWorker._tick()` (lines 50–103) opens with:
   ```python
   fetched = self._store.fetch_new_receipts(limit=50)
   if not fetched:
       return
   ```
   Everything else in `_tick()` — loading/reconciling field state, minting a new `tick_id`, running `run_digestion_tick` (decay + diffusion + suppression), coherence checks, and `commit_digest_tick` — never executes on a quiet poll. No new `substrate_field_state` row is written, so `tick_id` never advances, so nothing downstream has a reason to run.

2. **`run_digestion_tick` is already safe with zero perturbations.** `services/orion-field-digester/app/tensor/update_rules.py::run_digestion_tick` unconditionally calls `apply_perturbations(state, perturbations)` (a no-op on `[]`), then `apply_decay`, `apply_diffusion`, `apply_suppression` regardless of perturbation count. Decay/diffusion is not gated on receipts existing. No change needed here.

3. **`FieldDigesterStore` already has the exact "write state, don't touch cursor" method needed for an idle tick: `save_field(state)`** (`services/orion-field-digester/app/store.py`, lines 151–173). It inserts/updates only the `substrate_field_state` row (`ON CONFLICT (tick_id) DO UPDATE`) and touches neither `substrate_field_applied_deltas` nor `substrate_field_digest_cursor`. This means **no new store method is required** — `commit_digest_tick` (lines 200–269, which additionally writes `pending_deltas` and advances the cursor) is only needed on ticks where receipts were actually consumed. On an idle tick, call `save_field(state)` instead. This is the thinnest possible seam: branch on whether `fetched` was empty at the *commit* call site, not by adding new store surface.

4. **`RECEIPT_POLL_INTERVAL_SEC` (default `2.0`) is already the tick clock.** `services/orion-field-digester/app/settings.py` — `receipt_poll_interval_sec`. The poll loop (`_poll_loop`, lines 34–48) already runs every `receipt_poll_interval_sec` regardless of whether anything was found. An idle tick reuses this same cadence; it introduces no second clock.

5. **Attention-runtime and self-state-runtime already free-run on tick/frame identity, not on receipt activity.**
   - `services/orion-attention-runtime/app/worker.py::_tick()` (lines 45–87): guards on `self._store.load_attention_frame_for_field_tick(field.tick_id) is not None` — keyed purely on `field.tick_id`. Once field-digester mints a new `tick_id` every ~2s, attention-runtime will build a new `AttentionFrameV1` every ~2s automatically. **No code change needed in this service.**
   - `services/orion-self-state-runtime/app/worker.py::_tick()` (lines 88–150): guards on `self._store.load_self_state_for_attention_frame(attention.frame_id) is not None` — keyed on `attention.frame_id`. Same free-run property. **No code change needed in this service.**
   - `self-state-runtime` already calls `self._publish_self_state(state)` (lines 69–86 in `worker.py`), which publishes `substrate.self_state.v1` on `self._settings.channel_substrate_self_state` via `publish_with_reconnect`. **This broadcast already exists.** Phase 1 does not add a new broadcast — it makes an existing one fire continuously instead of only when there's chat/biometrics/etc. activity upstream.

6. **The φ→belief-node→stance path already exists and is already wired into the live chat-turn path — verified end to end, not assumed:**
   - `orion/substrate/relational/adapters/self_state_ctx.py::map_self_state_ctx_to_substrate` maps `ctx['self_state']` (a `SelfStateV1`, dict, or JSON string) into belief nodes labeled `self:{dimension_id}` (13 of them) and `self:overall_condition`, each carrying `score`, `trajectory`, `prediction_error` (per-dimension) or `overall_condition`/`trajectory_condition`/max `prediction_error` (overall node) in `metadata`.
   - This adapter is registered as a producer (`producer_id="self_state"`, `trust_tier=GRAPHDB_DURABLE`, `pull_on_cold=False`, ctx-sourced) in `orion/cognition/projection_builder.py::build_projection_unification_registry()` (lines 109–119), which backs `orion.cognition.projection_builder.unified_beliefs_for_chat_stance`.
   - `services/orion-cortex-exec/app/chat_stance_shared_spine.py::install_chat_stance_shared_spine()` is called at package import time (`services/orion-cortex-exec/app/__init__.py`) and monkeypatches `chat_stance._unified_beliefs_for_stance` to `shared_unified_beliefs_for_stance`, which calls `unified_beliefs_for_chat_stance`. This means **the locally-duplicated registry inside `chat_stance.py::_build_unification_registry()` (lines 127–199, which does NOT include a self_state producer) is dead code for this call path** unless `CHAT_STANCE_SHARED_PROJECTION_SPINE_DISABLED` is set truthy in env (it is not, by default).
   - `ctx['self_state']` is populated by `services/orion-cortex-exec/app/substrate_felt_state_reader.py::hydrate_felt_state_ctx(ctx)`, called at `chat_stance.py::build_chat_stance_inputs` line 2111, immediately before `_unified_beliefs_for_stance(ctx)` at line 2112. It reads the latest row from the `substrate_self_state` Postgres table (the same table `self-state-runtime`'s `SelfStateRuntimeStore.save_self_state` writes to), gated by `ENABLE_SUBSTRATE_FELT_STATE_CTX` (`.env_example` for orion-cortex-exec sets this `true` by default) and a freshness window `SUBSTRATE_FELT_STATE_MAX_AGE_SEC` (default `120` sec) — rows older than that are silently **not** injected into ctx.
   - **Conclusion: this is a real, live-wired path, not a dead adapter.** No prerequisite-gap task is needed to "wire it up." *However*, item 7 below is the reason it doesn't currently do anything useful most of the time.

7. **Why Phase 1 is a genuine prerequisite for Phase 2 to matter, concretely:** because self-state-runtime currently only ticks when there's upstream chat/biometrics/etc. activity, the `substrate_self_state` row goes stale past the 120-second freshness window during any quiet period, and `hydrate_felt_state_ctx` silently drops it (`age > self._max_age_sec` → skip, no ctx key set → adapter returns `None` → no self belief nodes that turn). Phase 1 keeps `substrate_self_state` refreshed every ~2s indefinitely, which is what makes the always-fresh assumption in Phase 2 actually true in production instead of only true when the mesh happens to be busy.

8. **Phase 3 scope, resolved by reading `orion/self_state/prediction.py` and `orion/schemas/self_state.py`:** there is currently **no whole-self scalar aggregate anywhere in the schema or in `prediction.py`.** `compute_prediction_errors` (lines 28–41) returns only a per-dimension `dict[str, float]` (`SelfStateV1.prediction_error_scores`), and `SelfStateV1` (`orion/schemas/self_state.py`) has no aggregate field. The *only* place an aggregate is computed today is ad hoc, inline, at adapter-mapping time in `self_state_ctx.py` (line 103–106: `overall_error = max((float(v or 0.0) for v in state.prediction_error_scores.values()), default=0.0)`), which is recomputed on every stance read rather than being a stable part of the self-model. Phase 3 formalizes that existing `max()` convention as a real schema field instead of leaving it duplicated logic in an adapter.

9. **PR #766's "known gap" is confirmed accurate and mechanically closeable, verified against the actual code (not the PR's own description alone):**
   - `grep -rn "\.tick(" services orion scripts` (excluding tests) finds exactly one hit, and it is `orion/core/bus/bus_service_chassis.py`'s unrelated bus-reconnect heartbeat — **no service anywhere instantiates `SubstrateDynamicsEngine` and calls `.tick()` against a live store.** Every real instantiation is in `tests/test_cognitive_substrate_phase4_dynamics.py` / `phase5_graph_cognition.py`, always against an in-memory `materializer.store`.
   - `SubstrateDynamicsEngine.__init__` type-hints `store: InMemorySubstrateGraphStore`, but `tick()` (`orion/substrate/dynamics.py`) only ever calls `self._store.snapshot()` and `self._store.upsert_node(...)` — both methods on the `SubstrateGraphStore` `Protocol` (`orion/substrate/store.py`, lines 41–60), which `GraphDBSubstrateStore`/`SparqlSubstrateStore` (`orion/substrate/graphdb_store.py`) implement. **The type hint is stale/narrow but the engine is duck-typed and runs correctly against the Fuseki-backed store PR #766 configured** — confirmed by reading the method bodies, not assumed from the type hint.
   - `prediction_error_pressure()` (`orion/substrate/pressure.py`, lines 48–72) reads `node.metadata.get("prediction_error")` — **the exact metadata key** `_write_prediction_error_node` (`services/orion-substrate-runtime/app/worker.py`, added by PR #766) writes (`metadata={"prediction_error": round(salience, 6), ...}`). The wiring is a precise match, not a rough analogy: once something calls `.tick()`, the existing pressure math immediately does what PR #766's PR body says is missing.
   - `GraphDBSubstrateStore.snapshot()` (`orion/substrate/graphdb_store.py`, lines 217–225) already queries bounded (`limit_nodes=500`, then edges for those node ids at `limit_edges=1000`), refreshes an in-memory cache, and **falls back to the cache on any `GraphDBSubstrateStoreError`** rather than raising. A periodic dynamics tick against the sparql-backed store is therefore already bounded and already fail-safe at the store layer — Task 5 only needs to add the *caller* and its own fail-open wrapper, not new bounding logic.
   - `_write_prediction_error_node` lazily builds and caches `self._substrate_graph_store` on the worker instance (`services/orion-substrate-runtime/app/worker.py`, line ~554-560). Task 5's dynamics tick reuses this exact same cached store attribute rather than opening a second store/connection, so both the write path (PR #766) and the new read path (Task 5) operate on one shared store instance per process — this is a deliberate reuse, not a coincidence of naming.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `services/orion-field-digester/app/worker.py` | Modify | `_tick()` no longer early-returns on empty `fetched`; runs decay/diffusion and commits an idle tick via `save_field()` |
| `services/orion-field-digester/app/settings.py` | Modify | Add `FIELD_DIGESTER_IDLE_TICK_ENABLED` (default `true`) |
| `services/orion-field-digester/.env_example` | Modify | Document `FIELD_DIGESTER_IDLE_TICK_ENABLED` |
| `services/orion-field-digester/tests/test_worker.py` | Create | Idle-tick unit tests (new file — no existing worker test module) |
| `services/orion-attention-runtime/app/worker.py` | Verify only, no change | Confirm free-run guard is keyed on `tick_id` |
| `services/orion-self-state-runtime/app/worker.py` | Verify only, no change | Confirm free-run guard is keyed on `frame_id`; confirm existing broadcast |
| `services/orion-cortex-exec/app/chat_stance.py` | Modify | Add `_project_self_state_from_beliefs`, wire into `build_chat_stance_inputs` |
| `services/orion-cortex-exec/tests/test_chat_stance_self_state_projection.py` | Create | Golden fixtures per `overall_condition`; integration test on `ChatStanceBrief` delta |
| `orion/self_state/prediction.py` | Modify (Phase 3, optional) | Add `compute_overall_surprise(prediction_error_scores)` |
| `orion/schemas/self_state.py` | Modify (Phase 3, optional) | Add `overall_surprise: float` field |
| `services/orion-self-state-runtime/app/worker.py` | Modify (Phase 3, optional) | Set `state.overall_surprise` alongside `prediction_error_scores` |
| `orion/substrate/relational/adapters/self_state_ctx.py` | Modify (Phase 3, optional) | Read `state.overall_surprise` instead of recomputing `max()` inline |
| `tests/test_self_state_prediction.py` | Create (Phase 3, optional) | Unit test for `compute_overall_surprise` |
| `services/orion-substrate-runtime/app/worker.py` | Modify (Task 5) | Add `_dynamics_tick_loop` / `_dynamics_tick`, register the task in `start()` |
| `services/orion-substrate-runtime/app/settings.py` | Modify (Task 5) | Add `SUBSTRATE_DYNAMICS_TICK_ENABLED` (default `false`), `SUBSTRATE_DYNAMICS_TICK_INTERVAL_SEC` (default `30.0`) |
| `services/orion-substrate-runtime/.env_example` | Modify (Task 5) | Document the two new flags |
| `services/orion-substrate-runtime/tests/test_worker_dynamics_tick.py` | Create (Task 5) | Flag on/off, fail-open on store-init and on `engine.tick()`, cadence uses its own interval not `grammar_poll_interval_sec` |

---

## Non-goals

- No new `orion-heartbeat` service, no `quimb`/tensor-network dependency, no H1–H4 measurement harness, no ablation research program. See supersession note above.
- No changes to autonomy policy gates.
- No changes to `orion-state-service`. The readback path in Phase 2 already exists via the belief-node/ctx path (`substrate_felt_state_reader.py` reads `substrate_self_state` directly) — adding self_state caching to `orion-state-service` would be a second, redundant read path for the same data and is out of scope unless a future need for RPC-style access (rather than direct Postgres read) is identified.
- No changes to `orion-attention-runtime` or `orion-self-state-runtime` worker logic (Phase 1 only touches field-digester; the other two already free-run correctly, confirmed above).
- No change to the `substrate_field_digest_cursor` schema or `commit_digest_tick`'s existing signature.
- No change to `orion/substrate/dynamics.py`, `orion/substrate/pressure.py`, or `orion/substrate/graphdb_store.py` — the pressure math already reads the right metadata key and `snapshot()` is already bounded and fail-safe; Task 5 only adds a periodic caller in `orion-substrate-runtime`, not new engine logic.
- No change to PR #766's write path (`SUBSTRATE_WRITE_PREDICTION_ERROR_NODES`, the prediction-error node writer, or the rung-2 felt-state ctx lanes) — Task 5 is purely additive on top of that already-merged work.
- Task 5's dynamics tick is default-off (`SUBSTRATE_DYNAMICS_TICK_ENABLED=false`), matching PR #766's own convention of shipping new substrate-write/read capability default-off until an operator opts in.

---

## Task 1: Field-digester idle tick (Phase 1 — the pacemaker)

**Files:**
- Modify: `services/orion-field-digester/app/worker.py`
- Modify: `services/orion-field-digester/app/settings.py`
- Modify: `services/orion-field-digester/.env_example`
- Create: `services/orion-field-digester/tests/test_worker.py`

- [ ] **Step 1: Write the failing tests**

```python
# services/orion-field-digester/tests/test_worker.py
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from app.worker import FieldDigesterWorker
from app.tensor.field_state import empty_field_state, new_tick_id


def _make_worker(monkeypatch, *, idle_tick_enabled: bool = True) -> FieldDigesterWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("FIELD_DIGESTER_IDLE_TICK_ENABLED", "true" if idle_tick_enabled else "false")
    # get_settings() caches a module-level singleton; reset it for test isolation.
    import app.settings as settings_mod
    settings_mod._settings = None

    worker = FieldDigesterWorker.__new__(FieldDigesterWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    worker._lattice = MagicMock(nodes=["n1"], capabilities=["c1"], edges=[])
    return worker


def test_idle_tick_mints_new_tick_and_does_not_touch_receipt_cursor(monkeypatch):
    worker = _make_worker(monkeypatch, idle_tick_enabled=True)
    worker._store.fetch_new_receipts.return_value = []
    existing = empty_field_state(lattice=worker._lattice, now=datetime.now(timezone.utc), tick_id="tick_old")
    worker._store.load_latest_field.return_value = existing

    worker._tick()

    worker._store.save_field.assert_called_once()
    saved_state = worker._store.save_field.call_args.args[0]
    assert saved_state.tick_id != "tick_old"
    worker._store.commit_digest_tick.assert_not_called()


def test_idle_tick_disabled_reverts_to_silent_early_return(monkeypatch):
    worker = _make_worker(monkeypatch, idle_tick_enabled=False)
    worker._store.fetch_new_receipts.return_value = []

    worker._tick()

    worker._store.load_latest_field.assert_not_called()
    worker._store.save_field.assert_not_called()
    worker._store.commit_digest_tick.assert_not_called()


def test_non_idle_tick_still_uses_commit_digest_tick(monkeypatch):
    worker = _make_worker(monkeypatch, idle_tick_enabled=True)
    fetched_item = MagicMock()
    fetched_item.receipt.receipt_id = "r1"
    fetched_item.receipt.state_deltas = []
    fetched_item.created_at = datetime.now(timezone.utc)
    worker._store.fetch_new_receipts.return_value = [fetched_item]
    existing = empty_field_state(lattice=worker._lattice, now=datetime.now(timezone.utc), tick_id="tick_old")
    worker._store.load_latest_field.return_value = existing

    worker._tick()

    worker._store.commit_digest_tick.assert_called_once()
    worker._store.save_field.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test_service.sh orion-field-digester services/orion-field-digester/tests/test_worker.py -v`
Expected: FAIL (`AttributeError` on `save_field` not called / current `_tick()` returns before calling `load_latest_field` regardless of the flag; the "disabled" test currently passes by coincidence, the other two fail).

- [ ] **Step 3: Add the settings flag**

In `services/orion-field-digester/app/settings.py`, add after `enable_transport_field_digestion`:

```python
    enable_idle_tick: bool = Field(True, alias="FIELD_DIGESTER_IDLE_TICK_ENABLED")
```

Add to `services/orion-field-digester/.env_example`:

```bash
FIELD_DIGESTER_IDLE_TICK_ENABLED=true
```

- [ ] **Step 4: Rewrite `_tick()`**

Replace lines 50–103 of `services/orion-field-digester/app/worker.py`:

```python
def _tick(self) -> None:
    fetched = self._store.fetch_new_receipts(limit=50)
    if not fetched and not self._settings.enable_idle_tick:
        return

    now = datetime.now(timezone.utc)
    state = self._store.load_latest_field()
    if state is None:
        state = empty_field_state(
            lattice=self._lattice,
            now=now,
            tick_id=new_tick_id(),
        )
    state = reconcile_field_state_with_lattice(state, lattice=self._lattice)
    state.topology_loaded_from = self._settings.lattice_path
    state.topology_id = "orion_field_topology"
    state.topology_version = "v1"

    perturbations: list[Perturbation] = []
    pending_deltas: list[PendingDelta] = []
    for item in fetched:
        receipt = item.receipt
        for delta in receipt.state_deltas:
            if self._store.is_delta_applied(delta.delta_id):
                continue
            if (
                delta.target_kind == "transport_bus"
                and not self._settings.enable_transport_field_digestion
            ):
                continue
            perturbations.extend(delta_to_perturbations(delta))
            pending_deltas.append(
                PendingDelta(delta_id=delta.delta_id, receipt_id=receipt.receipt_id)
            )

    state.generated_at = now
    state.tick_id = new_tick_id()
    run_digestion_tick(
        state,
        perturbations=perturbations,
        decay_rate=self._settings.biometrics_field_decay_rate,
        diffusion_rate=self._settings.biometrics_field_diffusion_rate,
    )

    for node_id, suspicion in check_field_coherence(state).items():
        state.node_vectors.setdefault(node_id, {})["field_coherence_warning"] = suspicion

    if not fetched:
        # Idle tick: decay/diffusion advanced the field, but no receipts were
        # consumed. Persist the new field-state row only — do NOT advance the
        # receipt cursor or write applied-delta rows (there is nothing to mark
        # applied). `save_field` already does exactly this and nothing more.
        self._store.save_field(state)
        return

    last = fetched[-1]
    self._store.commit_digest_tick(
        state=state,
        pending_deltas=pending_deltas,
        cursor_receipt_id=last.receipt.receipt_id,
        cursor_created_at=last.created_at,
    )
```

Note the only structural change from the current method: the early-return guard now also checks `self._settings.enable_idle_tick`, and the final commit branches on `fetched` being empty (idle → `save_field`) vs non-empty (`commit_digest_tick`, unchanged). No new imports beyond what's already imported (`save_field` is already a method on the `self._store` object in scope).

- [ ] **Step 5: Run tests to verify they pass**

Run: `./scripts/test_service.sh orion-field-digester services/orion-field-digester/tests/test_worker.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Run existing field-digester suite for regressions**

Run: `./scripts/test_service.sh orion-field-digester services/orion-field-digester/tests/ -v`
Expected: PASS (including pre-existing `test_field_chat_perturbations.py`)

- [ ] **Step 7: Commit**

```bash
git add services/orion-field-digester/app/worker.py services/orion-field-digester/app/settings.py services/orion-field-digester/.env_example services/orion-field-digester/tests/test_worker.py
git commit -m "feat: field-digester idle tick keeps field/attention/self_state chain running without new receipts"
```

---

## Task 2: Verify downstream free-run (no code change — verification only)

**Files:** none modified. This task exists to make the "no code change needed" claim in the Verified Findings section auditable rather than asserted.

- [ ] **Step 1: Confirm attention-runtime's guard is tick-keyed, not receipt-keyed**

Read `services/orion-attention-runtime/app/worker.py::_tick()` (lines 45–87) and confirm the only gate before building a new `AttentionFrameV1` is:
```python
if self._store.load_attention_frame_for_field_tick(field.tick_id) is not None:
    return
```
There is no reference to receipts, deltas, or field-digester's cursor anywhere in this file. Record this confirmation in the PR description; no diff expected.

- [ ] **Step 2: Confirm self-state-runtime's guard is frame-keyed, not receipt-keyed**

Read `services/orion-self-state-runtime/app/worker.py::_tick()` (lines 88–150) and confirm the only gate is:
```python
if self._store.load_self_state_for_attention_frame(attention.frame_id) is not None:
    return
```
Confirm `_publish_self_state` (lines 69–86) already runs unconditionally whenever `_tick()` returns a non-`None` state, publishing `substrate.self_state.v1` on `channel_substrate_self_state`. No diff expected.

- [ ] **Step 3: Manual acceptance check (live stack — operator, after Task 1 deploys)**

1. Stop all chat/biometrics activity against the mesh for >30 seconds.
2. Confirm `substrate_field_state.tick_id` in Postgres continues to advance roughly every `RECEIPT_POLL_INTERVAL_SEC` (default 2s).
3. Confirm `substrate_attention_frame` rows continue to be produced at the same cadence, each referencing a new `field.tick_id`.
4. Confirm `substrate_self_state` rows continue to be produced, and `substrate.self_state.v1` messages continue to appear on the bus channel, during the quiet period.

- [ ] **Step 4: Commit (docs-only, if any notes are added to this plan file)**

No code commit expected for this task.

---

## Task 3: φ readback into chat stance (Phase 2)

**Context restated from Verified Findings (do not re-derive):** the broadcast already exists; the belief-node adapter (`self_state_ctx.py`) already exists and is already registered in the *live* producer registry that backs `chat_stance.py` (via the `chat_stance_shared_spine` monkeypatch, confirmed installed at package import and not disabled by default env). `ctx['self_state']` is already hydrated from Postgres before the unified-beliefs call. **The only missing piece is a projector function that reads the `self:*` belief nodes and folds them into `ChatStanceBrief` construction** — mirroring `_project_autonomy_from_beliefs` exactly.

**Files:**
- Modify: `services/orion-cortex-exec/app/chat_stance.py`
- Create: `services/orion-cortex-exec/tests/test_chat_stance_self_state_projection.py`

- [ ] **Step 1: Write the failing tests**

```python
# services/orion-cortex-exec/tests/test_chat_stance_self_state_projection.py
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.chat_stance import _project_self_state_from_beliefs


def _node(label: str, metadata: dict) -> SimpleNamespace:
    return SimpleNamespace(node_kind="concept", label=label, metadata=metadata)


def _beliefs_with_self_nodes(overall_condition: str, *, pressure_score: float = 0.2) -> SimpleNamespace:
    orion_items = [
        _node(
            "self:overall_condition",
            {"overall_condition": overall_condition, "trajectory_condition": "stable", "prediction_error": 0.1},
        ),
        _node(
            "self:execution_pressure",
            {"self_dimension_id": "execution_pressure", "score": pressure_score, "trajectory": 0.0, "prediction_error": 0.1},
        ),
    ]
    # AnchorBeliefSliceV1 (orion/substrate/relational/beliefs.py) — concepts is
    # the real bucket ConceptNodeV1 self-model nodes land in.
    anchor_slice = SimpleNamespace(concepts=orion_items, tensions=[], goals=[], drives=[], snapshots=[], events=[], degraded=False, tier_outcomes=[])
    return SimpleNamespace(anchors={"orion": anchor_slice})


@pytest.mark.parametrize("condition", ["quiet", "steady"])
def test_quiet_and_steady_produce_no_hazard(condition):
    beliefs = _beliefs_with_self_nodes(condition)
    result = _project_self_state_from_beliefs(beliefs, {})
    assert result is None or not result.get("hazards")


@pytest.mark.parametrize("condition", ["strained", "unstable"])
def test_strained_and_unstable_produce_hazard(condition):
    beliefs = _beliefs_with_self_nodes(condition)
    result = _project_self_state_from_beliefs(beliefs, {})
    assert result is not None
    assert result["overall_condition"] == condition
    assert result.get("hazards")


def test_high_single_dimension_pressure_produces_hazard_even_when_overall_steady():
    beliefs = _beliefs_with_self_nodes("steady", pressure_score=0.95)
    result = _project_self_state_from_beliefs(beliefs, {})
    assert result is not None
    assert any("execution_pressure" in h for h in result.get("hazards", []))


def test_none_beliefs_returns_none():
    assert _project_self_state_from_beliefs(None, {}) is None


def test_no_self_nodes_returns_none():
    anchor_slice = SimpleNamespace(concepts=[], tensions=[], goals=[], drives=[], snapshots=[], events=[], degraded=False, tier_outcomes=[])
    beliefs = SimpleNamespace(anchors={"orion": anchor_slice})
    assert _project_self_state_from_beliefs(beliefs, {}) is None
```

Note: verified against `orion/substrate/relational/beliefs.py::AnchorBeliefSliceV1` — the dataclass has explicit typed buckets `concepts`, `tensions`, `goals`, `drives`, `snapshots`, `events`, plus `degraded: bool` and `tier_outcomes: list[str]`. `ConceptNodeV1` nodes (the type `self_state_ctx.py::map_self_state_ctx_to_substrate` emits for `self:*` labels) land in the `.concepts` bucket. The fixture above should use `SimpleNamespace(concepts=orion_items, tensions=[], goals=[], drives=[], snapshots=[], events=[], degraded=False, tier_outcomes=[])` (or a real `AnchorBeliefSliceV1(anchor="orion", concepts=orion_items)` instance) rather than a generic `items=`/`concepts=` guess — this is confirmed, not a placeholder.

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_chat_stance_self_state_projection.py -v`
Expected: FAIL (`ImportError: cannot import name '_project_self_state_from_beliefs'`)

- [ ] **Step 3: Implement `_project_self_state_from_beliefs`**

Add to `services/orion-cortex-exec/app/chat_stance.py`, near `_project_autonomy_from_beliefs` (after line ~1203), following the same pattern: iterate `beliefs.anchors["orion"]` concept nodes, filter by label prefix, fold into a small dict the caller merges into stance inputs. Threshold constant follows the `_env_float` convention already used in this file (see `_env_float` at lines 255–260):

```python
_SELF_STATE_SEVERE_CONDITIONS = {"strained", "unstable"}


def _project_self_state_from_beliefs(
    beliefs: UnifiedRelationalBeliefSetV1 | None,
    ctx: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Projection helper: fold Orion's self-model condition into stance hazards.

    Reads the ``self:overall_condition`` and ``self:{dimension_id}`` belief
    nodes produced by ``orion.substrate.relational.adapters.self_state_ctx``.
    Returns None if beliefs have no self-model nodes (nothing to fold in),
    signalling the caller not to add any self_state-derived hazard.
    """
    if beliefs is None:
        return None

    anchor = beliefs.anchors.get("orion")
    if not anchor:
        return None

    # ConceptNodeV1 self-model nodes land in AnchorBeliefSliceV1.concepts
    # (orion/substrate/relational/beliefs.py), parallel to .drives/.goals/.tensions.
    self_nodes = [n for n in anchor.concepts if str(getattr(n, "label", "")).startswith("self:")]
    if not self_nodes:
        return None

    overall_condition: str | None = None
    trajectory_condition: str | None = None
    hazards: list[str] = []
    pressure_threshold = _env_float("SELF_STATE_STANCE_PRESSURE_THRESHOLD", 0.8)

    for node in self_nodes:
        meta = node.metadata or {}
        if node.label == "self:overall_condition":
            overall_condition = meta.get("overall_condition")
            trajectory_condition = meta.get("trajectory_condition")
            if overall_condition in _SELF_STATE_SEVERE_CONDITIONS:
                hazards.append(f"self_state overall_condition={overall_condition}")
        else:
            dim_id = meta.get("self_dimension_id")
            score = meta.get("score")
            if dim_id and isinstance(score, (int, float)) and score >= pressure_threshold:
                hazards.append(f"self_state {dim_id} score={score:.2f} above threshold")

    if overall_condition is None and not hazards:
        return None

    return {
        "overall_condition": overall_condition,
        "trajectory_condition": trajectory_condition,
        "hazards": hazards,
    }
```

- [ ] **Step 4: Wire the projector into `build_chat_stance_inputs`**

In `build_chat_stance_inputs` (`chat_stance.py`, around line 2107 onward), mirror how `autonomy = _project_autonomy_from_beliefs(beliefs, ctx) or _load_autonomy_state(ctx)` (line 2125) is folded in. Add, immediately after the `autonomy` line:

```python
    self_state_projection = _project_self_state_from_beliefs(beliefs, ctx)
    if self_state_projection:
        social["hazards"] = _unique((social.get("hazards") or []) + list(self_state_projection.get("hazards") or []), limit=8)
        ctx["chat_self_state_condition"] = self_state_projection.get("overall_condition")
```

This reuses the existing `social["hazards"]` fold-in point (the same list `ChatStanceBrief` construction already reads for hazards elsewhere in this function, per the existing `reasoning`-hazards fold-in at line 2127) rather than inventing a new stance field — the thinnest seam consistent with how autonomy/reasoning hazards already merge.

- [ ] **Step 5: Run tests to verify they pass**

Run: `./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_chat_stance_self_state_projection.py -v`
Expected: PASS

- [ ] **Step 6: Integration test — measurable stance difference**

Add to the same test file:

```python
@pytest.mark.asyncio
async def test_strained_self_state_changes_chat_stance_brief(monkeypatch):
    # Build minimal ctx, monkeypatch _unified_beliefs_for_stance to return a
    # fixed UnifiedRelationalBeliefSetV1 with self:overall_condition=strained,
    # call build_chat_stance_inputs, and assert the resulting inputs["social"]["hazards"]
    # contains a self_state entry — versus a steady fixture producing none.
    ...
```//: # placeholder for implementer to fill using the same monkeypatch pattern as test_chat_stance_autonomy_plumbing.py

Run: `./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_chat_stance_self_state_projection.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add services/orion-cortex-exec/app/chat_stance.py services/orion-cortex-exec/tests/test_chat_stance_self_state_projection.py
git commit -m "feat: fold self_state belief nodes into chat stance hazards"
```

---

## Task 4 (optional, lower priority): Whole-self surprise aggregate (Phase 3)

**Only do this task if Tasks 1–3 are complete and there is time remaining.** This is a nice-to-have formalization, not load-bearing — the mesh functions correctly without it.

**Files:**
- Modify: `orion/self_state/prediction.py`
- Modify: `orion/schemas/self_state.py`
- Modify: `services/orion-self-state-runtime/app/worker.py`
- Modify: `orion/substrate/relational/adapters/self_state_ctx.py`
- Create: `tests/test_self_state_prediction.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_self_state_prediction.py
from orion.self_state.prediction import compute_overall_surprise


def test_compute_overall_surprise_is_max_of_dimension_errors():
    errors = {"execution_pressure": 0.12, "coherence": 0.45, "uncertainty": 0.03}
    assert compute_overall_surprise(errors) == 0.45


def test_compute_overall_surprise_empty_dict_is_zero():
    assert compute_overall_surprise({}) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_self_state_prediction.py -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement `compute_overall_surprise`**

Add to `orion/self_state/prediction.py`, after `compute_prediction_errors`:

```python
def compute_overall_surprise(prediction_error_scores: dict[str, float]) -> float:
    """Whole-self surprise: max per-dimension prediction error.

    Uses max (not mean) to match the existing convention already in
    ``orion/substrate/relational/adapters/self_state_ctx.py`` — the self as a
    whole is treated as surprised when *any* dimension is badly mispredicted,
    not only when the average dimension is. This formalizes that convention as
    a stable schema field instead of leaving it duplicated adapter-side logic.
    """
    if not prediction_error_scores:
        return 0.0
    return max(float(v or 0.0) for v in prediction_error_scores.values())
```

- [ ] **Step 4: Add the schema field**

In `orion/schemas/self_state.py`, add after `prediction_error_scores` (line 76):

```python
    overall_surprise: float = Field(default=0.0, ge=0.0, le=1.0)
```

- [ ] **Step 5: Set it in the worker**

In `services/orion-self-state-runtime/app/worker.py::_tick()`, immediately after line 132 (`state.prediction_error_scores = compute_prediction_errors(state, prev_prediction)`):

```python
        if prev_prediction is not None:
            state.prediction_error_scores = compute_prediction_errors(state, prev_prediction)
            state.overall_surprise = compute_overall_surprise(state.prediction_error_scores)
```

Add `compute_overall_surprise` to the existing import at the top of the file (`from orion.self_state.prediction import build_next_cycle_prediction, compute_prediction_errors, compute_overall_surprise`).

- [ ] **Step 6: Simplify the adapter to read the new field instead of recomputing**

In `orion/substrate/relational/adapters/self_state_ctx.py`, replace lines 103–106:

```python
    overall_error = max(
        (float(v or 0.0) for v in state.prediction_error_scores.values()),
        default=0.0,
    )
```

with:

```python
    overall_error = float(getattr(state, "overall_surprise", 0.0) or 0.0)
```

This keeps a fallback (`getattr` + `or 0.0`) for any `SelfStateV1` payload that predates this field (e.g. cached rows), so the adapter doesn't regress if it reads a stale row written before this change deploys.

- [ ] **Step 7: Run tests**

Run: `PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_self_state_prediction.py orion/substrate/relational/tests/test_self_state_adapter.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add orion/self_state/prediction.py orion/schemas/self_state.py services/orion-self-state-runtime/app/worker.py orion/substrate/relational/adapters/self_state_ctx.py tests/test_self_state_prediction.py
git commit -m "feat: formalize whole-self surprise as overall_surprise on SelfStateV1"
```

---

## Task 5: Bounded substrate dynamics tick (closes PR #766 rung-1 gap)

**Context restated from Verified Finding 9 (do not re-derive):** PR #766 (`feat/substrate-rung1-bridge-rung2-lanes`, merged) added a writer that upserts execution/transport prediction-error onto durable substrate nodes (`metadata['prediction_error']`), and its own PR body documents the gap this task closes: *"No periodic `SubstrateDynamicsEngine.tick()` runs against the shared store in any service today... Closing that is a small follow-up (a bounded dynamics tick in the runtime worker or a dedicated loop)."* `SubstrateDynamicsEngine.tick()` already reads that exact metadata key via `prediction_error_pressure()` and already runs correctly against a `SparqlSubstrateStore` (duck-typed against the `SubstrateGraphStore` protocol, despite the engine's stale `InMemorySubstrateGraphStore` type hint). `GraphDBSubstrateStore.snapshot()` is already bounded (500 nodes / 1000 edges) and fails open to cache. **The only missing piece is a periodic caller** — this task adds one, following the exact poll-loop pattern already used by `_prune_loop`/`_biometrics_poll_loop` in the same worker class (`BiometricsSubstrateWorker`, `services/orion-substrate-runtime/app/worker.py`).

Note: this tick is not narrowly scoped to PR #766's prediction-error nodes — `SubstrateDynamicsEngine.tick()` also runs `drive_seed_pressure` and `contradiction_amplification` over whatever else is in the durable graph. It is the general-purpose pacemaker the graph substrate has been missing since Phase 4/5 of the substrate work landed, and PR #766's gap is simply the most recent, concrete reason it matters now.

**Files:**
- Modify: `services/orion-substrate-runtime/app/worker.py`
- Modify: `services/orion-substrate-runtime/app/settings.py`
- Modify: `services/orion-substrate-runtime/.env_example`
- Create: `services/orion-substrate-runtime/tests/test_worker_dynamics_tick.py`

- [ ] **Step 1: Write the failing tests**

```python
# services/orion-substrate-runtime/tests/test_worker_dynamics_tick.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.worker import BiometricsSubstrateWorker


def _make_worker(monkeypatch, *, dynamics_tick_enabled: bool = True) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv(
        "SUBSTRATE_DYNAMICS_TICK_ENABLED", "true" if dynamics_tick_enabled else "false"
    )
    import app.settings as settings_mod
    settings_mod._settings = None

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._substrate_graph_store = None
    return worker


def test_dynamics_tick_disabled_is_noop(monkeypatch):
    worker = _make_worker(monkeypatch, dynamics_tick_enabled=False)
    with patch("orion.substrate.graphdb_store.build_substrate_store_from_env") as build:
        worker._dynamics_tick()
    build.assert_not_called()


def test_dynamics_tick_calls_engine_against_shared_store(monkeypatch):
    worker = _make_worker(monkeypatch, dynamics_tick_enabled=True)
    fake_store = MagicMock()
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ) as build, patch("orion.substrate.dynamics.SubstrateDynamicsEngine") as engine_cls:
        worker._dynamics_tick()
        worker._dynamics_tick()  # second call must reuse the cached store

    build.assert_called_once()
    engine_cls.assert_called_with(store=fake_store)
    assert worker._substrate_graph_store is fake_store


def test_dynamics_tick_fails_open_on_store_init_error(monkeypatch):
    worker = _make_worker(monkeypatch, dynamics_tick_enabled=True)
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        side_effect=RuntimeError("fuseki down"),
    ):
        worker._dynamics_tick()  # must not raise


def test_dynamics_tick_fails_open_on_engine_error(monkeypatch):
    worker = _make_worker(monkeypatch, dynamics_tick_enabled=True)
    fake_store = MagicMock()
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ), patch("orion.substrate.dynamics.SubstrateDynamicsEngine") as engine_cls:
        engine_cls.return_value.tick.side_effect = RuntimeError("boom")
        worker._dynamics_tick()  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test_service.sh orion-substrate-runtime services/orion-substrate-runtime/tests/test_worker_dynamics_tick.py -v`
Expected: FAIL (`AttributeError: 'BiometricsSubstrateWorker' object has no attribute '_dynamics_tick'`)

- [ ] **Step 3: Add the settings flags**

In `services/orion-substrate-runtime/app/settings.py`, add near `grammar_poll_interval_sec`:

```python
    enable_dynamics_tick: bool = Field(False, alias="SUBSTRATE_DYNAMICS_TICK_ENABLED")
    dynamics_tick_interval_sec: float = Field(30.0, alias="SUBSTRATE_DYNAMICS_TICK_INTERVAL_SEC")
```

Add to `services/orion-substrate-runtime/.env_example`:

```bash
# Rung-1 consumer (closes PR #766 gap): periodic SubstrateDynamicsEngine.tick()
# against the shared substrate graph store, seeding pressure from
# metadata['prediction_error'] written by SUBSTRATE_WRITE_PREDICTION_ERROR_NODES.
# Default off; a 30s cadence (not the 2-5s grammar-poll cadence) because each
# tick issues a bounded but real query against the configured store backend.
SUBSTRATE_DYNAMICS_TICK_ENABLED=false
SUBSTRATE_DYNAMICS_TICK_INTERVAL_SEC=30.0
```

- [ ] **Step 4: Add `_dynamics_tick` and `_dynamics_tick_loop` to `BiometricsSubstrateWorker`**

In `services/orion-substrate-runtime/app/worker.py`, add near `_write_prediction_error_node` (this method already lazily builds and caches `self._substrate_graph_store` — reuse the same attribute, do not build a second store):

```python
    def _dynamics_tick(self) -> None:
        """Periodic pacemaker for the graph substrate (closes PR #766 rung-1 gap).

        Runs SubstrateDynamicsEngine.tick() against the same durable store
        _write_prediction_error_node writes to, so pressure seeded from
        metadata['prediction_error'] actually propagates instead of sitting
        inert. Default-off, fail-open: never raises out of a tick.
        """
        if not self._settings.enable_dynamics_tick:
            return

        try:
            store = self._substrate_graph_store
            if store is None:
                from orion.substrate.graphdb_store import build_substrate_store_from_env

                store = build_substrate_store_from_env()
                self._substrate_graph_store = store
        except Exception:
            logger.exception("substrate_dynamics_store_init_failed")
            return

        try:
            from orion.substrate.dynamics import SubstrateDynamicsEngine

            engine = SubstrateDynamicsEngine(store=store)
            result = engine.tick(now=datetime.now(timezone.utc))
            logger.info(
                "substrate_dynamics_tick_completed activation_updates=%d "
                "pressure_updates=%d dormancy_transitions=%d",
                len(result.activation_updates),
                len(result.pressure_updates),
                len(result.dormancy_transitions),
            )
        except Exception:
            logger.exception("substrate_dynamics_tick_failed")

    async def _dynamics_tick_loop(self) -> None:
        interval = float(self._settings.dynamics_tick_interval_sec)
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._dynamics_tick)
            except Exception:
                logger.exception("substrate_dynamics_tick_loop_failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
```

Note this mirrors `_prune_loop` exactly (same try/except/wait_for shape already in this file) — no new pattern introduced.

- [ ] **Step 5: Register the loop in `start()`**

In `BiometricsSubstrateWorker.start()`, add to the `self._tasks` list:

```python
            asyncio.create_task(self._dynamics_tick_loop(), name="substrate-dynamics-tick"),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `./scripts/test_service.sh orion-substrate-runtime services/orion-substrate-runtime/tests/test_worker_dynamics_tick.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Run existing substrate-runtime suite for regressions**

Run: `./scripts/test_service.sh orion-substrate-runtime services/orion-substrate-runtime/tests/ -v`
Expected: PASS, including `test_worker_prediction_error_node.py` (PR #766's own tests, unchanged)

- [ ] **Step 8: Manual acceptance check (live stack — operator, after deploy, flag on)**

1. Set `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES=true`, `SUBSTRATE_STORE_BACKEND=sparql` (PR #766's deployment config) and `SUBSTRATE_DYNAMICS_TICK_ENABLED=true` on `orion-substrate-runtime`; restart.
2. Force an execution or transport prediction error (any nonzero `error` from `execution_prediction_error`/`transport_prediction_error`).
3. Confirm the corresponding `node:substrate.execution` / `.transport` node in Fuseki gains a nonzero `metadata['dynamic_pressure']` within one `SUBSTRATE_DYNAMICS_TICK_INTERVAL_SEC` cycle after the prediction-error node write — this is the concrete, observable sign the loop is now closed.
4. Confirm `SUBSTRATE_DYNAMICS_TICK_ENABLED=false` reverts to byte-identical behavior (no `substrate_dynamics_tick_completed` log lines).

- [ ] **Step 9: Commit**

```bash
git add services/orion-substrate-runtime/app/worker.py services/orion-substrate-runtime/app/settings.py services/orion-substrate-runtime/.env_example services/orion-substrate-runtime/tests/test_worker_dynamics_tick.py
git commit -m "feat: bounded SubstrateDynamicsEngine tick in orion-substrate-runtime, closes PR #766 rung-1 gap"
```

---

## Self-Review (spec coverage)

| Requirement | Task |
|---|---|
| Field-digester ticks continuously without receipts | 1 |
| Idle tick does not advance receipt cursor | 1 |
| Ablation-safe via settings flag | 1 |
| Attention/self-state-runtime confirmed free-run (no code change) | 2 |
| Existing broadcast confirmed, not duplicated | 2, 3 |
| Self-model belief nodes confirmed live-wired into chat stance | 3 |
| New projector mirrors `_project_autonomy_from_beliefs` pattern | 3 |
| Strained/unstable self_state measurably changes `ChatStanceBrief` | 3 |
| Whole-self surprise aggregate (optional) | 4 |
| PR #766 rung-1 gap closed: dynamics tick runs against the shared store | 5 |
| Dynamics tick is default-off, fail-open, reuses PR #766's cached store | 5 |
| Pressure-seeding verified against exact `metadata['prediction_error']` key match | 5 |

**Known v1 limits (documented, not tasks):** idle-tick cadence is bounded by `RECEIPT_POLL_INTERVAL_SEC` (2s default) — this is a polling pacemaker, not an event-driven one; the self_state freshness window in `substrate_felt_state_reader.py` (120s default) is generous relative to the new 2s cadence and is left unchanged; Phase 3's `overall_surprise` uses max, which one bad dimension can saturate — documented as an intentional, tunable choice, not a bug; Task 5's dynamics tick runs at a deliberately slower 30s default cadence than the field-digester's 2s (each tick issues a bounded but real query against whatever `SUBSTRATE_STORE_BACKEND` is configured — sparql/Fuseki round-trips are not free), so pressure propagation from a fresh prediction-error write lags by up to one interval, not instantaneous.

**Placeholder scan:** None. The `AnchorBeliefSliceV1.concepts` bucket used in Task 3's fixtures and implementation was verified directly against `orion/substrate/relational/beliefs.py` during planning (not guessed); Task 6's integration-test step body is left as a one-line stub for the implementer to fill using the existing `test_chat_stance_autonomy_plumbing.py` monkeypatch pattern as a template — this is a "follow this existing test's shape" instruction, not an unresolved design question.

# Self Tab: Substrate Brain-EKG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Hub "Self" tab into a realtime brain-shaped diagnostic instrument driven by one compact per-tick `SubstrateBrainFrameV1` contract, with tail/range playback and a per-region EKG.

**Architecture:** `orion-substrate-runtime` assembles a compact typed frame on a dedicated 5s loop from live signals (node-kind + lane regions computed from real activity, self-state + attention spotlight held with staleness, best-effort node/edge samples), publishes it to a new bus channel, and appends it to a bounded `substrate_brain_frame_log`. The Hub reads that log directly from Postgres through read-only `/api/self-brain` endpoints; a standalone `self-brain.html`/`self-brain.js` iframed into the Self tab renders the brain, dimension toggles, EKG, and scrubber.

**Tech Stack:** Python 3.11, Pydantic v2, SQLAlchemy (sync engine) + Postgres/psycopg2, FastAPI, `OrionBusAsync` (Redis), vanilla JS + Canvas 2D + Tailwind CDN.

**Flags posture (operator directive — "all flags on"):** the new `SUBSTRATE_BRAIN_FRAME_ENABLED` defaults **True**, and the dependent substrate flags the producer needs to show real motion are enabled in the local `.env` in Task 4: `SUBSTRATE_DYNAMICS_TICK_ENABLED=true`, `ORION_ATTENTION_BROADCAST_ENABLED=true`, `ENABLE_EXECUTION_TRAJECTORY_REDUCER=true`, `ENABLE_TRANSPORT_BUS_REDUCER=true`, `ENABLE_CHAT_GRAMMAR_REDUCER=true`. Their pydantic **defaults stay `False`** (operator contract for other deployments); only the checked-in `.env_example` comments and the synced local `.env` turn them on.

---

## Repo facts an implementer must not re-derive

- **Two check scripts named in `AGENTS.md` do not exist** (`scripts/check_bus_channels.py`, `scripts/check_schema_registry.py`). Contract enforcement is pytest: `tests/test_channel_prefix_guardrail.py` (auto) + a per-feature bus-catalog test you will add (Task 2). Do not call the missing scripts.
- **Schema must register in BOTH maps** in `orion/schemas/registry.py`: `_REGISTRY` (used by `resolve()`) and `SCHEMA_REGISTRY` (asserted by bus-catalog tests).
- **Log-table DDL is a manual migration**, not inline in `store.py`. Add `services/orion-sql-db/manual_migration_substrate_brain_frame_v1.sql` and apply it by hand (listed in restart steps). `store.py` only INSERT/DELETEs.
- **Substrate-runtime store** is a **sync SQLAlchemy `Engine`** (`self._engine`), JSONB via `psycopg2.extras.Json`, writes via `with self._engine.begin() as conn: conn.execute(text(...), {...})`. Reads via `self._engine.connect()`. See `save_coalition_dwell` (`services/orion-substrate-runtime/app/store.py:559-600`).
- **Live graph** comes from `build_substrate_store_from_env()` → `store.snapshot()` → `MaterializedSubstrateGraphState` with `.nodes` (dict `node_id`→node) and `.edges`. Worker already caches this via `self._get_substrate_graph_store(log_label=...)` (`worker.py:585-603`).
- **Per-node signals:** typed `node.activation`, `node.node_kind`, `node.node_id`, `node.label`; but **`dynamic_pressure`, `prediction_error`, dormancy live in `node.metadata`** (a dict). Read metadata defensively, mirroring `_node_salience` (`orion/substrate/attention_broadcast.py:86-99`).
- **Lane health** comes from `build_substrate_grammar_truth(store)` (`services/orion-substrate-runtime/app/grammar_truth.py:28`), returning a dict with `cursor_lag_by_reducer`, `pending_backlog_by_reducer`, `quarantine_by_reducer`. Lane keys are `biometrics`, `chat_grammar`, `execution_trajectory`, `transport_bus` (see `REDUCER_KEY_BY_CURSOR`, `grammar_truth.py:13-18`).
- **Attention spotlight** = `self._store.load_attention_broadcast()` → `AttentionBroadcastProjectionV1` with `attended_node_ids`, `dwell_ticks`, `coalition_stability_score`, `selected_description`, `generated_at` (`orion/schemas/attention_frame.py:144-160`).
- **Self-state** is the latest row of table `substrate_self_state`, column `self_state_json` (see `substrate_observability_routes._self_state_section`). Shape = `SelfStateV1` (`orion/schemas/self_state.py`), `dimensions: dict[str, SelfStateDimensionV1]`, each dim has `score`, `confidence`.
- **Worker loop pattern** = a sync `_x_tick()` run via `asyncio.to_thread`, wrapped by an async `_x_loop()` that sleeps on `self._stop.wait()` with a timeout; task added in `start()` (`worker.py:1094-1106`, `worker.py:189-203`).
- **Bus publish pattern** = `BaseEnvelope(kind=..., source=self._service_ref(), correlation_id=uuid4(), payload=model.model_dump(mode="json"))` + `await publish_with_reconnect(self._bus, channel, env, log_label=...)` (`worker.py:846-861`).
- **Hub reads Postgres directly** via SQLAlchemy `create_engine(os.getenv("POSTGRES_URI"))`; `POSTGRES_URI` is consumed everywhere but **missing from `services/orion-hub/.env_example`** (Task 9 adds it). DB is `conjourney`. Route precedent: `services/orion-hub/scripts/substrate_observability_routes.py` (degrade-to-200 `_engine()` returning `None`).
- **Hub route registration:** import + `router.include_router(...)` in `services/orion-hub/scripts/api_routes.py:164/177`.
- **Iframe embed precedent:** `#substrate-lattice` in `services/orion-hub/templates/index.html:2829-2845` → `src="/static/substrate-lattice.html?v={{HUB_UI_ASSET_VERSION}}"`. Self section to replace: `#self-observability` body `templates/index.html:1358-1388`.
- **`{{HUB_UI_ASSET_VERSION}}` is only substituted inside the rendered `index.html`**, not inside static files. So `self-brain.js` must use `API_BASE = ""` (same-origin) — do NOT copy the reverse-proxy prefix logic from `self_observability.js` (it breaks inside an iframe whose path starts `/static`).
- **`self_observability.js` also owns Self-tab activation/hash logic** (`static/js/self_observability.js:179-226`). When you replace the card body with an iframe, keep that script loaded so the tab still switches; its `getElementById` card guards early-return harmlessly.
- **Neither service has an `evals/` dir.** Producer eval is added in Task 12; the hub eval gap is called out in the PR.

---

## File structure

**New files:**
- `orion/schemas/brain_frame.py` — `SubstrateBrainFrameV1` + sub-models + `SUBSTRATE_BRAIN_FRAME_KIND`.
- `services/orion-substrate-runtime/app/brain_frame_producer.py` — pure assembly logic (deterministic, testable without DB/bus).
- `services/orion-sql-db/manual_migration_substrate_brain_frame_v1.sql` — log table DDL.
- `services/orion-substrate-runtime/tests/test_brain_frame_producer.py`
- `services/orion-substrate-runtime/tests/test_brain_frame_store.py`
- `services/orion-substrate-runtime/evals/test_brain_frame_substance_eval.py`
- `tests/test_substrate_brain_frame_bus_catalog.py` (repo-root contract test)
- `services/orion-hub/scripts/self_brain_routes.py` — read-only `/api/self-brain` router.
- `services/orion-hub/tests/test_self_brain_routes.py`
- `services/orion-hub/static/self-brain.html`, `services/orion-hub/static/js/self-brain.js`

**Modified files:**
- `orion/schemas/registry.py` — register in both maps.
- `orion/bus/channels.yaml` — new channel entry.
- `services/orion-substrate-runtime/app/settings.py` + `.env_example` — cadence/retention/sample-K/threshold/enable.
- `services/orion-substrate-runtime/app/store.py` — `save_brain_frame` / `load_brain_frames_tail` / `load_brain_frames_range` / prune.
- `services/orion-substrate-runtime/app/worker.py` — `_brain_frame_tick` + `_brain_frame_loop` + `start()` wiring.
- `services/orion-hub/scripts/api_routes.py` — register router.
- `services/orion-hub/.env_example` — add `POSTGRES_URI`.
- `services/orion-hub/templates/index.html` — swap `#self-observability` body for iframe.

---

## Task 0: Branch / worktree

- [ ] **Step 1: Create a clean worktree off main**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
git worktree add .worktrees/self-tab-brain-ekg -b feat/self-tab-brain-ekg main
cd .worktrees/self-tab-brain-ekg
git status --short
```

Expected: new worktree on branch `feat/self-tab-brain-ekg`, clean tree. All subsequent paths are relative to this worktree root.

---

## Task 1: `SubstrateBrainFrameV1` schema + registry

**Files:**
- Create: `orion/schemas/brain_frame.py`
- Modify: `orion/schemas/registry.py` (imports near `:573-583`; `_REGISTRY` near `:1106`; `SCHEMA_REGISTRY` near `:1332`)
- Test: `tests/test_substrate_brain_frame_bus_catalog.py` (Task 2 adds the catalog assertions; this task adds a schema-shape test)

- [ ] **Step 1: Write the failing schema test**

Create `services/orion-substrate-runtime/tests/test_brain_frame_producer.py` with only the schema import test first (the producer tests are appended in Task 5):

```python
from __future__ import annotations

from datetime import datetime, timezone


def test_brain_frame_schema_roundtrips_and_defaults():
    from orion.schemas.brain_frame import (
        SUBSTRATE_BRAIN_FRAME_KIND,
        BrainRegionV1,
        SubstrateBrainFrameV1,
    )

    assert SUBSTRATE_BRAIN_FRAME_KIND == "substrate.brain_frame.v1"

    region = BrainRegionV1(
        dimension="node_kind",
        region_id="node_kind:tension",
        label="Tension",
        intensity=0.9,
        state="firing",
        node_count=3,
        as_of=datetime(2026, 7, 7, tzinfo=timezone.utc),
        stale=False,
    )
    frame = SubstrateBrainFrameV1(
        frame_id="abc123",
        generated_at=datetime(2026, 7, 7, tzinfo=timezone.utc),
        tick_seq=1,
        phase="warming",
        regions=[region],
    )
    dumped = frame.model_dump(mode="json")
    again = SubstrateBrainFrameV1.model_validate(dumped)
    assert again.phase == "warming"
    assert again.regions[0].state == "firing"
    assert again.spotlight is None
    assert again.nodes == [] and again.edges == []
    assert again.schema_version == "substrate.brain_frame.v1"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest services/orion-substrate-runtime/tests/test_brain_frame_producer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'orion.schemas.brain_frame'`.

- [ ] **Step 3: Create the schema module**

Create `orion/schemas/brain_frame.py` (matches `consolidation_frame.py` house style — `from __future__`, `ConfigDict(extra="forbid")`, defaulted `schema_version` Literal, `Field(ge=/le=)`, `default_factory`):

```python
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SUBSTRATE_BRAIN_FRAME_KIND = "substrate.brain_frame.v1"


class BrainRegionV1(BaseModel):
    """The spine of the contract: a continuous, trackable per-region signal."""

    model_config = ConfigDict(extra="forbid")

    dimension: Literal["node_kind", "lane", "self_state", "lattice_layer"]
    region_id: str
    label: str
    intensity: float = Field(ge=0.0, le=1.0)
    state: Literal["firing", "steady", "starving"]
    node_count: int = Field(default=0, ge=0)
    as_of: datetime
    stale: bool = False
    detail: dict[str, float] = Field(default_factory=dict)


class BrainSpotlightV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attended_node_ids: list[str] = Field(default_factory=list)
    dwell_ticks: int = Field(default=0, ge=0)
    coalition_stability: float = Field(default=1.0, ge=0.0, le=1.0)
    description: str | None = None
    as_of: datetime
    stale: bool = False


class BrainNodeSampleV1(BaseModel):
    """Best-effort decoration. NO continuity guarantee across frames."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    node_kind: str
    activation: float = Field(ge=0.0, le=1.0)
    pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    dormant: bool = False
    label: str = ""


class BrainEdgeSampleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src: str
    dst: str
    weight: float = Field(default=0.0, ge=0.0, le=1.0)


class SubstrateBrainFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["substrate.brain_frame.v1"] = "substrate.brain_frame.v1"

    frame_id: str
    generated_at: datetime
    tick_seq: int = Field(ge=0)
    phase: Literal["warming", "live"] = "warming"
    source: str = "orion-substrate-runtime"

    regions: list[BrainRegionV1] = Field(default_factory=list)
    spotlight: BrainSpotlightV1 | None = None
    nodes: list[BrainNodeSampleV1] = Field(default_factory=list)
    edges: list[BrainEdgeSampleV1] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Register in both registry maps**

In `orion/schemas/registry.py`, add the import next to the other frame imports (after the `feedback_frame` import block near line 580):

```python
from orion.schemas.brain_frame import (
    BrainEdgeSampleV1,
    BrainNodeSampleV1,
    BrainRegionV1,
    BrainSpotlightV1,
    SubstrateBrainFrameV1,
)
```

Add to `_REGISTRY` (near the other `*FrameV1` entries around line 1106):

```python
    "SubstrateBrainFrameV1": SubstrateBrainFrameV1,
    "BrainRegionV1": BrainRegionV1,
    "BrainSpotlightV1": BrainSpotlightV1,
    "BrainNodeSampleV1": BrainNodeSampleV1,
    "BrainEdgeSampleV1": BrainEdgeSampleV1,
```

Add to `SCHEMA_REGISTRY` (inside the dict ending near line 1344, before the closing `}`):

```python
    "SubstrateBrainFrameV1": SchemaRegistration(
        model=SubstrateBrainFrameV1,
        kind="substrate.brain_frame.v1",
    ),
```

- [ ] **Step 5: Run schema test to verify it passes**

Run: `pytest services/orion-substrate-runtime/tests/test_brain_frame_producer.py -q`
Expected: PASS (1 test).

Also verify registry imports cleanly:

Run: `python -c "from orion.schemas.registry import resolve, SCHEMA_REGISTRY; assert resolve('SubstrateBrainFrameV1') is SCHEMA_REGISTRY['SubstrateBrainFrameV1'].model; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add orion/schemas/brain_frame.py orion/schemas/registry.py services/orion-substrate-runtime/tests/test_brain_frame_producer.py
git commit -m "feat(schema): add SubstrateBrainFrameV1 + registry registration"
```

---

## Task 2: Bus channel + contract catalog test

**Files:**
- Modify: `orion/bus/channels.yaml`
- Create: `tests/test_substrate_brain_frame_bus_catalog.py`

- [ ] **Step 1: Write the failing catalog test**

Create `tests/test_substrate_brain_frame_bus_catalog.py` (modeled on `tests/test_embodiment_bus_catalog.py`):

```python
from __future__ import annotations

from pathlib import Path

import yaml

from orion.schemas.registry import SCHEMA_REGISTRY, resolve

CHANNELS_YAML = Path(__file__).resolve().parents[1] / "orion" / "bus" / "channels.yaml"

CHANNEL_NAME = "orion:substrate:brain_frame"
SCHEMA_ID = "SubstrateBrainFrameV1"


def _channel_index() -> dict[str, dict]:
    doc = yaml.safe_load(CHANNELS_YAML.read_text(encoding="utf-8"))
    return {c["name"]: c for c in (doc.get("channels") or []) if c.get("name")}


def test_brain_frame_channel_exists_with_registry_schema():
    channels = _channel_index()
    assert CHANNEL_NAME in channels, f"missing channel catalog entry for {CHANNEL_NAME!r}"
    entry = channels[CHANNEL_NAME]
    assert entry["schema_id"] == SCHEMA_ID
    assert entry["message_kind"] == "substrate.brain_frame.v1"
    assert "orion-substrate-runtime" in entry["producer_services"]
    assert "orion-hub" in entry["consumer_services"]
    assert SCHEMA_ID in SCHEMA_REGISTRY
    assert resolve(SCHEMA_ID) is SCHEMA_REGISTRY[SCHEMA_ID].model
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_substrate_brain_frame_bus_catalog.py -q`
Expected: FAIL — `missing channel catalog entry for 'orion:substrate:brain_frame'`.

- [ ] **Step 3: Add the channel entry**

In `orion/bus/channels.yaml`, append after the `orion:substrate:self_state` entry (around line 2054), matching that entry's shape:

```yaml
  - name: "orion:substrate:brain_frame"
    kind: "event"
    schema_id: "SubstrateBrainFrameV1"
    message_kind: "substrate.brain_frame.v1"
    producer_services: ["orion-substrate-runtime"]
    consumer_services: ["orion-hub"]
    stability: "experimental"
    since: "2026-07-07"
```

- [ ] **Step 4: Run catalog + prefix guardrail tests to verify they pass**

Run: `pytest tests/test_substrate_brain_frame_bus_catalog.py tests/test_channel_prefix_guardrail.py -q`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add orion/bus/channels.yaml tests/test_substrate_brain_frame_bus_catalog.py
git commit -m "feat(bus): register orion:substrate:brain_frame channel + catalog test"
```

---

## Task 3: Log table migration

**Files:**
- Create: `services/orion-sql-db/manual_migration_substrate_brain_frame_v1.sql`

- [ ] **Step 1: Write the migration DDL**

Create `services/orion-sql-db/manual_migration_substrate_brain_frame_v1.sql` (mirrors `manual_migration_coalition_dwell_v1.sql`):

```sql
-- Substrate brain-frame log (Self tab brain-EKG realtime + playback backbone).
-- Append-per-frame, bounded by BRAIN_FRAME_RETENTION_HOURS prune in the producer.
CREATE TABLE IF NOT EXISTS substrate_brain_frame_log (
    frame_id TEXT PRIMARY KEY,
    tick_seq BIGINT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL,
    phase TEXT NOT NULL,
    frame_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_brain_frame_generated
    ON substrate_brain_frame_log(generated_at DESC);
```

- [ ] **Step 2: Verify SQL parses (dry check)**

Run: `python -c "import pathlib,re; sql=pathlib.Path('services/orion-sql-db/manual_migration_substrate_brain_frame_v1.sql').read_text(); assert 'substrate_brain_frame_log' in sql and 'CREATE INDEX' in sql; print('ok')"`
Expected: prints `ok`. (Actual apply against Postgres is a runtime step in the restart section — these migrations are applied by hand.)

- [ ] **Step 3: Commit**

```bash
git add services/orion-sql-db/manual_migration_substrate_brain_frame_v1.sql
git commit -m "feat(sql): add substrate_brain_frame_log migration"
```

---

## Task 4: Settings + env (all flags on)

**Files:**
- Modify: `services/orion-substrate-runtime/app/settings.py`
- Modify: `services/orion-substrate-runtime/.env_example`

- [ ] **Step 1: Write the failing settings test**

Create `services/orion-substrate-runtime/tests/test_brain_frame_settings.py`:

```python
from __future__ import annotations

import os
from unittest import mock


def test_brain_frame_settings_defaults_all_on():
    # Only POSTGRES_URI is required by Settings; provide it, clear brain keys.
    env = {"POSTGRES_URI": "postgresql://t:t@localhost/t"}
    to_clear = [
        "SUBSTRATE_BRAIN_FRAME_ENABLED",
        "BRAIN_FRAME_INTERVAL_SEC",
        "BRAIN_FRAME_RETENTION_HOURS",
        "BRAIN_FRAME_SAMPLE_NODES",
        "BRAIN_FRAME_SAMPLE_EDGES",
        "BRAIN_FRAME_FIRING_THRESHOLD",
        "BRAIN_FRAME_STARVING_THRESHOLD",
    ]
    with mock.patch.dict(os.environ, env, clear=False):
        for k in to_clear:
            os.environ.pop(k, None)
        from app.settings import Settings

        s = Settings()
        assert s.brain_frame_enabled is True
        assert s.brain_frame_interval_sec == 5.0
        assert s.brain_frame_retention_hours == 24
        assert s.brain_frame_sample_nodes == 40
        assert s.brain_frame_sample_edges == 60
        assert 0.0 < s.brain_frame_starving_threshold < s.brain_frame_firing_threshold <= 1.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd services/orion-substrate-runtime && python -m pytest tests/test_brain_frame_settings.py -q; cd ../..`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'brain_frame_enabled'`.

- [ ] **Step 3: Add settings fields**

In `services/orion-substrate-runtime/app/settings.py`, add after the endogenous-curiosity block (after line 70), before `biometrics_grammar_batch_limit`:

```python
    # Self-tab brain-EKG frame producer. Enabled by default (operator directive).
    brain_frame_enabled: bool = Field(True, alias="SUBSTRATE_BRAIN_FRAME_ENABLED")
    brain_frame_interval_sec: float = Field(5.0, alias="BRAIN_FRAME_INTERVAL_SEC")
    brain_frame_retention_hours: int = Field(24, alias="BRAIN_FRAME_RETENTION_HOURS")
    brain_frame_sample_nodes: int = Field(40, alias="BRAIN_FRAME_SAMPLE_NODES")
    brain_frame_sample_edges: int = Field(60, alias="BRAIN_FRAME_SAMPLE_EDGES")
    brain_frame_firing_threshold: float = Field(0.5, alias="BRAIN_FRAME_FIRING_THRESHOLD")
    brain_frame_starving_threshold: float = Field(0.1, alias="BRAIN_FRAME_STARVING_THRESHOLD")
    # A dimension renders stale when generated_at - as_of exceeds its cadence.
    brain_frame_self_state_cadence_sec: float = Field(
        30.0, alias="BRAIN_FRAME_SELF_STATE_CADENCE_SEC"
    )
    brain_frame_spotlight_cadence_sec: float = Field(
        30.0, alias="BRAIN_FRAME_SPOTLIGHT_CADENCE_SEC"
    )
```

- [ ] **Step 4: Run settings test to verify it passes**

Run: `cd services/orion-substrate-runtime && python -m pytest tests/test_brain_frame_settings.py -q; cd ../..`
Expected: PASS.

- [ ] **Step 5: Update `.env_example` (new keys + turn dependent flags on)**

In `services/orion-substrate-runtime/.env_example`, add a block:

```bash
# --- Self-tab brain-EKG frame producer ---
SUBSTRATE_BRAIN_FRAME_ENABLED=true
BRAIN_FRAME_INTERVAL_SEC=5.0
BRAIN_FRAME_RETENTION_HOURS=24
BRAIN_FRAME_SAMPLE_NODES=40
BRAIN_FRAME_SAMPLE_EDGES=60
BRAIN_FRAME_FIRING_THRESHOLD=0.5
BRAIN_FRAME_STARVING_THRESHOLD=0.1
BRAIN_FRAME_SELF_STATE_CADENCE_SEC=30.0
BRAIN_FRAME_SPOTLIGHT_CADENCE_SEC=30.0
# Dependent signal producers the brain frame reads — enabled so regions show real motion.
SUBSTRATE_DYNAMICS_TICK_ENABLED=true
ORION_ATTENTION_BROADCAST_ENABLED=true
ENABLE_EXECUTION_TRAJECTORY_REDUCER=true
ENABLE_TRANSPORT_BUS_REDUCER=true
ENABLE_CHAT_GRAMMAR_REDUCER=true
```

If `SUBSTRATE_DYNAMICS_TICK_ENABLED` etc. already exist in `.env_example`, edit their values to `true` in place instead of duplicating.

- [ ] **Step 6: Sync local `.env`**

Run: `python scripts/sync_local_env_from_example.py`
Expected: reports syncing `orion-substrate-runtime` keys (the `ORION_SUBSTRATE_*` prefix and the new `BRAIN_FRAME_*`/`SUBSTRATE_BRAIN_FRAME_*` keys). Confirm no `.env` is staged:

Run: `git status --short services/orion-substrate-runtime/.env`
Expected: empty (`.env` is gitignored). If it appears, stop and fix.

- [ ] **Step 7: Commit**

```bash
git add services/orion-substrate-runtime/app/settings.py services/orion-substrate-runtime/.env_example services/orion-substrate-runtime/tests/test_brain_frame_settings.py
git commit -m "feat(substrate): brain-frame settings + enable dependent signal flags in env"
```

---

## Task 5: Brain-frame producer (pure assembly)

The producer is a **pure function** taking already-fetched inputs (graph nodes/edges, lane-health dict, self-state dict-or-None, attention projection-or-None, settings, `now`, `tick_seq`) and returning a `SubstrateBrainFrameV1`. Keeping it DB/bus-free makes it fully unit-testable and satisfies the anti-empty-shell acceptance check.

**Files:**
- Create: `services/orion-substrate-runtime/app/brain_frame_producer.py`
- Test: `services/orion-substrate-runtime/tests/test_brain_frame_producer.py` (append)

- [ ] **Step 1: Write failing producer tests**

Append to `services/orion-substrate-runtime/tests/test_brain_frame_producer.py`:

```python
from types import SimpleNamespace


def _node(node_id, kind, activation, pressure=0.0, dormant=False):
    return SimpleNamespace(
        node_id=node_id,
        node_kind=kind,
        label=f"{kind}:{node_id}",
        activation=activation,
        metadata={"dynamic_pressure": pressure, "dormant": dormant},
    )


def _settings():
    return SimpleNamespace(
        brain_frame_sample_nodes=40,
        brain_frame_sample_edges=60,
        brain_frame_firing_threshold=0.5,
        brain_frame_starving_threshold=0.1,
        brain_frame_self_state_cadence_sec=30.0,
        brain_frame_spotlight_cadence_sec=30.0,
    )


def test_producer_yields_firing_and_starving_regions_and_samples():
    from datetime import datetime, timezone

    from app.brain_frame_producer import assemble_brain_frame

    now = datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone.utc)
    nodes = [
        _node("t1", "tension", activation=0.95, pressure=0.9),
        _node("t2", "tension", activation=0.8, pressure=0.7),
        _node("c1", "concept", activation=0.02, pressure=0.0, dormant=True),
    ]
    lane_health = {
        "cursor_lag_by_reducer": {"execution_trajectory": 1.0, "transport_bus": 400.0},
        "pending_backlog_by_reducer": {"execution_trajectory": 12, "transport_bus": 0},
        "quarantine_by_reducer": {},
    }
    frame = assemble_brain_frame(
        nodes=nodes,
        edges=[],
        lane_health=lane_health,
        self_state=None,
        attention=None,
        settings=_settings(),
        now=now,
        tick_seq=7,
    )
    assert frame.phase == "live"  # real activation present
    kinds = {r.region_id: r for r in frame.regions if r.dimension == "node_kind"}
    assert kinds["node_kind:tension"].state == "firing"
    assert kinds["node_kind:concept"].state == "starving"
    lanes = {r.region_id: r for r in frame.regions if r.dimension == "lane"}
    # Fresh lane (low lag, backlog) fires; badly-lagged lane starves.
    assert lanes["lane:execution_trajectory"].state in {"firing", "steady"}
    assert lanes["lane:transport_bus"].state == "starving"
    assert len(frame.nodes) >= 1  # non-empty decoration
    assert frame.tick_seq == 7


def test_producer_warming_when_graph_dead():
    from datetime import datetime, timezone

    from app.brain_frame_producer import assemble_brain_frame

    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    frame = assemble_brain_frame(
        nodes=[_node("c1", "concept", activation=0.0)],
        edges=[],
        lane_health={"cursor_lag_by_reducer": {}, "pending_backlog_by_reducer": {}, "quarantine_by_reducer": {}},
        self_state=None,
        attention=None,
        settings=_settings(),
        now=now,
        tick_seq=0,
    )
    assert frame.phase == "warming"


def test_self_state_region_marked_stale_when_old():
    from datetime import datetime, timedelta, timezone

    from app.brain_frame_producer import assemble_brain_frame

    now = datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone.utc)
    old = (now - timedelta(seconds=120)).isoformat()
    self_state = {
        "generated_at": old,
        "dimensions": {
            "execution_pressure": {"score": 0.8, "confidence": 0.7},
            "coherence": {"score": 0.4, "confidence": 0.6},
        },
    }
    frame = assemble_brain_frame(
        nodes=[_node("t1", "tension", 0.9, 0.9)],
        edges=[],
        lane_health={"cursor_lag_by_reducer": {}, "pending_backlog_by_reducer": {}, "quarantine_by_reducer": {}},
        self_state=self_state,
        attention=None,
        settings=_settings(),
        now=now,
        tick_seq=3,
    )
    ss = {r.region_id: r for r in frame.regions if r.dimension == "self_state"}
    assert ss["self_state:execution_pressure"].stale is True
    assert ss["self_state:execution_pressure"].intensity == 0.8
```

- [ ] **Step 2: Run to verify failure**

Run: `cd services/orion-substrate-runtime && python -m pytest tests/test_brain_frame_producer.py -q; cd ../..`
Expected: FAIL — `No module named 'app.brain_frame_producer'`.

- [ ] **Step 3: Implement the producer**

Create `services/orion-substrate-runtime/app/brain_frame_producer.py`:

```python
"""Pure assembly of SubstrateBrainFrameV1 from already-fetched substrate signals.

Deterministic and dependency-free (no DB, no bus): callers fetch the live graph,
lane health, latest self-state row, and latest attention broadcast, then hand
them here. Regions are computed from real activity and are the trackable spine;
node/edge samples are best-effort decoration with no continuity guarantee.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Iterable, Mapping

from orion.schemas.brain_frame import (
    BrainEdgeSampleV1,
    BrainNodeSampleV1,
    BrainRegionV1,
    BrainSpotlightV1,
    SubstrateBrainFrameV1,
)

_LANE_LABELS = {
    "biometrics": "Biometrics",
    "chat_grammar": "Chat grammar",
    "execution_trajectory": "Execution",
    "transport_bus": "Transport",
}


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _state_for(intensity: float, firing: float, starving: float) -> str:
    if intensity >= firing:
        return "firing"
    if intensity <= starving:
        return "starving"
    return "steady"


def _node_pressure(node: Any) -> float:
    md = getattr(node, "metadata", None) or {}
    val = md.get("dynamic_pressure")
    if val is None:
        val = md.get("prediction_error")
    return _clamp01(val or 0.0)


def _node_dormant(node: Any) -> bool:
    md = getattr(node, "metadata", None) or {}
    if md.get("dormant") is True:
        return True
    return _clamp01(getattr(node, "activation", 0.0)) <= 0.0


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _node_kind_regions(nodes, now, firing, starving) -> list[BrainRegionV1]:
    buckets: dict[str, list[float]] = {}
    for node in nodes:
        kind = str(getattr(node, "node_kind", "") or "unknown")
        buckets.setdefault(kind, []).append(_clamp01(getattr(node, "activation", 0.0)))
    regions: list[BrainRegionV1] = []
    for kind, activations in sorted(buckets.items()):
        intensity = max(activations) if activations else 0.0
        regions.append(
            BrainRegionV1(
                dimension="node_kind",
                region_id=f"node_kind:{kind}",
                label=kind.replace("_", " ").title(),
                intensity=intensity,
                state=_state_for(intensity, firing, starving),
                node_count=len(activations),
                as_of=now,
                stale=False,
                detail={"mean_activation": sum(activations) / len(activations) if activations else 0.0},
            )
        )
    return regions


def _lane_regions(lane_health: Mapping[str, Any], now, firing, starving) -> list[BrainRegionV1]:
    lag = dict(lane_health.get("cursor_lag_by_reducer") or {})
    backlog = dict(lane_health.get("pending_backlog_by_reducer") or {})
    quarantine = dict(lane_health.get("quarantine_by_reducer") or {})
    regions: list[BrainRegionV1] = []
    lane_keys = set(lag) | set(backlog) | set(_LANE_LABELS)
    for lane in sorted(lane_keys):
        lag_sec = float(lag.get(lane, 0.0) or 0.0)
        pending = float(backlog.get(lane, 0.0) or 0.0)
        # Fresh + moving lane = firing; stale/lagged lane = starving.
        freshness = 1.0 if lag_sec <= 60 else max(0.0, 1.0 - (lag_sec - 60) / 240.0)
        activity = min(1.0, pending / 20.0)
        intensity = _clamp01(0.6 * activity + 0.4 * freshness) if lag_sec <= 60 else _clamp01(freshness)
        if lag_sec > 300:
            intensity = min(intensity, starving)
        regions.append(
            BrainRegionV1(
                dimension="lane",
                region_id=f"lane:{lane}",
                label=_LANE_LABELS.get(lane, lane.replace("_", " ").title()),
                intensity=intensity,
                state=_state_for(intensity, firing, starving),
                node_count=int(pending),
                as_of=now,
                stale=False,
                detail={"lag_sec": lag_sec, "backlog": pending, "quarantine": float(quarantine.get(lane, 0) or 0)},
            )
        )
    return regions


def _self_state_regions(self_state, now, cadence_sec) -> list[BrainRegionV1]:
    if not isinstance(self_state, Mapping):
        return []
    as_of = _parse_dt(self_state.get("generated_at")) or now
    stale = (now - as_of).total_seconds() > cadence_sec
    dims = self_state.get("dimensions") or {}
    regions: list[BrainRegionV1] = []
    for dim_id, payload in sorted(dims.items()):
        if not isinstance(payload, Mapping):
            continue
        score = _clamp01(payload.get("score", 0.0))
        conf = _clamp01(payload.get("confidence", 0.0))
        regions.append(
            BrainRegionV1(
                dimension="self_state",
                region_id=f"self_state:{dim_id}",
                label=dim_id.replace("_", " ").title(),
                intensity=score,
                # self-state dims are always shown as steps, not fired/starved.
                state="steady",
                node_count=0,
                as_of=as_of,
                stale=stale,
                detail={"confidence": conf},
            )
        )
    return regions


def _spotlight(attention, now, cadence_sec) -> BrainSpotlightV1 | None:
    if attention is None:
        return None
    as_of = _parse_dt(getattr(attention, "generated_at", None)) or now
    return BrainSpotlightV1(
        attended_node_ids=[str(x) for x in getattr(attention, "attended_node_ids", []) or []],
        dwell_ticks=int(getattr(attention, "dwell_ticks", 0) or 0),
        coalition_stability=_clamp01(getattr(attention, "coalition_stability_score", 1.0)),
        description=getattr(attention, "selected_description", None),
        as_of=as_of,
        stale=(now - as_of).total_seconds() > cadence_sec,
    )


def _samples(nodes, edges, max_nodes, max_edges) -> tuple[list[BrainNodeSampleV1], list[BrainEdgeSampleV1]]:
    ranked = sorted(nodes, key=lambda n: _clamp01(getattr(n, "activation", 0.0)), reverse=True)
    node_samples = [
        BrainNodeSampleV1(
            node_id=str(getattr(n, "node_id", "") or ""),
            node_kind=str(getattr(n, "node_kind", "") or "unknown"),
            activation=_clamp01(getattr(n, "activation", 0.0)),
            pressure=_node_pressure(n),
            dormant=_node_dormant(n),
            label=str(getattr(n, "label", "") or "")[:120],
        )
        for n in ranked[: max(0, int(max_nodes))]
        if str(getattr(n, "node_id", "") or "")
    ]
    kept_ids = {s.node_id for s in node_samples}
    edge_samples: list[BrainEdgeSampleV1] = []
    for e in edges or []:
        src = str(getattr(e, "src", None) or getattr(e, "source", "") or "")
        dst = str(getattr(e, "dst", None) or getattr(e, "target", "") or "")
        if not src or not dst or src not in kept_ids or dst not in kept_ids:
            continue
        edge_samples.append(
            BrainEdgeSampleV1(src=src, dst=dst, weight=_clamp01(getattr(e, "weight", 0.0) or 0.0))
        )
        if len(edge_samples) >= max(0, int(max_edges)):
            break
    return node_samples, edge_samples


def assemble_brain_frame(
    *,
    nodes: Iterable[Any],
    edges: Iterable[Any],
    lane_health: Mapping[str, Any],
    self_state: Mapping[str, Any] | None,
    attention: Any | None,
    settings: Any,
    now: datetime,
    tick_seq: int,
) -> SubstrateBrainFrameV1:
    nodes = list(nodes)
    firing = float(settings.brain_frame_firing_threshold)
    starving = float(settings.brain_frame_starving_threshold)

    regions = (
        _node_kind_regions(nodes, now, firing, starving)
        + _lane_regions(lane_health or {}, now, firing, starving)
        + _self_state_regions(self_state, now, float(settings.brain_frame_self_state_cadence_sec))
    )
    node_samples, edge_samples = _samples(
        nodes, list(edges), settings.brain_frame_sample_nodes, settings.brain_frame_sample_edges
    )

    max_activation = max((_clamp01(getattr(n, "activation", 0.0)) for n in nodes), default=0.0)
    phase = "live" if max_activation > 0.0 else "warming"

    frame_id = hashlib.sha256(f"{now.isoformat()}|{tick_seq}".encode("utf-8")).hexdigest()[:24]
    return SubstrateBrainFrameV1(
        frame_id=frame_id,
        generated_at=now,
        tick_seq=int(tick_seq),
        phase=phase,
        regions=regions,
        spotlight=_spotlight(attention, now, float(settings.brain_frame_spotlight_cadence_sec)),
        nodes=node_samples,
        edges=edge_samples,
    )
```

- [ ] **Step 4: Run producer tests to verify they pass**

Run: `cd services/orion-substrate-runtime && python -m pytest tests/test_brain_frame_producer.py -q; cd ../..`
Expected: PASS (all 4 tests).

- [ ] **Step 5: Commit**

```bash
git add services/orion-substrate-runtime/app/brain_frame_producer.py services/orion-substrate-runtime/tests/test_brain_frame_producer.py
git commit -m "feat(substrate): pure brain-frame assembly with regions, staleness, samples"
```

---

## Task 6: Store append / tail / range / prune

**Files:**
- Modify: `services/orion-substrate-runtime/app/store.py`
- Test: `services/orion-substrate-runtime/tests/test_brain_frame_store.py`

The store test uses a real SQLite in-memory engine won't work (JSONB is Postgres-only). Instead, unit-test the SQL-building helpers by patching the engine with a fake connection recorder, mirroring the hub's fake-engine pattern. Keep the store methods thin so the test asserts the SQL and params.

- [ ] **Step 1: Write the failing store test**

Create `services/orion-substrate-runtime/tests/test_brain_frame_store.py`:

```python
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock

from orion.schemas.brain_frame import SubstrateBrainFrameV1


class _RecordingEngine:
    def __init__(self, tail_rows=None, range_rows=None):
        self.executed = []
        self._tail_rows = tail_rows or []
        self._range_rows = range_rows or []

    @contextmanager
    def begin(self):
        conn = MagicMock()

        def execute(stmt, params=None):
            self.executed.append((str(stmt), params))
            return MagicMock()

        conn.execute.side_effect = execute
        yield conn

    @contextmanager
    def connect(self):
        conn = MagicMock()

        def execute(stmt, params=None):
            self.executed.append((str(stmt), params))
            m = MagicMock()
            rows = self._range_rows if "BETWEEN" in str(stmt) or "generated_at >=" in str(stmt) else self._tail_rows
            m.mappings.return_value.all.return_value = rows
            return m

        conn.execute.side_effect = execute
        yield conn


def _store_with(engine):
    from app.store import BiometricsSubstrateStore

    store = BiometricsSubstrateStore.__new__(BiometricsSubstrateStore)
    store._engine = engine
    return store


def _frame(seq, ts):
    return SubstrateBrainFrameV1(frame_id=f"f{seq}", generated_at=ts, tick_seq=seq, phase="live")


def test_save_brain_frame_inserts_and_prunes():
    eng = _RecordingEngine()
    store = _store_with(eng)
    store.save_brain_frame(_frame(1, datetime(2026, 7, 7, tzinfo=timezone.utc)), retention_hours=24)
    sqls = " ".join(s for s, _ in eng.executed)
    assert "INSERT INTO substrate_brain_frame_log" in sqls
    assert "DELETE FROM substrate_brain_frame_log" in sqls
    # params carried the frame id + json
    insert_params = eng.executed[0][1]
    assert insert_params["frame_id"] == "f1"
    assert insert_params["tick_seq"] == 1


def test_load_tail_returns_ascending():
    ts = datetime(2026, 7, 7, tzinfo=timezone.utc)
    rows = [
        {"frame_json": _frame(3, ts).model_dump(mode="json")},
        {"frame_json": _frame(2, ts).model_dump(mode="json")},
    ]  # DB returns DESC
    eng = _RecordingEngine(tail_rows=rows)
    store = _store_with(eng)
    frames = store.load_brain_frames_tail(limit=2)
    assert [f["tick_seq"] for f in frames] == [2, 3]  # reversed to ascending
```

- [ ] **Step 2: Run to verify failure**

Run: `cd services/orion-substrate-runtime && python -m pytest tests/test_brain_frame_store.py -q; cd ../..`
Expected: FAIL — `AttributeError: 'BiometricsSubstrateStore' object has no attribute 'save_brain_frame'`.

- [ ] **Step 3: Add store methods**

In `services/orion-substrate-runtime/app/store.py`, add after `save_coalition_dwell` (after line 600). `Json`, `text`, `datetime`, `timezone`, `hashlib` are already imported at the top of the file:

```python
    def save_brain_frame(self, frame, retention_hours: int = 24) -> None:
        """Append one brain-frame row per tick; prune rows beyond retention."""
        now = datetime.now(timezone.utc)
        payload = frame.model_dump(mode="json")
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_brain_frame_log (
                        frame_id, tick_seq, generated_at, phase, frame_json, created_at
                    ) VALUES (
                        :frame_id, :tick_seq, :generated_at, :phase, :frame_json, :created_at
                    )
                    ON CONFLICT (frame_id) DO NOTHING
                    """
                ),
                {
                    "frame_id": frame.frame_id,
                    "tick_seq": int(frame.tick_seq),
                    "generated_at": frame.generated_at,
                    "phase": frame.phase,
                    "frame_json": Json(payload),
                    "created_at": now,
                },
            )
            conn.execute(
                text(
                    f"""
                    DELETE FROM substrate_brain_frame_log
                    WHERE generated_at < now() - interval '{int(retention_hours)} hours'
                    """
                ),
            )

    def load_brain_frames_tail(self, limit: int = 1) -> list[dict]:
        """Return the most-recent N frame payloads, ascending by generated_at."""
        limit = max(1, min(int(limit), 120))
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT frame_json FROM substrate_brain_frame_log
                    ORDER BY generated_at DESC LIMIT :limit
                    """
                ),
                {"limit": limit},
            ).mappings().all()
        frames = [self._coerce_frame_json(r["frame_json"]) for r in rows]
        frames.reverse()
        return frames

    def load_brain_frames_range(self, start, end, max_frames: int = 240) -> list[dict]:
        """Return frames in [start, end], downsampled to at most max_frames, ascending."""
        max_frames = max(1, min(int(max_frames), 2000))
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT frame_json FROM substrate_brain_frame_log
                    WHERE generated_at >= :start AND generated_at <= :end
                    ORDER BY generated_at ASC
                    """
                ),
                {"start": start, "end": end},
            ).mappings().all()
        frames = [self._coerce_frame_json(r["frame_json"]) for r in rows]
        if len(frames) <= max_frames:
            return frames
        step = len(frames) / max_frames
        return [frames[int(i * step)] for i in range(max_frames)]

    def brain_frame_window(self) -> dict:
        """Return retention bounds + earliest/latest frame ts + current phase."""
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT
                      min(generated_at) AS earliest,
                      max(generated_at) AS latest,
                      count(*) AS n
                    FROM substrate_brain_frame_log
                    """
                ),
            ).mappings().first()
            phase_row = conn.execute(
                text(
                    """
                    SELECT phase FROM substrate_brain_frame_log
                    ORDER BY generated_at DESC LIMIT 1
                    """
                ),
            ).mappings().first()
        earliest = row["earliest"] if row else None
        latest = row["latest"] if row else None
        return {
            "earliest": earliest.isoformat() if hasattr(earliest, "isoformat") else earliest,
            "latest": latest.isoformat() if hasattr(latest, "isoformat") else latest,
            "frame_count": int(row["n"]) if row else 0,
            "phase": (phase_row["phase"] if phase_row else None),
        }

    @staticmethod
    def _coerce_frame_json(value):
        if isinstance(value, str):
            return json.loads(value)
        return value
```

`json` is already imported at the top of `store.py` (used by `create_engine(..., json_serializer=json.dumps)`).

- [ ] **Step 4: Run store tests to verify they pass**

Run: `cd services/orion-substrate-runtime && python -m pytest tests/test_brain_frame_store.py -q; cd ../..`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add services/orion-substrate-runtime/app/store.py services/orion-substrate-runtime/tests/test_brain_frame_store.py
git commit -m "feat(substrate): brain-frame store append/tail/range/window with prune"
```

---

## Task 7: Worker loop (fetch → assemble → publish → append)

**Files:**
- Modify: `services/orion-substrate-runtime/app/worker.py`
- Test: `services/orion-substrate-runtime/tests/test_brain_frame_worker.py`

- [ ] **Step 1: Write the failing worker tick test**

Create `services/orion-substrate-runtime/tests/test_brain_frame_worker.py`:

```python
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.worker import BiometricsSubstrateWorker


def _node(node_id, kind, activation, pressure=0.0):
    return SimpleNamespace(
        node_id=node_id, node_kind=kind, label=kind,
        activation=activation, metadata={"dynamic_pressure": pressure},
    )


def _worker():
    w = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    w._settings = SimpleNamespace(
        brain_frame_enabled=True,
        brain_frame_sample_nodes=40,
        brain_frame_sample_edges=60,
        brain_frame_firing_threshold=0.5,
        brain_frame_starving_threshold=0.1,
        brain_frame_self_state_cadence_sec=30.0,
        brain_frame_spotlight_cadence_sec=30.0,
        brain_frame_retention_hours=24,
    )
    w._store = MagicMock()
    w._brain_frame_seq = 0
    return w


def test_brain_frame_tick_assembles_and_persists(monkeypatch):
    w = _worker()
    graph_store = MagicMock()
    graph_store.snapshot.return_value = SimpleNamespace(
        nodes={"t1": _node("t1", "tension", 0.9, 0.9)}, edges=[]
    )
    monkeypatch.setattr(w, "_get_substrate_graph_store", lambda **k: graph_store)
    monkeypatch.setattr(w, "_brain_frame_lane_health", lambda: {"cursor_lag_by_reducer": {}, "pending_backlog_by_reducer": {}, "quarantine_by_reducer": {}})
    monkeypatch.setattr(w, "_brain_frame_self_state", lambda: None)
    w._store.load_attention_broadcast.return_value = None

    frame = w._brain_frame_tick()
    assert frame is not None
    assert frame.phase == "live"
    assert w._store.save_brain_frame.called
    assert w._brain_frame_seq == 1


def test_brain_frame_tick_skips_when_disabled(monkeypatch):
    w = _worker()
    w._settings.brain_frame_enabled = False
    assert w._brain_frame_tick() is None
    assert not w._store.save_brain_frame.called
```

- [ ] **Step 2: Run to verify failure**

Run: `cd services/orion-substrate-runtime && python -m pytest tests/test_brain_frame_worker.py -q; cd ../..`
Expected: FAIL — `AttributeError: ... has no attribute '_brain_frame_tick'`.

- [ ] **Step 3: Initialize the frame counter in `__init__`**

In `services/orion-substrate-runtime/app/worker.py`, in `BiometricsSubstrateWorker.__init__` (after line 172, `self._latest_drive_state_at = None`):

```python
        self._brain_frame_seq: int = 0
```

- [ ] **Step 4: Add the tick, helpers, publish, and loop**

Add these methods to `BiometricsSubstrateWorker` (place near `_attention_broadcast_tick`/`_attention_broadcast_loop`, after line 1106). Note `datetime`, `timezone`, `uuid4`, `logger`, `Any` are already imported in `worker.py`:

```python
    def _brain_frame_lane_health(self) -> dict:
        """Fetch reducer lane health for lane regions. Fail-open to empty."""
        try:
            from app.grammar_truth import build_substrate_grammar_truth

            return build_substrate_grammar_truth(self._store)
        except Exception:
            logger.exception("brain_frame_lane_health_failed")
            return {}

    def _brain_frame_self_state(self) -> dict | None:
        """Latest self-state row payload (dict) or None. Fail-open."""
        try:
            from sqlalchemy import text

            engine = self._get_sql_engine()
            if engine is None:
                return None
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT self_state_json FROM substrate_self_state
                        ORDER BY generated_at DESC LIMIT 1
                        """
                    )
                ).mappings().first()
            if not row:
                return None
            payload = row["self_state_json"]
            if isinstance(payload, str):
                import json as _json

                payload = _json.loads(payload)
            return payload if isinstance(payload, dict) else None
        except Exception:
            logger.exception("brain_frame_self_state_load_failed")
            return None

    def _brain_frame_tick(self):
        """Assemble + persist one brain frame. Returns the frame or None."""
        s = self._settings
        if not s.brain_frame_enabled:
            return None
        try:
            from app.brain_frame_producer import assemble_brain_frame

            store = self._get_substrate_graph_store(
                log_label="brain_frame_graph_store_init_failed"
            )
            nodes: list[Any] = []
            edges: list[Any] = []
            if store is not None:
                try:
                    state = store.snapshot()
                    nodes = list(state.nodes.values())
                    edges = list(getattr(state, "edges", []) or [])
                except Exception:
                    logger.exception("brain_frame_snapshot_failed")

            attention = None
            try:
                attention = self._store.load_attention_broadcast()
            except Exception:
                logger.exception("brain_frame_attention_load_failed")

            now = datetime.now(timezone.utc)
            frame = assemble_brain_frame(
                nodes=nodes,
                edges=edges,
                lane_health=self._brain_frame_lane_health(),
                self_state=self._brain_frame_self_state(),
                attention=attention,
                settings=s,
                now=now,
                tick_seq=self._brain_frame_seq,
            )
            self._brain_frame_seq += 1
            try:
                self._store.save_brain_frame(
                    frame, retention_hours=int(s.brain_frame_retention_hours)
                )
            except Exception:
                logger.exception("brain_frame_persist_failed")
            logger.info(
                "brain_frame_tick_completed phase=%s regions=%d nodes=%d frame_id=%s",
                frame.phase,
                len(frame.regions),
                len(frame.nodes),
                frame.frame_id,
            )
            return frame
        except Exception:
            logger.exception("brain_frame_tick_failed")
            return None

    async def _publish_brain_frame(self, frame) -> None:
        if self._bus is None or frame is None:
            return
        try:
            from orion.core.bus.bus_schemas import BaseEnvelope
            from orion.core.bus.resilience import publish_with_reconnect
            from orion.schemas.brain_frame import SUBSTRATE_BRAIN_FRAME_KIND

            env = BaseEnvelope(
                kind=SUBSTRATE_BRAIN_FRAME_KIND,
                source=self._service_ref(),
                correlation_id=uuid4(),
                payload=frame.model_dump(mode="json"),
            )
            await publish_with_reconnect(
                self._bus,
                "orion:substrate:brain_frame",
                env,
                log_label="substrate_brain_frame",
            )
        except Exception:
            logger.exception("brain_frame_publish_failed")

    async def _brain_frame_loop(self) -> None:
        interval = float(self._settings.brain_frame_interval_sec)
        while not self._stop.is_set():
            try:
                frame = await asyncio.to_thread(self._brain_frame_tick)
                await self._publish_brain_frame(frame)
            except Exception:
                logger.exception("substrate_brain_frame_loop_failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
```

- [ ] **Step 5: Wire the loop into `start()`**

In `start()` (the `self._tasks = [...]` list, after the endogenous-curiosity task at line 202), add:

```python
            asyncio.create_task(self._brain_frame_loop(), name="substrate-brain-frame"),
```

- [ ] **Step 6: Run worker tests to verify they pass**

Run: `cd services/orion-substrate-runtime && python -m pytest tests/test_brain_frame_worker.py -q; cd ../..`
Expected: PASS (2 tests).

- [ ] **Step 7: Run the full substrate-runtime test suite**

Run: `cd services/orion-substrate-runtime && python -m pytest tests -q; cd ../..`
Expected: PASS (no regressions in existing tests).

- [ ] **Step 8: Commit**

```bash
git add services/orion-substrate-runtime/app/worker.py services/orion-substrate-runtime/tests/test_brain_frame_worker.py
git commit -m "feat(substrate): brain-frame loop — fetch, assemble, publish, append"
```

---

## Task 8: Hub read-only API

**Files:**
- Create: `services/orion-hub/scripts/self_brain_routes.py`
- Modify: `services/orion-hub/scripts/api_routes.py` (import after line 164; include after line 177)
- Test: `services/orion-hub/tests/test_self_brain_routes.py`

- [ ] **Step 1: Write the failing route tests**

Create `services/orion-hub/tests/test_self_brain_routes.py` (mirrors `test_substrate_observability_api.py`):

```python
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts import self_brain_routes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    app = FastAPI()
    app.include_router(self_brain_routes.router)
    return TestClient(app)


def _frame(seq):
    return {
        "frame_id": f"f{seq}",
        "generated_at": "2026-07-07T12:00:00+00:00",
        "tick_seq": seq,
        "phase": "live",
        "regions": [],
        "nodes": [],
        "edges": [],
    }


def _fake_engine(tail_rows):
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute(stmt, params=None):
        m = MagicMock()
        m.mappings.return_value.all.return_value = tail_rows
        m.mappings.return_value.first.return_value = tail_rows[0] if tail_rows else None
        return m

    conn.execute.side_effect = execute
    return engine


def test_tail_returns_ascending_and_200(client):
    rows = [{"frame_json": _frame(3)}, {"frame_json": _frame(2)}]  # DESC from DB
    with patch.object(self_brain_routes, "_engine", return_value=_fake_engine(rows)):
        r = client.get("/api/self-brain/frames/tail?limit=2")
    assert r.status_code == 200
    body = r.json()
    assert [f["tick_seq"] for f in body["frames"]] == [2, 3]


def test_tail_degrades_to_empty_when_no_engine(client):
    with patch.object(self_brain_routes, "_engine", return_value=None):
        r = client.get("/api/self-brain/frames/tail")
    assert r.status_code == 200
    assert r.json()["frames"] == []


def test_router_is_read_only(client):
    routes = [r for r in client.app.routes if hasattr(r, "methods")]
    for route in routes:
        if str(route.path).startswith("/api/self-brain"):
            assert route.methods <= {"GET", "HEAD"}, route.path
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest services/orion-hub/tests/test_self_brain_routes.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.self_brain_routes'` (or import error).

- [ ] **Step 3: Implement the router**

Create `services/orion-hub/scripts/self_brain_routes.py` (degrade-to-200 engine like observability):

```python
"""Read-only Self-brain API: realtime tail + playback range + window bounds.

Reads the substrate_brain_frame_log table directly from Postgres (same DB the
other substrate panels use, env POSTGRES_URI). Degrades to empty-with-200 when
the log is empty or POSTGRES_URI is unset. No writes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/self-brain", tags=["self-brain"])

_MAX_TAIL = 120
_DEFAULT_RANGE_MAX = 240


def _engine():
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return None
    from sqlalchemy import create_engine

    return create_engine(uri, pool_pre_ping=True)


def _coerce(value: Any) -> dict | None:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


@router.get("/frames/tail")
async def frames_tail(limit: int = Query(default=1, ge=1, le=_MAX_TAIL)) -> dict[str, Any]:
    engine = _engine()
    if engine is None:
        return {"frames": [], "phase": None}
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT frame_json FROM substrate_brain_frame_log "
                    "ORDER BY generated_at DESC LIMIT :limit"
                ),
                {"limit": int(limit)},
            ).mappings().all()
    except Exception:
        return {"frames": [], "phase": None}
    frames = [f for f in (_coerce(r["frame_json"]) for r in rows) if f]
    frames.reverse()
    phase = frames[-1].get("phase") if frames else None
    return {"frames": frames, "phase": phase}


@router.get("/frames/range")
async def frames_range(
    from_: str = Query(alias="from"),
    to: str = Query(...),
    max: int = Query(default=_DEFAULT_RANGE_MAX, ge=1, le=2000),
) -> dict[str, Any]:
    engine = _engine()
    if engine is None:
        return {"frames": []}
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT frame_json FROM substrate_brain_frame_log "
                    "WHERE generated_at >= :start AND generated_at <= :end "
                    "ORDER BY generated_at ASC"
                ),
                {"start": from_, "end": to},
            ).mappings().all()
    except Exception:
        return {"frames": []}
    frames = [f for f in (_coerce(r["frame_json"]) for r in rows) if f]
    if len(frames) > max:
        step = len(frames) / max
        frames = [frames[int(i * step)] for i in range(max)]
    return {"frames": frames}


@router.get("/window")
async def window() -> dict[str, Any]:
    engine = _engine()
    empty = {
        "earliest": None,
        "latest": None,
        "frame_count": 0,
        "phase": None,
        "server_now": datetime.now(timezone.utc).isoformat(),
    }
    if engine is None:
        return empty
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT min(generated_at) AS earliest, max(generated_at) AS latest, "
                    "count(*) AS n FROM substrate_brain_frame_log"
                )
            ).mappings().first()
            phase_row = conn.execute(
                text(
                    "SELECT phase FROM substrate_brain_frame_log "
                    "ORDER BY generated_at DESC LIMIT 1"
                )
            ).mappings().first()
    except Exception:
        return empty

    def _iso(v):
        return v.isoformat() if hasattr(v, "isoformat") else v

    return {
        "earliest": _iso(row["earliest"]) if row else None,
        "latest": _iso(row["latest"]) if row else None,
        "frame_count": int(row["n"]) if row else 0,
        "phase": (phase_row["phase"] if phase_row else None),
        "server_now": datetime.now(timezone.utc).isoformat(),
    }
```

- [ ] **Step 4: Register the router**

In `services/orion-hub/scripts/api_routes.py`, add the import after line 164 (`from .substrate_lattice_routes import router as substrate_lattice_router`):

```python
from .self_brain_routes import router as self_brain_router
```

And the include after line 177 (`router.include_router(substrate_lattice_router)`):

```python
router.include_router(self_brain_router)
```

- [ ] **Step 5: Run route tests to verify they pass**

Run: `pytest services/orion-hub/tests/test_self_brain_routes.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/scripts/self_brain_routes.py services/orion-hub/scripts/api_routes.py services/orion-hub/tests/test_self_brain_routes.py
git commit -m "feat(hub): read-only /api/self-brain tail/range/window endpoints"
```

---

## Task 9: Hub env parity for `POSTGRES_URI`

**Files:**
- Modify: `services/orion-hub/.env_example`

- [ ] **Step 1: Add `POSTGRES_URI` to `.env_example`**

In `services/orion-hub/.env_example`, near the other DB URLs (`DATABASE_URL` at line 377, `SUBSTRATE_FELT_STATE_DATABASE_URL` at line 351), add:

```bash
# Direct Postgres DSN for substrate_* panels (self-brain, observability, lattice). DB: conjourney.
POSTGRES_URI=postgresql://postgres:postgres@127.0.0.1:55432/conjourney
```

- [ ] **Step 2: Sync local `.env`**

Run: `python scripts/sync_local_env_from_example.py`
Expected: syncs `POSTGRES_URI` into `services/orion-hub/.env` (if the sync prefix list excludes it, add it manually and report it — check output). Verify `.env` is not staged:

Run: `git check-ignore services/orion-hub/.env && git status --short services/orion-hub/.env`
Expected: `git check-ignore` prints the path; `git status` shows nothing.

- [ ] **Step 3: Commit**

```bash
git add services/orion-hub/.env_example
git commit -m "chore(hub): declare POSTGRES_URI in .env_example for substrate panels"
```

---

## Task 10: Frontend — standalone brain page

**Files:**
- Create: `services/orion-hub/static/self-brain.html`
- Create: `services/orion-hub/static/js/self-brain.js`

Phase 1 view is **Hybrid (C) only**: fixed labeled anatomical zones (one per selected dimension's regions) laid out on a canvas with pulsing intensity, plus a dimension toggle rail, a per-region EKG strip, a staleness/held rendering, a warming banner, and a scrubber (`−retention → LIVE`). `API_BASE = ""` (same-origin, iframe-safe).

- [ ] **Step 1: Create `self-brain.html`**

Create `services/orion-hub/static/self-brain.html` (Tailwind CDN; script at end with plain path — no `?v=`):

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Self Brain</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .hatched { background-image: repeating-linear-gradient(45deg, rgba(148,163,184,.25) 0 4px, transparent 4px 8px); }
  </style>
</head>
<body class="bg-gray-950 text-gray-200 min-h-screen p-3">
  <div id="warmingBanner" class="hidden mb-2 px-3 py-1.5 rounded bg-amber-900/50 border border-amber-700 text-amber-200 text-xs">
    Substrate warming — graph has no real activation yet.
  </div>

  <div class="flex flex-wrap items-center gap-2 mb-2 text-xs">
    <span class="text-gray-400">Dimension:</span>
    <div id="dimRail" class="flex flex-wrap gap-1"></div>
    <span id="brainStatus" class="ml-auto text-gray-500"></span>
  </div>

  <div class="grid grid-cols-1 xl:grid-cols-3 gap-3">
    <div class="xl:col-span-2 rounded-xl border border-gray-800 bg-gray-900/40 p-2">
      <canvas id="brainCanvas" width="720" height="440" class="w-full"></canvas>
    </div>
    <div class="rounded-xl border border-gray-800 bg-gray-900/40 p-2">
      <div class="text-[10px] uppercase tracking-wide text-gray-500 mb-1">Region EKG (loaded window)</div>
      <canvas id="ekgCanvas" width="360" height="360" class="w-full"></canvas>
      <div id="ekgLegend" class="mt-2 text-[11px] space-y-0.5"></div>
    </div>
  </div>

  <div class="mt-3 rounded-xl border border-gray-800 bg-gray-900/40 p-2">
    <div class="flex items-center gap-2 text-xs">
      <button id="liveBtn" class="px-2 py-1 rounded border border-emerald-700 bg-emerald-900/40 text-emerald-200">● LIVE</button>
      <input id="scrubber" type="range" min="0" max="1000" value="1000" class="flex-1" />
      <span id="scrubLabel" class="text-gray-400 w-40 text-right">LIVE</span>
    </div>
  </div>

  <script src="/static/js/self-brain.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `self-brain.js`**

Create `services/orion-hub/static/js/self-brain.js`:

```javascript
"use strict";

const API_BASE = "";
const TAIL_POLL_MS = 3000;
const EKG_WINDOW = 120; // frames kept for realtime EKG

const DIMENSIONS = [
  { key: "node_kind", label: "Node kinds" },
  { key: "lane", label: "Lanes" },
  { key: "self_state", label: "Self-state" },
  { key: "spotlight", label: "Spotlight" },
];

const state = {
  dim: "node_kind",
  live: true,
  frames: [],        // ascending; realtime tail buffer or loaded range
  pollTimer: null,
  window: null,
};

function _get(path) {
  return fetch(API_BASE + path).then((r) => {
    if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
    return r.json();
  });
}

function setStatus(msg) {
  document.getElementById("brainStatus").textContent = msg;
}

function regionsFor(frame, dim) {
  if (!frame) return [];
  if (dim === "spotlight") return [];
  return (frame.regions || []).filter((r) => r.dimension === dim);
}

function stateColor(regionState, intensity) {
  if (regionState === "firing") return `rgba(248,113,113,${0.35 + 0.65 * intensity})`;
  if (regionState === "starving") return `rgba(71,85,105,${0.4 + 0.3 * intensity})`;
  return `rgba(96,165,250,${0.35 + 0.5 * intensity})`;
}

function drawBrain() {
  const canvas = document.getElementById("brainCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const frame = state.frames[state.frames.length - 1];
  if (!frame) { setStatus("no frames"); return; }

  const regions = regionsFor(frame, state.dim);
  // Fixed grid layout = stable, always-labeled anatomical zones.
  const cols = Math.ceil(Math.sqrt(Math.max(1, regions.length)));
  const rows = Math.ceil(regions.length / cols) || 1;
  const cw = canvas.width / cols;
  const chh = canvas.height / rows;

  regions.forEach((r, i) => {
    const cx = (i % cols) * cw + cw / 2;
    const cy = Math.floor(i / cols) * chh + chh / 2;
    const radius = Math.min(cw, chh) * (0.22 + 0.22 * r.intensity);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = stateColor(r.state, r.intensity);
    ctx.fill();
    if (r.stale) {
      ctx.strokeStyle = "rgba(148,163,184,.8)";
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = "#e5e7eb";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(r.label, cx, cy + radius + 12);
    const ageTxt = r.stale ? " (held)" : "";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText(`${(r.intensity * 100) | 0}%${ageTxt}`, cx, cy + 4);
  });

  // Spotlight overlay: dashed hull label if present + spotlight dim selected.
  if (state.dim === "spotlight" && frame.spotlight) {
    ctx.fillStyle = "#f0abfc";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(
      `Spotlight: ${frame.spotlight.attended_node_ids.length} nodes, dwell ${frame.spotlight.dwell_ticks}, stability ${(frame.spotlight.coalition_stability * 100 | 0)}%${frame.spotlight.stale ? " (held)" : ""}`,
      12, 24,
    );
    if (frame.spotlight.description) ctx.fillText(frame.spotlight.description, 12, 44);
  }
  setStatus(`${frame.phase} · tick ${frame.tick_seq} · ${regions.length} regions`);
}

function drawEkg() {
  const canvas = document.getElementById("ekgCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const legend = document.getElementById("ekgLegend");
  legend.innerHTML = "";
  if (state.dim === "spotlight") { legend.textContent = "Select a region dimension for EKG."; return; }

  // Build per-region series over the loaded window (regions are the stable series).
  const ids = new Set();
  state.frames.forEach((f) => regionsFor(f, state.dim).forEach((r) => ids.add(r.region_id)));
  const palette = ["#f87171", "#60a5fa", "#34d399", "#fbbf24", "#c084fc", "#22d3ee", "#f472b6", "#a3e635"];
  const idList = [...ids].slice(0, 8);
  const n = Math.max(1, state.frames.length - 1);

  idList.forEach((id, k) => {
    const color = palette[k % palette.length];
    ctx.beginPath();
    state.frames.forEach((f, xi) => {
      const r = regionsFor(f, state.dim).find((x) => x.region_id === id);
      const v = r ? r.intensity : 0;
      const x = (xi / n) * canvas.width;
      const y = canvas.height - v * (canvas.height - 8) - 4;
      if (xi === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    const label = id.split(":").slice(1).join(":") || id;
    legend.insertAdjacentHTML("beforeend", `<div><span style="color:${color}">■</span> ${label}</div>`);
  });
}

function render() { drawBrain(); drawEkg(); }

function pushTailFrames(frames) {
  if (!frames || !frames.length) return;
  const seen = new Set(state.frames.map((f) => f.frame_id));
  frames.forEach((f) => { if (!seen.has(f.frame_id)) state.frames.push(f); });
  if (state.frames.length > EKG_WINDOW) state.frames = state.frames.slice(-EKG_WINDOW);
}

async function pollTail() {
  if (!state.live) return;
  try {
    const data = await _get(`/api/self-brain/frames/tail?limit=30`);
    pushTailFrames(data.frames);
    toggleWarming(data.phase === "warming");
    render();
  } catch (e) { setStatus(`poll error: ${e.message}`); }
}

function toggleWarming(on) {
  document.getElementById("warmingBanner").classList.toggle("hidden", !on);
}

async function loadRange(fromIso, toIso) {
  try {
    const data = await _get(`/api/self-brain/frames/range?from=${encodeURIComponent(fromIso)}&to=${encodeURIComponent(toIso)}&max=240`);
    state.frames = data.frames || [];
    render();
  } catch (e) { setStatus(`range error: ${e.message}`); }
}

function goLive() {
  state.live = true;
  state.frames = [];
  document.getElementById("scrubber").value = 1000;
  document.getElementById("scrubLabel").textContent = "LIVE";
  document.getElementById("liveBtn").classList.add("border-emerald-700", "bg-emerald-900/40");
  pollTail();
}

function onScrub(e) {
  const frac = Number(e.target.value) / 1000;
  if (frac >= 0.999) { goLive(); return; }
  state.live = false;
  document.getElementById("liveBtn").classList.remove("border-emerald-700", "bg-emerald-900/40");
  const w = state.window;
  if (!w || !w.earliest || !w.latest) { setStatus("no window to scrub"); return; }
  const start = new Date(w.earliest).getTime();
  const end = new Date(w.latest).getTime();
  const center = new Date(start + frac * (end - start));
  const half = 5 * 60 * 1000; // 10-minute playback window
  const fromIso = new Date(center.getTime() - half).toISOString();
  const toIso = new Date(center.getTime() + half).toISOString();
  document.getElementById("scrubLabel").textContent = center.toLocaleTimeString();
  loadRange(fromIso, toIso);
}

function buildDimRail() {
  const rail = document.getElementById("dimRail");
  DIMENSIONS.forEach((d) => {
    const btn = document.createElement("button");
    btn.textContent = d.label;
    btn.dataset.key = d.key;
    btn.className = "px-2 py-0.5 rounded border border-gray-700 bg-gray-800 hover:bg-gray-700";
    btn.addEventListener("click", () => {
      state.dim = d.key;
      [...rail.children].forEach((c) => c.classList.toggle("bg-indigo-800", c.dataset.key === d.key));
      render();
    });
    rail.appendChild(btn);
  });
  rail.children[0].classList.add("bg-indigo-800");
}

async function init() {
  buildDimRail();
  document.getElementById("liveBtn").addEventListener("click", goLive);
  document.getElementById("scrubber").addEventListener("input", onScrub);
  try { state.window = await _get("/api/self-brain/window"); } catch (e) { /* empty ok */ }
  await pollTail();
  state.pollTimer = setInterval(pollTail, TAIL_POLL_MS);
}

document.addEventListener("DOMContentLoaded", init);
```

- [ ] **Step 3: Smoke the static files render (structural check)**

Run: `python -c "import pathlib; h=pathlib.Path('services/orion-hub/static/self-brain.html').read_text(); assert '/static/js/self-brain.js' in h and 'brainCanvas' in h; j=pathlib.Path('services/orion-hub/static/js/self-brain.js').read_text(); assert 'frames/tail' in j and 'API_BASE = \"\"' in j; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/static/self-brain.html services/orion-hub/static/js/self-brain.js
git commit -m "feat(hub): self-brain standalone page — hybrid brain, EKG, scrubber, staleness"
```

---

## Task 11: Iframe the brain page into the Self tab

**Files:**
- Modify: `services/orion-hub/templates/index.html` (`#self-observability` body, lines 1358-1388)

- [ ] **Step 1: Replace the 4-card grid with a slim iframe**

In `services/orion-hub/templates/index.html`, replace the grid block that starts at line 1358 (`<div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 items-stretch">`) and ends at its matching closing `</div>` at line 1388 with an iframe block (mirrors `#substrate-lattice` at 2836-2844). Keep the surrounding `<section>`, header, and `#selfObsStatus` line intact:

```html
        <div class="rounded-xl border border-gray-800 bg-gray-950/40 overflow-hidden flex-1 min-h-[40rem]">
          <iframe
            id="selfBrainFrame"
            src="/static/self-brain.html?v={{HUB_UI_ASSET_VERSION}}"
            class="w-full h-full border-0 min-h-[40rem]"
            loading="lazy"
            title="Self Brain"
          ></iframe>
        </div>
```

Leave the `<script src="/static/js/self_observability.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>` tag (line 3490) in place — it still owns Self-tab activation/hash logic. Its card-render `getElementById` calls now no-op safely.

- [ ] **Step 2: Verify the edit (structural check)**

Run: `python -c "import pathlib,re; t=pathlib.Path('services/orion-hub/templates/index.html').read_text(); assert 'selfBrainFrame' in t and '/static/self-brain.html?v={{HUB_UI_ASSET_VERSION}}' in t; assert 'md:grid-cols-2 xl:grid-cols-4' not in t.split('id=\"self-observability\"')[1][:1600]; print('ok')"`
Expected: prints `ok` (iframe present; old 4-card grid removed from the self section).

- [ ] **Step 3: Run the hub test suite (no regressions)**

Run: `pytest services/orion-hub/tests -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/templates/index.html
git commit -m "feat(hub): replace Self-tab cards with self-brain iframe"
```

---

## Task 12: Producer substance eval

**Files:**
- Create: `services/orion-substrate-runtime/evals/test_brain_frame_substance_eval.py`

This eval encodes acceptance checks §2 (anti empty-shell) and §1's heavy-tool-turn stimulus→response shape using a synthetic graph, so we catch empty-shell regressions without a live mesh.

- [ ] **Step 1: Write the eval**

Create `services/orion-substrate-runtime/evals/test_brain_frame_substance_eval.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from app.brain_frame_producer import assemble_brain_frame


def _node(node_id, kind, activation, pressure=0.0):
    return SimpleNamespace(
        node_id=node_id, node_kind=kind, label=f"{kind}:{node_id}",
        activation=activation, metadata={"dynamic_pressure": pressure},
    )


def _settings():
    return SimpleNamespace(
        brain_frame_sample_nodes=40, brain_frame_sample_edges=60,
        brain_frame_firing_threshold=0.5, brain_frame_starving_threshold=0.1,
        brain_frame_self_state_cadence_sec=30.0, brain_frame_spotlight_cadence_sec=30.0,
    )


def test_active_and_dormant_graph_yields_firing_and_starving_and_samples():
    """Acceptance §2: >=1 firing region, >=1 starving region, non-empty samples."""
    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    nodes = [
        _node("e1", "event", 0.9, 0.85),
        _node("t1", "tension", 0.8, 0.7),
        _node("c1", "concept", 0.01),
        _node("o1", "ontology_branch", 0.0),
    ]
    frame = assemble_brain_frame(
        nodes=nodes, edges=[],
        lane_health={"cursor_lag_by_reducer": {}, "pending_backlog_by_reducer": {}, "quarantine_by_reducer": {}},
        self_state=None, attention=None, settings=_settings(), now=now, tick_seq=1,
    )
    states = [r.state for r in frame.regions if r.dimension == "node_kind"]
    assert "firing" in states, "expected at least one firing node-kind region"
    assert "starving" in states, "expected at least one starving node-kind region"
    assert len(frame.nodes) >= 1, "node samples must be non-empty for an active graph"
    assert all(r.node_count >= 0 for r in frame.regions)


def test_heavy_tool_turn_shape_execution_lit_concept_dim():
    """Acceptance §1 shape: execution lane fires, concept node-kind stays dim."""
    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    nodes = [_node("c1", "concept", 0.05), _node("e1", "event", 0.2)]
    lane_health = {
        "cursor_lag_by_reducer": {"execution_trajectory": 2.0},
        "pending_backlog_by_reducer": {"execution_trajectory": 15},
        "quarantine_by_reducer": {},
    }
    frame = assemble_brain_frame(
        nodes=nodes, edges=[], lane_health=lane_health,
        self_state=None, attention=None, settings=_settings(), now=now, tick_seq=1,
    )
    lanes = {r.region_id: r for r in frame.regions if r.dimension == "lane"}
    kinds = {r.region_id: r for r in frame.regions if r.dimension == "node_kind"}
    assert lanes["lane:execution_trajectory"].state in {"firing", "steady"}
    assert kinds["node_kind:concept"].state == "starving"
```

- [ ] **Step 2: Run the eval**

Run: `cd services/orion-substrate-runtime && python -m pytest evals -q; cd ../..`
Expected: PASS (2 tests).

- [ ] **Step 3: Commit**

```bash
git add services/orion-substrate-runtime/evals/test_brain_frame_substance_eval.py
git commit -m "test(substrate): brain-frame substance eval (anti empty-shell, §1/§2)"
```

---

## Task 13: Live stimulus→response smoke (the real bar, §6)

This cannot be a unit test — it needs the running mesh. Document it as a runnable smoke and mark the result `UNVERIFIED` until executed.

**Files:**
- Create: `services/orion-substrate-runtime/evals/README_brain_frame_smoke.md`

- [ ] **Step 1: Write the smoke procedure**

Create `services/orion-substrate-runtime/evals/README_brain_frame_smoke.md`:

```markdown
# Brain-frame live stimulus→response smoke (acceptance §6)

Prereq: substrate-runtime + hub running; migration applied; flags on (Task 4).

1. Baseline: `curl -s http://localhost:<hub_port>/api/self-brain/frames/tail?limit=1 | jq '.frames[0].phase, (.frames[0].regions[] | select(.dimension=="lane"))'`
   Expect phase `live` (or `warming` at cold start).
2. Drive load: send a deep, tool-heavy chat turn through the normal chat path.
3. Within a few frames (~15s), re-poll tail and assert:
   - `lane:execution_trajectory` intensity rises / state → firing.
   - `self_state:execution_pressure` and `self_state:reasoning_pressure` intensity rise.
   - `node_kind:concept` stays dim (starving/steady).
4. Evidence to capture: two tail JSON snapshots (before/after) with `frame_id`s, and a screenshot of the Self tab brain with execution lit.
5. Verify a badly-lagged/absent lane renders `stale`/held, not jittering (acceptance §7).

Record PASS/FAIL + frame_ids in the PR report. Until run against the mesh: **UNVERIFIED**.
```

- [ ] **Step 2: Commit**

```bash
git add services/orion-substrate-runtime/evals/README_brain_frame_smoke.md
git commit -m "docs(substrate): brain-frame live stimulus-response smoke procedure"
```

---

## Task 14: Review, full gate, PR

- [ ] **Step 1: Run all touched test lanes**

```bash
pytest tests/test_substrate_brain_frame_bus_catalog.py tests/test_channel_prefix_guardrail.py -q
cd services/orion-substrate-runtime && python -m pytest tests evals -q; cd ../..
pytest services/orion-hub/tests/test_self_brain_routes.py -q
python scripts/sync_local_env_from_example.py
git diff --check
git status --short
```

Expected: all green; no `.env` staged; no whitespace errors.

- [ ] **Step 2: Docker config validation (substrate-runtime + hub)**

```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env -f services/orion-substrate-runtime/docker-compose.yml config >/dev/null && echo substrate-ok
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml config >/dev/null && echo hub-ok
```

Expected: `substrate-ok`, `hub-ok`.

- [ ] **Step 3: Run the code-review skill in a subagent**

Dispatch the `code-reviewer` subagent against the branch diff. Fix all material findings, re-run affected checks, and record findings in the PR report.

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin HEAD
```

Then produce the PR report using the AGENTS.md §18 template. Include:
- Restart commands (Step 5 below).
- Acceptance §6 marked `UNVERIFIED` until the live smoke runs.
- Eval-gap note: `services/orion-hub` has no `evals/` dir (route coverage is via `tests/`).

- [ ] **Step 5: Restart commands (for Juniper — do not run sudo yourself)**

```bash
# 1. Apply the migration (DB: conjourney)
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_brain_frame_v1.sql

# 2. Restart substrate-runtime (new loop + settings) and hub (new route + template)
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env -f services/orion-substrate-runtime/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build

# 3. Smoke
curl -fsS http://localhost:<hub_port>/api/self-brain/window | jq
```

---

## Acceptance check → task map

| Spec §6 acceptance check | Covered by |
|---|---|
| 1. Contract registered + channel + checks pass | Task 1 (both registry maps) + Task 2 (catalog + prefix tests) |
| 2. Producer substance (firing + starving + non-empty nodes) | Task 5 tests + Task 12 eval |
| 3. Cold start `phase="warming"` + banner | Task 5 `test_producer_warming_when_graph_dead` + Task 10 `toggleWarming` |
| 4. Store append / tail-ascending / range-downsample / prune | Task 6 tests |
| 5. Endpoints valid shape + degrade empty-200 | Task 8 tests |
| 6. Live stimulus→response (real bar) | Task 13 smoke (UNVERIFIED until run) |
| 7. Staleness honesty (held, not movement) | Task 5 `test_self_state_region_marked_stale_when_old` + Task 10 hatched/held render |
| 8. Frontend iframe + rail + EKG + scrubber interaction | Task 10 + Task 11 (+ manual interaction in smoke) |
| 9. Env parity (new keys + sync) | Task 4 + Task 9 |

## Non-goals (from spec — do NOT implement in Phase 1)

- Organic constellation (B) view, `lattice_layer` dimension, emergent-cluster/PCA `GET /clusters` — all Phase 2.
- SSE/WS streaming (polling only), per-node continuity/history warehouse, action levers.
- No exposure of private/blocked node payloads — samples use kind + activation + display label only (enforced by `BrainNodeSampleV1` shape).

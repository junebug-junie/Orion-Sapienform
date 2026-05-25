# Node-Scoped Biometrics Grammar Ingress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire existing `orion-biometrics` per-node sample/summary/induction into the grammar substrate as node-scoped `GrammarEventV1` traces on `orion:grammar:event`, with canonical node identity (atlas/athena/circe/prometheus) and capability surfaces—without a new organ or mesh-average reducer.

**Architecture:** After each `publish_metrics` tick, resolve `sample.node` through a YAML node catalog, build one trace per node (`biometrics.node:{node_id}:{timestamp}`) with five atoms and six edges, and publish `grammar.event.v1` envelopes alongside existing biometrics channels. Grammar publish failures are logged and swallowed so telemetry never breaks.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2 (`orion/schemas/grammar.py`, `orion/schemas/telemetry/biometrics.py`), Redis bus (`OrionBusAsync`, `BaseEnvelope`), PyYAML 6.0.2, pytest 8.3.x.

**Design source:** User handoff “Node-Scoped Biometrics Grammar Ingress” (2026-05-24).

**Non-goals (do not implement):** New organ, new pressure formulas, cluster-as-grammar-source, mesh-average traces, UI, signal-registry redesign (`orion/signals/registry.py`).

---

## Worktree isolation (mandatory)

All implementation commits happen **only** in a dedicated worktree. Do not checkout the feature branch in the main workspace.

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin main
git worktree add .worktrees/feat-biometrics-node-grammar-ingress \
  -b feat/biometrics-node-grammar-ingress origin/main
cd .worktrees/feat-biometrics-node-grammar-ingress
git check-ignore -q .worktrees   # must succeed
```

**Rules:**
- Never bleed changed files back to the main checkout except copying `.env` keys locally (see below).
- When editing `services/orion-biometrics/.env_example`, also copy new keys into `services/orion-biometrics/.env` in the **worktree** (gitignored; operator sync).
- PR and push from `feat/biometrics-node-grammar-ingress` only.

---

## Preflight findings (2026-05-24)

| Question | Finding |
|----------|---------|
| Biometrics publish path | `services/orion-biometrics/app/main.py` → `publish_metrics()` |
| Grammar schemas | `orion/schemas/grammar.py` — closed `AtomType` / `RelationType` literals |
| Grammar bus channel | `orion:grammar:event` exists in `orion/bus/channels.yaml`; producers today: vision services + hub — **add `orion-biometrics`** |
| Registry | `GrammarEventV1` already in `orion/schemas/registry.py` `_REGISTRY` — **no registry change required** |
| sql-writer | Already routes `grammar.event.v1` → `GrammarEventSQL` |
| PyYAML in biometrics | **Not** in `requirements.txt` today — add `PyYAML==6.0.2` |
| Tests dir | **None** under `services/orion-biometrics/` — create `tests/` |
| Dockerfile | Copies `app/` + `orion/` only — must COPY `config/biometrics` + `tests/` |
| `collect_biometrics()` node | Uses `settings.NODE_NAME` in `app/metrics.py` |

---

## File structure

| Path | Responsibility |
|------|----------------|
| `config/biometrics/node_catalog.yaml` | Canonical node IDs, aliases, roles, capabilities |
| `services/orion-biometrics/app/node_catalog.py` | Load YAML, `resolve(raw_node) → NodeProfile` |
| `services/orion-biometrics/app/grammar_emit.py` | `build_biometrics_node_grammar_events(...) → list[GrammarEventV1]` |
| `services/orion-biometrics/app/main.py` | Wire grammar publish after sample/summary/induction |
| `services/orion-biometrics/app/settings.py` | `PUBLISH_BIOMETRICS_GRAMMAR`, channel, catalog path |
| `services/orion-biometrics/.env_example` | New env keys |
| `services/orion-biometrics/.env` | Operator sync (not committed) |
| `services/orion-biometrics/docker-compose.yml` | Env passthrough + config volume |
| `services/orion-biometrics/Dockerfile` | COPY config + tests |
| `services/orion-biometrics/requirements.txt` | PyYAML + pytest |
| `services/orion-biometrics/README.md` | Grammar channel docs |
| `services/orion-biometrics/tests/test_node_catalog.py` | Catalog resolution tests |
| `services/orion-biometrics/tests/test_biometrics_grammar_emit.py` | Trace shape + schema literal tests |
| `orion/bus/channels.yaml` | Add `orion-biometrics` to `orion:grammar:event` producers |
| `scripts/smoke_biometrics_grammar.sh` | Bus tap for grammar events |

**Do not modify:** `publish_cluster()`, `BiometricsClusterV1` grammar path, `orion/signals/registry.py`.

---

# Phase 1 — Node catalog

### Task 1: YAML node catalog

**Files:**
- Create: `config/biometrics/node_catalog.yaml`

- [ ] **Step 1: Create catalog file**

```yaml
schema_version: biometrics_node_catalog.v1

defaults:
  expected_online: true
  role: unknown
  capabilities: {}

nodes:
  atlas:
    aliases:
      - atlas
      - atlas.tail348bbe.ts.net
    role: inference_gpu
    expected_online: true
    capabilities:
      local_llm_heavy: true
      local_llm_quick: true
      embedding: true
      batch_inference: true
      graphdb: false
      postgres: false
      hub: false
      monitoring: false

  athena:
    aliases:
      - athena
      - athena.tail348bbe.ts.net
    role: orchestration
    expected_online: true
    capabilities:
      hub: true
      redis_bus: true
      graphdb: true
      postgres: true
      orchestration: true
      local_llm_heavy: false
      local_llm_quick: false
      monitoring: false

  circe:
    aliases:
      - circe
      - circe.tail348bbe.ts.net
    role: burst_gpu
    expected_online: false
    capabilities:
      local_llm_heavy: true
      local_llm_quick: true
      training: true
      batch_inference: true
      dream_batch: true
      graphdb: false
      postgres: false
      hub: false

  prometheus:
    aliases:
      - prometheus
      - prometheous
      - prometheus.tail348bbe.ts.net
      - prometheous.tail348bbe.ts.net
    role: observability
    expected_online: true
    capabilities:
      monitoring: true
      logs: true
      metrics: true
      orchestration: false
      local_llm_heavy: false
```

- [ ] **Step 2: Commit**

```bash
git add config/biometrics/node_catalog.yaml
git commit -m "feat(biometrics): add node catalog for grammar ingress"
```

---

### Task 2: Node catalog module + tests

**Files:**
- Create: `services/orion-biometrics/app/node_catalog.py`
- Create: `services/orion-biometrics/tests/test_node_catalog.py`
- Modify: `services/orion-biometrics/requirements.txt`

- [ ] **Step 1: Add PyYAML and pytest to requirements**

Append to `services/orion-biometrics/requirements.txt`:

```text
PyYAML==6.0.2
pytest==8.3.4
```

- [ ] **Step 2: Write failing tests**

```python
# services/orion-biometrics/tests/test_node_catalog.py
from __future__ import annotations

from pathlib import Path

import pytest

from app.node_catalog import NodeCatalog

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"


@pytest.fixture
def catalog() -> NodeCatalog:
    return NodeCatalog.load(CATALOG_PATH)


def test_resolves_atlas_alias(catalog: NodeCatalog) -> None:
    p = catalog.resolve("atlas.tail348bbe.ts.net")
    assert p.node_id == "atlas"
    assert p.role == "inference_gpu"
    assert p.capabilities["local_llm_heavy"] is True
    assert p.known is True
    assert p.raw_node == "atlas.tail348bbe.ts.net"


def test_resolves_prometheus_typo(catalog: NodeCatalog) -> None:
    p = catalog.resolve("prometheous")
    assert p.node_id == "prometheus"
    assert p.role == "observability"
    assert p.known is True


def test_unknown_node_gets_fallback(catalog: NodeCatalog) -> None:
    p = catalog.resolve("weirdbox")
    assert p.node_id == "weirdbox"
    assert p.known is False
    assert p.role == "unknown"


def test_circe_expected_offline(catalog: NodeCatalog) -> None:
    p = catalog.resolve("circe")
    assert p.expected_online is False
    assert p.node_id == "circe"
```

- [ ] **Step 3: Run tests — expect FAIL**

```bash
cd .worktrees/feat-biometrics-node-grammar-ingress
PYTHONPATH="services/orion-biometrics:." pytest services/orion-biometrics/tests/test_node_catalog.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.node_catalog'`

- [ ] **Step 4: Implement `node_catalog.py`**

```python
# services/orion-biometrics/app/node_catalog.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class NodeProfile:
    node_id: str
    raw_node: str | None
    role: str
    expected_online: bool
    capabilities: dict[str, bool] = field(default_factory=dict)
    known: bool = True


class NodeCatalog:
    def __init__(
        self,
        profiles: dict[str, NodeProfile],
        aliases: dict[str, str],
        defaults: dict[str, Any],
    ) -> None:
        self.profiles = profiles
        self.aliases = aliases
        self.defaults = defaults

    @classmethod
    def load(cls, path: str | Path) -> "NodeCatalog":
        data = yaml.safe_load(Path(path).read_text()) or {}
        defaults = data.get("defaults") or {}
        profiles: dict[str, NodeProfile] = {}
        aliases: dict[str, str] = {}

        for node_id, spec in (data.get("nodes") or {}).items():
            canonical = str(node_id).strip().lower()
            role = str(spec.get("role") or defaults.get("role") or "unknown")
            expected_online = bool(
                spec.get("expected_online", defaults.get("expected_online", True))
            )
            capabilities = {
                str(k): bool(v) for k, v in (spec.get("capabilities") or {}).items()
            }

            profiles[canonical] = NodeProfile(
                node_id=canonical,
                raw_node=None,
                role=role,
                expected_online=expected_online,
                capabilities=capabilities,
                known=True,
            )

            aliases[canonical] = canonical
            for alias in spec.get("aliases") or []:
                aliases[str(alias).strip().lower()] = canonical

        return cls(profiles=profiles, aliases=aliases, defaults=defaults)

    def resolve(self, raw_node: str | None) -> NodeProfile:
        raw = str(raw_node or "").strip()
        key = raw.lower()
        canonical = self.aliases.get(key)

        if canonical and canonical in self.profiles:
            base = self.profiles[canonical]
            return NodeProfile(
                node_id=base.node_id,
                raw_node=raw or None,
                role=base.role,
                expected_online=base.expected_online,
                capabilities=dict(base.capabilities),
                known=True,
            )

        fallback_id = key or "unknown"
        return NodeProfile(
            node_id=fallback_id,
            raw_node=raw or None,
            role=str(self.defaults.get("role") or "unknown"),
            expected_online=bool(self.defaults.get("expected_online", True)),
            capabilities={
                str(k): bool(v)
                for k, v in (self.defaults.get("capabilities") or {}).items()
            },
            known=False,
        )
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
PYTHONPATH="services/orion-biometrics:." pytest services/orion-biometrics/tests/test_node_catalog.py -v
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add services/orion-biometrics/app/node_catalog.py \
  services/orion-biometrics/tests/test_node_catalog.py \
  services/orion-biometrics/requirements.txt
git commit -m "feat(biometrics): node catalog resolver with tests"
```

---

# Phase 2 — Grammar emitter

### Task 3: Grammar emitter + tests

**Files:**
- Create: `services/orion-biometrics/app/grammar_emit.py`
- Create: `services/orion-biometrics/tests/test_biometrics_grammar_emit.py`

- [ ] **Step 1: Write failing tests**

```python
# services/orion-biometrics/tests/test_biometrics_grammar_emit.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

import pytest

from app.grammar_emit import build_biometrics_node_grammar_events
from app.node_catalog import NodeCatalog
from orion.schemas.grammar import AtomType, RelationType
from orion.schemas.telemetry.biometrics import (
    BiometricsInductionMetricV1,
    BiometricsInductionV1,
    BiometricsSampleV1,
    BiometricsSummaryV1,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"
FIXED_TS = datetime(2026, 5, 24, 20, 6, 18, 624380, tzinfo=timezone.utc)


@pytest.fixture
def catalog() -> NodeCatalog:
    return NodeCatalog.load(CATALOG_PATH)


def _fixtures(node: str, *, strain: float = 0.42):
    sample = BiometricsSampleV1(timestamp=FIXED_TS, node=node, cpu={"util": 0.1})
    summary = BiometricsSummaryV1(
        timestamp=FIXED_TS,
        node=node,
        composites={"strain": strain},
        telemetry_error_rate=0.0,
    )
    induction = BiometricsInductionV1(
        timestamp=FIXED_TS,
        node=node,
        metrics={
            "cpu": BiometricsInductionMetricV1(
                level=0.5, trend=0.5, volatility=0.1, spike_rate=0.0
            )
        },
    )
    return sample, summary, induction


def test_builds_node_scoped_trace_for_atlas(catalog: NodeCatalog) -> None:
    sample, summary, induction = _fixtures("atlas")
    profile = catalog.resolve("atlas")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
        code_version="0.1.0",
    )
    assert events
    assert all(e.trace_id.startswith("biometrics.node:atlas:") for e in events)
    atoms = [e.atom for e in events if e.atom]
    node_context = next(a for a in atoms if a.semantic_role == "node_context")
    assert node_context.text_value == "atlas"
    assert "capability" in node_context.dimensions


def test_uses_allowed_atom_types_only(catalog: NodeCatalog) -> None:
    sample, summary, induction = _fixtures("atlas")
    profile = catalog.resolve("atlas")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    allowed = set(get_args(AtomType))
    for event in events:
        if event.atom:
            assert event.atom.atom_type in allowed


def test_uses_allowed_relation_types_only(catalog: NodeCatalog) -> None:
    sample, summary, induction = _fixtures("atlas")
    profile = catalog.resolve("atlas")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    allowed = set(get_args(RelationType))
    for event in events:
        if event.edge:
            assert event.edge.relation_type in allowed


def test_athena_capability_surface_mentions_graphdb_not_heavy_llm(
    catalog: NodeCatalog,
) -> None:
    sample, summary, induction = _fixtures("athena")
    profile = catalog.resolve("athena")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    cap = next(
        e.atom
        for e in events
        if e.atom and e.atom.semantic_role == "capability_surface"
    )
    assert "graphdb" in cap.summary
    assert "local_llm_heavy" not in cap.summary


def test_trace_has_start_atoms_edges_end(catalog: NodeCatalog) -> None:
    sample, summary, induction = _fixtures("prometheous")
    profile = catalog.resolve("prometheous")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    kinds = [e.event_kind for e in events]
    assert kinds[0] == "trace_started"
    assert "atom_emitted" in kinds
    assert "edge_emitted" in kinds
    assert kinds[-1] == "trace_ended"
    assert events[0].trace_id.startswith("biometrics.node:prometheus:")


def test_circe_node_availability_reflects_expected_offline(catalog: NodeCatalog) -> None:
    sample, summary, induction = _fixtures("circe")
    profile = catalog.resolve("circe")
    events = build_biometrics_node_grammar_events(
        sample=sample,
        summary=summary,
        induction=induction,
        node_profile=profile,
        source_channel="orion:biometrics:induction",
    )
    avail = next(
        e.atom
        for e in events
        if e.atom and e.atom.semantic_role == "node_availability"
    )
    assert "expected offline" in avail.summary.lower()
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
PYTHONPATH="services/orion-biometrics:." pytest services/orion-biometrics/tests/test_biometrics_grammar_emit.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.grammar_emit'`

- [ ] **Step 3: Implement `grammar_emit.py`**

Create `services/orion-biometrics/app/grammar_emit.py` with the handoff skeleton, plus `_availability_summary`:

```python
def _availability_summary(
    node_id: str,
    node_profile: NodeProfile,
    summary: BiometricsSummaryV1,
) -> str:
    if summary.telemetry_error_rate and summary.telemetry_error_rate > 0.1:
        status = "DEGRADED"
    else:
        status = "OK"
    online = "expected online" if node_profile.expected_online else "expected offline"
    known = "known node" if node_profile.known else "unknown node"
    return f"{node_id} telemetry status {status} ({online}, {known})"
```

Use that for `node_availability` atom `summary` (not arbitrary payload blobs). Keep all `atom_type` / `relation_type` values within `orion/schemas/grammar.py` literals. Full module body matches handoff §2 emitter skeleton (trace_id, five atoms, six edges, trace_started → atoms → edges → trace_ended).

- [ ] **Step 4: Run tests — expect PASS**

```bash
PYTHONPATH="services/orion-biometrics:." pytest services/orion-biometrics/tests/test_biometrics_grammar_emit.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add services/orion-biometrics/app/grammar_emit.py \
  services/orion-biometrics/tests/test_biometrics_grammar_emit.py
git commit -m "feat(biometrics): emit node-scoped grammar traces from biometrics"
```

---

# Phase 3 — Settings, Docker, bus catalog

### Task 4: Settings and env

**Files:**
- Modify: `services/orion-biometrics/app/settings.py`
- Modify: `services/orion-biometrics/.env_example`
- Modify: `services/orion-biometrics/.env` (worktree only, not committed)

- [ ] **Step 1: Add settings fields**

After biometrics channel fields in `settings.py`:

```python
    PUBLISH_BIOMETRICS_GRAMMAR: bool = Field(default=True)
    GRAMMAR_EVENT_CHANNEL: str = Field(default="orion:grammar:event")
    NODE_CATALOG_PATH: str = Field(
        default="/app/config/biometrics/node_catalog.yaml"
    )
```

- [ ] **Step 2: Update `.env_example`**

```env
PUBLISH_BIOMETRICS_GRAMMAR=true
GRAMMAR_EVENT_CHANNEL=orion:grammar:event
NODE_CATALOG_PATH=/app/config/biometrics/node_catalog.yaml
```

- [ ] **Step 3: Sync `.env` in worktree**

```bash
grep -q PUBLISH_BIOMETRICS_GRAMMAR services/orion-biometrics/.env 2>/dev/null || \
  cat >> services/orion-biometrics/.env <<'EOF'
PUBLISH_BIOMETRICS_GRAMMAR=true
GRAMMAR_EVENT_CHANNEL=orion:grammar:event
NODE_CATALOG_PATH=/app/config/biometrics/node_catalog.yaml
EOF
```

- [ ] **Step 4: Commit**

```bash
git add services/orion-biometrics/app/settings.py services/orion-biometrics/.env_example
git commit -m "feat(biometrics): settings for grammar publish and node catalog path"
```

---

### Task 5: Dockerfile and docker-compose

**Files:**
- Modify: `services/orion-biometrics/Dockerfile`
- Modify: `services/orion-biometrics/docker-compose.yml`

- [ ] **Step 1: Dockerfile — COPY config and tests**

After `COPY orion /app/orion` add:

```dockerfile
COPY config/biometrics /app/config/biometrics
COPY services/orion-biometrics/tests ./tests
```

(Build context remains repo root `../..` per existing compose.)

- [ ] **Step 2: docker-compose — env + volume**

Under `environment:` add:

```yaml
      - PUBLISH_BIOMETRICS_GRAMMAR=${PUBLISH_BIOMETRICS_GRAMMAR:-true}
      - GRAMMAR_EVENT_CHANNEL=${GRAMMAR_EVENT_CHANNEL:-orion:grammar:event}
      - NODE_CATALOG_PATH=${NODE_CATALOG_PATH:-/app/config/biometrics/node_catalog.yaml}
```

Under `volumes:` add (allows live catalog edits without rebuild):

```yaml
      - ../../config/biometrics:/app/config/biometrics:ro
```

- [ ] **Step 3: Commit**

```bash
git add services/orion-biometrics/Dockerfile services/orion-biometrics/docker-compose.yml
git commit -m "chore(biometrics): mount node catalog and pass grammar env to container"
```

---

### Task 6: Bus channel producer

**Files:**
- Modify: `orion/bus/channels.yaml` (line ~1793 `producer_services` for `orion:grammar:event`)

- [ ] **Step 1: Add biometrics producer**

Change:

```yaml
    producer_services: ["orion-vision-retina", "orion-hub", "orion-vision-edge", "orion-vision-window"]
```

To:

```yaml
    producer_services:
      - orion-vision-retina
      - orion-hub
      - orion-vision-edge
      - orion-vision-window
      - orion-biometrics
```

- [ ] **Step 2: Commit**

```bash
git add orion/bus/channels.yaml
git commit -m "chore(bus): register orion-biometrics as grammar event producer"
```

---

# Phase 4 — Wire main.py

### Task 7: Publish grammar from `publish_metrics`

**Files:**
- Modify: `services/orion-biometrics/app/main.py`

- [ ] **Step 1: Add imports and catalog init**

After `_pipeline = BiometricsPipeline(...)` block:

```python
from app.node_catalog import NodeCatalog
from app.grammar_emit import build_biometrics_node_grammar_events

_NODE_CATALOG = NodeCatalog.load(settings.NODE_CATALOG_PATH)
```

- [ ] **Step 2: Add grammar publish block**

Inside `publish_metrics`, after induction publish and before `logger.debug`:

```python
        if settings.PUBLISH_BIOMETRICS_GRAMMAR:
            try:
                node_profile = _NODE_CATALOG.resolve(
                    sample.node or summary.node or induction.node
                )
                grammar_events = build_biometrics_node_grammar_events(
                    sample=sample,
                    summary=summary,
                    induction=induction,
                    node_profile=node_profile,
                    source_channel=settings.BIOMETRICS_INDUCTION_CHANNEL,
                    code_version=settings.SERVICE_VERSION,
                )
                for event in grammar_events:
                    await _publish(
                        bus,
                        settings.GRAMMAR_EVENT_CHANNEL,
                        "grammar.event.v1",
                        event,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to publish biometrics grammar events: %s",
                    exc,
                    exc_info=True,
                )
```

- [ ] **Step 3: Extend `/health` response (optional but useful)**

Add to health dict:

```python
        "grammar_event_channel": settings.GRAMMAR_EVENT_CHANNEL,
        "publish_biometrics_grammar": settings.PUBLISH_BIOMETRICS_GRAMMAR,
```

- [ ] **Step 4: Run full unit suite**

```bash
PYTHONPATH="services/orion-biometrics:." pytest services/orion-biometrics/tests/ -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-biometrics/app/main.py
git commit -m "feat(biometrics): publish node-scoped grammar events on biometrics tick"
```

---

# Phase 5 — Docs and smoke

### Task 8: README

**Files:**
- Modify: `services/orion-biometrics/README.md`

- [ ] **Step 1: Document grammar channel**

Add row to Published Channels table:

| `orion:grammar:event` | `GRAMMAR_EVENT_CHANNEL` | `grammar.event.v1` | Node-scoped grammar trace (one trace per observed node per tick). |

Add env rows for `PUBLISH_BIOMETRICS_GRAMMAR`, `GRAMMAR_EVENT_CHANNEL`, `NODE_CATALOG_PATH`.

Note: catalog lives at `config/biometrics/node_catalog.yaml`; aliases canonicalize e.g. `prometheous` → `prometheus`.

- [ ] **Step 2: Commit**

```bash
git add services/orion-biometrics/README.md
git commit -m "docs(biometrics): document grammar ingress channel and settings"
```

---

### Task 9: Smoke script

**Files:**
- Create: `scripts/smoke_biometrics_grammar.sh`

- [ ] **Step 1: Create smoke script**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== unit tests (host) =="
PYTHONPATH="services/orion-biometrics:." pytest \
  services/orion-biometrics/tests/test_node_catalog.py \
  services/orion-biometrics/tests/test_biometrics_grammar_emit.py -q

echo "== optional: container tests =="
if docker compose -f services/orion-biometrics/docker-compose.yml ps -q biometrics 2>/dev/null | grep -q .; then
  docker compose -f services/orion-biometrics/docker-compose.yml exec -T biometrics \
    pytest tests/test_node_catalog.py tests/test_biometrics_grammar_emit.py -q
fi

echo "== bus tap (manual, 30s) =="
echo "Run: redis-cli SUBSCRIBE orion:grammar:event"
echo "Expect: schema_version grammar_event.v1, trace_id biometrics.node:<node>:..., provenance.source_service orion-biometrics"
```

```bash
chmod +x scripts/smoke_biometrics_grammar.sh
```

- [ ] **Step 2: Commit**

```bash
git add scripts/smoke_biometrics_grammar.sh
git commit -m "test(biometrics): smoke script for grammar ingress unit tests"
```

---

# Phase 6 — Verification, code review, PR

### Task 10: Runtime verification (worktree)

- [ ] **Step 1: Rebuild and restart biometrics**

```bash
cd services/orion-biometrics
docker compose build biometrics
docker compose up -d biometrics
```

- [ ] **Step 2: Subscribe and wait one telemetry interval**

```bash
redis-cli SUBSCRIBE orion:grammar:event
```

Verify within ~30s (per `TELEMETRY_INTERVAL`):

- `kind` = `grammar.event.v1`
- `payload.schema_version` = `grammar_event.v1`
- `payload.trace_id` starts with `biometrics.node:{NODE_NAME}:`
- `payload.provenance.source_service` = `orion-biometrics`
- Atom semantic roles include `node_context`, `body_state`, `capability_surface`

If verification cannot run (no Redis/GPU host), status for PR body: **UNVERIFIED** with blocker named.

---

### Task 11: Code review subagent

**REQUIRED SUB-SKILL:** `requesting-code-review` — dispatch code-reviewer (or generalPurpose with review checklist) against the full diff `origin/main...HEAD`.

- [ ] **Step 1: Run review subagent**

Prompt the reviewer with:
- Handoff acceptance criteria (10 items)
- Files changed list
- Require fixes for: schema literal violations, mesh-average grammar, grammar failure breaking telemetry, missing tests

- [ ] **Step 2: Fix all reported issues**

Implement fixes in worktree; re-run:

```bash
PYTHONPATH="services/orion-biometrics:." pytest services/orion-biometrics/tests/ -v
```

- [ ] **Step 3: Commit review fixes**

```bash
git commit -m "fix(biometrics): address grammar ingress code review"
```

---

### Task 12: PR report and GitHub PR

**Files:**
- Create: `docs/superpowers/pr-reports/2026-05-24-node-scoped-biometrics-grammar-ingress-pr.md`

- [ ] **Step 1: Write PR report markdown**

Template:

```markdown
# PR: Node-scoped biometrics grammar ingress

**Branch:** `feat/biometrics-node-grammar-ingress`
**Base:** `main`
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-biometrics-node-grammar-ingress`

## Summary
- Publishes per-node `GrammarEventV1` traces on `orion:grammar:event` from existing biometrics sample/summary/induction (not cluster aggregate).
- YAML node catalog canonicalizes aliases (e.g. prometheous → prometheus) and attaches role/capability/expected_online context.
- Grammar publish failures are non-fatal to biometrics telemetry.

## Architecture
collect_biometrics → pipeline → sample/summary/induction bus publish → NodeCatalog.resolve → build_biometrics_node_grammar_events → orion:grammar:event

## Test plan
- [ ] pytest node_catalog + grammar_emit (host)
- [ ] redis SUBSCRIBE shows biometrics.node traces
- [ ] sql-writer ingests grammar events (if sql-writer running)

## Verification evidence
(paste pytest + redis sample or UNVERIFIED + blocker)
```

- [ ] **Step 2: Push and open PR**

```bash
git push -u origin feat/biometrics-node-grammar-ingress
gh pr create --base main --title "feat(biometrics): node-scoped grammar ingress" --body "$(cat docs/superpowers/pr-reports/2026-05-24-node-scoped-biometrics-grammar-ingress-pr.md)"
```

Return PR URL to operator.

---

## Acceptance criteria mapping

| # | Criterion | Task |
|---|-----------|------|
| 1 | Atlas trace `biometrics.node:atlas:...` | Task 3, 7, 10 |
| 2 | Separate Athena trace | Task 3 (per-node trace_id), multi-node via NODE_NAME per deployment |
| 3 | Circe `expected_online=false` | Task 1–2, test `test_circe_expected_offline` |
| 4 | prometheous → prometheus | Task 1–2, test `test_resolves_prometheus_typo` |
| 5 | Valid GrammarEventV1 fields only | Task 3 schema literal tests |
| 6 | payload_ref not blob stuffing | Task 3 emitter |
| 7 | Grammar failure non-fatal | Task 7 try/except |
| 8 | No new organ | Scope |
| 9 | No cluster grammar source | Non-goal |
| 10 | Distinct capability surfaces | Task 3 athena test |

---

## Self-review (plan author checklist)

| Check | Status |
|-------|--------|
| Spec coverage — all 10 acceptance criteria mapped | OK |
| Placeholder scan — no TBD/TODO/similar-to | OK |
| Type consistency — AtomType/RelationType from `grammar.py` | OK |
| PyYAML added; Dockerfile copies config | OK |
| Registry unchanged (already has GrammarEventV1) | OK |
| Worktree + .env sync rules documented | OK |

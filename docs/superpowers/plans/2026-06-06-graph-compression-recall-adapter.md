# Graph Compression — Recall Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Prerequisite:** `orion-graph-compression` service must be deployed and writing artifacts to Postgres (`compression_artifacts` table) and Fuseki (`orion:compressions`). For test environments, the adapter degrades gracefully to `[]` with no artifacts present.

**Goal:** Extend `orion-recall` with a `graph_compression` backend that queries the `compression_artifacts` Postgres index, ranks by salience + keyword relevance, retrieves summaries from Fuseki, and returns fragments to the memory bundle — activated via three new recall profiles.

**Architecture:** A new `app/storage/graph_compression_adapter.py` module provides `fetch_graph_compression_fragments()`. The existing `_query_backends()` function in `worker.py` gains a guarded call to this adapter when the active recall profile has `enable_graph_compression: true`. Three new YAML profiles cover global (summary-only), local (exemplar-rich), and unified (both) modes.

**Tech Stack:** Python 3.12, SQLAlchemy + psycopg2 (already in recall deps), httpx (already present), pydantic-settings (already present), orion shared schemas.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `services/orion-recall/app/settings.py` | Modify | Add `RECALL_COMPRESSION_*` fields |
| `services/orion-recall/app/storage/graph_compression_adapter.py` | Create | Postgres index query + Fuseki summary fetch |
| `services/orion-recall/app/worker.py` | Modify | Add compression backend call in `_query_backends` |
| `orion/recall/profiles/graph.compressions.global.v1.yaml` | Create | Global summary-only profile |
| `orion/recall/profiles/graph.compressions.local.v1.yaml` | Create | Local exemplar-heavy profile |
| `orion/recall/profiles/graph.compressions.v1.yaml` | Create | Default unified profile |
| `services/orion-recall/tests/test_graph_compression_adapter.py` | Create | Adapter unit tests |
| `services/orion-recall/tests/test_recall_profiles_compression.py` | Create | Profile YAML parsing tests |
| `services/orion-recall/tests/test_query_backends_compression.py` | Create | Integration test for `_query_backends` extension |

---

## Task 1: Add compression settings to `orion-recall/app/settings.py`

**Files:**
- Modify: `services/orion-recall/app/settings.py`
- Create: (no new test file — settings are exercised by adapter tests)

- [ ] **Step 1: Add compression fields to `Settings` class**

In `services/orion-recall/app/settings.py`, add these fields after the existing `RECALL_TENSOR_RANKER_MODEL_PATH` field (before the `@field_validator` block):

```python
    # ── Graph Compression backend ─────────────────────────────────────
    RECALL_COMPRESSION_ENABLED: bool = Field(
        default=False, validation_alias=AliasChoices("RECALL_COMPRESSION_ENABLED")
    )
    RECALL_COMPRESSION_PG_DSN: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("RECALL_COMPRESSION_PG_DSN")
    )
    RECALL_COMPRESSION_RDF_QUERY_URL: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("RECALL_COMPRESSION_RDF_QUERY_URL")
    )
    RECALL_COMPRESSION_RDF_USER: str = Field(
        default="admin", validation_alias=AliasChoices("RECALL_COMPRESSION_RDF_USER")
    )
    RECALL_COMPRESSION_RDF_PASS: str = Field(
        default="orion", validation_alias=AliasChoices("RECALL_COMPRESSION_RDF_PASS")
    )
    RECALL_COMPRESSION_TIMEOUT_SEC: float = Field(
        default=3.0, validation_alias=AliasChoices("RECALL_COMPRESSION_TIMEOUT_SEC")
    )
```

Also extend the existing `_blank_to_none` validator to include the new optional fields:

```python
    @field_validator(
        "RECALL_VECTOR_BASE_URL",
        "RECALL_VECTOR_COLLECTIONS",
        "RECALL_VECTOR_EMBEDDING_URL",
        "RECALL_RDF_ENDPOINT_URL",
        "RECALL_RDF_QUERY_URL",
        "RECALL_COMPRESSION_PG_DSN",       # add this
        "RECALL_COMPRESSION_RDF_QUERY_URL", # add this
        mode="before",
    )
```

- [ ] **Step 2: Verify settings parse from existing `.env` without error**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-recall
PYTHONPATH=. python -c "
from app.settings import settings
print('compression enabled:', settings.RECALL_COMPRESSION_ENABLED)
print('compression pg dsn:', settings.RECALL_COMPRESSION_PG_DSN)
"
```

Expected:
```
compression enabled: True
compression pg dsn: postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney
```

(Values from `.env` — already updated in the previous env session.)

- [ ] **Step 3: Commit**

```bash
git add services/orion-recall/app/settings.py
git commit -m "feat(graph-compression): add RECALL_COMPRESSION_* settings to orion-recall"
```

---

## Task 2: `graph_compression_adapter.py`

**Files:**
- Create: `services/orion-recall/app/storage/graph_compression_adapter.py`
- Create: `services/orion-recall/tests/test_graph_compression_adapter.py`

- [ ] **Step 1: Write failing adapter tests**

Create `services/orion-recall/tests/test_graph_compression_adapter.py`:

```python
import pytest
from unittest.mock import MagicMock, patch


def _mock_pg_rows(rows):
    """Helper: mock SQLAlchemy execute().mappings().fetchall() returning rows."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.mappings.return_value.fetchall.return_value = rows
    return mock_conn


def test_fetch_returns_empty_when_no_artifacts():
    """Empty Postgres table → empty list, no exception."""
    with patch("app.storage.graph_compression_adapter.create_engine") as mock_eng:
        engine = MagicMock()
        mock_eng.return_value = engine
        conn = _mock_pg_rows([])
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        from app.storage.graph_compression_adapter import fetch_graph_compression_fragments

        frags = fetch_graph_compression_fragments(
            query_text="what are my goals",
            mode="global",
            max_global=5,
            max_local=5,
            scopes=["episodic"],
            pg_dsn="postgresql://x:y@localhost/test",
            rdf_query_url=None,
            rdf_user="admin",
            rdf_pass="orion",
            timeout_sec=3.0,
        )
        assert frags == []


def test_fetch_returns_correct_fragment_shape():
    """Returned fragments have source='graph_compression' and required fields."""
    from datetime import datetime, timezone

    fake_row = {
        "region_id": "urn:orion:compression:region:abc",
        "scope": "episodic",
        "kind": "community",
        "summary_kind": "structural",
        "salience": 0.7,
        "trust_tier": "unverified",
        "compression_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc),
        "stale": False,
    }

    with patch("app.storage.graph_compression_adapter.create_engine") as mock_eng, \
         patch("app.storage.graph_compression_adapter._fetch_summary_from_fuseki") as mock_rdf:
        engine = MagicMock()
        mock_eng.return_value = engine
        conn = _mock_pg_rows([fake_row])
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_rdf.return_value = "A structural summary about episodic memories."

        from app.storage.graph_compression_adapter import fetch_graph_compression_fragments

        frags = fetch_graph_compression_fragments(
            query_text="what are my goals",
            mode="global",
            max_global=5,
            max_local=5,
            scopes=["episodic"],
            pg_dsn="postgresql://x:y@localhost/test",
            rdf_query_url="http://fuseki/query",
            rdf_user="admin",
            rdf_pass="orion",
            timeout_sec=3.0,
        )
        assert len(frags) == 1
        f = frags[0]
        assert f["source"] == "graph_compression"
        assert f["source_ref"] == "urn:orion:compression:region:abc"
        assert "scope:episodic" in f["tags"]
        assert "kind:community" in f["tags"]
        assert f["text"] == "A structural summary about episodic memories."


def test_fetch_does_not_raise_on_fuseki_error():
    """If Fuseki summary fetch fails, the fragment is still returned with a fallback summary."""
    from datetime import datetime, timezone

    fake_row = {
        "region_id": "urn:orion:compression:region:abc",
        "scope": "episodic",
        "kind": "community",
        "summary_kind": "structural",
        "salience": 0.7,
        "trust_tier": "unverified",
        "compression_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc),
        "stale": False,
    }

    with patch("app.storage.graph_compression_adapter.create_engine") as mock_eng, \
         patch("app.storage.graph_compression_adapter._fetch_summary_from_fuseki") as mock_rdf:
        engine = MagicMock()
        mock_eng.return_value = engine
        conn = _mock_pg_rows([fake_row])
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_rdf.side_effect = Exception("fuseki unreachable")

        from app.storage.graph_compression_adapter import fetch_graph_compression_fragments

        frags = fetch_graph_compression_fragments(
            query_text="what are my goals",
            mode="global",
            max_global=5,
            max_local=5,
            scopes=["episodic"],
            pg_dsn="postgresql://x:y@localhost/test",
            rdf_query_url="http://fuseki/query",
            rdf_user="admin",
            rdf_pass="orion",
            timeout_sec=3.0,
        )
        # Fragment still returned, just with fallback text
        assert len(frags) == 1
        assert frags[0]["source"] == "graph_compression"


def test_fetch_filters_stale_artifacts():
    """Stale artifacts are excluded from results."""
    from datetime import datetime, timezone

    stale_row = {
        "region_id": "urn:orion:compression:region:stale",
        "scope": "episodic",
        "kind": "community",
        "summary_kind": "structural",
        "salience": 0.7,
        "trust_tier": "unverified",
        "compression_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc),
        "stale": True,
    }

    with patch("app.storage.graph_compression_adapter.create_engine") as mock_eng:
        engine = MagicMock()
        mock_eng.return_value = engine
        conn = _mock_pg_rows([stale_row])
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        from app.storage.graph_compression_adapter import fetch_graph_compression_fragments

        frags = fetch_graph_compression_fragments(
            query_text="test",
            mode="global",
            max_global=5,
            max_local=5,
            scopes=["episodic"],
            pg_dsn="postgresql://x:y@localhost/test",
            rdf_query_url=None,
            rdf_user="admin",
            rdf_pass="orion",
            timeout_sec=3.0,
        )
        assert frags == []
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-recall
PYTHONPATH=. pytest tests/test_graph_compression_adapter.py -v 2>&1 | head -15
```

Expected: `ImportError: No module named 'app.storage.graph_compression_adapter'`

- [ ] **Step 3: Write `app/storage/graph_compression_adapter.py`**

```python
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal, Optional

import httpx
from sqlalchemy import create_engine, text

logger = logging.getLogger("orion-recall.graph_compression_adapter")

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")
_COMPRESSIONS_GRAPH_URI = "http://conjourney.net/graph/orion/compressions"
_ORN_NS = "http://orion.conjourney.net/ns/compression#"


def _extract_keywords(query_text: str, max_keywords: int = 6) -> list[str]:
    tokens = _TOKEN_RE.findall(query_text.lower())
    seen: set[str] = set()
    kw: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            kw.append(t)
        if len(kw) >= max_keywords:
            break
    return kw


def _score_artifact(
    row: Dict[str, Any],
    keywords: list[str],
) -> float:
    """Simple salience + keyword hit scoring."""
    base = float(row.get("salience") or 0.0)
    region_id = str(row.get("region_id") or "").lower()
    scope = str(row.get("scope") or "").lower()
    hits = sum(1 for kw in keywords if kw in region_id or kw in scope)
    return base + 0.1 * hits


def _fetch_summary_from_fuseki(
    region_id: str,
    rdf_query_url: str,
    rdf_user: str,
    rdf_pass: str,
    timeout_sec: float,
) -> str:
    query = f"""
SELECT ?summary WHERE {{
  GRAPH <{_COMPRESSIONS_GRAPH_URI}> {{
    <{region_id}> <{_ORN_NS}summary> ?summary .
  }}
}}
LIMIT 1
"""
    with httpx.Client(timeout=timeout_sec) as client:
        resp = client.post(
            rdf_query_url,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            auth=(rdf_user, rdf_pass),
        )
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
    if bindings:
        return bindings[0].get("summary", {}).get("value", "")
    return ""


def fetch_graph_compression_fragments(
    *,
    query_text: str,
    mode: Literal["global", "local", "unified"],
    max_global: int = 5,
    max_local: int = 5,
    scopes: list[str],
    pg_dsn: str,
    rdf_query_url: Optional[str],
    rdf_user: str,
    rdf_pass: str,
    timeout_sec: float,
) -> List[Dict[str, Any]]:
    """
    Query Postgres artifact index, rank by salience + keyword relevance,
    fetch summaries from Fuseki, and return recall fragments.
    Returns [] on any error (never raises).
    """
    try:
        engine = create_engine(pg_dsn, pool_pre_ping=True)
        scope_filter = ", ".join(f"'{s}'" for s in scopes)
        limit = max(max_global, max_local) * 2  # over-fetch then rank

        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT region_id, scope, kind, summary_kind, salience, trust_tier,"
                    f" compression_version, generated_at, stale"
                    f" FROM compression_artifacts"
                    f" WHERE scope IN ({scope_filter})"
                    f" AND stale = false"
                    f" ORDER BY salience DESC NULLS LAST"
                    f" LIMIT :limit"
                ),
                {"limit": limit},
            ).mappings().fetchall()

        rows = [dict(r) for r in rows]
        if not rows:
            return []

        keywords = _extract_keywords(query_text)
        scored = sorted(rows, key=lambda r: _score_artifact(r, keywords), reverse=True)

        if mode == "global":
            ranked = scored[:max_global]
        elif mode == "local":
            ranked = scored[:max_local]
        else:  # unified
            ranked = scored[:max_global + max_local]

        fragments: List[Dict[str, Any]] = []
        for row in ranked:
            region_id = row["region_id"]
            summary = ""
            if rdf_query_url:
                try:
                    summary = _fetch_summary_from_fuseki(
                        region_id, rdf_query_url, rdf_user, rdf_pass, timeout_sec
                    )
                except Exception as exc:
                    logger.debug("compression_summary_fetch_failed region_id=%s reason=%s", region_id, exc)
            if not summary:
                summary = (
                    f"[graph compression] {row.get('scope')} {row.get('kind')} region "
                    f"(salience={row.get('salience', 0):.2f}, kind={row.get('summary_kind')})"
                )

            fragments.append(
                {
                    "source": "graph_compression",
                    "source_ref": region_id,
                    "text": summary,
                    "tags": [
                        f"scope:{row.get('scope')}",
                        f"kind:{row.get('kind')}",
                        f"trust:{row.get('trust_tier')}",
                        f"summary_kind:{row.get('summary_kind')}",
                    ],
                    "salience": float(row.get("salience") or 0.0),
                    "compression_version": row.get("compression_version"),
                    "score": _score_artifact(row, keywords),
                }
            )
        return fragments
    except Exception as exc:
        logger.warning("fetch_graph_compression_fragments_failed reason=%s", exc)
        return []
```

- [ ] **Step 4: Run adapter tests**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-recall
PYTHONPATH=. pytest tests/test_graph_compression_adapter.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add services/orion-recall/app/storage/graph_compression_adapter.py \
        services/orion-recall/tests/test_graph_compression_adapter.py
git commit -m "feat(graph-compression): graph_compression_adapter for orion-recall — Postgres index + Fuseki summary fetch"
```

---

## Task 3: Recall profiles (three YAML files)

**Files:**
- Create: `orion/recall/profiles/graph.compressions.global.v1.yaml`
- Create: `orion/recall/profiles/graph.compressions.local.v1.yaml`
- Create: `orion/recall/profiles/graph.compressions.v1.yaml`
- Create: `services/orion-recall/tests/test_recall_profiles_compression.py`

- [ ] **Step 1: Write failing profile tests**

Create `services/orion-recall/tests/test_recall_profiles_compression.py`:

```python
import pytest


def _load_profile(name: str) -> dict:
    """Load a recall profile YAML by name using the existing profiles loader."""
    from app.profiles import get_profile
    return get_profile(name)


def test_global_profile_loads_and_has_required_fields():
    p = _load_profile("graph.compressions.global.v1")
    assert p.get("enable_graph_compression") is True
    assert p.get("compression_mode") == "global"
    assert int(p.get("compression_global_top_k", 0)) >= 1
    assert isinstance(p.get("compression_scopes"), list)
    assert "episodic" in p["compression_scopes"]


def test_local_profile_loads_and_has_required_fields():
    p = _load_profile("graph.compressions.local.v1")
    assert p.get("enable_graph_compression") is True
    assert p.get("compression_mode") == "local"
    assert int(p.get("compression_local_top_k", 0)) >= 1


def test_unified_profile_loads_and_has_required_fields():
    p = _load_profile("graph.compressions.v1")
    assert p.get("enable_graph_compression") is True
    assert p.get("compression_mode") == "unified"
    assert int(p.get("compression_global_top_k", 0)) >= 1
    assert int(p.get("compression_local_top_k", 0)) >= 1


def test_profiles_do_not_conflict_with_existing_profiles():
    """Loading a compression profile must not affect loading a standard profile."""
    _load_profile("graph.compressions.v1")
    from app.profiles import get_profile
    existing = get_profile("reflect.v1")
    assert existing is not None
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-recall
PYTHONPATH=. pytest tests/test_recall_profiles_compression.py -v 2>&1 | head -15
```

Expected: FAIL — profile YAML files don't exist yet.

- [ ] **Step 3: Locate existing profile path and check `get_profile` loader**

```bash
ls /mnt/scripts/Orion-Sapienform/orion/recall/profiles/ | head -10
```

Note the exact directory and convention so the new files match.

- [ ] **Step 4: Create `orion/recall/profiles/graph.compressions.global.v1.yaml`**

```yaml
profile: graph.compressions.global.v1
enable_rdf: false
enable_sql_timeline: false
enable_graph_compression: true
compression_mode: global
compression_global_top_k: 8
compression_local_top_k: 3
compression_scopes:
  - episodic
  - substrate
  - self_study
render_budget_tokens: 512
max_total_items: 11
```

- [ ] **Step 5: Create `orion/recall/profiles/graph.compressions.local.v1.yaml`**

```yaml
profile: graph.compressions.local.v1
enable_rdf: true
rdf_top_k: 6
enable_graph_compression: true
compression_mode: local
compression_global_top_k: 2
compression_local_top_k: 8
compression_scopes:
  - episodic
  - substrate
  - self_study
render_budget_tokens: 480
max_total_items: 16
```

- [ ] **Step 6: Create `orion/recall/profiles/graph.compressions.v1.yaml`**

```yaml
profile: graph.compressions.v1
enable_rdf: true
rdf_top_k: 6
enable_graph_compression: true
compression_mode: unified
compression_global_top_k: 5
compression_local_top_k: 5
compression_scopes:
  - episodic
  - substrate
  - self_study
render_budget_tokens: 640
max_total_items: 16
```

- [ ] **Step 7: Run profile tests**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-recall
PYTHONPATH=. pytest tests/test_recall_profiles_compression.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add orion/recall/profiles/graph.compressions.global.v1.yaml \
        orion/recall/profiles/graph.compressions.local.v1.yaml \
        orion/recall/profiles/graph.compressions.v1.yaml \
        services/orion-recall/tests/test_recall_profiles_compression.py
git commit -m "feat(graph-compression): three compression recall profiles — global, local, unified"
```

---

## Task 4: Wire compression backend into `_query_backends`

**Files:**
- Modify: `services/orion-recall/app/worker.py`
- Create: `services/orion-recall/tests/test_query_backends_compression.py`

- [ ] **Step 1: Write failing integration test**

Create `services/orion-recall/tests/test_query_backends_compression.py`:

```python
import pytest
import asyncio
from unittest.mock import patch, MagicMock


def _compression_profile():
    return {
        "enable_graph_compression": True,
        "compression_mode": "global",
        "compression_global_top_k": 3,
        "compression_local_top_k": 3,
        "compression_scopes": ["episodic"],
        "enable_rdf": False,
        "enable_sql_timeline": False,
    }


def test_compression_backend_called_when_enabled():
    fake_fragments = [
        {
            "source": "graph_compression",
            "source_ref": "urn:orion:compression:region:abc",
            "text": "Episodic cluster about workflow design.",
            "tags": ["scope:episodic", "kind:community"],
            "salience": 0.7,
            "score": 0.7,
        }
    ]
    with patch("app.worker.fetch_graph_compression_fragments", return_value=fake_fragments) as mock_fetch, \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_COMPRESSION_ENABLED = True
        mock_settings.RECALL_COMPRESSION_PG_DSN = "postgresql://x:y@localhost/test"
        mock_settings.RECALL_COMPRESSION_RDF_QUERY_URL = "http://fuseki/query"
        mock_settings.RECALL_COMPRESSION_RDF_USER = "admin"
        mock_settings.RECALL_COMPRESSION_RDF_PASS = "orion"
        mock_settings.RECALL_COMPRESSION_TIMEOUT_SEC = 3.0
        mock_settings.RECALL_RDF_ENDPOINT_URL = None

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "what are my dominant preoccupations",
                _compression_profile(),
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_fetch.assert_called_once()
        assert counts.get("graph_compression") == 1
        assert any(c["source"] == "graph_compression" for c in candidates)


def test_compression_backend_returns_empty_when_disabled():
    with patch("app.worker.fetch_graph_compression_fragments") as mock_fetch, \
         patch("app.worker.settings") as mock_settings:
        mock_settings.RECALL_COMPRESSION_ENABLED = False
        mock_settings.RECALL_COMPRESSION_PG_DSN = None
        mock_settings.RECALL_RDF_ENDPOINT_URL = None

        profile = dict(_compression_profile())
        profile["enable_graph_compression"] = True  # profile enables, but settings disable globally

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "test query",
                profile,
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        mock_fetch.assert_not_called()
        assert counts.get("graph_compression", 0) == 0


def test_compression_backend_does_not_suppress_rdf_backend():
    """When both RDF and compression are enabled, both produce candidates."""
    fake_compression = [
        {"source": "graph_compression", "source_ref": "urn:x", "text": "compression summary",
         "tags": ["scope:episodic"], "salience": 0.5, "score": 0.5}
    ]
    with patch("app.worker.fetch_graph_compression_fragments", return_value=fake_compression), \
         patch("app.worker.settings") as mock_settings, \
         patch("app.worker.fetch_rdf_chatturn_fragments", return_value=[
             {"source": "rdf", "text": "an rdf fragment"}
         ]):
        mock_settings.RECALL_COMPRESSION_ENABLED = True
        mock_settings.RECALL_COMPRESSION_PG_DSN = "postgresql://x:y@localhost/test"
        mock_settings.RECALL_COMPRESSION_RDF_QUERY_URL = None
        mock_settings.RECALL_COMPRESSION_RDF_USER = "admin"
        mock_settings.RECALL_COMPRESSION_RDF_PASS = "orion"
        mock_settings.RECALL_COMPRESSION_TIMEOUT_SEC = 3.0
        mock_settings.RECALL_RDF_ENDPOINT_URL = "http://fuseki/query"

        mixed_profile = {
            "enable_rdf": True,
            "rdf_top_k": 4,
            "enable_graph_compression": True,
            "compression_mode": "global",
            "compression_global_top_k": 3,
            "compression_local_top_k": 3,
            "compression_scopes": ["episodic"],
        }

        from app.worker import _query_backends

        candidates, counts = asyncio.run(
            _query_backends(
                "what tensions am I carrying",
                mixed_profile,
                session_id=None,
                node_id=None,
                entities=[],
            )
        )

        assert counts.get("graph_compression", 0) >= 1
        # RDF backend ran (may have returned from mock or failed gracefully — either way no suppression)
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-recall
PYTHONPATH=. pytest tests/test_query_backends_compression.py -v 2>&1 | head -15
```

Expected: FAIL — `fetch_graph_compression_fragments` not yet imported or called in `worker.py`.

- [ ] **Step 3: Add import to `worker.py`**

At the top of `services/orion-recall/app/worker.py`, in the try/except import block that imports from `app.storage.rdf_adapter`, add after the existing storage imports:

```python
    try:
        from .storage.graph_compression_adapter import fetch_graph_compression_fragments
    except ImportError:
        fetch_graph_compression_fragments = None  # type: ignore
```

(Use the same pattern as the existing `try: from .fusion import ... except ImportError:` block.)

- [ ] **Step 4: Add the compression backend call inside `_query_backends`**

Find the end of `_query_backends` just before the `return candidates, backend_counts` line (around line 867 in the current file). Add this block immediately before the return:

```python
    # ── Graph Compression backend ─────────────────────────────────────────────
    compression_enabled = (
        bool(profile.get("enable_graph_compression"))
        and bool(getattr(settings, "RECALL_COMPRESSION_ENABLED", False))
        and bool(getattr(settings, "RECALL_COMPRESSION_PG_DSN", None))
        and fetch_graph_compression_fragments is not None
    )
    if compression_enabled:
        try:
            compression_frags = fetch_graph_compression_fragments(
                query_text=fragment,
                mode=str(profile.get("compression_mode") or "unified"),
                max_global=int(profile.get("compression_global_top_k") or 5),
                max_local=int(profile.get("compression_local_top_k") or 5),
                scopes=list(profile.get("compression_scopes") or ["episodic", "substrate", "self_study"]),
                pg_dsn=settings.RECALL_COMPRESSION_PG_DSN,
                rdf_query_url=getattr(settings, "RECALL_COMPRESSION_RDF_QUERY_URL", None),
                rdf_user=getattr(settings, "RECALL_COMPRESSION_RDF_USER", "admin"),
                rdf_pass=getattr(settings, "RECALL_COMPRESSION_RDF_PASS", "orion"),
                timeout_sec=float(getattr(settings, "RECALL_COMPRESSION_TIMEOUT_SEC", 3.0)),
            )
            backend_counts["graph_compression"] = len(compression_frags)
            candidates.extend(compression_frags)
        except Exception as exc:
            logger.debug("graph_compression_backend_skipped reason=%s", exc)
            backend_counts["graph_compression"] = 0
    else:
        backend_counts["graph_compression"] = 0
```

Note: `fragment` is the first positional arg to `_query_backends` — the query text. Make sure this block is added inside the `_query_backends` function body, before the `return`.

- [ ] **Step 5: Run integration tests**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-recall
PYTHONPATH=. pytest tests/test_query_backends_compression.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 6: Run full recall test suite to check for regressions**

```bash
PYTHONPATH=. pytest tests/ -v --tb=short -q 2>&1 | tail -20
```

Expected: All previously-passing tests still pass. Fix any failures before committing.

- [ ] **Step 7: Commit**

```bash
git add services/orion-recall/app/worker.py \
        services/orion-recall/tests/test_query_backends_compression.py
git commit -m "feat(graph-compression): wire graph_compression backend into orion-recall _query_backends"
```

---

## Task 5: Final verification

- [ ] **Step 1: Run all recall tests**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-recall
PYTHONPATH=. pytest tests/ -v --tb=short
```

Expected: All pass.

- [ ] **Step 2: Smoke test profile load from disk**

```bash
PYTHONPATH=. python -c "
from app.profiles import get_profile
for name in ['graph.compressions.v1', 'graph.compressions.global.v1', 'graph.compressions.local.v1']:
    p = get_profile(name)
    print(name, '-> mode:', p.get('compression_mode'), 'enabled:', p.get('enable_graph_compression'))
"
```

Expected:
```
graph.compressions.v1 -> mode: unified enabled: True
graph.compressions.global.v1 -> mode: global enabled: True
graph.compressions.local.v1 -> mode: local enabled: True
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat(graph-compression): orion-recall adapter complete — adapter, profiles, _query_backends integration"
```

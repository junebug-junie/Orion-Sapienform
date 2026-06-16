# Memory Consolidation Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Durable automated memory consolidation on **every** `chat_history_log` turn — classify via logprobs, merge scores by original `correlation_id`, close windows by situational phase, run `memory_graph_suggest`, persist drafts for operator review.

**Architecture:** Hub publishes `orion:chat:history:turn` → sql-writer commits row → sql-writer emits `orion:memory:turn:persisted` (same `correlation_id`) → `orion-memory-consolidation` classifies (gateway quick lane), patches spark_meta back through sql-writer, tracks windows in Postgres, closes on phase+boundary rules, runs suggest, writes draft rows. Spark introspector is **not** on this path.

**Tech Stack:** Python 3.12, FastAPI, pydantic v2, asyncpg, Orion bus (`Hunter`, `OrionBusAsync`, `BaseEnvelope`), existing `orion/schemas`, `orion/memory_graph`, sql-writer route map.

**Design spec:** `docs/superpowers/specs/2026-06-16-memory-consolidation-design.md`

**Worktree:** Implement in an isolated worktree (`using-superpowers:using-git-worktrees`) before touching main.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `orion/schemas/memory_consolidation.py` | Create | `MemoryTurnPersistedV1`, `ChatHistorySparkMetaPatchV1`, window/draft DTOs |
| `orion/schemas/registry.py` | Modify | Register new schemas |
| `orion/bus/channels.yaml` | Modify | Two new channels + consumer/producer lists |
| `tests/test_memory_consolidation_bus_catalog.py` | Create | Catalog/registry contract test |
| `orion/memory/consolidation_classify.py` | Create | Logprob YES/NO extraction + prompt builder |
| `tests/test_consolidation_classify.py` | Create | Unit tests for softmax + parsing |
| `services/orion-sql-writer/app/settings.py` | Modify | Route map + subscribe channels + emit flags |
| `services/orion-sql-writer/app/worker.py` | Modify | Post-commit outbox + patch merge handler |
| `services/orion-sql-writer/tests/test_memory_turn_persisted_outbox.py` | Create | Outbox emit after chat.history write |
| `services/orion-sql-writer/tests/test_spark_meta_patch.py` | Create | Patch merge by correlation_id |
| `orion/memory_graph/draft_repository.py` | Create | Insert/list pending `SuggestDraftV1` drafts |
| `services/orion-sql-db/manual_migration_memory_consolidation_v1.sql` | Create | `memory_consolidation_windows`, `memory_graph_suggest_drafts` |
| `services/orion-memory-consolidation/` | Create | New service (see spec layout) |
| `.env_example` | Modify | Root channel env vars |
| `services/orion-memory-consolidation/.env_example` | Create | Service overrides |
| `scripts/sync_local_env_from_example.py` | Run after env changes | Local `.env` sync |

---

## Task 1: Shared schemas + bus catalog

**Files:**
- Create: `orion/schemas/memory_consolidation.py`
- Modify: `orion/schemas/registry.py`
- Modify: `orion/bus/channels.yaml`
- Create: `tests/test_memory_consolidation_bus_catalog.py`

- [ ] **Step 1: Write schema module**

Create `orion/schemas/memory_consolidation.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

MEMORY_TURN_PERSISTED_KIND = "memory.turn.persisted.v1"
CHAT_HISTORY_SPARK_META_PATCH_KIND = "chat.history.spark_meta.patch.v1"


class MemoryTurnPersistedV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    prompt: str
    response: str
    spark_meta: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    created_at: Optional[datetime] = None


class ChatHistorySparkMetaPatchV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    spark_meta: Dict[str, Any] = Field(default_factory=dict)


class MemoryConsolidationWindowV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_window_id: str
    turn_correlation_ids: List[str] = Field(default_factory=list)
    status: Literal["open", "closed", "consolidated", "failed"] = "open"
    phase_change_at_close: Optional[str] = None
    consolidation_status: Optional[Literal["pending", "ok", "failed"]] = None
    draft_id: Optional[str] = None
    created_at: datetime
    closed_at: Optional[datetime] = None


class MemoryGraphSuggestDraftRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_id: str
    memory_window_id: str
    status: Literal["pending_review", "approved", "rejected"] = "pending_review"
    draft: Dict[str, Any]
    turn_correlation_ids: List[str] = Field(default_factory=list)
    created_at: datetime
```

- [ ] **Step 2: Register in `orion/schemas/registry.py`**

Add import and registry entries:

```python
from orion.schemas.memory_consolidation import (
    ChatHistorySparkMetaPatchV1,
    MemoryConsolidationWindowV1,
    MemoryGraphSuggestDraftRecordV1,
    MemoryTurnPersistedV1,
)
# in _REGISTRY dict:
"MemoryTurnPersistedV1": MemoryTurnPersistedV1,
"ChatHistorySparkMetaPatchV1": ChatHistorySparkMetaPatchV1,
"MemoryConsolidationWindowV1": MemoryConsolidationWindowV1,
"MemoryGraphSuggestDraftRecordV1": MemoryGraphSuggestDraftRecordV1,
```

- [ ] **Step 3: Add channels to `orion/bus/channels.yaml`**

Insert near other `orion:memory:*` entries:

```yaml
  - name: "orion:memory:turn:persisted"
    kind: "event"
    schema_id: "MemoryTurnPersistedV1"
    message_kind: "memory.turn.persisted.v1"
    producer_services: ["orion-sql-writer"]
    consumer_services: ["orion-memory-consolidation"]
    stability: "experimental"
    since: "2026-06-16"

  - name: "orion:chat:history:spark_meta:patch"
    kind: "event"
    schema_id: "ChatHistorySparkMetaPatchV1"
    message_kind: "chat.history.spark_meta.patch.v1"
    producer_services: ["orion-memory-consolidation"]
    consumer_services: ["orion-sql-writer"]
    stability: "experimental"
    since: "2026-06-16"
```

- [ ] **Step 4: Write catalog test**

Create `tests/test_memory_consolidation_bus_catalog.py` (pattern from crystallization catalog test):

```python
from pathlib import Path
import yaml
from orion.schemas.memory_consolidation import MemoryTurnPersistedV1, ChatHistorySparkMetaPatchV1
from orion.schemas.registry import _REGISTRY, resolve

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = {
    "orion:memory:turn:persisted": ("MemoryTurnPersistedV1", "memory.turn.persisted.v1"),
    "orion:chat:history:spark_meta:patch": ("ChatHistorySparkMetaPatchV1", "chat.history.spark_meta.patch.v1"),
}

def _channels():
    doc = yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text(encoding="utf-8")) or {}
    return {e["name"]: e for e in doc.get("channels") or [] if isinstance(e, dict) and "name" in e}

def test_memory_consolidation_schemas_registered():
    assert resolve("MemoryTurnPersistedV1") is MemoryTurnPersistedV1
    assert resolve("ChatHistorySparkMetaPatchV1") is ChatHistorySparkMetaPatchV1

def test_memory_consolidation_channels_cataloged():
    ch = _channels()
    for name, (schema_id, message_kind) in CHANNELS.items():
        entry = ch[name]
        assert entry["schema_id"] == schema_id
        assert entry["message_kind"] == message_kind
```

- [ ] **Step 5: Run tests**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_memory_consolidation_bus_catalog.py -q
```

Expected: PASS

---

## Task 2: Logprob classification helper

**Files:**
- Create: `orion/memory/consolidation_classify.py`
- Create: `tests/test_consolidation_classify.py`

- [ ] **Step 1: Write failing tests**

```python
import math
from orion.memory.consolidation_classify import (
    binary_score_from_top_logprobs,
    parse_classify_lines,
    build_classify_prompt,
)

def test_binary_score_from_top_logprobs_yes_wins():
    tops = [{"token": "YES", "logprob": -0.2}, {"token": "NO", "logprob": -2.0}]
    score = binary_score_from_top_logprobs(tops)
    assert score == pytest.approx(math.exp(-0.2) / (math.exp(-0.2) + math.exp(-2.0)), rel=1e-3)

def test_parse_classify_lines():
    text = "MEMORY: YES\nBOUNDARY: NO\n"
    mem, bnd = parse_classify_lines(text)
    assert mem == "YES"
    assert bnd == "NO"

def test_build_classify_prompt_includes_phase():
    p = build_classify_prompt(prompt="hi", response="hello", spark_meta={"conversation_phase": {"phase_change": "same_breath"}})
    assert "same_breath" in p
    assert "MEMORY:" in p
```

- [ ] **Step 2: Implement helper**

```python
# orion/memory/consolidation_classify.py
import math
import re
from typing import Any

_MEMORY_LINE = re.compile(r"^MEMORY:\s*(YES|NO)\s*$", re.I | re.M)
_BOUNDARY_LINE = re.compile(r"^BOUNDARY:\s*(YES|NO)\s*$", re.I | re.M)

def binary_score_from_top_logprobs(top_logprobs: list[dict]) -> float | None:
    yes_lp = no_lp = None
    for entry in top_logprobs or []:
        tok = str(entry.get("token") or "").strip().upper()
        lp = entry.get("logprob")
        if not isinstance(lp, (int, float)):
            continue
        if tok == "YES":
            yes_lp = float(lp)
        elif tok == "NO":
            no_lp = float(lp)
    if yes_lp is None or no_lp is None:
        return None
    ey, en = math.exp(yes_lp), math.exp(no_lp)
    return ey / (ey + en)

def parse_classify_lines(text: str) -> tuple[str | None, str | None]:
    mem = _MEMORY_LINE.search(text or "")
    bnd = _BOUNDARY_LINE.search(text or "")
    return (
        mem.group(1).upper() if mem else None,
        bnd.group(1).upper() if bnd else None,
    )

def build_classify_prompt(*, prompt: str, response: str, spark_meta: dict[str, Any]) -> str:
    phase = ((spark_meta.get("conversation_phase") or {}).get("phase_change")
             or spark_meta.get("temporal_phase") or "unknown")
    phi_after = spark_meta.get("phi_after") or {}
    novelty = phi_after.get("novelty", spark_meta.get("spark_phi_novelty", "?"))
    coherence = phi_after.get("coherence", spark_meta.get("spark_phi_coherence", "?"))
    return (
        "Classify this turn. Output exactly two lines:\n"
        "MEMORY: YES or NO\n"
        "BOUNDARY: YES or NO\n\n"
        f"phase={phase} novelty={novelty} coherence={coherence}\n"
        f"User: {prompt[:300]!r}\n"
        f"Orion: {response[:300]!r}\n"
    )
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_consolidation_classify.py -q
```

Expected: PASS

---

## Task 3: sql-writer outbox (`memory.turn.persisted`)

**Files:**
- Modify: `services/orion-sql-writer/app/settings.py`
- Modify: `services/orion-sql-writer/app/worker.py`
- Create: `services/orion-sql-writer/tests/test_memory_turn_persisted_outbox.py`

- [ ] **Step 1: Add settings**

In `settings.py`:

```python
# DEFAULT_ROUTE_MAP add:
"memory.turn.persisted.v1": "__emit_only__",  # or skip route — emit-only event
"chat.history.spark_meta.patch.v1": "__patch_chat_history__",

sql_writer_emit_memory_turn_persisted: bool = Field(True, alias="SQL_WRITER_EMIT_MEMORY_TURN_PERSISTED")
channel_memory_turn_persisted: str = Field("orion:memory:turn:persisted", alias="CHANNEL_MEMORY_TURN_PERSISTED")
channel_chat_history_spark_meta_patch: str = Field(
    "orion:chat:history:spark_meta:patch", alias="CHANNEL_CHAT_HISTORY_SPARK_META_PATCH"
)
```

Add `orion:chat:history:spark_meta:patch` to `sql_writer_subscribe_channels`.

- [ ] **Step 2: Write failing outbox test**

Test that after `_write_row(ChatHistoryLogSQL, ...)` succeeds for `kind=chat.history`, worker publishes envelope with:
- `kind=memory.turn.persisted.v1`
- `correlation_id == payload.correlation_id == trace_id`
- payload contains prompt/response/spark_meta

Mock `bus.publish` and assert call args.

- [ ] **Step 3: Implement post-commit emit in `worker.py`**

After existing journal/collapse post-commit block (~line 1940), add:

```python
if env.kind == "chat.history" and write_ok and bus is not None and settings.sql_writer_emit_memory_turn_persisted:
    from orion.schemas.memory_consolidation import MemoryTurnPersistedV1, MEMORY_TURN_PERSISTED_KIND
    corr = str(extra_sql_fields.get("correlation_id") or env.correlation_id or "")
    if corr:
        persisted = MemoryTurnPersistedV1(
            correlation_id=corr,
            prompt=str(data_to_process.get("prompt") or ""),
            response=str(data_to_process.get("response") or ""),
            spark_meta=data_to_process.get("spark_meta") if isinstance(data_to_process.get("spark_meta"), dict) else {},
            session_id=data_to_process.get("session_id"),
        )
        out_env = env.derive_child(
            kind=MEMORY_TURN_PERSISTED_KIND,
            source=ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name),
            payload=persisted.model_dump(mode="json"),
            reply_to=None,
        )
        # Envelope correlation_id MUST match turn correlation_id
        out_env.correlation_id = corr  # if derive_child preserves; else set explicitly
        await bus.publish(settings.channel_memory_turn_persisted, out_env)
```

**Critical:** `out_env.correlation_id` and `payload["correlation_id"]` must both equal hub `trace_id`. Add explicit assert/log if they diverge.

- [ ] **Step 4: Run test**

```bash
./scripts/test_service.sh orion-sql-writer services/orion-sql-writer/tests/test_memory_turn_persisted_outbox.py -q
```

Expected: PASS

---

## Task 4: sql-writer spark_meta patch handler

**Files:**
- Modify: `services/orion-sql-writer/app/worker.py`
- Create: `services/orion-sql-writer/tests/test_spark_meta_patch.py`

- [ ] **Step 1: Write failing test**

Given existing `ChatHistoryLogSQL` row with `correlation_id=T` and `spark_meta={"foo": 1}`, apply patch envelope:

```python
ChatHistorySparkMetaPatchV1(correlation_id=T, spark_meta={"memory_significance_score": 0.8})
```

Expect merged `spark_meta={"foo": 1, "memory_significance_score": 0.8}`.

Missing row → handler returns False / logs error, no insert.

- [ ] **Step 2: Implement handler**

In envelope dispatch (where `env.kind` routed), before generic `_write`:

```python
if env.kind == "chat.history.spark_meta.patch.v1":
    from orion.schemas.memory_consolidation import ChatHistorySparkMetaPatchV1
    patch = ChatHistorySparkMetaPatchV1.model_validate(payload)
    corr = str(patch.correlation_id)
    existing = sess.query(ChatHistoryLogSQL).filter(ChatHistoryLogSQL.correlation_id == corr).first()
    if existing is None:
        logger.error("spark_meta_patch_missing_row correlation_id=%s", corr)
        return True  # ack to avoid poison; this is a bug if outbox ordering correct
    merged = _merge_spark_meta(getattr(existing, "spark_meta", None), patch.spark_meta)
    sess.execute(update(ChatHistoryLogSQL).where(ChatHistoryLogSQL.correlation_id == corr).values(spark_meta=merged))
    sess.commit()
    return True
```

- [ ] **Step 3: Run test**

```bash
./scripts/test_service.sh orion-sql-writer services/orion-sql-writer/tests/test_spark_meta_patch.py -q
```

Expected: PASS

---

## Task 5: Postgres tables + draft repository

**Files:**
- Create: `services/orion-sql-db/manual_migration_memory_consolidation_v1.sql`
- Create: `orion/memory_graph/draft_repository.py`
- Create: `tests/test_memory_graph_draft_repository.py`

- [ ] **Step 1: Migration SQL**

```sql
CREATE TABLE IF NOT EXISTS memory_consolidation_windows (
    memory_window_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'open',
    turn_correlation_ids JSONB NOT NULL DEFAULT '[]',
    phase_change_at_close TEXT,
    consolidation_status TEXT,
    draft_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS memory_graph_suggest_drafts (
    draft_id TEXT PRIMARY KEY,
    memory_window_id TEXT NOT NULL REFERENCES memory_consolidation_windows(memory_window_id),
    status TEXT NOT NULL DEFAULT 'pending_review',
    draft JSONB NOT NULL,
    turn_correlation_ids JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (memory_window_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_consolidation_windows_status ON memory_consolidation_windows(status);
CREATE INDEX IF NOT EXISTS idx_memory_graph_suggest_drafts_status ON memory_graph_suggest_drafts(status);
```

Note: Hub manual suggest today returns draft JSON only — **no table exists**. This migration is required for automated pending_review storage.

- [ ] **Step 2: Draft repository**

```python
# orion/memory_graph/draft_repository.py
import json
from uuid import uuid4
from datetime import datetime, timezone
import asyncpg
from orion.memory_graph.dto import SuggestDraftV1

async def insert_pending_draft(
    pool: asyncpg.Pool,
    *,
    memory_window_id: str,
    draft: SuggestDraftV1,
    turn_correlation_ids: list[str],
) -> str:
    draft_id = str(uuid4())
    await pool.execute(
        """
        INSERT INTO memory_graph_suggest_drafts
          (draft_id, memory_window_id, status, draft, turn_correlation_ids, created_at)
        VALUES ($1, $2, 'pending_review', $3::jsonb, $4::jsonb, $5)
        ON CONFLICT (memory_window_id) DO NOTHING
        """,
        draft_id,
        memory_window_id,
        json.dumps(draft.model_dump(mode="json")),
        json.dumps(turn_correlation_ids),
        datetime.now(timezone.utc),
    )
    return draft_id
```

- [ ] **Step 3: Unit test with asyncpg mock or sqlite-free pure SQL string test**

- [ ] **Step 4: Run test**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest tests/test_memory_graph_draft_repository.py -q
```

---

## Task 6: Scaffold `orion-memory-consolidation` service

**Files:**
- Create: `services/orion-memory-consolidation/Dockerfile`
- Create: `services/orion-memory-consolidation/docker-compose.yml`
- Create: `services/orion-memory-consolidation/requirements.txt`
- Create: `services/orion-memory-consolidation/.env_example`
- Create: `services/orion-memory-consolidation/app/settings.py`
- Create: `services/orion-memory-consolidation/app/main.py`

Mirror `orion-memory-crystallizer` chassis pattern:
- `Hunter` subscribes `orion:memory:turn:persisted`
- asyncpg pool for window + draft tables
- health endpoint

Settings (read from root env):

```python
CHANNEL_MEMORY_TURN_PERSISTED: str = "orion:memory:turn:persisted"
CHANNEL_CHAT_HISTORY_SPARK_META_PATCH: str = "orion:chat:history:spark_meta:patch"
CHANNEL_LLM_INTAKE: str = "orion:exec:request:LLMGatewayService"
POSTGRES_URI: str
MEMORY_CONSOLIDATION_ENABLED: bool = True
MEMORY_CLASSIFY_TIMEOUT_SEC: float = 8.0
MEMORY_BOUNDARY_SCORE_THRESHOLD: float = 0.70
```

- [ ] **Step 1: Create minimal service that starts and subscribes**

- [ ] **Step 2: Verify compile**

```bash
python -m compileall services/orion-memory-consolidation
```

Expected: exit 0

---

## Task 7: Classify + patch worker

**Files:**
- Create: `services/orion-memory-consolidation/app/classify.py`
- Create: `services/orion-memory-consolidation/app/worker.py`
- Create: `services/orion-memory-consolidation/tests/test_worker_classify_patch.py`

- [ ] **Step 1: Gateway RPC classify**

Follow `orion-graph-compression/app/summarizer.py` bus RPC pattern. Use **new UUID only for LLM RPC reply routing**, not for turn identity:

```python
async def classify_turn(bus, *, turn: MemoryTurnPersistedV1, settings) -> dict:
    prompt = build_classify_prompt(prompt=turn.prompt, response=turn.response, spark_meta=turn.spark_meta)
    rpc_corr = str(uuid4())  # LLM RPC only
    reply = f"orion:memory:consolidation:llm:reply:{rpc_corr}"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "return_logprobs": True,
            "logprobs_top_k": 2,
            "logprob_summary_only": False,
            "max_tokens": 3,
            "llm_route": "quick",
        },
    }
    # rpc_request to CHANNEL_LLM_INTAKE, parse logprobs.content for YES/NO tokens
```

Map to patch dict:

```python
{
    "memory_significance_score": mem_score,
    "conversation_boundary_score": bnd_score,
    "memory_classify_status": "ok",
    "memory_classify_ts": datetime.now(timezone.utc).isoformat(),
}
```

On failure after 2 retries: `"memory_classify_status": "degraded"`.

- [ ] **Step 2: Publish patch with turn correlation_id**

```python
patch_env = BaseEnvelope(
    kind="chat.history.spark_meta.patch.v1",
    correlation_id=turn.correlation_id,  # SACRED — hub trace_id
    source=svc_ref,
    payload=ChatHistorySparkMetaPatchV1(
        correlation_id=turn.correlation_id,
        spark_meta=patch_fields,
    ).model_dump(mode="json"),
)
await bus.publish(settings.channel_chat_history_spark_meta_patch, patch_env)
```

- [ ] **Step 3: Worker handler**

```python
async def handle_memory_turn_persisted(env: BaseEnvelope, *, bus, settings, window_store, suggest_runner):
    turn = MemoryTurnPersistedV1.model_validate(env.payload)
    assert str(env.correlation_id) == turn.correlation_id, "correlation_id mismatch"
    patch_fields = await classify_turn(bus, turn=turn, settings=settings)
    await publish_spark_meta_patch(bus, turn.correlation_id, patch_fields, settings)
    await window_store.append_turn(turn, scores=patch_fields)
    if should_close_window(turn, patch_fields, settings):
        closed = await window_store.close_current_window(turn.correlation_id)
        await suggest_runner.consolidate_window(closed, bus=bus)
```

- [ ] **Step 4: Unit test with mocked bus + mocked classify**

---

## Task 8: Window state + boundary logic

**Files:**
- Create: `services/orion-memory-consolidation/app/window_state.py`
- Create: `services/orion-memory-consolidation/app/boundary.py`
- Create: `services/orion-memory-consolidation/tests/test_boundary.py`

- [ ] **Step 1: Implement `should_close_window` per spec**

```python
def should_close_window(turn: MemoryTurnPersistedV1, scores: dict, settings) -> bool:
    phase = ((turn.spark_meta.get("conversation_phase") or {}).get("phase_change") or "unknown")
    bnd = float(scores.get("conversation_boundary_score") or 0.0)
    if phase in {"long_gap", "next_day", "stale_thread"} and bnd >= settings.memory_boundary_score_threshold:
        return True
    if phase == "unknown" and bnd >= settings.memory_boundary_llm_only_threshold:
        return True
    if phase in {"same_breath", "short_pause", "resumed_thread"} and bnd >= settings.memory_boundary_override_threshold:
        return True
    return False
```

- [ ] **Step 2: Window store**

Postgres-backed:
- One `open` window at a time (v1 single-operator).
- `append_turn` adds `correlation_id` + scores snapshot to JSON array.
- `close_current_window` sets status `closed`, returns window excluding closing turn (closing turn opens next window).

- [ ] **Step 3: Tests for boundary matrix**

Cover all phase × score combinations from spec acceptance table.

---

## Task 9: Suggest runner + draft persistence + turn stamps

**Files:**
- Create: `services/orion-memory-consolidation/app/suggest_runner.py`
- Create: `services/orion-memory-consolidation/app/window_fetch.py`
- Modify: `services/orion-memory-consolidation/app/worker.py`

- [ ] **Step 1: Build annotated transcript from window turn list**

```python
def build_window_transcript(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        sig = t.get("memory_significance_score")
        prefix = f"[sig={sig:.2f}] " if isinstance(sig, (int, float)) else ""
        lines.append(f"{prefix}User: {t['prompt']}\nOrion: {t['response']}\n")
    return "\n".join(lines)
```

- [ ] **Step 2: Call cortex `memory_graph_suggest`**

Reuse logic from `services/orion-hub/scripts/memory_graph_suggest.py` — extract minimal `suggest_once(cortex_bus, transcript, ...)` into shared `orion/memory_graph/suggest_runner.py` OR invoke cortex orch RPC from consolidation service (same verb, no hub HTTP).

Validate with `SuggestDraftV1.model_validate` + `sanitize_suggest_draft_dict`.

- [ ] **Step 3: Persist draft + stamp turns**

```python
draft_id = await insert_pending_draft(pool, memory_window_id=window.memory_window_id, draft=draft, turn_correlation_ids=window.turn_correlation_ids)
for corr in window.turn_correlation_ids:
    await publish_spark_meta_patch(bus, corr, {
        "memory_window_id": window.memory_window_id,
        "memory_consolidated_at": now_iso,
    }, settings)
await window_store.mark_consolidated(window.memory_window_id, draft_id=draft_id)
```

- [ ] **Step 4: Integration test with mocked cortex response** using fixture `tests/fixtures/memory_graph/joey_cats_draft.json`

---

## Task 10: Failed-window retry (failures only)

**Files:**
- Create: `services/orion-memory-consolidation/app/retry_failed_windows.py`

- [ ] **Step 1: Periodic task (30 min)**

```sql
SELECT memory_window_id FROM memory_consolidation_windows
WHERE consolidation_status = 'failed' AND status = 'closed'
ORDER BY closed_at LIMIT 20;
```

Re-run suggest for each. **Not** a scan for missing classification coverage.

- [ ] **Step 2: Test mark failed on suggest exception + retry succeeds**

---

## Task 11: Root env + docker wiring

**Files:**
- Modify: `.env_example`
- Modify: `services/orion-sql-writer/.env_example`
- Create: `services/orion-memory-consolidation/.env_example`

- [ ] **Step 1: Add root keys**

```
CHANNEL_MEMORY_TURN_PERSISTED=orion:memory:turn:persisted
CHANNEL_CHAT_HISTORY_SPARK_META_PATCH=orion:chat:history:spark_meta:patch
SQL_WRITER_EMIT_MEMORY_TURN_PERSISTED=true
MEMORY_CONSOLIDATION_ENABLED=true
LLM_LOGPROB_SUMMARY_ENABLED=true
```

- [ ] **Step 2: Sync local env**

```bash
python scripts/sync_local_env_from_example.py
```

- [ ] **Step 3: docker-compose for new service** (pass-through `${POSTGRES_URI}`, `${ORION_BUS_URL}`, channel vars from root)

```bash
docker compose --env-file .env --env-file services/orion-bus/.env \
  -f services/orion-bus/docker-compose.yml build
```

---

## Task 12: End-to-end smoke test

**Files:**
- Create: `scripts/smoke_memory_consolidation_pipeline.py`

- [ ] **Step 1: Script publishes `chat.history` turn envelope** (same shape as hub) with known `correlation_id=smoke-{uuid}`

- [ ] **Step 2: Assert chain**

1. Row in `chat_history_log` with that `correlation_id`
2. `spark_meta.memory_significance_score` or `memory_classify_status=degraded` within timeout
3. For forced boundary fixture (inject `phase_change=long_gap` + high boundary score on second turn), draft row in `memory_graph_suggest_drafts` with `status=pending_review`

- [ ] **Step 3: Run smoke (requires live stack)**

Document in plan README section; CI can run unit/integration portions only.

---

## Spec coverage self-review

| Spec requirement | Task |
|------------------|------|
| Every chat_history_log turn classified | Task 3 outbox + Task 7 handler |
| correlation_id sacred | Tasks 3, 4, 7 explicit asserts |
| No spark candidate dependency | Architecture — consolidation subscribes only to outbox |
| Situational phase bounds | Task 8 |
| 90-min fallback | Task 9 window_fetch.py |
| Draft pending_review | Task 5 + 9 |
| Failed suggest retry only | Task 10 |
| Registry + channels | Task 1 |
| Root env pattern | Task 11 |

**Gap fixed in plan:** Spec referenced "hub memory-graph draft store" but none exists — Task 5 adds `memory_graph_suggest_drafts` table.

---

## Verification commands (full suite)

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  tests/test_memory_consolidation_bus_catalog.py \
  tests/test_consolidation_classify.py \
  tests/test_memory_graph_draft_repository.py \
  services/orion-sql-writer/tests/test_memory_turn_persisted_outbox.py \
  services/orion-sql-writer/tests/test_spark_meta_patch.py \
  services/orion-memory-consolidation/tests/ -q
```

Expected: all PASS before merge.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-16-memory-consolidation-pipeline.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration

2. **Inline Execution** — implement tasks in this session using executing-plans, batch execution with checkpoints

Which approach?

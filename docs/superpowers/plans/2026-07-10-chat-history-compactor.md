# Chat History Compactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add workflow `chat_history_compactor_pass` that compacts bounded `chat_history_log` windows into **indexed** (upsert-by-`compactor_index`) high-recall memory cards, with optional journal append, Skill Runner entries, and a daily 06:00 `America/Denver` schedule bootstrap.

**Architecture:** Reuse `skills.chat.discussion_window.v1` for SQL fetch. Shared pure helpers under `orion/cognition/compactor/` (window keys + budget) and `orion/cognition/chat_history_compactor/` (trim/quiet/parse). Brain-lane verb `chat_history_compactor_digest_v1` with orch-side **chat → quick** retry. DAL upsert-by-index (not GitHub’s supersede-slot). No new bus channels, schema registry kinds, or env keys.

**Tech Stack:** Python 3, Pydantic v2, asyncpg (`RECALL_PG_DSN` on cortex-orch), pytest + pytest-asyncio, Jinja2 verb prompt, zoneinfo for Denver day windows.

**Design source:** `docs/superpowers/specs/2026-07-09-chat-history-compactor-design.md`

**Sibling pattern:** `docs/superpowers/plans/2026-07-08-github-compactor.md` — copy structure, but **indexed upsert** instead of `compactor_slot` supersede.

---

## Context for the implementer (read before starting)

Work in a clean branch/worktree from repo root `/mnt/scripts/Orion-Sapienform`.

Verified facts:

- Discussion window skill: `skills.chat.discussion_window.v1` in `services/orion-cortex-exec/app/verb_adapters.py`; request/result schemas in `orion/schemas/discussion_window.py`.
- Journal discussion workflow already calls that skill from `services/orion-cortex-orch/app/workflow_runtime.py` (`_execute_journal_discussion_window_pass`) — copy the skill-call envelope pattern.
- GitHub compactor uses `compactor_slot` + `supersede_and_insert_compactor_card`. Chat must use **`compactor_index` + in-place update or insert** (many coexisting cards).
- Digest LLM: set `options["llm_route"]="chat"` first; on empty/invalid/structured rejection, one retry with `llm_route="quick"`.
- Structured digest output: orch parses `final_text` JSON (optional metadata key `chat_history_compactor_digest` for tests). Add verb to `_structured_output_expected` in `services/orion-cortex-exec/app/router.py`.
- Schedule store: `WorkflowScheduleStore.upsert_from_dispatch(WorkflowDispatchRequestV1)` in `services/orion-actions/app/workflow_schedule_store.py`. Bootstrap after store init in `lifespan` in `services/orion-actions/app/main.py`.
- **No** changes to `orion/bus/channels.yaml` or `orion/schemas/registry.py`.
- **No** new env keys — reuse `DATABASE_URL` / `ENDOGENOUS_RUNTIME_SQL_DATABASE_URL` (exec) and `RECALL_PG_DSN` (orch).

### Distinct from GitHub compactor

| | GitHub | Chat (this plan) |
|--|--------|------------------|
| Key | `compactor_slot=repo_dev_snapshot` | `compactor_index=chat_compactor:…` |
| Persist | supersede old + insert new | update same card or insert |
| Source | GitHub PRs | `chat_history_log` via discussion window |
| Quiet day | journal stub; card unchanged | skip card; optional journal stub |
| Schedule | operator-created | bootstrap daily 06:00 Denver |

### Test commands

```bash
pytest orion/cognition/compactor/tests -q
pytest orion/cognition/chat_history_compactor/tests -q
pytest tests/test_indexed_compactor_memory_cards.py -q
pytest services/orion-cortex-exec/tests/test_router_final_text_assembly.py -k chat_history_compactor -q
pytest services/orion-cortex-orch/tests/test_workflow_lane.py -k chat_history_compactor -q
pytest services/orion-hub/tests/test_workflow_request_builder.py -k chat_history_compactor -q
pytest services/orion-actions/tests -k chat_history_compactor_schedule_bootstrap -q
```

Full gate before PR:

```bash
pytest orion/cognition/compactor/tests -q
pytest orion/cognition/chat_history_compactor/tests -q
pytest tests/test_indexed_compactor_memory_cards.py -q
pytest services/orion-cortex-exec/tests/test_router_final_text_assembly.py -q
pytest services/orion-cortex-orch/tests/test_workflow_lane.py -q
pytest services/orion-hub/tests/test_workflow_request_builder.py -q
pytest services/orion-actions/tests -k chat_history_compactor_schedule_bootstrap -q
```

---

## File structure

| File | Responsibility |
|------|----------------|
| `orion/cognition/compactor/__init__.py` | Package export for index helpers |
| `orion/cognition/compactor/index.py` | `build_compactor_index` (day / rolling keys) |
| `orion/cognition/compactor/budget.py` | Generic char-budget assertion |
| `orion/cognition/compactor/tests/test_index.py` | Key determinism tests |
| `orion/cognition/compactor/tests/test_budget.py` | Budget fail-loud tests |
| `orion/schemas/actions/chat_history_compactor.py` | `ChatHistoryCompactorDigestV1` |
| `orion/cognition/chat_history_compactor/constants.py` | Tag, budgets, max turns, defaults |
| `orion/cognition/chat_history_compactor/window.py` | Resolve day/rolling window + lookback from request |
| `orion/cognition/chat_history_compactor/digest.py` | Trim input, quiet stub, parse, journal entry id |
| `orion/cognition/chat_history_compactor/tests/test_window.py` | Window resolution tests |
| `orion/cognition/chat_history_compactor/tests/test_digest.py` | Digest helper tests |
| `orion/core/contracts/memory_cards.py` | Add `chat_compactor` provenance |
| `orion/core/storage/memory_cards.py` | `find_active_card_by_compactor_index`, `upsert_indexed_compactor_card` |
| `orion/core/storage/__init__.py` | Re-export new DAL helpers |
| `tests/test_indexed_compactor_memory_cards.py` | Mocked asyncpg upsert tests |
| `orion/cognition/prompts/chat_history_compactor_digest_v1.j2` | LLM prompt |
| `orion/cognition/verbs/chat_history_compactor_digest_v1.yaml` | Verb registration |
| `services/orion-cortex-exec/app/router.py` | Structured-output allowlist |
| `services/orion-cortex-exec/app/executor.py` | Max-tokens branch for digest verb |
| `services/orion-cortex-orch/app/chat_history_compactor_memory.py` | Pool + upsert wrapper |
| `services/orion-cortex-orch/app/workflow_runtime.py` | `_execute_chat_history_compactor_pass` + digest runner + dispatch |
| `orion/cognition/workflows/registry.py` | Register workflow |
| `services/orion-hub/templates/index.html` | Skill Runner options |
| `services/orion-hub/static/js/app.js` | Workflow catalogue |
| `services/orion-hub/static/js/workflow-ui.js` | Display label |
| `docs/operator_skill_prompt_catalogue.md` | Document new prompts |
| `services/orion-actions/app/workflow_schedule_bootstrap.py` | Seed daily schedule |
| `services/orion-actions/app/main.py` | Call bootstrap on startup |
| `services/orion-actions/README.md` | Operator docs |
| `services/orion-actions/tests/test_chat_history_compactor_schedule_bootstrap.py` | Bootstrap idempotency |

Build order: index/budget → schema/constants/window/digest → DAL → LLM verb → orch workflow → Hub → schedule bootstrap.

---

### Task 1: Shared `compactor_index` builders

**Files:**
- Create: `orion/cognition/compactor/__init__.py`
- Create: `orion/cognition/compactor/index.py`
- Create: `orion/cognition/compactor/tests/__init__.py`
- Create: `orion/cognition/compactor/tests/test_index.py`

- [ ] **Step 1: Write the failing index tests**

```python
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from orion.cognition.compactor.index import build_compactor_index


def test_build_compactor_index_day() -> None:
    key = build_compactor_index(
        kind="chat_history_log",
        mode="day",
        calendar_date="2026-07-08",
    )
    assert key == "chat_compactor:day:2026-07-08"


def test_build_compactor_index_rolling_floors_to_minute() -> None:
    start = datetime(2026, 7, 9, 4, 0, 37, tzinfo=ZoneInfo("America/Denver"))
    key = build_compactor_index(
        kind="chat_history_log",
        mode="rolling",
        lookback_hours=6,
        window_start=start,
    )
    assert key == "chat_compactor:rolling:6h:2026-07-09T04:00:00-06:00"


def test_build_compactor_index_day_requires_calendar_date() -> None:
    with pytest.raises(ValueError, match="calendar_date"):
        build_compactor_index(kind="chat_history_log", mode="day")


def test_build_compactor_index_rolling_requires_hours_and_start() -> None:
    with pytest.raises(ValueError, match="lookback_hours"):
        build_compactor_index(
            kind="chat_history_log",
            mode="rolling",
            window_start=datetime(2026, 7, 9, 4, 0, tzinfo=ZoneInfo("America/Denver")),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/cognition/compactor/tests/test_index.py -v`

Expected: FAIL with `ModuleNotFoundError: orion.cognition.compactor`.

- [ ] **Step 3: Implement index helpers**

Create `orion/cognition/compactor/index.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Literal


def build_compactor_index(
    *,
    kind: str,
    mode: Literal["day", "rolling"],
    calendar_date: str | None = None,
    lookback_hours: int | None = None,
    window_start: datetime | None = None,
) -> str:
    """Stable window key for indexed compactor memory cards.

    v1 kind is always ``chat_history_log``. Future kinds (e.g. journal) may
    add filter dimensions to the key — do not invent them here.
    """
    _ = kind  # reserved for future key namespaces; v1 always chat_compactor prefix
    if mode == "day":
        date = (calendar_date or "").strip()
        if not date:
            raise ValueError("calendar_date required for mode=day")
        return f"chat_compactor:day:{date}"
    if mode == "rolling":
        if lookback_hours is None or int(lookback_hours) < 1:
            raise ValueError("lookback_hours required for mode=rolling")
        if window_start is None:
            raise ValueError("window_start required for mode=rolling")
        floored = window_start.replace(second=0, microsecond=0)
        start_iso = floored.isoformat()
        return f"chat_compactor:rolling:{int(lookback_hours)}h:{start_iso}"
    raise ValueError(f"unsupported_compactor_mode:{mode}")
```

Create `orion/cognition/compactor/__init__.py`:

```python
from orion.cognition.compactor.index import build_compactor_index

__all__ = ["build_compactor_index"]
```

Create empty `orion/cognition/compactor/tests/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest orion/cognition/compactor/tests/test_index.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/cognition/compactor/
git commit -m "feat: add shared compactor_index window key builders"
```

---

### Task 2: Schema, constants, budget, window resolution

**Files:**
- Create: `orion/schemas/actions/chat_history_compactor.py`
- Create: `orion/cognition/compactor/budget.py`
- Create: `orion/cognition/compactor/tests/test_budget.py`
- Create: `orion/cognition/chat_history_compactor/__init__.py`
- Create: `orion/cognition/chat_history_compactor/constants.py`
- Create: `orion/cognition/chat_history_compactor/window.py`
- Create: `orion/cognition/chat_history_compactor/tests/__init__.py`
- Create: `orion/cognition/chat_history_compactor/tests/test_window.py`
- Create: `orion/cognition/chat_history_compactor/tests/test_digest.py` (schema test only first)
- Modify: `orion/core/contracts/memory_cards.py` (add provenance)

- [ ] **Step 1: Write failing schema + budget + window tests**

Create `orion/cognition/chat_history_compactor/tests/test_digest.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.schemas.actions.chat_history_compactor import ChatHistoryCompactorDigestV1


def test_chat_history_compactor_digest_v1_rejects_empty_card_summary() -> None:
    with pytest.raises(ValidationError):
        ChatHistoryCompactorDigestV1(
            card_summary="",
            journal_title="Title",
            journal_body="Body",
            turn_refs=["corr-1"],
        )
```

Create `orion/cognition/compactor/tests/test_budget.py`:

```python
from __future__ import annotations

import pytest

from orion.cognition.compactor.budget import assert_fields_within_budget


def test_assert_fields_within_budget_raises_named_error() -> None:
    with pytest.raises(ValueError, match="compactor_output_over_budget:card_summary"):
        assert_fields_within_budget({"card_summary": ("x" * 10, 5)})
```

Create `orion/cognition/chat_history_compactor/tests/test_window.py`:

```python
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from orion.cognition.chat_history_compactor.window import (
    ResolvedChatCompactorWindow,
    resolve_chat_compactor_window,
)


def test_resolve_day_window_uses_yesterday_denver() -> None:
    now = datetime(2026, 7, 9, 6, 0, tzinfo=ZoneInfo("America/Denver"))
    resolved = resolve_chat_compactor_window(
        window_mode="day",
        lookback_hours=None,
        now=now,
        user_text="",
        workflow_request={"window_mode": "day"},
    )
    assert isinstance(resolved, ResolvedChatCompactorWindow)
    assert resolved.mode == "day"
    assert resolved.calendar_date == "2026-07-08"
    assert resolved.compactor_index == "chat_compactor:day:2026-07-08"
    assert resolved.lookback_seconds == 86400
    assert resolved.window_end.astimezone(ZoneInfo("America/Denver")).date().isoformat() == "2026-07-08"


def test_resolve_rolling_from_prompt_six_hours() -> None:
    now = datetime(2026, 7, 9, 10, 0, tzinfo=ZoneInfo("America/Denver"))
    resolved = resolve_chat_compactor_window(
        window_mode=None,
        lookback_hours=None,
        now=now,
        user_text="Compact the last 6 hours of chat into a memory digest.",
        workflow_request={},
    )
    assert resolved.mode == "rolling"
    assert resolved.lookback_hours == 6
    assert resolved.compactor_index.startswith("chat_compactor:rolling:6h:")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest orion/cognition/chat_history_compactor/tests/test_digest.py::test_chat_history_compactor_digest_v1_rejects_empty_card_summary \
  orion/cognition/compactor/tests/test_budget.py \
  orion/cognition/chat_history_compactor/tests/test_window.py -v
```

Expected: FAIL with import errors.

- [ ] **Step 3: Implement schema, constants, budget, window, provenance**

Create `orion/schemas/actions/chat_history_compactor.py`:

```python
from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatHistoryCompactorDigestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_summary: str = Field(..., min_length=1)
    journal_title: str = Field(default="")
    journal_body: str = Field(default="")
    turn_refs: List[str] = Field(default_factory=list)

    @field_validator("turn_refs")
    @classmethod
    def _normalize_turn_refs(cls, value: List[str]) -> List[str]:
        return [str(item).strip() for item in value if str(item).strip()]
```

Create `orion/cognition/compactor/budget.py`:

```python
from __future__ import annotations


def assert_fields_within_budget(fields: dict[str, tuple[str, int]]) -> None:
    """Raise ``compactor_output_over_budget:<field>`` when any value exceeds max chars."""
    for name, (value, max_chars) in fields.items():
        if len(value or "") > int(max_chars):
            raise ValueError(f"compactor_output_over_budget:{name}")
```

Create `orion/cognition/chat_history_compactor/constants.py`:

```python
from __future__ import annotations

CHAT_DEV_DIGEST_TAG = "chat_dev_digest"
COMPACTOR_KIND = "chat_history_log"
DEFAULT_TIMEZONE = "America/Denver"
DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_MAX_TURNS = 30

CARD_SUMMARY_MAX_CHARS = 800
JOURNAL_TITLE_MAX_CHARS = 120
JOURNAL_BODY_MAX_CHARS = 4000

# Per-turn trim before digest LLM
DIGEST_TURN_PROMPT_MAX_CHARS = 400
DIGEST_TURN_RESPONSE_MAX_CHARS = 600
```

Create `orion/cognition/chat_history_compactor/window.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, time, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo

from orion.cognition.chat_history_compactor.constants import (
    DEFAULT_LOOKBACK_HOURS,
    DEFAULT_TIMEZONE,
)
from orion.cognition.compactor.index import build_compactor_index


@dataclass(frozen=True)
class ResolvedChatCompactorWindow:
    mode: Literal["day", "rolling"]
    compactor_index: str
    window_start: datetime
    window_end: datetime
    lookback_seconds: int
    lookback_hours: int | None
    calendar_date: str | None
    timezone_name: str


_HOURS_RE = re.compile(
    r"\blast\s+(\d+)\s+hours?\b|\b(\d+)\s*h(?:our)?s?\b",
    re.IGNORECASE,
)


def _parse_lookback_hours_from_text(user_text: str) -> int | None:
    m = _HOURS_RE.search(user_text or "")
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    try:
        hours = int(raw)
    except (TypeError, ValueError):
        return None
    if hours < 1:
        return None
    return min(hours, 24 * 14)


def resolve_chat_compactor_window(
    *,
    window_mode: str | None,
    lookback_hours: int | None,
    now: datetime,
    user_text: str,
    workflow_request: dict[str, Any],
) -> ResolvedChatCompactorWindow:
    tz_name = DEFAULT_TIMEZONE
    tz = ZoneInfo(tz_name)
    now_local = now.astimezone(tz) if now.tzinfo else now.replace(tzinfo=timezone.utc).astimezone(tz)

    req = workflow_request if isinstance(workflow_request, dict) else {}
    mode_raw = (window_mode or req.get("window_mode") or "").strip().lower()
    if not mode_raw:
        mode_raw = "day" if req.get("scheduled_dispatch") else "rolling"

    hours_raw = lookback_hours if lookback_hours is not None else req.get("lookback_hours")
    hours: int | None = None
    if hours_raw is not None:
        try:
            hours = max(1, int(hours_raw))
        except (TypeError, ValueError):
            hours = None
    if hours is None:
        hours = _parse_lookback_hours_from_text(user_text)

    if mode_raw == "day":
        yesterday = now_local.date() - timedelta(days=1)
        start_local = datetime.combine(yesterday, time.min, tzinfo=tz)
        end_local = datetime.combine(yesterday, time(23, 59, 59, 999000), tzinfo=tz)
        calendar_date = yesterday.isoformat()
        index = build_compactor_index(
            kind="chat_history_log",
            mode="day",
            calendar_date=calendar_date,
        )
        lookback_seconds = int((end_local - start_local).total_seconds()) + 1
        return ResolvedChatCompactorWindow(
            mode="day",
            compactor_index=index,
            window_start=start_local.astimezone(timezone.utc),
            window_end=end_local.astimezone(timezone.utc),
            lookback_seconds=lookback_seconds,
            lookback_hours=None,
            calendar_date=calendar_date,
            timezone_name=tz_name,
        )

    # rolling (default for on-demand)
    hours = hours or DEFAULT_LOOKBACK_HOURS
    end_local = now_local
    start_local = end_local - timedelta(hours=hours)
    index = build_compactor_index(
        kind="chat_history_log",
        mode="rolling",
        lookback_hours=hours,
        window_start=start_local,
    )
    return ResolvedChatCompactorWindow(
        mode="rolling",
        compactor_index=index,
        window_start=start_local.astimezone(timezone.utc),
        window_end=end_local.astimezone(timezone.utc),
        lookback_seconds=max(1, int(hours * 3600)),
        lookback_hours=hours,
        calendar_date=None,
        timezone_name=tz_name,
    )
```

Create `orion/cognition/chat_history_compactor/__init__.py`:

```python
from orion.cognition.chat_history_compactor.constants import CHAT_DEV_DIGEST_TAG

__all__ = ["CHAT_DEV_DIGEST_TAG"]
```

In `orion/core/contracts/memory_cards.py`, extend `MemoryProvenance`:

```python
MemoryProvenance = Literal[
    "operator_highlight",
    "operator_distiller",
    "auto_extractor",
    "imported",
    "repo_compactor",
    "chat_compactor",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest orion/cognition/chat_history_compactor/tests/test_digest.py::test_chat_history_compactor_digest_v1_rejects_empty_card_summary \
  orion/cognition/compactor/tests/test_budget.py \
  orion/cognition/chat_history_compactor/tests/test_window.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/actions/chat_history_compactor.py orion/cognition/compactor/budget.py \
  orion/cognition/compactor/tests/test_budget.py orion/cognition/chat_history_compactor/ \
  orion/core/contracts/memory_cards.py
git commit -m "feat: add chat history compactor schema, budget, and window resolution"
```

---

### Task 3: Indexed memory-card DAL upsert

**Files:**
- Modify: `orion/core/storage/memory_cards.py`
- Modify: `orion/core/storage/__init__.py`
- Create: `tests/test_indexed_compactor_memory_cards.py`

- [ ] **Step 1: Write failing DAL tests**

Create `tests/test_indexed_compactor_memory_cards.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orion.core.contracts.memory_cards import MemoryCardCreateV1
from orion.core.storage import memory_cards as mc_dal


INDEX = "chat_compactor:day:2026-07-08"


@pytest.mark.asyncio
async def test_upsert_indexed_compactor_card_updates_existing() -> None:
    existing_id = uuid4()
    conn = AsyncMock()
    # FOR UPDATE lookup returns existing card_id
    conn.fetchrow = AsyncMock(
        side_effect=[
            {"card_id": existing_id},
            # update_card path snapshots — simplified: update returns row via fetchrow after UPDATE
            {
                "card_id": existing_id,
                "slug": "chat-digest",
                "types": ["fact"],
                "anchor_class": "event",
                "status": "active",
                "confidence": "likely",
                "sensitivity": "private",
                "priority": "high_recall",
                "visibility_scope": ["chat"],
                "time_horizon": None,
                "provenance": "chat_compactor",
                "trust_source": None,
                "project": None,
                "title": "Chat digest",
                "summary": "Updated summary",
                "still_true": None,
                "anchors": None,
                "tags": ["chat_dev_digest"],
                "evidence": [],
                "subschema": {"compactor_index": INDEX},
                "created_at": "2026-07-09T00:00:00+00:00",
                "updated_at": "2026-07-09T01:00:00+00:00",
            },
            {"j": {"card_id": str(existing_id), "summary": "old"}},
            {"j": {"card_id": str(existing_id), "summary": "Updated summary"}},
        ]
    )
    conn.execute = AsyncMock()
    conn.transaction = MagicMock()
    conn.transaction.return_value.__aenter__ = AsyncMock(return_value=None)
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)

    pool = AsyncMock()
    pool.acquire = MagicMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = acquire_cm

    card = MemoryCardCreateV1(
        types=["fact"],
        anchor_class="event",
        status="active",
        priority="high_recall",
        provenance="chat_compactor",
        title="Chat digest (2026-07-08)",
        summary="Updated summary",
        tags=["chat_dev_digest"],
        subschema={"compactor_index": INDEX, "compactor_kind": "chat_history_log"},
    )

    result_id = await mc_dal.upsert_indexed_compactor_card(
        pool,
        index=INDEX,
        card=card,
        actor="chat_history_compactor_pass",
    )
    assert result_id == existing_id
    # Must not supersede
    for call in conn.execute.await_args_list:
        sql = str(call.args[0]).lower()
        assert "superseded" not in sql


@pytest.mark.asyncio
async def test_upsert_indexed_compactor_card_inserts_when_missing() -> None:
    new_id = uuid4()
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(side_effect=[None])  # no existing
    conn.execute = AsyncMock()
    conn.transaction = MagicMock()
    conn.transaction.return_value.__aenter__ = AsyncMock(return_value=None)
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)

    pool = AsyncMock()
    pool.acquire = MagicMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = acquire_cm

    card = MemoryCardCreateV1(
        types=["fact"],
        anchor_class="event",
        status="active",
        priority="high_recall",
        provenance="chat_compactor",
        title="Chat digest",
        summary="Fresh digest",
        tags=["chat_dev_digest"],
        subschema={"compactor_index": INDEX},
    )

    async def _fake_insert(_conn, _card, *, actor, op="create"):
        assert op == "create"
        return new_id

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mc_dal, "_insert_card_on_conn", _fake_insert)
    try:
        result_id = await mc_dal.upsert_indexed_compactor_card(
            pool,
            index=INDEX,
            card=card,
            actor="chat_history_compactor_pass",
        )
    finally:
        monkeypatch.undo()

    assert result_id == new_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_indexed_compactor_memory_cards.py -v`

Expected: FAIL with `AttributeError: upsert_indexed_compactor_card`.

- [ ] **Step 3: Implement DAL helpers**

Append to `orion/core/storage/memory_cards.py` (near the existing `supersede_and_insert_compactor_card`):

```python
async def find_active_card_by_compactor_index(pool: asyncpg.Pool, index: str) -> Optional[MemoryCardV1]:
    key = (index or "").strip()
    if not key:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM memory_cards
            WHERE status = 'active'
              AND subschema ->> 'compactor_index' = $1
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            key,
        )
        return _row_to_card(row) if row else None


async def upsert_indexed_compactor_card(
    pool: asyncpg.Pool,
    *,
    index: str,
    card: MemoryCardCreateV1,
    actor: str,
) -> UUID:
    """Update the active card for ``compactor_index`` in place, or insert a new one.

    Unlike ``supersede_and_insert_compactor_card``, this keeps one live card per
    index key and writes history op ``update`` or ``create`` (never supersede).
    """
    key = (index or "").strip()
    if not key:
        raise ValueError("compactor index required")
    sub = dict(card.subschema or {})
    sub["compactor_index"] = key
    card = card.model_copy(update={"subschema": sub})

    async with pool.acquire() as conn:
        async with conn.transaction():
            existing = await conn.fetchrow(
                """
                SELECT card_id FROM memory_cards
                WHERE status = 'active'
                  AND subschema ->> 'compactor_index' = $1
                ORDER BY updated_at DESC
                LIMIT 1
                FOR UPDATE
                """,
                key,
            )
            if existing is None:
                return await _insert_card_on_conn(conn, card, actor=actor, op="create")

            cid = existing["card_id"]
            before = await conn.fetchrow(
                "SELECT to_jsonb(mc.*) AS j FROM memory_cards mc WHERE mc.card_id = $1",
                cid,
            )
            th_val = _jsonb_param(
                card.time_horizon.model_dump(mode="json") if card.time_horizon else None
            )
            evidence_val = _jsonb_param([e.model_dump(mode="json") for e in card.evidence])
            sub_val = _jsonb_param(dict(card.subschema or {}))
            await conn.fetchrow(
                """
                UPDATE memory_cards SET
                  title = $2,
                  summary = $3,
                  tags = $4,
                  evidence = $5::jsonb,
                  time_horizon = $6::jsonb,
                  subschema = $7::jsonb,
                  priority = $8,
                  provenance = $9,
                  updated_at = now()
                WHERE card_id = $1
                RETURNING *
                """,
                cid,
                card.title,
                card.summary,
                card.tags,
                evidence_val,
                th_val,
                sub_val,
                card.priority,
                card.provenance,
            )
            after = await conn.fetchrow(
                "SELECT to_jsonb(mc.*) AS j FROM memory_cards mc WHERE mc.card_id = $1",
                cid,
            )
            await conn.execute(
                """
                INSERT INTO memory_card_history (card_id, edge_id, op, actor, "before", "after")
                VALUES ($1, NULL, 'update', $2, $3::jsonb, $4::jsonb)
                """,
                cid,
                actor,
                json.dumps(before["j"] if before else {}),
                json.dumps(after["j"] if after else {}),
            )
            return cid
```

Update `orion/core/storage/__init__.py` exports:

```python
from orion.core.storage.memory_cards import (  # noqa: F401
    apply_memory_cards_schema,
    add_edge,
    card_exists_by_fingerprint,
    find_active_card_by_compactor_index,
    find_active_card_by_compactor_slot,
    get_card,
    insert_card,
    list_cards,
    list_edges,
    list_history,
    remove_edge,
    reverse_history,
    supersede_and_insert_compactor_card,
    update_card,
    upsert_indexed_compactor_card,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_indexed_compactor_memory_cards.py -v`

Expected: PASS

If the update-path mock is brittle against `_row_to_card`, simplify the update test to only assert: (1) `FOR UPDATE` SQL contains `compactor_index`, (2) no `superseded` in execute SQL, (3) returned UUID equals `existing_id`. Prefer adjusting the mock over weakening the contract.

- [ ] **Step 5: Commit**

```bash
git add orion/core/storage/memory_cards.py orion/core/storage/__init__.py tests/test_indexed_compactor_memory_cards.py
git commit -m "feat: upsert memory cards by compactor_index without supersede"
```

---

### Task 4: Digest pure helpers

**Files:**
- Create: `orion/cognition/chat_history_compactor/digest.py`
- Modify: `orion/cognition/chat_history_compactor/tests/test_digest.py`

- [ ] **Step 1: Append failing digest helper tests**

Append to `orion/cognition/chat_history_compactor/tests/test_digest.py`:

```python
from orion.cognition.chat_history_compactor.constants import CARD_SUMMARY_MAX_CHARS
from orion.cognition.chat_history_compactor.digest import (
    assert_chat_compactor_digest_within_budget,
    build_quiet_day_chat_digest,
    parse_chat_history_compactor_digest_json,
    stable_chat_compactor_journal_entry_id,
    trim_chat_history_compactor_input,
)
from orion.schemas.discussion_window import DiscussionWindowResultV1, DiscussionWindowTurnV1
from datetime import datetime, timezone


def test_assert_chat_compactor_digest_within_budget() -> None:
    digest = ChatHistoryCompactorDigestV1(
        card_summary="x" * (CARD_SUMMARY_MAX_CHARS + 1),
        journal_title="t",
        journal_body="b",
        turn_refs=[],
    )
    with pytest.raises(ValueError, match="compactor_output_over_budget:card_summary"):
        assert_chat_compactor_digest_within_budget(digest)


def test_build_quiet_day_chat_digest() -> None:
    digest = build_quiet_day_chat_digest(window_label="2026-07-08")
    assert "No Hub chat turns" in digest.card_summary or "quiet" in digest.card_summary.lower() or "No" in digest.card_summary
    assert digest.turn_refs == []


def test_stable_chat_compactor_journal_entry_id_is_deterministic() -> None:
    a = stable_chat_compactor_journal_entry_id(
        workflow_id="chat_history_compactor_pass",
        compactor_index="chat_compactor:day:2026-07-08",
    )
    b = stable_chat_compactor_journal_entry_id(
        workflow_id="chat_history_compactor_pass",
        compactor_index="chat_compactor:day:2026-07-08",
    )
    assert a == b


def test_parse_chat_history_compactor_digest_json() -> None:
    raw = '{"card_summary":"Talked about memory cards.","journal_title":"Chat digest","journal_body":"Details.","turn_refs":["c1"]}'
    digest = parse_chat_history_compactor_digest_json(raw)
    assert digest.card_summary.startswith("Talked")
    assert digest.turn_refs == ["c1"]


def test_trim_chat_history_compactor_input_bounds_turns() -> None:
    turns = [
        DiscussionWindowTurnV1(
            created_at=datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc),
            correlation_id=f"c{i}",
            prompt="p" * 2000,
            response="r" * 2000,
        )
        for i in range(5)
    ]
    window = DiscussionWindowResultV1(
        window_start_utc=datetime(2026, 7, 8, 0, 0, tzinfo=timezone.utc),
        window_end_utc=datetime(2026, 7, 8, 23, 59, tzinfo=timezone.utc),
        turn_count=5,
        turns=turns,
        transcript_text="ignored",
    )
    trimmed = trim_chat_history_compactor_input(window, max_turns=3)
    assert len(trimmed["turns"]) == 3
    assert trimmed["turn_count"] == 5
    assert trimmed["turns_truncated_for_digest"] is True
    assert len(trimmed["turns"][0]["prompt"]) <= 401
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest orion/cognition/chat_history_compactor/tests/test_digest.py -v`

Expected: FAIL with `ModuleNotFoundError: orion.cognition.chat_history_compactor.digest`.

- [ ] **Step 3: Implement digest helpers**

Create `orion/cognition/chat_history_compactor/digest.py`:

```python
from __future__ import annotations

import json
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from orion.cognition.chat_history_compactor.constants import (
    CARD_SUMMARY_MAX_CHARS,
    DEFAULT_MAX_TURNS,
    DIGEST_TURN_PROMPT_MAX_CHARS,
    DIGEST_TURN_RESPONSE_MAX_CHARS,
    JOURNAL_BODY_MAX_CHARS,
    JOURNAL_TITLE_MAX_CHARS,
)
from orion.cognition.compactor.budget import assert_fields_within_budget
from orion.schemas.actions.chat_history_compactor import ChatHistoryCompactorDigestV1
from orion.schemas.discussion_window import DiscussionWindowResultV1

_COMPACTOR_JOURNAL_ENTRY_NS = NAMESPACE_URL


def trim_chat_history_compactor_input(
    window: DiscussionWindowResultV1,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> dict[str, Any]:
    turns = list(window.turns or [])
    total = len(turns)
    # Keep newest suffix (discussion window already contiguous-suffix oriented)
    selected = turns[-max_turns:] if max_turns > 0 else []
    compact_turns = []
    for turn in selected:
        prompt = str(turn.prompt or "")
        response = str(turn.response or "")
        if len(prompt) > DIGEST_TURN_PROMPT_MAX_CHARS:
            prompt = prompt[:DIGEST_TURN_PROMPT_MAX_CHARS].rstrip() + "…"
        if len(response) > DIGEST_TURN_RESPONSE_MAX_CHARS:
            response = response[:DIGEST_TURN_RESPONSE_MAX_CHARS].rstrip() + "…"
        compact_turns.append(
            {
                "created_at": turn.created_at.isoformat() if turn.created_at else None,
                "correlation_id": turn.correlation_id,
                "user_id": turn.user_id,
                "source": turn.source,
                "prompt": prompt,
                "response": response,
            }
        )
    payload: dict[str, Any] = {
        "window_start_utc": window.window_start_utc.isoformat(),
        "window_end_utc": window.window_end_utc.isoformat(),
        "turn_count": int(window.turn_count),
        "selection_strategy": window.selection_strategy,
        "turns": compact_turns,
    }
    if total > len(selected):
        payload["turns_truncated_for_digest"] = True
        payload["turns_total"] = total
    return payload


def assert_chat_compactor_digest_within_budget(digest: ChatHistoryCompactorDigestV1) -> None:
    assert_fields_within_budget(
        {
            "card_summary": (digest.card_summary, CARD_SUMMARY_MAX_CHARS),
            "journal_title": (digest.journal_title or "", JOURNAL_TITLE_MAX_CHARS),
            "journal_body": (digest.journal_body or "", JOURNAL_BODY_MAX_CHARS),
        }
    )


def build_quiet_day_chat_digest(*, window_label: str) -> ChatHistoryCompactorDigestV1:
    label = (window_label or "window").strip()
    return ChatHistoryCompactorDigestV1(
        card_summary=f"No Hub chat turns in {label}.",
        journal_title=f"Chat digest — {label} (quiet)",
        journal_body=(
            f"No chat_history_log turns were found for {label}. "
            "No indexed chat digest memory card was written."
        ),
        turn_refs=[],
    )


def parse_chat_history_compactor_digest_json(raw: str) -> ChatHistoryCompactorDigestV1:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("compactor_digest_not_object")
    return ChatHistoryCompactorDigestV1.model_validate(payload)


def stable_chat_compactor_journal_entry_id(*, workflow_id: str, compactor_index: str) -> str:
    payload = "|".join([workflow_id.strip(), compactor_index.strip()])
    return str(uuid5(_COMPACTOR_JOURNAL_ENTRY_NS, payload))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest orion/cognition/chat_history_compactor/tests/test_digest.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/cognition/chat_history_compactor/digest.py orion/cognition/chat_history_compactor/tests/test_digest.py
git commit -m "feat: add chat history compactor digest helpers"
```

---

### Task 5: Brain-lane digest verb

**Files:**
- Create: `orion/cognition/prompts/chat_history_compactor_digest_v1.j2`
- Create: `orion/cognition/verbs/chat_history_compactor_digest_v1.yaml`
- Modify: `services/orion-cortex-exec/app/router.py`
- Modify: `services/orion-cortex-exec/app/executor.py`
- Modify: `services/orion-cortex-exec/tests/test_router_final_text_assembly.py`

- [ ] **Step 1: Write failing router allowlist test**

Append to `services/orion-cortex-exec/tests/test_router_final_text_assembly.py`:

```python
def test_router_allows_chat_history_compactor_digest_v1() -> None:
    assert _structured_output_expected("chat_history_compactor_digest_v1") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-cortex-exec/tests/test_router_final_text_assembly.py::test_router_allows_chat_history_compactor_digest_v1 -v`

Expected: FAIL (assertion False).

- [ ] **Step 3: Add prompt, verb YAML, router allowlist, max-tokens branch**

Create `orion/cognition/prompts/chat_history_compactor_digest_v1.j2`:

```jinja2
You compact a bounded Hub chat discussion window into Orion's indexed chat digest.

BOUNDARY
- Use ONLY the payload below.
- Do NOT use recall memory or unstated assumptions.
- Do not invent turns not present in input.

GROUNDED PAYLOAD (authoritative)
Input JSON is in metadata key `chat_history_compactor_input`. It contains:
- window_start_utc / window_end_utc
- turn_count
- turns: list of {created_at, correlation_id, prompt, response, user_id?, source?}

{{ metadata.get("chat_history_compactor_input", {}) | tojson(indent=2) }}

TASK
Return exactly one JSON object.
- The response MUST be valid JSON.
- The response MUST begin with `{` and end with `}`.
- Do NOT wrap the JSON in markdown fences.
- Do NOT include any explanatory prose before or after the JSON.
- Do NOT output `<think>` tags, chain-of-thought, reasoning traces, or hidden-analysis markers.
- Use exactly these keys at top level: `card_summary`, `journal_title`, `journal_body`, `turn_refs`.
- Set `card_summary` to a plain-language digest of what was discussed (max 800 chars).
- Set `journal_title` to a concise title (max 120 chars); may be empty for quiet paths.
- Set `journal_body` to a richer narrative (max 4000 chars); may be empty.
- Set `turn_refs` to an array of correlation_id strings from turns you relied on.

RESPONSE SHAPE (example)
{"card_summary":"<compact summary>","journal_title":"<title>","journal_body":"<narrative>","turn_refs":["corr-1"]}

RULES
- Prefer themes and decisions over quoting every turn.
- Never fabricate topics or correlation_ids not present in the payload.
- Global house digest: do not invent per-user privacy labels.
```

Create `orion/cognition/verbs/chat_history_compactor_digest_v1.yaml`:

```yaml
name: chat_history_compactor_digest_v1
label: Chat History Compactor Digest
description: >
  Compact a bounded chat_history_log discussion window into card + journal digest JSON.
  Uses only chat_history_compactor_input payload passed in metadata.

category: Generative
priority: high

requires_gpu: true
requires_memory: false
timeout_ms: 90000

services:
  - LLMGatewayService

prompt_template: chat_history_compactor_digest_v1.j2

steps:
  - name: chat_history_compactor_digest_v1
    order: 0
    services: [LLMGatewayService]
    prompt_template: chat_history_compactor_digest_v1.j2
```

In `services/orion-cortex-exec/app/router.py`, add `"chat_history_compactor_digest_v1"` to `_structured_output_expected` set (beside `github_compactor_digest_v1`).

In `services/orion-cortex-exec/app/executor.py`, beside the github max-tokens branch (~line 395):

```python
    if step.verb_name == "chat_history_compactor_digest_v1":
        return int(settings.llm_chat_general_max_tokens), requested, "settings.llm_chat_general_max_tokens_chat_history_compactor_digest"
```

- [ ] **Step 4: Run router test**

Run: `pytest services/orion-cortex-exec/tests/test_router_final_text_assembly.py::test_router_allows_chat_history_compactor_digest_v1 -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/cognition/prompts/chat_history_compactor_digest_v1.j2 \
  orion/cognition/verbs/chat_history_compactor_digest_v1.yaml \
  services/orion-cortex-exec/app/router.py \
  services/orion-cortex-exec/app/executor.py \
  services/orion-cortex-exec/tests/test_router_final_text_assembly.py
git commit -m "feat: add chat history compactor digest brain-lane verb"
```

---

### Task 6: Orch memory wrapper + workflow executor

**Files:**
- Create: `services/orion-cortex-orch/app/chat_history_compactor_memory.py`
- Modify: `services/orion-cortex-orch/app/workflow_runtime.py`
- Modify: `orion/cognition/workflows/registry.py`
- Modify: `services/orion-cortex-orch/tests/test_workflow_lane.py`

- [ ] **Step 1: Write failing workflow lane tests**

Append to `services/orion-cortex-orch/tests/test_workflow_lane.py` (reuse `DummyBus`, `DummyVerbResult`, `_req`):

```python
def test_chat_history_compactor_pass_upserts_card_and_writes_journal(monkeypatch) -> None:
    bus = DummyBus()
    card_calls: dict = {}
    digest_routes: list[str] = []

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs["client_request"]
        if req.verb == "skills.chat.discussion_window.v1":
            skill = {
                "window_start_utc": "2026-07-09T04:00:00+00:00",
                "window_end_utc": "2026-07-09T10:00:00+00:00",
                "turn_count": 1,
                "turns": [
                    {
                        "created_at": "2026-07-09T05:00:00+00:00",
                        "correlation_id": "corr-a",
                        "prompt": "How is the memory card upsert going?",
                        "response": "Indexed by compactor_index.",
                    }
                ],
                "transcript_text": "user: How is the memory card upsert going?\norion: Indexed by compactor_index.",
                "selection_strategy": "time_bound_then_contiguous_suffix",
            }
            return DummyVerbResult(
                payload={
                    "result": {
                        "status": "success",
                        "final_text": json.dumps(skill),
                        "metadata": {"skill_result": skill},
                    }
                }
            )
        if req.verb == "chat_history_compactor_digest_v1":
            digest_routes.append(str((req.options or {}).get("llm_route") or ""))
            digest = {
                "card_summary": "Discussed indexed memory card upserts.",
                "journal_title": "Chat digest — rolling 6h",
                "journal_body": "Talked through compactor_index upsert semantics.",
                "turn_refs": ["corr-a"],
            }
            return DummyVerbResult(
                payload={
                    "result": {
                        "status": "success",
                        "final_text": json.dumps(digest),
                        "metadata": {"chat_history_compactor_digest": digest},
                    }
                }
            )
        raise AssertionError(f"unexpected verb {req.verb}")

    async def _fake_persist_card(**kwargs):
        card_calls.update(kwargs)
        return uuid4()

    monkeypatch.setattr(
        "app.workflow_runtime.persist_chat_history_compactor_memory_card",
        _fake_persist_card,
    )

    req = _req("chat_history_compactor_pass")
    req.context.raw_user_text = "Compact the last 6 hours of chat into a memory digest."
    req.context.user_message = req.context.raw_user_text

    result = asyncio.run(
        execute_chat_workflow(
            bus=bus,
            source=ServiceRef(name="cortex-orch"),
            req=req,
            correlation_id="00000000-0000-0000-0000-000000000301",
            causality_chain=[],
            trace={},
            call_verb_runtime=_fake_call_verb_runtime,
        )
    )

    assert result.ok is True
    assert result.metadata["workflow"]["workflow_id"] == "chat_history_compactor_pass"
    assert result.metadata["workflow"]["turn_count"] == 1
    assert card_calls.get("digest") is not None
    assert any(ch == "orion:journal:write" for ch, _ in bus.published)
    assert digest_routes[0] == "chat"


def test_chat_history_compactor_pass_quiet_day_skips_card(monkeypatch) -> None:
    bus = DummyBus()
    card_called = {"n": 0}

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs["client_request"]
        if req.verb == "skills.chat.discussion_window.v1":
            skill = {
                "window_start_utc": "2026-07-08T06:00:00+00:00",
                "window_end_utc": "2026-07-09T05:59:59.999000+00:00",
                "turn_count": 0,
                "turns": [],
                "transcript_text": "",
                "selection_strategy": "time_bound_then_contiguous_suffix",
            }
            return DummyVerbResult(
                payload={
                    "result": {
                        "status": "success",
                        "final_text": json.dumps(skill),
                        "metadata": {"skill_result": skill},
                    }
                }
            )
        raise AssertionError(f"unexpected verb {req.verb}")

    async def _fake_persist_card(**kwargs):
        card_called["n"] += 1
        return uuid4()

    monkeypatch.setattr(
        "app.workflow_runtime.persist_chat_history_compactor_memory_card",
        _fake_persist_card,
    )

    req = _req("chat_history_compactor_pass")
    req.context.metadata["workflow_request"]["window_mode"] = "day"
    req.context.metadata["workflow_request"]["scheduled_dispatch"] = {"source": "orion-actions"}

    result = asyncio.run(
        execute_chat_workflow(
            bus=bus,
            source=ServiceRef(name="cortex-orch"),
            req=req,
            correlation_id="00000000-0000-0000-0000-000000000302",
            causality_chain=[],
            trace={},
            call_verb_runtime=_fake_call_verb_runtime,
        )
    )
    assert result.ok is True
    assert card_called["n"] == 0
    assert result.metadata["workflow"]["turn_count"] == 0


def test_chat_history_compactor_pass_digest_chat_then_quick_retry(monkeypatch) -> None:
    bus = DummyBus()
    routes: list[str] = []

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs["client_request"]
        if req.verb == "skills.chat.discussion_window.v1":
            skill = {
                "window_start_utc": "2026-07-09T04:00:00+00:00",
                "window_end_utc": "2026-07-09T10:00:00+00:00",
                "turn_count": 1,
                "turns": [
                    {
                        "created_at": "2026-07-09T05:00:00+00:00",
                        "correlation_id": "corr-b",
                        "prompt": "hi",
                        "response": "hello",
                    }
                ],
                "transcript_text": "user: hi\norion: hello",
                "selection_strategy": "time_bound_then_contiguous_suffix",
            }
            return DummyVerbResult(
                payload={
                    "result": {
                        "status": "success",
                        "final_text": json.dumps(skill),
                        "metadata": {"skill_result": skill},
                    }
                }
            )
        if req.verb == "chat_history_compactor_digest_v1":
            route = str((req.options or {}).get("llm_route") or "")
            routes.append(route)
            if route == "chat":
                return DummyVerbResult(
                    payload={"result": {"status": "success", "final_text": "not-json", "metadata": {}}}
                )
            digest = {
                "card_summary": "Quick-route digest recovered.",
                "journal_title": "Chat digest",
                "journal_body": "Recovered on quick.",
                "turn_refs": ["corr-b"],
            }
            return DummyVerbResult(
                payload={
                    "result": {
                        "status": "success",
                        "final_text": json.dumps(digest),
                        "metadata": {"chat_history_compactor_digest": digest},
                    }
                }
            )
        raise AssertionError(f"unexpected verb {req.verb}")

    async def _fake_persist_card(**kwargs):
        return uuid4()

    monkeypatch.setattr(
        "app.workflow_runtime.persist_chat_history_compactor_memory_card",
        _fake_persist_card,
    )

    result = asyncio.run(
        execute_chat_workflow(
            bus=bus,
            source=ServiceRef(name="cortex-orch"),
            req=_req("chat_history_compactor_pass"),
            correlation_id="00000000-0000-0000-0000-000000000303",
            causality_chain=[],
            trace={},
            call_verb_runtime=_fake_call_verb_runtime,
        )
    )
    assert result.ok is True
    assert routes == ["chat", "quick"]
    assert result.metadata["workflow"].get("digest_llm_route") == "quick"


def test_chat_history_compactor_pass_over_budget_fails_without_persist(monkeypatch) -> None:
    bus = DummyBus()
    card_called = {"n": 0}

    async def _fake_call_verb_runtime(*args, **kwargs):
        req = kwargs["client_request"]
        if req.verb == "skills.chat.discussion_window.v1":
            skill = {
                "window_start_utc": "2026-07-09T04:00:00+00:00",
                "window_end_utc": "2026-07-09T10:00:00+00:00",
                "turn_count": 1,
                "turns": [
                    {
                        "created_at": "2026-07-09T05:00:00+00:00",
                        "correlation_id": "corr-c",
                        "prompt": "hi",
                        "response": "hello",
                    }
                ],
                "transcript_text": "user: hi\norion: hello",
                "selection_strategy": "time_bound_then_contiguous_suffix",
            }
            return DummyVerbResult(
                payload={
                    "result": {
                        "status": "success",
                        "final_text": json.dumps(skill),
                        "metadata": {"skill_result": skill},
                    }
                }
            )
        if req.verb == "chat_history_compactor_digest_v1":
            digest = {
                "card_summary": "x" * 801,
                "journal_title": "Title",
                "journal_body": "Body",
                "turn_refs": ["corr-c"],
            }
            return DummyVerbResult(
                payload={
                    "result": {
                        "status": "success",
                        "final_text": json.dumps(digest),
                        "metadata": {"chat_history_compactor_digest": digest},
                    }
                }
            )
        raise AssertionError(f"unexpected verb {req.verb}")

    async def _fake_persist_card(**kwargs):
        card_called["n"] += 1
        return uuid4()

    monkeypatch.setattr(
        "app.workflow_runtime.persist_chat_history_compactor_memory_card",
        _fake_persist_card,
    )

    with pytest.raises(Exception, match="compactor_output_over_budget"):
        asyncio.run(
            execute_chat_workflow(
                bus=bus,
                source=ServiceRef(name="cortex-orch"),
                req=_req("chat_history_compactor_pass"),
                correlation_id="00000000-0000-0000-0000-000000000304",
                causality_chain=[],
                trace={},
                call_verb_runtime=_fake_call_verb_runtime,
            )
        )
    assert card_called["n"] == 0
    assert not any(ch == "orion:journal:write" for ch, _ in bus.published)
```

Ensure `uuid4` and `json` are already imported in the test file (they are for github tests).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest services/orion-cortex-orch/tests/test_workflow_lane.py -k chat_history_compactor -v`

Expected: FAIL (`unknown_workflow` or missing symbol).

- [ ] **Step 3: Implement memory wrapper, workflow, registry**

Create `services/orion-cortex-orch/app/chat_history_compactor_memory.py`:

```python
from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from orion.cognition.chat_history_compactor.constants import CHAT_DEV_DIGEST_TAG, COMPACTOR_KIND
from orion.cognition.chat_history_compactor.window import ResolvedChatCompactorWindow
from orion.core.contracts.memory_cards import EvidenceItemV1, MemoryCardCreateV1, TimeHorizonV1
from orion.core.storage import memory_cards as mc_dal
from orion.schemas.actions.chat_history_compactor import ChatHistoryCompactorDigestV1

from .memory_extractor import _get_memory_pool

logger = logging.getLogger("orion.cortex.chat_history_compactor_memory")


async def persist_chat_history_compactor_memory_card(
    *,
    digest: ChatHistoryCompactorDigestV1,
    window: ResolvedChatCompactorWindow,
    turn_count: int,
    actor: str = "chat_history_compactor_pass",
) -> Optional[UUID]:
    pool = await _get_memory_pool()
    if pool is None:
        raise RuntimeError("recall_pg_dsn_unavailable")

    window_label = window.calendar_date or (
        f"{window.lookback_hours}h" if window.lookback_hours else "window"
    )
    evidence = [
        EvidenceItemV1(source=ref, ts=window_label)
        for ref in (digest.turn_refs or [])[:12]
    ]
    card = MemoryCardCreateV1(
        types=["fact"],
        anchor_class="event",
        status="active",
        priority="high_recall",
        provenance="chat_compactor",
        title=f"Chat digest ({window_label})",
        summary=digest.card_summary,
        tags=[CHAT_DEV_DIGEST_TAG],
        time_horizon=TimeHorizonV1(
            kind="era_bound",
            start=window.window_start.isoformat(),
            end=window.window_end.isoformat(),
        ),
        evidence=evidence,
        subschema={
            "compactor_index": window.compactor_index,
            "compactor_kind": COMPACTOR_KIND,
            "window_mode": window.mode,
            "turn_count": int(turn_count),
            "window_start": window.window_start.isoformat(),
            "window_end": window.window_end.isoformat(),
        },
    )
    return await mc_dal.upsert_indexed_compactor_card(
        pool,
        index=window.compactor_index,
        card=card,
        actor=actor,
    )
```

Register workflow in `orion/cognition/workflows/registry.py` (before `_WORKFLOW_INDEX`):

```python
    WorkflowDefinition(
        workflow_id="chat_history_compactor_pass",
        display_name="Chat History Compactor",
        description="Compact bounded chat_history_log windows into indexed memory cards (optional journal).",
        aliases=[
            "compact chat history",
            "compact last 24 hours of chat",
            "compact the last 24 hours of chat into a memory digest",
            "compact the last 6 hours of chat into a memory digest",
            "chat history compactor",
            "what have we been talking about",
        ],
        user_invocable=True,
        autonomous_invocable=True,
        execution_mode="sync",
        may_call_actions=False,
        persistence_policy="Upsert one indexed memory card per window key; optional append-only journal entry.",
        result_surface="Return turn count, compactor_index, card summary preview, card_id.",
        steps=[
            WorkflowStepDefinition(step_id="fetch_window", description="Fetch discussion window from SQL.", adapter="verb:skills.chat.discussion_window.v1"),
            WorkflowStepDefinition(step_id="compact", description="LLM compact digest JSON.", adapter="verb:chat_history_compactor_digest_v1"),
            WorkflowStepDefinition(step_id="persist_card", description="Upsert indexed memory card.", adapter="memory_cards:compactor_index"),
            WorkflowStepDefinition(step_id="persist_journal", description="Optional journal append.", adapter="journaler:append_only_write"),
        ],
        planner_hints=["Use when the operator asks to compact recent Hub chat into a durable memory digest."],
    ),
```

In `workflow_runtime.py`:

1. Add imports for chat history compactor helpers + `persist_chat_history_compactor_memory_card` + `resolve_chat_compactor_window` + `ChatHistoryCompactorDigestV1` + `DEFAULT_MAX_TURNS`.
2. Implement `_run_chat_history_compactor_digest` that:
   - Sets verb `chat_history_compactor_digest_v1`
   - Sets `metadata["chat_history_compactor_input"] = trim_chat_history_compactor_input(window)`
   - Sets `options["llm_route"] = route` (`"chat"` then `"quick"`)
   - Parses digest from metadata key or `final_text`
   - Calls `assert_chat_compactor_digest_within_budget`
   - On first-route failure (verb not ok, structured_output_rejected, invalid JSON), retry once with `quick`; if still failing raise `WorkflowExecutionError("chat_compactor_digest_failed:…")`
   - Returns `(digest, route_used)`
3. Implement `_execute_chat_history_compactor_pass` mirroring journal discussion fetch + github persist/journal pattern:

```python
async def _execute_chat_history_compactor_pass(
    *,
    bus: OrionBusAsync,
    source: ServiceRef,
    correlation_id: str,
    causality_chain: list | None,
    trace: dict | None,
    req: CortexClientRequest,
    call_verb_runtime,
) -> CortexClientResult:
    workflow_id = "chat_history_compactor_pass"
    user_text = (req.context.raw_user_text or req.context.user_message or "").strip()
    request = _workflow_request(req)
    window_spec = resolve_chat_compactor_window(
        window_mode=str(request.get("window_mode") or "") or None,
        lookback_hours=request.get("lookback_hours"),
        now=datetime.now(timezone.utc),
        user_text=user_text,
        workflow_request=request,
    )
    dw_req = DiscussionWindowRequestV1(
        lookback_seconds=int(window_spec.lookback_seconds),
        end_time_utc=window_spec.window_end,
        user_id=None,  # global scope v1
        source=None,
        max_turns=DEFAULT_MAX_TURNS,
        require_prompt_and_response=True,
    )
    # ... same skill_client envelope as journal_discussion_window_pass ...
    # validate DiscussionWindowResultV1 from skill_result

    card_id = None
    card_persist_skipped_reason = None
    digest_route = None
    persisted: List[str] = []

    if window.turn_count <= 0 or not (window.transcript_text or "").strip():
        digest = build_quiet_day_chat_digest(
            window_label=window_spec.calendar_date or window_spec.compactor_index
        )
        # optional quiet journal (write stub); skip card
    else:
        digest, digest_route = await _run_chat_history_compactor_digest(...)
        try:
            persisted_card_id = await persist_chat_history_compactor_memory_card(
                digest=digest,
                window=window_spec,
                turn_count=window.turn_count,
            )
            card_id = str(persisted_card_id) if persisted_card_id else None
        except RuntimeError as exc:
            if "recall_pg_dsn_unavailable" not in str(exc):
                raise
            card_persist_skipped_reason = "recall_pg_dsn_unavailable"

    # Journal when journal_body non-empty (quiet stub counts)
    if (digest.journal_body or "").strip():
        draft = JournalEntryDraftV1(
            mode="digest",
            title=(digest.journal_title or f"Chat digest — {window_spec.compactor_index}")[:120] or "Chat digest",
            body=digest.journal_body,
        )
        entry_id = stable_chat_compactor_journal_entry_id(
            workflow_id=workflow_id,
            compactor_index=window_spec.compactor_index,
        )
        write = build_write_payload(
            draft,
            trigger=build_manual_trigger(
                summary=f"Chat history compactor {window_spec.compactor_index}",
                source_ref=f"chat_history_compactor_pass:{window_spec.compactor_index}",
            ),
            correlation_id=correlation_id,
            author=req.context.user_id or "orion",
            entry_id=entry_id,
        )
        await _publish_journal_entry_write_or_fail(...)
        persisted.append(f"journal.entry.write.v1:{write.entry_id}")

    if card_id:
        persisted.insert(0, f"memory_card:{card_id}")

    # metadata.workflow includes: turn_count, compactor_index, card_id,
    # card_summary_preview, digest_llm_route, window_mode, lookback_hours,
    # selection_strategy, card_persist_skipped_reason?
```

4. Add dispatch branch in `execute_chat_workflow`:

```python
        elif workflow_id == "chat_history_compactor_pass":
            result = await _execute_chat_history_compactor_pass(...)
```

Implement `_run_chat_history_compactor_digest` fully (do not leave a stub). Mirror `_run_github_compactor_digest` but loop routes `("chat", "quick")` and break on first success after budget assert.

- [ ] **Step 4: Run workflow tests**

Run: `pytest services/orion-cortex-orch/tests/test_workflow_lane.py -k chat_history_compactor -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-orch/app/chat_history_compactor_memory.py \
  services/orion-cortex-orch/app/workflow_runtime.py \
  orion/cognition/workflows/registry.py \
  services/orion-cortex-orch/tests/test_workflow_lane.py
git commit -m "feat: wire chat_history_compactor_pass workflow with chat→quick digest"
```

---

### Task 7: Hub Skill Runner + catalogue docs

**Files:**
- Modify: `services/orion-hub/templates/index.html`
- Modify: `services/orion-hub/static/js/app.js`
- Modify: `services/orion-hub/static/js/workflow-ui.js`
- Modify: `services/orion-hub/tests/test_workflow_request_builder.py`
- Modify: `docs/operator_skill_prompt_catalogue.md`

Note: These are **Cognitive workflows** options (`data-workflow-id`), not entries in `SKILL_RUNNER_CATALOGUE_VERBS` (that map is skills-only). Spec mentions catalogue.py for completeness — do **not** add workflow prompts to the deterministic skill map; that would force the skills-only lane. Workflow prompts must stay on the normal chat/workflow path so the digest subverb can use chat→quick.

- [ ] **Step 1: Write failing alias routing test**

In `services/orion-hub/tests/test_workflow_request_builder.py`, extend the parameterized alias cases (same pattern as github):

```python
        ("Compact the last 24 hours of chat into a memory digest", "chat_history_compactor_pass"),
        ("Compact the last 6 hours of chat into a memory digest", "chat_history_compactor_pass"),
        ("what have we been talking about", "chat_history_compactor_pass"),
```

- [ ] **Step 2: Run test to verify it fails** (if aliases not yet registered — Task 6 should already register them; if Task 6 done, this may already pass — still add Hub UI before claiming done)

Run: `pytest services/orion-hub/tests/test_workflow_request_builder.py -k chat_history_compactor -v`

- [ ] **Step 3: Add Hub UI entries + docs**

In `services/orion-hub/templates/index.html` Cognitive workflows optgroup, add:

```html
<option value="Compact the last 24 hours of chat into a memory digest." data-workflow-id="chat_history_compactor_pass">Workflow · Compact last 24h chat digest.</option>
<option value="Compact the last 6 hours of chat into a memory digest." data-workflow-id="chat_history_compactor_pass">Workflow · Compact last 6h chat digest.</option>
```

In `services/orion-hub/static/js/app.js` workflow catalogue array, add:

```javascript
{ workflowId: 'chat_history_compactor_pass', prompt: 'Compact the last 24 hours of chat into a memory digest.', label: 'Workflow · Compact last 24h chat digest.' },
{ workflowId: 'chat_history_compactor_pass', prompt: 'Compact the last 6 hours of chat into a memory digest.', label: 'Workflow · Compact last 6h chat digest.' },
```

In `services/orion-hub/static/js/workflow-ui.js` label map:

```javascript
chat_history_compactor_pass: 'Chat history compactor',
```

In `docs/operator_skill_prompt_catalogue.md`, add a short **Cognitive workflows** note (or extend existing workflow section) listing the two compact prompts and that they invoke `chat_history_compactor_pass` (not the deterministic skills lane).

- [ ] **Step 4: Run Hub tests**

Run: `pytest services/orion-hub/tests/test_workflow_request_builder.py -k "chat_history_compactor or github_compactor" -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/templates/index.html services/orion-hub/static/js/app.js \
  services/orion-hub/static/js/workflow-ui.js \
  services/orion-hub/tests/test_workflow_request_builder.py \
  docs/operator_skill_prompt_catalogue.md
git commit -m "feat: expose chat history compactor in Hub Skill Runner"
```

---

### Task 8: Daily schedule bootstrap in orion-actions

**Files:**
- Create: `services/orion-actions/app/workflow_schedule_bootstrap.py`
- Create: `services/orion-actions/tests/test_chat_history_compactor_schedule_bootstrap.py`
- Modify: `services/orion-actions/app/main.py`
- Modify: `services/orion-actions/README.md`

- [ ] **Step 1: Write failing bootstrap tests**

Create `services/orion-actions/tests/test_chat_history_compactor_schedule_bootstrap.py`:

```python
from __future__ import annotations

from pathlib import Path

from app.workflow_schedule_bootstrap import ensure_chat_history_compactor_daily_schedule
from app.workflow_schedule_store import WorkflowScheduleStore


def test_chat_history_compactor_schedule_bootstrap_creates_once(tmp_path: Path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    first = ensure_chat_history_compactor_daily_schedule(store)
    assert first is not None
    assert first.workflow_id == "chat_history_compactor_pass"
    assert first.execution_policy.schedule is not None
    assert first.execution_policy.schedule.kind == "recurring"
    assert first.execution_policy.schedule.cadence == "daily"
    assert first.execution_policy.schedule.hour_local == 6
    assert first.execution_policy.schedule.minute_local == 0
    assert first.execution_policy.schedule.timezone == "America/Denver"
    assert first.workflow_request.get("window_mode") == "day"

    second = ensure_chat_history_compactor_daily_schedule(store)
    assert second is None
    active = [s for s in store.list_schedules() if s.workflow_id == "chat_history_compactor_pass"]
    assert len(active) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-actions/tests/test_chat_history_compactor_schedule_bootstrap.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement bootstrap + wire lifespan + README**

Create `services/orion-actions/app/workflow_schedule_bootstrap.py`:

```python
from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from orion.schemas.workflow_execution import (
    WorkflowDispatchRequestV1,
    WorkflowExecutionPolicyV1,
    WorkflowScheduleSpecV1,
)

from .workflow_schedule_store import WorkflowScheduleStore

logger = logging.getLogger("orion.actions.workflow_schedule_bootstrap")

CHAT_HISTORY_COMPACTOR_WORKFLOW_ID = "chat_history_compactor_pass"
_BOOTSTRAP_REQUEST_ID = "bootstrap:chat_history_compactor_pass:daily:06:00:America/Denver"


def _is_matching_daily_chat_compactor(schedule) -> bool:
    if schedule.workflow_id != CHAT_HISTORY_COMPACTOR_WORKFLOW_ID:
        return False
    if schedule.state in {"cancelled", "completed"}:
        return False
    spec = schedule.execution_policy.schedule if schedule.execution_policy else None
    if spec is None or spec.kind != "recurring" or spec.cadence != "daily":
        return False
    if int(spec.hour_local or -1) != 6 or int(spec.minute_local or -1) != 0:
        return False
    if (spec.timezone or "") != "America/Denver":
        return False
    return True


def ensure_chat_history_compactor_daily_schedule(
    store: WorkflowScheduleStore,
) -> object | None:
    """Idempotently seed daily 06:00 America/Denver chat history compactor schedule.

    Returns the created record, or None if a matching schedule already exists.
    """
    existing = [s for s in store.list_schedules(include_inactive=True) if _is_matching_daily_chat_compactor(s)]
    if existing:
        logger.info(
            "chat_history_compactor_schedule_bootstrap_skip existing_schedule_id=%s",
            existing[0].schedule_id,
        )
        return None

    policy = WorkflowExecutionPolicyV1(
        workflow_id=CHAT_HISTORY_COMPACTOR_WORKFLOW_ID,
        invocation_mode="scheduled",
        schedule=WorkflowScheduleSpecV1(
            kind="recurring",
            timezone="America/Denver",
            cadence="daily",
            hour_local=6,
            minute_local=0,
            label="Chat history compactor (daily Denver)",
        ),
        notify_on="completion",
        recipient_group="juniper_primary",
        requested_from_chat=False,
        policy_summary="Bootstrap: daily chat_history_log digest into indexed memory card",
    )
    request = WorkflowDispatchRequestV1(
        request_id=_BOOTSTRAP_REQUEST_ID,
        workflow_id=CHAT_HISTORY_COMPACTOR_WORKFLOW_ID,
        workflow_request={
            "workflow_id": CHAT_HISTORY_COMPACTOR_WORKFLOW_ID,
            "workflow_display_name": "Chat History Compactor",
            "window_mode": "day",
            "execution_policy": policy.model_dump(mode="json"),
        },
        execution_policy=policy,
        correlation_id=str(uuid4()),
        source_service="orion-actions",
        source_kind="actions_scheduler",
        created_at=datetime.now(timezone.utc),
    )
    record = store.upsert_from_dispatch(request)
    logger.info(
        "chat_history_compactor_schedule_bootstrap_created schedule_id=%s next_run_at=%s",
        getattr(record, "schedule_id", None),
        getattr(record, "next_run_at", None),
    )
    return record
```

In `services/orion-actions/app/main.py`, after `workflow_schedule_store = WorkflowScheduleStore(...)`:

```python
    from .workflow_schedule_bootstrap import ensure_chat_history_compactor_daily_schedule

    try:
        ensure_chat_history_compactor_daily_schedule(workflow_schedule_store)
    except Exception:
        logger.exception("chat_history_compactor_schedule_bootstrap_failed")
```

In `services/orion-actions/README.md`, after the `github_compactor_pass` section, add:

```markdown
### `chat_history_compactor_pass`
- **Purpose**: compact bounded Hub `chat_history_log` windows into indexed `high_recall` memory cards (optional journal).
- **Example phrases**: “Compact the last 24 hours of chat into a memory digest”, “what have we been talking about”.
- **Schedulable**: yes. On actions startup, if absent, bootstraps a recurring daily schedule at **06:00 `America/Denver`** with `window_mode=day` (yesterday’s calendar day).
- **Notify**: bootstrap default `notify_on=completion` (editable/pausable in Hub schedule inventory).
- **Typical result**: turn count, `compactor_index`, card summary preview, card_id, optional journal entry id.
- **Persisted meaning**: upserts one active card per `compactor_index` (not supersede); optional append-only journal (`journal.entry.write.v1`).
- **Requires**: cortex-exec SQL (`DATABASE_URL` / `ENDOGENOUS_RUNTIME_SQL_DATABASE_URL`) for discussion window fetch; cortex-orch `RECALL_PG_DSN` for card writes.
```

Also add `chat_history_compactor_pass` to the named cognition workflows bullet near the top of the README if that list exists.

- [ ] **Step 4: Run bootstrap tests**

Run: `pytest services/orion-actions/tests/test_chat_history_compactor_schedule_bootstrap.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-actions/app/workflow_schedule_bootstrap.py \
  services/orion-actions/app/main.py \
  services/orion-actions/README.md \
  services/orion-actions/tests/test_chat_history_compactor_schedule_bootstrap.py
git commit -m "feat: bootstrap daily chat history compactor schedule at 06:00 Denver"
```

---

### Task 9: Full gate + operator smoke checklist

**Files:** none new (verification only)

- [ ] **Step 1: Run full gate**

```bash
pytest orion/cognition/compactor/tests -q
pytest orion/cognition/chat_history_compactor/tests -q
pytest tests/test_indexed_compactor_memory_cards.py -q
pytest services/orion-cortex-exec/tests/test_router_final_text_assembly.py -k chat_history_compactor -q
pytest services/orion-cortex-orch/tests/test_workflow_lane.py -k chat_history_compactor -q
pytest services/orion-hub/tests/test_workflow_request_builder.py -k chat_history_compactor -q
pytest services/orion-actions/tests/test_chat_history_compactor_schedule_bootstrap.py -q
```

Expected: all PASS

- [ ] **Step 2: Confirm no contract drift**

```bash
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
git diff --check
```

Expected: no new registry/channel requirements; whitespace clean.

- [ ] **Step 3: Document restart commands in PR body (do not run sudo yourself)**

```bash
# After merge / deploy:
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-cortex-orch/.env \
  -f services/orion-cortex-orch/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-actions/.env \
  -f services/orion-actions/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

Operator smoke (post-deploy):

1. Skill Runner → “Compact the last 24 hours…” → Hub Memory shows card with `compactor_index` / tag `chat_dev_digest`.
2. Rerun same window → same `card_id`, updated summary.
3. Run 6-hour prompt → second active card coexists.
4. After actions restart → schedule inventory shows daily 06:00 `America/Denver`.
5. Scheduled/forced day run → `chat_compactor:day:{yesterday}` card.

- [ ] **Step 4: Commit any leftover doc/test fixes only if needed; otherwise stop**

No empty commit.

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|------------------|------|
| `build_compactor_index` day/rolling | Task 1 |
| `ChatHistoryCompactorDigestV1` + budgets | Tasks 2, 4 |
| Indexed upsert DAL (not supersede) | Task 3 |
| Digest trim / quiet / stable journal id | Task 4 |
| Brain verb + structured allowlist | Task 5 |
| Workflow fetch → digest → card → journal | Task 6 |
| chat → quick retry | Task 6 |
| Global scope (no user/source filter) | Task 6 |
| Hub Skill Runner + aliases | Task 7 |
| Daily 06:00 Denver bootstrap | Task 8 |
| No bus/registry/env keys | Task 9 check |
| Quiet day skips card | Task 6 test |
| Over-budget fails without persist | Task 6 test |

**Placeholder scan:** no TBD/TODO steps; code blocks included for each implement step.

**Type consistency:** `compactor_index` (not slot); provenance `chat_compactor`; tag `chat_dev_digest`; metadata input key `chat_history_compactor_input`; error token `chat_compactor_digest_failed`; workflow id `chat_history_compactor_pass`.

**Intentional Hub note:** workflow prompts are **not** added to `SKILL_RUNNER_CATALOGUE_VERBS` — that map is deterministic `skills.*` only; putting the workflow there would bypass the digest LLM path. Spec’s catalogue mention is satisfied via registry aliases + Hub Cognitive workflows optgroup + operator docs.

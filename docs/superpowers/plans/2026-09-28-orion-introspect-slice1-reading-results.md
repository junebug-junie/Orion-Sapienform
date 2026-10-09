# Orion Introspect — Slice 1 (foundation + reading results) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Orion an `orion-introspect` MCP server whose first tool, `reading_results`, returns what Orion actually learned from a reading, answered by the Hub over the existing reading bus channel.

**Architecture:** A new stdio MCP server (`orion/introspect/`) sends `ReadingToolRequestV1(operation="reading_result")` on the existing `orion:reading:tool:request` channel. The Hub's `ReadingListener` answers with an `IntrospectResultV1` built by a new read-only query module (`orion/world_pulse_read/introspect.py`). The harness attaches the server only when a new flag is on and the turn has a reading binding; a per-turn `IntrospectToolBindingV1` is derived from runtime facts (never the model), including `memory_allowed = not HARNESS_AITOWN_ENABLED` for slice 5.

**Tech Stack:** Python 3.12, pydantic v2, asyncpg, `mcp` (stdio server), Orion bus (`OrionBusAsync`, `BaseEnvelope`, `OrionCodec`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-orion-introspect-mcp-design.md`. Slices 2–5 (dreams, reveries, curiosity, memories) get their own plans.

## Global Constraints

- Work only in the worktree `/mnt/scripts/Orion-Sapienform-introspect-mcp-spec` on branch `docs/introspect-mcp-spec` (already created; rename to `feat/introspect-slice1` in Task 1 Step 0). Never commit from `/mnt/scripts/Orion-Sapienform`.
- Test interpreter: `PY=/tmp/introspect-venv/bin/python`, created in Task 1 Step 0 from `services/orion-hub/tests/requirements-reading.txt` (system python lacks `mcp`).
- All test commands run from the worktree root with `PYTHONPATH=.`.
- Read-only: nothing in this slice writes, re-queues, or changes any row.
- Bounds: `MAX_ITEMS = 5`, `DEFAULT_LIMIT = 5`, `DEFAULT_TEXT_CAP = 900` chars, `JOURNAL_EXCERPT_CHARS = 600`, `SHORT_FIELD_CAP = 200` (title, why_now), `URL_CAP = 500`. Reason: `ORION_FCC_MCP_TOOL_RESULT_MAX_CHARS=12000` truncates any single tool result above 12,000 chars; measured worst case at 10 items was 18,416 chars, so the spec's 10 items × 1,200 chars cannot fit. 5 items × (900 text + 500 url + 2×200 short fields + overhead) ≈ 11k.
- Empty ≠ unknown: success with nothing found is `ok=True, items=[], total_available=0`. Any transport/owner failure raises `IntrospectUnknownError` whose message contains `answer unknown`. Never return `items=[]` for a failure.
- Reading results are always `epistemic_status="unsettled"`, `kind="reading_result"`.
- A Stage 1 handoff counts as learned only when the row's `status == 'done'` (same rule as `operator.read_detail`'s `handoff_accepted`).
- New env key: `HARNESS_FCC_INTROSPECT_ENABLED`, default `false` everywhere (`.env_example`, compose default, settings default).
- Bus URL in examples/tests: `redis://100.92.216.81:6379/0` (Tailscale node; never `bus-core`/`redis`).
- MCP server name `orion-introspect`; module `orion.introspect.mcp_server`; env var `ORION_INTROSPECT_BINDING`.
- No new bus channels in this slice. `IntrospectRequestV1` / `IntrospectBusOperation` from the spec are deliberately deferred to slice 2 (first slice with its own channel); `IntrospectOperation` here is `Literal["reading_result"]` and each later slice extends it.
- Never `git commit --no-verify`. Never stage `.env`.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `orion/schemas/introspect.py` | Create | Binding, item, result, `ReadingResultArguments`, bounds, `clip_text` |
| `orion/schemas/reading.py` | Modify | `ReadingToolRequestV1` gains `reading_result`, `limit`, `since` |
| `orion/schemas/registry.py` | Modify | Register the four new models |
| `orion/bus/channels.yaml` | Modify | Document `reading_result` on the reading tool channels |
| `orion/world_pulse_read/introspect.py` | Create | Read-only SQL: recent finished reads, or one read by URL/request_id |
| `services/orion-hub/scripts/reading_listener.py` | Modify | Route `reading_result` to the query module, log `introspect op=...` |
| `orion/introspect/__init__.py` | Create | Package marker |
| `orion/introspect/tools.py` | Create | `IntrospectTools` bus client, tool specs, `IntrospectUnknownError` |
| `orion/introspect/mcp_server.py` | Create | Stdio MCP adapter |
| `orion/introspect/binding.py` | Create | `introspect_enabled`, `outward_tools_attached`, `introspect_binding_for_turn` |
| `orion/introspect/brief.py` | Create | Harness usage brief lines |
| `orion/fcc/mcp_config.py` | Modify | Render `orion-introspect`; outward-memory guard |
| `orion/harness/fcc_motor.py` | Modify | Pass binding into render; count tool as context-gathering |
| `orion/harness/prefix.py` | Modify | Append introspect brief |
| `services/orion-harness-governor/{.env_example,docker-compose.yml,app/settings.py,README.md}` | Modify | New flag |
| `.github/workflows/orion-reading-tests.yml` | Modify | Run new tests, trigger on `orion/introspect/**` |
| `scripts/smoke_introspect.py` | Create | Live bus smoke for `reading_results` |
| Tests | Create/Modify | `orion/introspect/tests/*`, `services/orion-hub/tests/test_reading_ingress.py`, `services/orion-hub/tests/test_reading_postgres.py` |

---

### Task 1: Introspect contracts

**Files:**
- Create: `orion/schemas/introspect.py`
- Modify: `orion/schemas/reading.py` (class `ReadingToolRequestV1`, lines ~77–92)
- Modify: `orion/schemas/registry.py` (import block ~line 752; `_REGISTRY` dict ~line 1406)
- Modify: `orion/bus/channels.yaml` (entries `orion:reading:tool:request` ~3551 and `orion:reading:tool:result:*` ~3561)
- Test: `orion/introspect/tests/test_introspect_schemas.py`

**Interfaces:**
- Produces:
  - `orion.schemas.introspect`: `MAX_ITEMS: int = 5`, `DEFAULT_LIMIT: int = 5`, `DEFAULT_TEXT_CAP: int = 900`, `SHORT_FIELD_CAP: int = 200`, `URL_CAP: int = 500`, `IntrospectOperation = Literal["reading_result"]`, `clip_text(text: str | None, cap: int = DEFAULT_TEXT_CAP) -> tuple[str, bool]`, `IntrospectToolBindingV1(invocation_context, parent_run_id, parent_trace_id, memory_allowed)`, `IntrospectItemV1(id, occurred_at, kind, epistemic_status, text, truncated, sensitivity, extra)`, `IntrospectResultV1(ok, operation, as_of, total_available, items, error)`, `ReadingResultArguments(request_id, url, limit, since)`.
  - `ReadingToolRequestV1.operation` accepts `"reading_result"`; new optional fields `limit: int | None`, `since: datetime | None`.

- [ ] **Step 0: Prepare branch and test venv**

```bash
cd /mnt/scripts/Orion-Sapienform-introspect-mcp-spec
git branch -m docs/introspect-mcp-spec feat/introspect-slice1
python3 -m venv /tmp/introspect-venv
/tmp/introspect-venv/bin/pip install -q -r services/orion-hub/tests/requirements-reading.txt
export PY=/tmp/introspect-venv/bin/python
PYTHONPATH=. $PY -m pytest orion/fcc/tests/test_reading_mcp.py -q
```

Expected: 4 passed (baseline green). The spec + plan commits stay on this branch; the PR will contain them.

- [ ] **Step 1: Write the failing tests**

Create `orion/introspect/tests/test_introspect_schemas.py`:

```python
"""Introspect contracts: bounded, empty-vs-unknown, caller-bound."""
import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from orion.schemas.introspect import (
    DEFAULT_TEXT_CAP,
    MAX_ITEMS,
    SHORT_FIELD_CAP,
    URL_CAP,
    IntrospectItemV1,
    IntrospectResultV1,
    IntrospectToolBindingV1,
    ReadingResultArguments,
    clip_text,
)
from orion.schemas.reading import ReadingToolRequestV1
from orion.schemas.registry import resolve

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
MCP_TOOL_RESULT_MAX_CHARS = 12000


def _item(**overrides):
    fields = dict(
        id="reading:1", occurred_at=NOW, kind="reading_result",
        epistemic_status="unsettled", text="learned", truncated=False,
    )
    fields.update(overrides)
    return IntrospectItemV1(**fields)


def test_clip_text_marks_truncation():
    assert clip_text("short") == ("short", False)
    assert clip_text(None) == ("", False)
    body, truncated = clip_text("x" * (DEFAULT_TEXT_CAP + 5))
    assert truncated and len(body) == DEFAULT_TEXT_CAP


def test_binding_is_frozen_and_rejects_extra_fields():
    binding = IntrospectToolBindingV1(
        invocation_context="curiosity", parent_run_id="r", parent_trace_id="t", memory_allowed=False,
    )
    with pytest.raises(ValidationError):
        binding.memory_allowed = True
    with pytest.raises(ValidationError):
        IntrospectToolBindingV1(
            invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t",
            memory_allowed=True, lane="all",
        )


def test_ok_result_requires_total_and_no_error():
    empty = IntrospectResultV1(ok=True, operation="reading_result", as_of=NOW, total_available=0)
    assert empty.items == []
    with pytest.raises(ValidationError):
        IntrospectResultV1(ok=True, operation="reading_result", as_of=NOW)
    with pytest.raises(ValidationError):
        IntrospectResultV1(ok=True, operation="reading_result", as_of=NOW, total_available=0, items=[_item()])


def test_failed_result_carries_only_an_error():
    IntrospectResultV1(ok=False, operation="reading_result", as_of=NOW, error="owner down")
    with pytest.raises(ValidationError):
        IntrospectResultV1(ok=False, operation="reading_result", as_of=NOW, error="x", total_available=0)
    with pytest.raises(ValidationError):
        IntrospectResultV1(ok=False, operation="reading_result", as_of=NOW)


def test_timestamps_must_be_timezone_aware():
    with pytest.raises(ValidationError):
        _item(occurred_at=datetime(2026, 9, 28, 12, 0))
    with pytest.raises(ValidationError):
        IntrospectResultV1(ok=True, operation="reading_result", as_of=datetime(2026, 9, 28), total_available=0)


def test_items_are_capped():
    with pytest.raises(ValidationError):
        IntrospectResultV1(
            ok=True, operation="reading_result", as_of=NOW,
            total_available=MAX_ITEMS + 1, items=[_item(id=f"r{i}") for i in range(MAX_ITEMS + 1)],
        )


def test_worst_case_result_fits_the_mcp_tool_result_cap():
    item = _item(
        text="x" * DEFAULT_TEXT_CAP, truncated=True,
        extra={
            "url": "https://example.org/" + "p" * (URL_CAP - 20),
            "title": "t" * SHORT_FIELD_CAP,
            "why_now": "w" * SHORT_FIELD_CAP,
            "reading_status": "landing_pending",
            "learned": True,
            "request_id": str(uuid4()),
        },
    )
    result = IntrospectResultV1(
        ok=True, operation="reading_result", as_of=NOW, total_available=500,
        items=[item.model_copy(update={"id": f"reading:{i}"}) for i in range(MAX_ITEMS)],
    )
    assert len(json.dumps(result.model_dump(mode="json"))) < MCP_TOOL_RESULT_MAX_CHARS


@pytest.mark.parametrize("args", [
    {"request_id": str(uuid4()), "url": "https://example.org/a"},
    {"url": "https://example.org/a", "since": "2026-09-01T00:00:00+00:00"},
    {"since": "2026-09-01T00:00:00"},
    {"limit": 0},
    {"limit": MAX_ITEMS + 1},
    {"memory_allowed": True},
])
def test_reading_result_arguments_reject_bad_shapes(args):
    with pytest.raises(ValidationError):
        ReadingResultArguments.model_validate(args)


def test_reading_result_arguments_defaults_to_recent_mode():
    args = ReadingResultArguments.model_validate({})
    assert (args.request_id, args.url, args.limit, args.since) == (None, None, 5, None)


def test_reading_tool_request_accepts_reading_result_selectors():
    ReadingToolRequestV1(operation="reading_result")
    ReadingToolRequestV1(operation="reading_result", url="https://example.org/a", limit=3)
    ReadingToolRequestV1(operation="reading_result", request_id=uuid4())
    ReadingToolRequestV1(operation="reading_result", since=NOW, limit=MAX_ITEMS)


@pytest.mark.parametrize("fields", [
    {"operation": "reading_result", "url": "https://example.org/a", "request_id": uuid4()},
    {"operation": "reading_result", "url": "https://example.org/a", "since": NOW},
    {"operation": "reading_result", "since": datetime(2026, 9, 1)},
    {"operation": "reading_status", "url": "https://example.org/a", "limit": 3},
    {"operation": "reading_status", "url": "https://example.org/a", "since": NOW},
])
def test_reading_tool_request_rejects_mixed_operation_arguments(fields):
    with pytest.raises(ValidationError):
        ReadingToolRequestV1(**fields)


def test_new_models_are_registered():
    for name in ("IntrospectToolBindingV1", "IntrospectItemV1", "IntrospectResultV1", "ReadingResultArguments"):
        assert resolve(name) is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. $PY -m pytest orion/introspect/tests/test_introspect_schemas.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'orion.schemas.introspect'`.

- [ ] **Step 3: Create `orion/schemas/introspect.py`**

```python
"""Introspect tool contracts: Orion reading back their own recorded activity.

Design: docs/superpowers/specs/2026-09-28-orion-introspect-mcp-design.md.
Slice 1 carries only ``reading_result``; each later slice extends
``IntrospectOperation`` when its owning service gains a responder.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

# MAX_ITEMS full-size items must fit under ORION_FCC_MCP_TOOL_RESULT_MAX_CHARS
# (12000) or the harness truncates the JSON mid-item.
MAX_ITEMS = 5
DEFAULT_LIMIT = 5
DEFAULT_TEXT_CAP = 900
SHORT_FIELD_CAP = 200
URL_CAP = 500

IntrospectOperation = Literal["reading_result"]


def clip_text(text: str | None, cap: int = DEFAULT_TEXT_CAP) -> tuple[str, bool]:
    body = (text or "").strip()
    if len(body) <= cap:
        return body, False
    return body[:cap], True


def _require_tz(value: datetime | None, field: str) -> None:
    if value is not None and value.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")


class IntrospectToolBindingV1(BaseModel):
    """Server-authored turn context; never part of model tool arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    invocation_context: Literal["unified_chat", "curiosity"]
    parent_run_id: str
    parent_trace_id: str
    memory_allowed: bool


class IntrospectItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    occurred_at: datetime
    kind: str = Field(min_length=1)
    epistemic_status: Literal["record", "unsettled"]
    text: str
    truncated: bool = False
    sensitivity: Literal["public", "private", "intimate"] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _aware(self):
        _require_tz(self.occurred_at, "occurred_at")
        return self


class IntrospectResultV1(BaseModel):
    """``ok=True`` with no items means nothing matched; ``ok=False`` means unknown."""

    model_config = ConfigDict(extra="forbid")
    ok: bool
    operation: IntrospectOperation
    as_of: datetime
    total_available: int | None = Field(default=None, ge=0)
    items: list[IntrospectItemV1] = Field(default_factory=list, max_length=MAX_ITEMS)
    error: str | None = None

    @model_validator(mode="after")
    def _coherent(self):
        _require_tz(self.as_of, "as_of")
        if self.ok:
            if self.error is not None or self.total_available is None:
                raise ValueError("ok result requires total_available and no error")
            if self.total_available < len(self.items):
                raise ValueError("total_available cannot be smaller than items returned")
        elif not self.error or self.items or self.total_available is not None:
            raise ValueError("failed result carries only an error")
        return self


class ReadingResultArguments(BaseModel):
    """Model-supplied arguments for the ``reading_results`` tool."""

    model_config = ConfigDict(extra="forbid")
    request_id: UUID | None = None
    url: str | None = Field(default=None, min_length=1, max_length=8192)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_ITEMS)
    since: datetime | None = None

    @model_validator(mode="after")
    def _selectors(self):
        if self.request_id is not None and self.url is not None:
            raise ValueError("reading_results takes at most one of request_id or url")
        _require_tz(self.since, "since")
        if self.since is not None and (self.request_id is not None or self.url is not None):
            raise ValueError("since applies only to recent reads (no request_id or url)")
        return self
```

- [ ] **Step 4: Extend `ReadingToolRequestV1` in `orion/schemas/reading.py`**

Add the import near the top (after the pydantic import):

```python
from orion.schemas.introspect import MAX_ITEMS
```

Replace the whole `ReadingToolRequestV1` class with:

```python
class ReadingToolRequestV1(BaseModel):
    """Internal ephemeral RPC. Only its post-commit reply proves acceptance."""
    model_config = ConfigDict(extra="forbid")
    operation: Literal["recommend_reading", "reading_status", "reading_result"]
    request: ReadingRequestedV1 | None = None
    request_id: UUID | None = None
    url: str | None = Field(default=None, min_length=1, max_length=8192)
    limit: int | None = Field(default=None, ge=1, le=MAX_ITEMS)
    since: datetime | None = None

    @model_validator(mode="after")
    def operation_arguments(self):
        if self.operation == "recommend_reading":
            if self.request is None or self.request_id is not None or self.url is not None:
                raise ValueError("recommend_reading requires only request")
        elif self.operation == "reading_status":
            if self.request is not None or (self.request_id is None) == (self.url is None):
                raise ValueError("reading_status requires exactly one of request_id or url")
        else:
            if self.request is not None:
                raise ValueError("reading_result never carries a reading request")
            if self.request_id is not None and self.url is not None:
                raise ValueError("reading_result takes at most one of request_id or url")
            if self.since is not None:
                if self.since.tzinfo is None:
                    raise ValueError("since must include a timezone")
                if self.request_id is not None or self.url is not None:
                    raise ValueError("since applies only to recent reads")
        if self.operation != "reading_result" and (self.limit is not None or self.since is not None):
            raise ValueError(f"{self.operation} takes no limit or since")
        return self
```

- [ ] **Step 5: Register models in `orion/schemas/registry.py`**

After the `from orion.schemas.reading import (...)` block (~line 752), add:

```python
from orion.schemas.introspect import (
    IntrospectItemV1, IntrospectResultV1, IntrospectToolBindingV1, ReadingResultArguments,
)
```

In the `_REGISTRY` dict, directly after `"ReadingLifecycleV1": ReadingLifecycleV1,` (~line 1409), add:

```python
    "IntrospectToolBindingV1": IntrospectToolBindingV1,
    "IntrospectItemV1": IntrospectItemV1,
    "IntrospectResultV1": IntrospectResultV1,
    "ReadingResultArguments": ReadingResultArguments,
```

- [ ] **Step 6: Update channel descriptions in `orion/bus/channels.yaml`**

Replace the `description:` line of `orion:reading:tool:request` with:

```yaml
    description: "Ephemeral internal tool RPC; no acceptance until a durable queue receipt arrives. Status accepts exactly one request_id or url; URL selects the newest matching request without enqueueing. reading_result (orion-introspect) is read-only: one read by request_id or url, or recent finished reads (limit, since)."
```

Replace the `description:` line of `orion:reading:tool:result:*` with:

```yaml
    description: "Durable queue receipt, durable status, or an IntrospectResultV1 for reading_result. A timeout means acceptance (or the introspection answer) is unknown."
```

- [ ] **Step 7: Run tests to verify they pass**

Run:
```bash
PYTHONPATH=. $PY -m pytest orion/introspect/tests/test_introspect_schemas.py services/orion-hub/tests/test_reading_ingress.py tests/test_unified_turn_bus_catalog.py -q
```
Expected: all pass (existing reading contract tests still green).

- [ ] **Step 8: Commit**

```bash
git add orion/schemas/introspect.py orion/schemas/reading.py orion/schemas/registry.py orion/bus/channels.yaml orion/introspect/tests/test_introspect_schemas.py
git diff --cached --check
git commit -m "feat(introspect): add introspect contracts and reading_result operation"
```

---

### Task 2: Read-only reading-results query (Hub-owned data)

**Files:**
- Create: `orion/world_pulse_read/introspect.py`
- Test: `services/orion-hub/tests/test_reading_postgres.py` (append one test; real disposable PostgreSQL)

**Interfaces:**
- Consumes: `IntrospectItemV1`, `IntrospectResultV1`, `clip_text`, `DEFAULT_LIMIT`, `SHORT_FIELD_CAP`, `URL_CAP` (Task 1); `queue.reading_status(conn, request_id=None, *, url=None) -> dict`; `queue.derive_reading_status(s1, s2, landing_at) -> str`; `operator._journal_entries(conn, refs: list[str]) -> list[dict]` (ordered by `created_at`, keys `entry_id, created_at, title, body, source_ref`).
- Produces: `async def reading_results(conn, *, request_id: UUID | None = None, url: str | None = None, limit: int = DEFAULT_LIMIT, since: datetime | None = None) -> IntrospectResultV1`; constant `JOURNAL_EXCERPT_CHARS = 600`.

- [ ] **Step 1: Write the failing test**

Append to `services/orion-hub/tests/test_reading_postgres.py` (module already imports `asyncio`, `json`, `datetime`, `timezone`, `queue`, and defines `db()` and `request()`):

```python
def test_reading_results_introspection_is_read_only_and_distinguishes_states(local_pg):
    from orion.schemas.introspect import DEFAULT_TEXT_CAP
    from orion.world_pulse_read.introspect import reading_results

    async def run():
        conn, _ = await db(local_pg)
        # Production journal_entries (sql-writer) has these columns; the shared
        # helper's minimal table does not.
        await conn.execute(
            "ALTER TABLE journal_entries ADD COLUMN created_at timestamptz NOT NULL DEFAULT now(), "
            "ADD COLUMN title text"
        )
        done = request("https://example.org/done")
        queued = request("https://example.org/queued")
        failed = request("https://example.org/failed")
        for r in (done, queued, failed):
            await queue.enqueue_reading(conn, r)
        await conn.execute(
            """UPDATE world_pulse_read_seed
               SET status='done', stage2_status='done',
                   handoff_json=$2::jsonb, stage2_result_json=$3::jsonb,
                   handoff_at=now(), stage2_completed_at=now(), landing_at=now(),
                   trace_id='t1', stage2_trace_id='t2'
               WHERE request_id=$1""",
            done.request_id,
            json.dumps({"what_i_learned": "stage one note"}),
            json.dumps({"summary": "S" * (DEFAULT_TEXT_CAP + 300)}),
        )
        await conn.execute(
            "INSERT INTO journal_entries (entry_id, source_ref, body) "
            "VALUES ('j1', 'world_pulse_read_stage2:t2', 'Journal body about the source')"
        )
        await conn.execute(
            "UPDATE world_pulse_read_seed SET status='failed', last_error='boom', handoff_json=$2::jsonb "
            "WHERE request_id=$1",
            failed.request_id, json.dumps({"what_i_learned": "rejected handoff"}),
        )
        before = await conn.fetch("SELECT * FROM world_pulse_read_seed ORDER BY seed_id")

        recent = await reading_results(conn)
        assert recent.ok and recent.total_available == 1
        [item] = recent.items
        assert item.kind == "reading_result" and item.epistemic_status == "unsettled"
        assert item.extra["url"] == "https://example.org/done"
        assert item.extra["learned"] is True
        assert item.truncated and len(item.text) == DEFAULT_TEXT_CAP
        assert "journal_excerpt" not in item.extra

        by_url = await reading_results(conn, url="https://EXAMPLE.org/done#x")
        assert by_url.total_available == 1
        assert by_url.items[0].extra["journal_excerpt"] == "Journal body about the source"
        assert by_url.items[0].extra["request_id"] == str(done.request_id)

        pending = await reading_results(conn, request_id=queued.request_id)
        assert pending.items[0].text == ""
        assert pending.items[0].extra["learned"] is False
        assert pending.items[0].extra["reading_status"] == "queued"

        rejected = await reading_results(conn, request_id=failed.request_id)
        assert rejected.items[0].extra["reading_status"] == "failed"
        assert rejected.items[0].extra["learned"] is False
        assert rejected.items[0].text == ""

        missing = await reading_results(conn, url="https://example.org/never")
        assert missing.ok and missing.items == [] and missing.total_available == 0

        future = await reading_results(conn, since=datetime(2999, 1, 1, tzinfo=timezone.utc))
        assert future.ok and future.items == [] and future.total_available == 0

        assert await conn.fetch("SELECT * FROM world_pulse_read_seed ORDER BY seed_id") == before
        await conn.close()

    asyncio.run(run())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `RUN_READING_POSTGRES=1 PYTHONPATH=. $PY -m pytest services/orion-hub/tests/test_reading_postgres.py -k reading_results_introspection -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'orion.world_pulse_read.introspect'`.

- [ ] **Step 3: Create `orion/world_pulse_read/introspect.py`**

```python
"""Read-only "what did I learn" lookups for the orion-introspect reading_results tool.

Plain SELECTs over world_pulse_read_seed and journal_entries; never enqueues,
retries, or charges a wallet.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from orion.schemas.introspect import (
    DEFAULT_LIMIT,
    SHORT_FIELD_CAP,
    URL_CAP,
    IntrospectItemV1,
    IntrospectResultV1,
    clip_text,
)
from orion.world_pulse_read.operator import _journal_entries
from orion.world_pulse_read.queue import derive_reading_status, reading_status

JOURNAL_EXCERPT_CHARS = 600

_COLUMNS = """
s.seed_id, s.request_id, s.url, s.title, s.status, s.stage2_status,
s.created_at, s.handoff_at, s.stage2_completed_at, s.landing_at,
s.handoff_json, s.stage2_result_json, s.trace_id, s.stage2_trace_id,
s.request_json->>'why_now' AS why_now
"""
_OCCURRED = "COALESCE(s.landing_at, s.stage2_completed_at, s.handoff_at, s.created_at)"

_ROW_SQL = f"SELECT {_COLUMNS} FROM world_pulse_read_seed s WHERE s.seed_id = $1"

# count(*) OVER () is evaluated before LIMIT, so ``total`` is the full match count.
_RECENT_SQL = f"""
SELECT {_COLUMNS}, count(*) OVER () AS total
FROM world_pulse_read_seed s
WHERE s.duplicate_of IS NULL
  AND s.status = 'done'
  AND (s.stage2_result_json IS NOT NULL OR s.handoff_json IS NOT NULL)
  AND ($2::timestamptz IS NULL OR {_OCCURRED} >= $2)
ORDER BY {_OCCURRED} DESC, s.seed_id DESC
LIMIT $1
"""


def _obj(raw: Any) -> dict[str, Any]:
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


def _learned(row: Any) -> str:
    result = _obj(row["stage2_result_json"])
    # A handoff on a row that did not finish Stage 1 was rejected -- never "learned".
    handoff = _obj(row["handoff_json"]) if row["status"] == "done" else {}
    return str(result.get("summary") or handoff.get("what_i_learned") or "")


def _item(row: Any, *, request_id: str | None, journal_excerpt: str | None = None) -> IntrospectItemV1:
    learned = _learned(row)
    text, truncated = clip_text(learned)
    extra: dict[str, Any] = {
        "url": clip_text(row["url"], URL_CAP)[0],
        "title": clip_text(row["title"], SHORT_FIELD_CAP)[0],
        "why_now": clip_text(row["why_now"], SHORT_FIELD_CAP)[0],
        "reading_status": derive_reading_status(row["status"], row["stage2_status"], row["landing_at"]),
        "learned": bool(text),
        "request_id": request_id,
    }
    if journal_excerpt:
        extra["journal_excerpt"] = journal_excerpt
    return IntrospectItemV1(
        id=row["seed_id"],
        occurred_at=row["landing_at"] or row["stage2_completed_at"] or row["handoff_at"] or row["created_at"],
        kind="reading_result",
        epistemic_status="unsettled",
        text=text,
        truncated=truncated,
        extra=extra,
    )


async def reading_results(
    conn: Any,
    *,
    request_id: UUID | None = None,
    url: str | None = None,
    limit: int = DEFAULT_LIMIT,
    since: datetime | None = None,
) -> IntrospectResultV1:
    as_of = datetime.now(timezone.utc)
    if request_id is None and url is None:
        rows = await conn.fetch(_RECENT_SQL, limit, since)
        return IntrospectResultV1(
            ok=True, operation="reading_result", as_of=as_of,
            total_available=int(rows[0]["total"]) if rows else 0,
            items=[
                _item(r, request_id=str(r["request_id"]) if r["request_id"] else None)
                for r in rows
            ],
        )
    status = await (reading_status(conn, url=url) if url is not None else reading_status(conn, request_id))
    if status["status"] == "not_found":
        return IntrospectResultV1(ok=True, operation="reading_result", as_of=as_of, total_available=0)
    row = await conn.fetchrow(_ROW_SQL, status.get("duplicate_of") or status["seed_id"])
    if row is None:
        raise RuntimeError("reading row vanished between status and detail lookup")
    refs = []
    if row["trace_id"]:
        refs.append(f"world_pulse_read:{row['trace_id']}")
    if row["stage2_trace_id"]:
        refs.append(f"world_pulse_read_stage2:{row['stage2_trace_id']}")
    journal = await _journal_entries(conn, refs)
    excerpt = clip_text(journal[-1]["body"], JOURNAL_EXCERPT_CHARS)[0] if journal else None
    return IntrospectResultV1(
        ok=True, operation="reading_result", as_of=as_of,
        total_available=int(status.get("matched_request_count") or 1),
        items=[_item(row, request_id=status.get("request_id"), journal_excerpt=excerpt)],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `RUN_READING_POSTGRES=1 PYTHONPATH=. $PY -m pytest services/orion-hub/tests/test_reading_postgres.py -q`
Expected: all pass, including the new test.

- [ ] **Step 5: Commit**

```bash
git add orion/world_pulse_read/introspect.py services/orion-hub/tests/test_reading_postgres.py
git diff --cached --check
git commit -m "feat(reading): read-only reading_results query for introspection"
```

---

### Task 3: Introspect bus client and MCP server

**Files:**
- Create: `orion/introspect/__init__.py`, `orion/introspect/tools.py`, `orion/introspect/mcp_server.py`
- Test: `orion/introspect/tests/test_introspect_tools.py`, `orion/introspect/tests/test_introspect_mcp_server.py`

**Interfaces:**
- Consumes: `IntrospectToolBindingV1`, `IntrospectResultV1`, `ReadingResultArguments` (Task 1); `ReadingToolRequestV1`, `ReadingToolResultV1`; `TOOL_CHANNEL = "orion:reading:tool:request"`, `TOOL_RESULT_PREFIX = "orion:reading:tool:result:"` from `orion.world_pulse_read.events`; `normalize_source_url` from `orion.world_pulse_read.urls`.
- Produces:
  - `orion.introspect.tools`: `RPC_TIMEOUT_SEC = 15.0`, `READING_RESULTS_DESCRIPTION: str`, `@dataclass(frozen=True) ToolSpec(name: str, description: str, arguments: type[BaseModel])`, `class IntrospectUnknownError(RuntimeError)`, `class IntrospectTools(bus, binding)` with `tool_specs() -> list[ToolSpec]` and `async invoke(name: str, arguments: dict) -> dict` (returns `IntrospectResultV1.model_dump(mode="json")`).
  - `orion.introspect.mcp_server`: `build_server(tools) -> mcp.server.Server` (tools needs `tool_specs()` and `invoke()`), `async run()`.

- [ ] **Step 1: Write the failing client tests**

Create `orion/introspect/tests/test_introspect_tools.py`:

```python
"""IntrospectTools over a fake bus: correct transport, and failures are 'unknown', never empty."""
import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.introspect.tools import RPC_TIMEOUT_SEC, IntrospectTools, IntrospectUnknownError
from orion.schemas.introspect import IntrospectResultV1, IntrospectToolBindingV1
from orion.schemas.reading import ReadingToolResultV1
from orion.world_pulse_read.events import TOOL_CHANNEL, TOOL_RESULT_PREFIX

BINDING = IntrospectToolBindingV1(
    invocation_context="unified_chat", parent_run_id="run-1", parent_trace_id="trace-1", memory_allowed=True,
)
NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


def _ok_payload():
    result = IntrospectResultV1(ok=True, operation="reading_result", as_of=NOW, total_available=0)
    return ReadingToolResultV1(ok=True, result=result.model_dump(mode="json")).model_dump(mode="json")


class ReplyBus:
    codec = OrionCodec()

    def __init__(self, payload=None, *, raise_exc=None, wrong_correlation=False):
        self.payload = payload
        self.raise_exc = raise_exc
        self.wrong_correlation = wrong_correlation
        self.sent = []

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
        self.sent.append((channel, envelope, reply_channel, timeout_sec))
        if self.raise_exc is not None:
            raise self.raise_exc
        reply = BaseEnvelope(
            kind="reading.tool.result.v1",
            correlation_id=uuid4() if self.wrong_correlation else envelope.correlation_id,
            source=ServiceRef(name="orion-hub"),
            payload=self.payload,
        )
        return {"data": self.codec.encode(reply)}


def _invoke(bus, name="reading_results", args=None):
    return asyncio.run(IntrospectTools(bus, BINDING).invoke(name, args or {}))


def test_tool_specs_list_only_reading_results():
    assert [s.name for s in IntrospectTools(ReplyBus(), BINDING).tool_specs()] == ["reading_results"]


def test_reading_results_uses_reading_channel_with_normalized_url():
    bus = ReplyBus(_ok_payload())
    out = _invoke(bus, args={"url": "https://EXAMPLE.org/a#frag", "limit": 3})
    assert out["ok"] is True and out["items"] == [] and out["total_available"] == 0
    [(channel, envelope, reply_channel, timeout)] = bus.sent
    assert channel == TOOL_CHANNEL
    assert envelope.kind == "reading.tool.request.v1"
    assert envelope.reply_to == reply_channel == f"{TOOL_RESULT_PREFIX}{envelope.correlation_id}"
    assert envelope.source.name == "orion-harness-governor"
    assert timeout == RPC_TIMEOUT_SEC
    assert envelope.payload["operation"] == "reading_result"
    assert envelope.payload["url"] == "https://example.org/a"
    assert envelope.payload["limit"] == 3


@pytest.mark.parametrize("bus", [
    ReplyBus(raise_exc=TimeoutError("no reply")),
    ReplyBus(ReadingToolResultV1(ok=False, error="reading_queue_unavailable").model_dump(mode="json")),
    ReplyBus(_ok_payload(), wrong_correlation=True),
    ReplyBus(ReadingToolResultV1(ok=True, result={"garbage": 1}).model_dump(mode="json")),
    ReplyBus({"not": "a reading tool result"}),
])
def test_every_failure_is_unknown_never_empty(bus):
    with pytest.raises(IntrospectUnknownError, match="answer unknown"):
        _invoke(bus)


def test_unknown_tool_is_rejected_without_sending():
    bus = ReplyBus(_ok_payload())
    with pytest.raises(ValueError, match="unknown introspect tool"):
        _invoke(bus, name="memories")
    assert bus.sent == []


def test_model_cannot_supply_binding_fields():
    bus = ReplyBus(_ok_payload())
    with pytest.raises(ValidationError):
        _invoke(bus, args={"memory_allowed": True})
    assert bus.sent == []
```

- [ ] **Step 2: Write the failing MCP protocol test**

Create `orion/introspect/tests/test_introspect_mcp_server.py`:

```python
"""Real MCP protocol over an in-memory session (no bus, no model)."""
import asyncio
import json

import pytest

from orion.introspect.tools import READING_RESULTS_DESCRIPTION, ToolSpec
from orion.schemas.introspect import ReadingResultArguments


class FakeTools:
    def __init__(self):
        self.calls = []

    def tool_specs(self):
        return [ToolSpec("reading_results", READING_RESULTS_DESCRIPTION, ReadingResultArguments)]

    async def invoke(self, name, arguments):
        self.calls.append((name, arguments))
        return {"ok": True, "operation": "reading_result", "as_of": "2026-09-28T12:00:00+00:00",
                "total_available": 0, "items": [], "error": None}


def test_protocol_lists_one_tool_and_rejects_extra_or_unlisted_calls():
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    from orion.introspect.mcp_server import build_server

    tools = FakeTools()

    async def run():
        async with create_connected_server_and_client_session(build_server(tools)) as client:
            listing = await client.list_tools()
            assert {t.name for t in listing.tools} == {"reading_results"}
            schema = listing.tools[0].inputSchema
            assert set(schema["properties"]) == {"request_id", "url", "limit", "since"}
            assert schema["additionalProperties"] is False
            bad = await client.call_tool("reading_results", {"memory_allowed": True})
            assert bad.isError
            unlisted = await client.call_tool("memories", {"query": "x"})
            assert unlisted.isError
            assert tools.calls == []
            good = await client.call_tool("reading_results", {"url": "https://example.org/a"})
            assert not good.isError
            assert json.loads(good.content[0].text)["total_available"] == 0
            assert tools.calls == [("reading_results", {"url": "https://example.org/a"})]

    asyncio.run(run())
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=. $PY -m pytest orion/introspect/tests/test_introspect_tools.py orion/introspect/tests/test_introspect_mcp_server.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'orion.introspect'`.

- [ ] **Step 4: Create `orion/introspect/__init__.py`**

```python
"""orion-introspect: Orion reading back their own recorded activity over the bus."""
```

- [ ] **Step 5: Create `orion/introspect/tools.py`**

```python
"""Caller-bound introspection tools over the existing internal Orion bus trust boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.introspect import IntrospectResultV1, IntrospectToolBindingV1, ReadingResultArguments
from orion.schemas.reading import ReadingToolRequestV1, ReadingToolResultV1
from orion.world_pulse_read.events import TOOL_CHANNEL, TOOL_RESULT_PREFIX
from orion.world_pulse_read.urls import normalize_source_url

RPC_TIMEOUT_SEC = 15.0

READING_RESULTS_DESCRIPTION = (
    "Look up what you actually learned from sources read through your reading pipeline. "
    "Pass url or request_id for one reading, or neither for your most recent finished reads "
    "(optional since=<ISO timestamp with timezone>, limit up to 5). Each item's text is the "
    "learned summary; learned=false means no output exists yet, so report its reading_status "
    "instead. Results are source-attributed candidates, not settled beliefs. items=[] means "
    "nothing matched; a tool error means the answer is unknown, never that nothing happened."
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    arguments: type[BaseModel]


class IntrospectUnknownError(RuntimeError):
    """The owning service could not be asked, or did not answer coherently."""


class IntrospectTools:
    def __init__(self, bus, binding: IntrospectToolBindingV1):
        self.bus = bus
        self.binding = binding

    def tool_specs(self) -> list[ToolSpec]:
        return [ToolSpec("reading_results", READING_RESULTS_DESCRIPTION, ReadingResultArguments)]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name != "reading_results":
            raise ValueError(f"unknown introspect tool: {name}")
        # Validate before transport: extra fields are rejected, never ignored.
        args = ReadingResultArguments.model_validate(arguments)
        command = ReadingToolRequestV1(
            operation="reading_result",
            request_id=args.request_id,
            url=normalize_source_url(args.url) if args.url is not None else None,
            limit=args.limit,
            since=args.since,
        )
        return (await self._reading_rpc(command)).model_dump(mode="json")

    async def _reading_rpc(self, command: ReadingToolRequestV1) -> IntrospectResultV1:
        correlation_id = uuid4()
        reply = f"{TOOL_RESULT_PREFIX}{correlation_id}"
        try:
            raw = await self.bus.rpc_request(
                TOOL_CHANNEL,
                BaseEnvelope(
                    kind="reading.tool.request.v1", correlation_id=correlation_id,
                    reply_to=reply, source=ServiceRef(name="orion-harness-governor"),
                    payload=command.model_dump(mode="json"),
                ),
                reply_channel=reply, timeout_sec=RPC_TIMEOUT_SEC,
            )
        except Exception as exc:
            raise IntrospectUnknownError(
                f"reading_results: answer unknown (no reply: {type(exc).__name__})"
            ) from exc
        decoded = self.bus.codec.decode(raw.get("data"))
        if not decoded.ok or str(decoded.envelope.correlation_id) != str(correlation_id):
            raise IntrospectUnknownError("reading_results: answer unknown (malformed reply)")
        try:
            envelope_result = ReadingToolResultV1.model_validate(decoded.envelope.payload)
        except ValueError as exc:
            raise IntrospectUnknownError("reading_results: answer unknown (malformed reply)") from exc
        if not envelope_result.ok:
            raise IntrospectUnknownError(f"reading_results: answer unknown ({envelope_result.error})")
        try:
            result = IntrospectResultV1.model_validate(envelope_result.result)
        except ValueError as exc:
            raise IntrospectUnknownError("reading_results: answer unknown (malformed result)") from exc
        if not result.ok or result.operation != "reading_result":
            raise IntrospectUnknownError("reading_results: answer unknown (mismatched result)")
        return result
```

- [ ] **Step 6: Create `orion/introspect/mcp_server.py`**

```python
"""Stdio MCP adapter for orion-introspect, following the FCC per-turn tool transport."""
from __future__ import annotations

import asyncio
import json
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from orion.core.bus.async_service import OrionBusAsync
from orion.introspect.tools import IntrospectTools
from orion.schemas.introspect import IntrospectToolBindingV1


def build_server(tools) -> Server:
    server = Server("orion-introspect")
    specs = {spec.name: spec for spec in tools.tool_specs()}

    @server.list_tools()
    async def list_tools():
        return [
            Tool(name=s.name, description=s.description, inputSchema=s.arguments.model_json_schema())
            for s in specs.values()
        ]

    @server.call_tool()
    async def call_tool(name, arguments):
        if name not in specs:
            raise ValueError(f"tool not available this turn: {name}")
        result = await tools.invoke(name, arguments or {})
        return [TextContent(type="text", text=json.dumps(result, default=str, ensure_ascii=False))]

    return server


async def run():
    binding = IntrospectToolBindingV1.model_validate_json(os.environ["ORION_INTROSPECT_BINDING"])
    bus = OrionBusAsync(os.environ["ORION_BUS_URL"])
    try:
        await bus.connect()
        server = build_server(IntrospectTools(bus, binding))
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())
    finally:
        await bus.close()


if __name__ == "__main__":
    asyncio.run(run())
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `PYTHONPATH=. $PY -m pytest orion/introspect/tests -q`
Expected: all pass. If the unlisted-tool assertion fails because the installed `mcp` returns a non-error for unknown names, keep the `ValueError` in `call_tool` and assert on the error text instead — the requirement is that `tools.calls` stays empty.

- [ ] **Step 8: Commit**

```bash
git add orion/introspect/__init__.py orion/introspect/tools.py orion/introspect/mcp_server.py orion/introspect/tests/test_introspect_tools.py orion/introspect/tests/test_introspect_mcp_server.py
git diff --cached --check
git commit -m "feat(introspect): orion-introspect MCP server with reading_results tool"
```

---

### Task 4: Hub listener answers `reading_result`

**Files:**
- Modify: `services/orion-hub/scripts/reading_listener.py` (imports lines 10–18; operation dispatch lines 90–115)
- Test: `services/orion-hub/tests/test_reading_ingress.py` (append)

**Interfaces:**
- Consumes: `reading_results(conn, *, request_id, url, limit, since) -> IntrospectResultV1` (Task 2); `IntrospectTools` (Task 3); `DEFAULT_LIMIT` (Task 1).
- Produces: listener replies `ReadingToolResultV1(ok=True, result=IntrospectResultV1.model_dump(mode="json"))`; on failure the existing `_SAFE_ERROR` reply; INFO log `introspect op=reading_result corr=<id> items=<n> total=<n>`.

- [ ] **Step 1: Write the failing tests**

Append to `services/orion-hub/tests/test_reading_ingress.py`:

```python
def _introspect_tools(bus):
    from orion.introspect.tools import IntrospectTools
    from orion.schemas.introspect import IntrospectToolBindingV1
    return IntrospectTools(bus, IntrospectToolBindingV1(
        invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t", memory_allowed=True,
    ))


def test_reading_result_round_trips_through_real_listener(monkeypatch, caplog):
    import logging
    from datetime import datetime, timezone
    from orion.schemas.introspect import IntrospectResultV1

    seen = {}

    async def fake_results(conn, **kwargs):
        seen.update(kwargs)
        return IntrospectResultV1(
            ok=True, operation="reading_result", as_of=datetime.now(timezone.utc), total_available=0,
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("reading_result must never enqueue")

    monkeypatch.setitem(ReadingListener.handle.__globals__, "reading_results", fake_results)
    monkeypatch.setitem(ReadingListener.handle.__globals__, "enqueue_reading", forbidden)
    bus = RpcBus(_FakeConn())
    caplog.set_level(logging.INFO, logger="scripts.reading_listener")
    out = asyncio.run(_introspect_tools(bus).invoke(
        "reading_results", {"url": "https://EXAMPLE.org/a#frag", "limit": 3},
    ))
    assert out["ok"] is True and out["items"] == [] and out["total_available"] == 0
    assert seen == {"request_id": None, "url": "https://example.org/a", "limit": 3, "since": None}
    assert "introspect op=reading_result" in caplog.text
    assert f"corr={bus.commands[0].correlation_id}" in caplog.text


def test_reading_result_failure_is_unknown_and_sanitized(monkeypatch, caplog):
    from orion.introspect.tools import IntrospectUnknownError

    async def boom(conn, **kwargs):
        raise RuntimeError("lost postgres://orion:hunter2@db/orion")

    monkeypatch.setitem(ReadingListener.handle.__globals__, "reading_results", boom)
    bus = RpcBus(_FakeConn())
    with pytest.raises(IntrospectUnknownError, match="answer unknown"):
        asyncio.run(_introspect_tools(bus).invoke("reading_results", {}))
    assert "hunter2" not in caplog.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. $PY -m pytest services/orion-hub/tests/test_reading_ingress.py -k reading_result -q`
Expected: FAIL — `monkeypatch.setitem` succeeds but the listener still routes `reading_result` into the status branch (`reading_status requires exactly one...` → `_SAFE_ERROR`), so the round-trip test raises `IntrospectUnknownError`.

- [ ] **Step 3: Route the operation in `reading_listener.py`**

Add imports:

```python
from orion.schemas.introspect import DEFAULT_LIMIT
from orion.world_pulse_read.introspect import reading_results
```

Replace the status `else:` branch (the block starting `else:` / `phase = "status"` inside `async with pool.acquire() as conn:`) with two branches:

```python
                elif command.operation == "reading_status":
                    phase = "status"
                    if command.url is not None:
                        result = await reading_status(conn, url=command.url)
                    else:
                        result = await reading_status(conn, command.request_id)
                    try:
                        receipt = ReadingStatusReceiptV1.model_validate(result)
                    except ValueError as exc:
                        raise RuntimeError("status returned a malformed receipt") from exc
                    if command.request_id is not None and receipt.request_id != command.request_id:
                        raise RuntimeError("status returned a mismatched request_id")
                    if command.url is not None:
                        if result.get("lookup_url") != normalize_source_url(command.url):
                            raise RuntimeError("status returned a mismatched URL")
                else:
                    phase = "reading_result"
                    introspection = await reading_results(
                        conn, request_id=command.request_id, url=command.url,
                        limit=command.limit or DEFAULT_LIMIT, since=command.since,
                    )
                    result = introspection.model_dump(mode="json")
                    logger.info(
                        "introspect op=reading_result corr=%s items=%d total=%s",
                        envelope.correlation_id, len(introspection.items), introspection.total_available,
                    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. $PY -m pytest services/orion-hub/tests/test_reading_ingress.py services/orion-hub/tests/test_reading_turn_listener.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/reading_listener.py services/orion-hub/tests/test_reading_ingress.py
git diff --cached --check
git commit -m "feat(hub): reading listener answers read-only reading_result"
```

---

### Task 5: Harness wiring, outward guard, brief, and flag

**Files:**
- Create: `orion/introspect/binding.py`, `orion/introspect/brief.py`
- Modify: `orion/fcc/mcp_config.py` (`render_mcp_config` signature line ~136; block after the `reading_binding` block ~line 185)
- Modify: `orion/harness/fcc_motor.py` (`_CONTEXT_GATHERING_MCP_PREFIXES` line 335; `_maybe_render_mcp_config` lines 654–688)
- Modify: `orion/harness/prefix.py` (imports; after `append_reading_mcp_harness_brief(...)` line ~245)
- Modify: `services/orion-harness-governor/.env_example` (after `HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED=true`), `docker-compose.yml` (after line 122), `app/settings.py` (after `harness_fcc_context_mode_hooks_enabled`), `README.md` (FCC MCP section)
- Modify: `.github/workflows/orion-reading-tests.yml`
- Test: `orion/introspect/tests/test_introspect_harness_wiring.py`

**Interfaces:**
- Consumes: `IntrospectToolBindingV1` (Task 1); `ReadingToolBindingV1`; `harness_mcp_enabled()` from `orion.fcc.github_repo_context`; `McpPreflightError(error_code, message)`.
- Produces:
  - `orion.introspect.binding`: `introspect_enabled() -> bool`, `outward_tools_attached() -> bool`, `introspect_binding_for_turn(reading_binding: ReadingToolBindingV1 | None, *, reading_only: bool = False) -> IntrospectToolBindingV1 | None`.
  - `orion.introspect.brief`: `introspect_brief_lines(binding: IntrospectToolBindingV1) -> list[str]`, `append_introspect_harness_brief(parts: list[str], *, binding: IntrospectToolBindingV1 | None) -> None`.
  - `render_mcp_config(..., introspect_binding=None, introspect_bus_url=None)`; error codes `fcc_introspect_bus_missing`, `fcc_introspect_outward_memory`.

- [ ] **Step 1: Write the failing tests**

Create `orion/introspect/tests/test_introspect_harness_wiring.py`:

```python
"""The harness attaches orion-introspect only when it should, and the model never sets the binding."""
import json

import pytest

from orion.fcc import mcp_config
from orion.fcc.mcp_config import McpPreflightError
from orion.harness import fcc_motor as motor
from orion.harness.prefix import compile_harness_prefix
from orion.harness.tests.fixtures import make_thought
from orion.introspect.binding import introspect_binding_for_turn
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.introspect import IntrospectToolBindingV1
from orion.schemas.reading import ReadingToolBindingV1

READING = ReadingToolBindingV1(invocation_context="curiosity", parent_run_id="run-9", parent_trace_id="trace-9")
BUS = "redis://100.92.216.81:6379/0"
FCC_ENV = {
    "GITHUB_PAT": "x", "FIRECRAWL_API_KEY": "y",
    "AITOWN_CONVEX_URL": "http://convex.test", "AITOWN_ADMIN_KEY": "k", "AITOWN_WORLD_ID": "w",
}


@pytest.fixture
def harness_env(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_config.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(mcp_config, "_TMP_ROOT", tmp_path)
    monkeypatch.setattr(mcp_config, "_probe_convex_version", lambda *a, **k: None)
    monkeypatch.setattr(mcp_config, "_probe_convex_auth", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", lambda _path: dict(FCC_ENV))
    monkeypatch.setenv("HARNESS_FCC_MCP_ENABLED", "true")
    monkeypatch.setenv("HARNESS_FCC_INTROSPECT_ENABLED", "true")
    monkeypatch.setenv("HARNESS_AITOWN_ENABLED", "false")
    monkeypatch.setenv("ORION_BUS_URL", BUS)
    for key in ("HARNESS_FCC_GITNEXUS_ENABLED", "HARNESS_FCC_CONTEXT_MODE_ENABLED",
                "HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED", "HARNESS_AITOWN_CONVEX_URL"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _servers(**kwargs):
    path = motor._maybe_render_mcp_config(correlation_id="corr-introspect", **kwargs)
    return json.loads(path.read_text())["mcpServers"]


def test_attached_with_flag_and_binding(harness_env):
    server = _servers(reading_binding=READING)["orion-introspect"]
    assert server["type"] == "stdio"
    assert server["args"] == ["-P", "-m", "orion.introspect.mcp_server"]
    assert set(server["env"]) == {"ORION_BUS_URL", "PYTHONPATH", "ORION_INTROSPECT_BINDING"}
    assert server["env"]["ORION_BUS_URL"] == BUS
    binding = IntrospectToolBindingV1.model_validate_json(server["env"]["ORION_INTROSPECT_BINDING"])
    assert binding == IntrospectToolBindingV1(
        invocation_context="curiosity", parent_run_id="run-9", parent_trace_id="trace-9", memory_allowed=True,
    )


def test_ai_town_turns_get_memory_disallowed(harness_env):
    harness_env.setenv("HARNESS_AITOWN_ENABLED", "true")
    servers = _servers(reading_binding=READING)
    assert "orion-aitown" in servers
    binding = IntrospectToolBindingV1.model_validate_json(servers["orion-introspect"]["env"]["ORION_INTROSPECT_BINDING"])
    assert binding.memory_allowed is False


def test_absent_without_flag_or_binding(harness_env):
    assert "orion-introspect" not in _servers()
    harness_env.setenv("HARNESS_FCC_INTROSPECT_ENABLED", "false")
    assert "orion-introspect" not in _servers(reading_binding=READING)


def test_reading_only_turns_stay_empty(harness_env):
    assert _servers(reading_binding=READING, reading_only=True) == {}


def test_render_refuses_memory_access_alongside_ai_town(harness_env, tmp_path):
    risky = IntrospectToolBindingV1(
        invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t", memory_allowed=True,
    )
    with pytest.raises(McpPreflightError) as exc:
        mcp_config.render_mcp_config(
            correlation_id="c", fcc_env=dict(FCC_ENV), tmp_dir=tmp_path,
            include_aitown=True, introspect_binding=risky, introspect_bus_url=BUS,
        )
    assert exc.value.error_code == "fcc_introspect_outward_memory"


def test_render_requires_bus_url(harness_env, tmp_path):
    binding = introspect_binding_for_turn(READING)
    with pytest.raises(McpPreflightError) as exc:
        mcp_config.render_mcp_config(
            correlation_id="c", fcc_env=dict(FCC_ENV), tmp_dir=tmp_path, introspect_binding=binding,
        )
    assert exc.value.error_code == "fcc_introspect_bus_missing"


def test_binding_requires_master_mcp_flag(harness_env):
    harness_env.delenv("HARNESS_FCC_MCP_ENABLED")
    assert introspect_binding_for_turn(READING) is None


def test_introspect_calls_count_as_context_gathering():
    assert motor.classify_step_tool_kind("mcp__orion-introspect__reading_results") == "context_gathering"


def _prefix(**kwargs):
    return compile_harness_prefix(
        make_thought(imperative="What did you learn from that article?"),
        repair_overlay=HarnessRepairOverlayV1(),
        **kwargs,
    )


def test_brief_present_only_when_server_attached(harness_env):
    harness_env.delenv("ORION_GITHUB_OWNER", raising=False)
    harness_env.delenv("ORION_GITHUB_REPO", raising=False)
    prompt = _prefix(reading_binding=READING)
    assert "orion-introspect" in prompt
    assert "reading_results" in prompt
    assert "the answer is unknown" in prompt
    assert "not settled beliefs" in prompt
    assert "orion-introspect" not in _prefix(reading_binding=READING, reading_only=True)
    assert "orion-introspect" not in _prefix()
    harness_env.setenv("HARNESS_FCC_INTROSPECT_ENABLED", "false")
    assert "orion-introspect" not in _prefix(reading_binding=READING)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. $PY -m pytest orion/introspect/tests/test_introspect_harness_wiring.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'orion.introspect.binding'`.

- [ ] **Step 3: Create `orion/introspect/binding.py`**

```python
"""Derive the per-turn introspect binding from runtime facts, never from the model."""
from __future__ import annotations

import os

from orion.fcc.github_repo_context import harness_mcp_enabled
from orion.schemas.introspect import IntrospectToolBindingV1
from orion.schemas.reading import ReadingToolBindingV1

_TRUTHY = {"1", "true", "yes", "on"}


def _env_truthy(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in _TRUTHY


def introspect_enabled() -> bool:
    return harness_mcp_enabled() and _env_truthy("HARNESS_FCC_INTROSPECT_ENABLED")


def outward_tools_attached() -> bool:
    """True when this turn can speak to someone other than Juniper (AI Town today)."""
    return _env_truthy("HARNESS_AITOWN_ENABLED")


def introspect_binding_for_turn(
    reading_binding: ReadingToolBindingV1 | None, *, reading_only: bool = False,
) -> IntrospectToolBindingV1 | None:
    if reading_only or reading_binding is None or not introspect_enabled():
        return None
    return IntrospectToolBindingV1(
        invocation_context=reading_binding.invocation_context,
        parent_run_id=reading_binding.parent_run_id,
        parent_trace_id=reading_binding.parent_trace_id,
        memory_allowed=not outward_tools_attached(),
    )
```

- [ ] **Step 4: Create `orion/introspect/brief.py`**

```python
"""Harness usage brief for orion-introspect, appended only when the server is attached."""
from __future__ import annotations

from orion.schemas.introspect import IntrospectToolBindingV1


def introspect_brief_lines(binding: IntrospectToolBindingV1) -> list[str]:
    return [
        (
            "Introspect MCP (orion-introspect) is available: reading_results returns what you "
            "actually learned from sources read through your reading pipeline -- by url or "
            "request_id, or your most recent finished reads when neither is given. Call it before "
            "describing what you learned from a reading instead of reconstructing it; "
            "reading_status says where a request is in the queue, reading_results says what came "
            "out of it. Results are source-attributed candidates, not settled beliefs. items=[] "
            "means nothing matched; a tool error means the answer is unknown -- say so, never "
            "report an error as nothing having happened. learned=false means no output exists "
            "yet: report its reading_status."
        ),
    ]


def append_introspect_harness_brief(
    parts: list[str], *, binding: IntrospectToolBindingV1 | None,
) -> None:
    if binding is None:
        return
    parts.extend(introspect_brief_lines(binding))
```

- [ ] **Step 5: Render the server in `orion/fcc/mcp_config.py`**

Add two keyword parameters to `render_mcp_config` after `reading_bus_url: Optional[str] = None,`:

```python
    introspect_binding: Any = None,
    introspect_bus_url: Optional[str] = None,
```

Insert directly after the `if reading_binding is not None:` block:

```python
    if introspect_binding is not None:
        from orion.schemas.introspect import IntrospectToolBindingV1
        ib = IntrospectToolBindingV1.model_validate(introspect_binding)
        if not introspect_bus_url:
            raise McpPreflightError("fcc_introspect_bus_missing", "ORION_BUS_URL required for introspect tools")
        if include_aitown and ib.memory_allowed:
            raise McpPreflightError(
                "fcc_introspect_outward_memory",
                "introspect memory access must be off when an outward-facing MCP (AI Town) is attached",
            )
        rendered["mcpServers"]["orion-introspect"] = {
            "type": "stdio", "command": "python3",
            "args": ["-P", "-m", "orion.introspect.mcp_server"],
            "env": {"ORION_BUS_URL": introspect_bus_url,
                    "PYTHONPATH": str(_TEMPLATE_PATH.parents[2]),
                    "ORION_INTROSPECT_BINDING": ib.model_dump_json()},
        }
```

- [ ] **Step 6: Wire the motor in `orion/harness/fcc_motor.py`**

Change line 335 to:

```python
_CONTEXT_GATHERING_MCP_PREFIXES = ("mcp__gitnexus__", "mcp__firecrawl__", "mcp__orion-introspect__")
```

Update the comment above it (lines 330–333) so the list of read-only servers reads: `(gitnexus: graph queries only; firecrawl: search/scrape only; orion-introspect: SELECT-only lookups)`.

In `_maybe_render_mcp_config`, add the import next to `render_mcp_config`:

```python
    from orion.introspect.binding import introspect_binding_for_turn
```

and add two arguments to the final `render_mcp_config(...)` call, after `reading_bus_url=os.environ.get("ORION_BUS_URL"),`:

```python
        introspect_binding=introspect_binding_for_turn(reading_binding, reading_only=reading_only),
        introspect_bus_url=os.environ.get("ORION_BUS_URL"),
```

- [ ] **Step 7: Append the brief in `orion/harness/prefix.py`**

Add imports:

```python
from orion.introspect.binding import introspect_binding_for_turn
from orion.introspect.brief import append_introspect_harness_brief
```

Directly after the `append_reading_mcp_harness_brief(...)` call, add:

```python
    append_introspect_harness_brief(
        parts, binding=introspect_binding_for_turn(reading_binding, reading_only=reading_only)
    )
```

In the `compile_harness_prefix` docstring, change "enabled MCP tool briefs" to "enabled MCP tool briefs (including orion-introspect)".

- [ ] **Step 8: Run tests to verify they pass**

Run:
```bash
PYTHONPATH=. $PY -m pytest orion/introspect/tests orion/fcc/tests/test_mcp_config.py orion/fcc/tests/test_reading_mcp.py orion/harness/tests/test_fcc_motor_mcp.py orion/harness/tests/test_harness_prefix.py -q
```
Expected: all pass.

- [ ] **Step 9: Add the flag to the governor config surfaces**

`services/orion-harness-governor/.env_example` — after the `HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED=true` line:

```bash
# Orion introspect MCP (design: docs/superpowers/specs/2026-09-28-orion-introspect-mcp-design.md).
# Lets Orion look up their own recorded activity over the bus (slice 1:
# reading_results -- what was learned from a reading, answered by orion-hub).
# Needs HARNESS_FCC_MCP_ENABLED and a reading binding on the turn. Read-only.
# Off by default.
HARNESS_FCC_INTROSPECT_ENABLED=false
```

`services/orion-harness-governor/docker-compose.yml` — after the `HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED` line:

```yaml
      - HARNESS_FCC_INTROSPECT_ENABLED=${HARNESS_FCC_INTROSPECT_ENABLED:-false}
```

`services/orion-harness-governor/app/settings.py` — after the `harness_fcc_context_mode_hooks_enabled` field:

```python
    # orion-introspect MCP; read at spawn time by orion/introspect/binding.py.
    harness_fcc_introspect_enabled: bool = Field(False, alias="HARNESS_FCC_INTROSPECT_ENABLED")
```

`services/orion-harness-governor/README.md` — at the end of the "FCC MCP (Orion mode)" intro paragraph, add a new paragraph:

```markdown
`HARNESS_FCC_INTROSPECT_ENABLED=true` adds `orion-introspect`, a read-only MCP that lets Orion look up their own recorded activity over the bus. Slice 1 exposes `reading_results` (what was learned from a reading; answered by orion-hub on `orion:reading:tool:request`). It is attached only on turns with a reading binding, never on `reading_only` turns. Design: `docs/superpowers/specs/2026-09-28-orion-introspect-mcp-design.md`.
```

- [ ] **Step 10: Sync local env and run env gates**

```bash
python3 scripts/sync_local_env_from_example.py
python3 scripts/check_env_template_parity.py
python3 scripts/check_env_key_single_source.py
grep -n HARNESS_FCC_INTROSPECT_ENABLED services/orion-harness-governor/.env
git check-ignore services/orion-harness-governor/.env
```

Expected: sync reports the new key added (or already present); both checks exit 0; the grep shows `HARNESS_FCC_INTROSPECT_ENABLED=false`; `check-ignore` prints the path. Note that the sync ran against the worktree's `.env`; the shared checkout's `services/orion-harness-governor/.env` also needs the key — run the same sync script from `/mnt/scripts/Orion-Sapienform` after merge (read-only on git, writes only the ignored `.env`), and report it in the PR.

- [ ] **Step 11: Add new tests to CI**

In `.github/workflows/orion-reading-tests.yml`, add under `paths:` (after `"orion/fcc/**"`):

```yaml
      - "orion/introspect/**"
```

and add a line to the pytest list after `orion/fcc/tests/test_mcp_config.py \`:

```yaml
            orion/introspect/tests \
```

- [ ] **Step 12: Commit**

```bash
git add orion/introspect/binding.py orion/introspect/brief.py orion/introspect/tests/test_introspect_harness_wiring.py orion/fcc/mcp_config.py orion/harness/fcc_motor.py orion/harness/prefix.py services/orion-harness-governor/.env_example services/orion-harness-governor/docker-compose.yml services/orion-harness-governor/app/settings.py services/orion-harness-governor/README.md .github/workflows/orion-reading-tests.yml
git status --short
git diff --cached --check
git commit -m "feat(harness): attach orion-introspect behind HARNESS_FCC_INTROSPECT_ENABLED"
```

Expected: `git status --short` shows no `.env` staged.

---

### Task 6: Live smoke, full gates, review, PR

**Files:**
- Create: `scripts/smoke_introspect.py`

**Interfaces:**
- Consumes: `IntrospectTools`, `IntrospectUnknownError` (Task 3); `IntrospectToolBindingV1` (Task 1); `OrionBusAsync`.
- Produces: CLI `python scripts/smoke_introspect.py [--url URL] [--limit N]`, exit 0 on a coherent answer, exit 2 on `answer unknown`, exit 1 on a degenerate answer (ok but an item with `learned=true` and empty text).

- [ ] **Step 1: Create `scripts/smoke_introspect.py`**

```python
#!/usr/bin/env python3
"""Live smoke: ask orion-hub for reading_results over the real bus.

    ORION_BUS_URL=redis://100.92.216.81:6379/0 python scripts/smoke_introspect.py --limit 3

Read-only. Exit 0 = coherent answer, 1 = degenerate answer, 2 = answer unknown.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from orion.core.bus.async_service import OrionBusAsync
from orion.introspect.tools import IntrospectTools, IntrospectUnknownError
from orion.schemas.introspect import IntrospectToolBindingV1


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    bus_url = os.environ.get("ORION_BUS_URL")
    if not bus_url:
        print("ORION_BUS_URL is required (redis://<tailscale-node-ip>:6379/0)", file=sys.stderr)
        return 2
    binding = IntrospectToolBindingV1(
        invocation_context="unified_chat", parent_run_id="smoke-introspect",
        parent_trace_id="smoke-introspect", memory_allowed=False,
    )
    arguments = {"url": args.url} if args.url else {"limit": args.limit}
    bus = OrionBusAsync(bus_url)
    await bus.connect()
    try:
        result = await IntrospectTools(bus, binding).invoke("reading_results", arguments)
    except IntrospectUnknownError as exc:
        print(f"UNKNOWN: {exc}", file=sys.stderr)
        return 2
    finally:
        await bus.close()
    print(json.dumps(result, indent=2))
    degenerate = [i["id"] for i in result["items"] if i["extra"].get("learned") and not i["text"]]
    if degenerate:
        print(f"DEGENERATE: learned=true with empty text: {degenerate}", file=sys.stderr)
        return 1
    print(f"OK items={len(result['items'])} total_available={result['total_available']} as_of={result['as_of']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 2: Run the full focused suite**

```bash
RUN_READING_POSTGRES=1 PYTHONPATH=. $PY -m pytest -q \
  orion/introspect/tests \
  services/orion-hub/tests/test_reading_ingress.py \
  services/orion-hub/tests/test_reading_postgres.py \
  services/orion-hub/tests/test_reading_turn_listener.py \
  orion/world_pulse_read/tests \
  orion/fcc/tests/test_reading_mcp.py orion/fcc/tests/test_mcp_config.py \
  orion/harness/tests/test_fcc_motor_mcp.py orion/harness/tests/test_harness_prefix.py \
  tests/test_unified_turn_bus_catalog.py tests/test_unified_turn_schemas.py
python3 scripts/check_bus_reply_channels.py
git diff --check origin/main...HEAD
```

Expected: all pass; checks exit 0.

- [ ] **Step 3: Commit the smoke script**

```bash
git add scripts/smoke_introspect.py
git diff --cached --check
git commit -m "chore(introspect): live bus smoke for reading_results"
```

- [ ] **Step 4: Deploy Hub from the worktree and run the live smoke**

The Hub responder must be running this branch for the live smoke to mean anything. Deploy only through the wrapper (refuses the shared checkout):

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. $PY scripts/smoke_introspect.py --limit 3
docker logs orion-hub --since 5m 2>&1 | grep "introspect op=reading_result"
```

Expected: smoke exits 0 with `OK items=<n> total_available=<n>`, and the Hub log shows a line with the same correlation ID. If deploying Hub is not approved in this session, record the smoke as `UNVERIFIED` in the PR and print the commands for Juniper instead of running them.

- [ ] **Step 5: Refresh the knowledge graph**

```bash
scripts/safe_graphify_update.sh
```

Expected: node count does not drop more than 10% (wrapper enforces).

- [ ] **Step 6: Code review in a subagent**

Dispatch the review-agent skill (`/home/athena/.codex/skills/.system/review-agent/SKILL.md`) against `origin/main...HEAD`. Ask it to check specifically: (a) any path where a failure reaches Orion as `items=[]`; (b) any way the model can influence `IntrospectToolBindingV1`; (c) `reading_status` behavior unchanged for existing callers; (d) worst-case payload vs the 12,000-char MCP cap; (e) the `reading_result` SQL is SELECT-only. Fix every material finding, re-run Step 2, commit fixes.

- [ ] **Step 7: Push and open the PR**

```bash
git push -u origin feat/introspect-slice1
gh pr create --title "feat(introspect): orion-introspect MCP, slice 1 (reading_results)" --body-file /tmp/introspect-slice1-pr.md
```

Write `/tmp/introspect-slice1-pr.md` using the AGENTS.md §18 template. It must include: env key `HARNESS_FCC_INTROSPECT_ENABLED` (added, default false, local `.env` synced); no new bus channels; `ReadingToolRequestV1` gained `reading_result`/`limit`/`since` (old Hub rejects `reading_result` → tool reports "answer unknown", graceful); text cap 900 vs spec 1200 and why; eval status (model-in-loop truth eval is scheduled for slice 2 per spec; this slice's evidence is the real-SQL state matrix + live smoke); live smoke result or `UNVERIFIED`; restart commands:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
# then, to turn it on: set HARNESS_FCC_INTROSPECT_ENABLED=true in services/orion-harness-governor/.env and
scripts/safe_docker_build.sh orion-harness-governor up -d
```

- [ ] **Step 8: Watch CI and resolve conflicts**

```bash
gh pr checks --watch
gh pr view --json mergeable,mergeStateStatus
```

Expected: `orion-reading tests` green; mergeable. Fix any failure at its cause and push.

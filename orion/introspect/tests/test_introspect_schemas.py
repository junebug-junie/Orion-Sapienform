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
            "source_read": True,
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


def test_existing_reading_operations_keep_their_pre_introspect_wire_keys():
    # A Hub older than this contract forbids unknown keys, even null ones.
    status = ReadingToolRequestV1(operation="reading_status", url="https://example.org/a")
    assert set(status.model_dump(mode="json")) == {"operation", "request", "request_id", "url"}
    recent = ReadingToolRequestV1(operation="reading_result", limit=3, since=NOW)
    wire = recent.model_dump(mode="json")
    assert wire["limit"] == 3 and wire["since"].startswith("2026-")
    assert ReadingToolRequestV1.model_validate(wire) == recent


def test_new_models_are_registered():
    for name in ("IntrospectToolBindingV1", "IntrospectItemV1", "IntrospectResultV1", "ReadingResultArguments"):
        assert resolve(name) is not None

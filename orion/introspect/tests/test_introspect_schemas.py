"""Introspect contracts: bounded, empty-vs-unknown, caller-bound."""
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from orion.schemas.introspect import (
    DEFAULT_TEXT_CAP,
    MAX_ITEMS,
    SHORT_FIELD_CAP,
    QUERY_CAP,
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


def test_reading_result_arguments_accept_query_and_strip_it():
    args = ReadingResultArguments(query="  graphics cards  ", since=NOW)
    assert args.query == "graphics cards"
    assert args.since == NOW


@pytest.mark.parametrize(
    "fields",
    [
        {"query": "   "},
        {"query": "x" * 501},
        {"query": "gpus", "url": "https://example.org/a"},
        {"query": "gpus", "request_id": "00000000-0000-0000-0000-000000000001"},
    ],
)
def test_reading_result_arguments_reject_bad_query(fields):
    with pytest.raises(ValidationError):
        ReadingResultArguments(**fields)


def test_reading_tool_request_carries_query_only_for_reading_result():
    wire = ReadingToolRequestV1(operation="reading_result", query="gpus", limit=2).model_dump(mode="json")
    assert wire["query"] == "gpus"
    assert ReadingToolRequestV1.model_validate(wire).query == "gpus"
    with pytest.raises(ValidationError):
        ReadingToolRequestV1(operation="reading_status", url="https://example.org/a", query="gpus")
    with pytest.raises(ValidationError):
        ReadingToolRequestV1(operation="reading_result", url="https://example.org/a", query="gpus")


def test_null_query_never_reaches_the_wire():
    # A pre-1b Hub forbids unknown keys, even null ones.
    recent = ReadingToolRequestV1(operation="reading_result", limit=3).model_dump(mode="json")
    assert "query" not in recent
    status = ReadingToolRequestV1(operation="reading_status", url="https://example.org/a")
    assert set(status.model_dump(mode="json")) == {"operation", "request", "request_id", "url"}


def test_query_length_is_checked_after_stripping_on_both_contracts():
    padded = " " + "x" * QUERY_CAP + " "
    assert ReadingResultArguments(query=padded).query == "x" * QUERY_CAP
    wire = ReadingToolRequestV1(operation="reading_result", query="  gpus  ")
    assert wire.query == "gpus"
    with pytest.raises(ValidationError):
        ReadingToolRequestV1(operation="reading_result", query="   ")


from orion.introspect.transport import DREAM_REQUEST_CHANNEL, REQUEST_KIND, RESULT_KIND, RESULT_PREFIX  # noqa: E402
from orion.schemas.introspect import (  # noqa: E402
    FULL_TEXT_CAP,
    DreamsArguments,
    IntrospectRequestV1,
    IntrospectResultV1,
    IntrospectToolBindingV1,
)

_BINDING = IntrospectToolBindingV1(
    invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t", memory_allowed=True,
)


def test_transport_constants():
    assert DREAM_REQUEST_CHANNEL == "orion:introspect:dream:request"
    assert RESULT_PREFIX == "orion:introspect:result:"
    assert (REQUEST_KIND, RESULT_KIND) == ("introspect.tool.request.v1", "introspect.tool.result.v1")


def test_dreams_arguments_defaults_and_modes():
    assert DreamsArguments().limit == 5
    assert DreamsArguments(query="  vision  ").query == "vision"
    assert DreamsArguments(dream_id="dream:19").dream_id == "dream:19"
    assert DreamsArguments(dream_id="dh-33f002b0db4c").dream_id == "dh-33f002b0db4c"
    assert DreamsArguments(kind="hypothesis", since=NOW).kind == "hypothesis"


@pytest.mark.parametrize(
    "fields",
    [
        {"query": "   "},
        {"query": "x" * 501},
        {"dream_id": "19"},
        {"dream_id": "dh-XYZ"},
        {"dream_id": "dream:19", "query": "vision"},
        {"dream_id": "dream:19", "kind": "narrative"},
        {"dream_id": "dream:19", "since": NOW},
        {"kind": "control"},
        {"limit": 6},
        {"since": NOW.replace(tzinfo=None)},
        {"arm": "dream"},
    ],
)
def test_dreams_arguments_reject_bad_input(fields):
    with pytest.raises(ValidationError):
        DreamsArguments(**fields)


def test_request_carries_only_bus_operations():
    req = IntrospectRequestV1(operation="dreams", binding=_BINDING, args={"limit": 2})
    assert req.model_dump(mode="json")["operation"] == "dreams"
    with pytest.raises(ValidationError):
        IntrospectRequestV1(operation="reading_result", binding=_BINDING, args={})


def test_result_accepts_dreams_operation():
    result = IntrospectResultV1(ok=True, operation="dreams", as_of=NOW, total_available=0)
    assert result.operation == "dreams"
    assert FULL_TEXT_CAP == 4000


def test_registry_and_channels_know_the_dream_contract():
    import yaml
    from pathlib import Path

    from orion.schemas.registry import _REGISTRY

    assert {"IntrospectRequestV1", "DreamsArguments", "IntrospectResultV1"} <= set(_REGISTRY)
    channels = yaml.safe_load((Path(__file__).resolve().parents[3] / "orion/bus/channels.yaml").read_text())["channels"]
    by_name = {c["name"]: c for c in channels}
    req = by_name["orion:introspect:dream:request"]
    assert (req["schema_id"], req["message_kind"], req["consumer_services"]) == (
        "IntrospectRequestV1", "introspect.tool.request.v1", ["orion-dream"],
    )
    res = by_name["orion:introspect:result:*"]
    assert (res["schema_id"], res["message_kind"]) == ("IntrospectResultV1", "introspect.tool.result.v1")
    assert "orion-dream" in by_name["orion:vector:semantic:upsert"]["producer_services"]


from orion.introspect.transport import CURIOSITY_REQUEST_CHANNEL  # noqa: E402
from orion.schemas.introspect import (  # noqa: E402
    CURIOSITY_FULL_JSON_BUDGET,
    CURIOSITY_RUN_ID_PATTERN,
    CuriosityArguments,
    clip_json_text,
)


def test_curiosity_run_id_pattern_matches_the_atlas_guard():
    from orion.curiosity.atlas import _RUN_ID_RE

    assert _RUN_ID_RE.pattern == CURIOSITY_RUN_ID_PATTERN


def test_curiosity_arguments_defaults_and_modes():
    args = CuriosityArguments()
    assert (args.kind, args.limit, args.line, args.query) == ("run", 5, None, None)
    assert CuriosityArguments(query="  bees  ").query == "bees"
    assert CuriosityArguments(run_id="3dc94088912b").run_id == "3dc94088912b"
    assert CuriosityArguments(run_id="r1", kind="run").kind == "run"
    recent = datetime.now(timezone.utc) - timedelta(days=3)
    assert CuriosityArguments(line="self_inquiry", since=recent, query="x").line == "self_inquiry"
    assert CuriosityArguments(kind="self_question", since=recent, limit=2).kind == "self_question"


@pytest.mark.parametrize(
    "fields",
    [
        {"query": "   "},
        {"query": "x" * 501},
        {"run_id": "bad id"},
        {"run_id": "r1' OR 1=1"},
        {"run_id": "x" * 65},
        {"run_id": "r1", "query": "bees"},
        {"run_id": "r1", "since": datetime.now(timezone.utc)},
        {"run_id": "r1", "line": "investigate"},
        {"run_id": "r1", "kind": "self_question"},
        {"kind": "self_question", "query": "bees"},
        {"kind": "self_question", "line": "self_inquiry"},
        {"kind": "candidate"},
        {"line": "reflect"},
        {"limit": 0},
        {"limit": 6},
        {"since": datetime.now().replace(tzinfo=None)},
        {"status": "failed"},
    ],
)
def test_curiosity_arguments_reject_bad_input(fields):
    with pytest.raises(ValidationError):
        CuriosityArguments(**fields)


def test_clip_json_text_bounds_the_serialized_length():
    assert clip_json_text("short", 100) == ("short", False)
    assert clip_json_text(None, 100) == ("", False)
    nasty = 'He said "no"\n\\ é ' * 2000
    body, truncated = clip_json_text(nasty, CURIOSITY_FULL_JSON_BUDGET)
    assert truncated
    assert len(json.dumps(body, ensure_ascii=False)) <= CURIOSITY_FULL_JSON_BUDGET
    # Longest prefix: one more character would not fit.
    assert len(json.dumps(nasty.strip()[: len(body) + 1], ensure_ascii=False)) > CURIOSITY_FULL_JSON_BUDGET


def test_worst_case_full_curiosity_run_fits_the_mcp_tool_result_cap():
    nasty = '"Answer"\n\tbackslash \\ accent é ' * 1000
    text, truncated = clip_json_text(nasty, CURIOSITY_FULL_JSON_BUDGET)
    item = _item(
        id="x" * 64, kind="curiosity_run", text=text, truncated=truncated,
        extra={
            "line": "self_inquiry", "status": "failed", "error": "e" * SHORT_FIELD_CAP,
            "hops": 12, "findings": 3, "revisions": 2,
            "prior_touched": {"claim": '"c"\n' * (SHORT_FIELD_CAP // 4), "from": 0.4, "to": 0.7},
            "outcome_kind": "reached_out_blocked", "reach_out": "blocked:quiet_hours",
            "turn_ok": True, "n_tested": 4, "n_moved": 2, "n_formed": 1, "unknown_reason": "u" * SHORT_FIELD_CAP,
        },
    )
    result = IntrospectResultV1(ok=True, operation="curiosity", as_of=NOW, total_available=1, items=[item])
    assert len(json.dumps(result.model_dump(mode="json"))) < MCP_TOOL_RESULT_MAX_CHARS
    assert len(json.dumps(result.model_dump(mode="json"), ensure_ascii=False)) < MCP_TOOL_RESULT_MAX_CHARS


def test_registry_and_channels_know_the_curiosity_contract():
    import yaml
    from pathlib import Path

    from orion.schemas.registry import _REGISTRY

    assert CURIOSITY_REQUEST_CHANNEL == "orion:introspect:curiosity:request"
    assert IntrospectRequestV1(operation="curiosity", binding=_BINDING, args={}).operation == "curiosity"
    assert _REGISTRY["CuriosityArguments"] is CuriosityArguments
    channels = yaml.safe_load((Path(__file__).resolve().parents[3] / "orion/bus/channels.yaml").read_text())["channels"]
    by_name = {c["name"]: c for c in channels}
    req = by_name[CURIOSITY_REQUEST_CHANNEL]
    assert (req["schema_id"], req["message_kind"], req["consumer_services"]) == (
        "IntrospectRequestV1", "introspect.tool.request.v1", ["orion-hub"],
    )
    assert "orion-hub" in by_name["orion:introspect:result:*"]["producer_services"]
    assert "orion-hub" in by_name["orion:vector:semantic:upsert"]["producer_services"]


def test_curiosity_since_must_fall_inside_the_90_day_window():
    """The run join keeps 90 days; an older since would silently undercount."""
    from orion.schemas.introspect import CURIOSITY_WINDOW_DAYS

    now = datetime.now(timezone.utc)
    assert CURIOSITY_WINDOW_DAYS == 90
    CuriosityArguments(since=now - timedelta(days=89))
    for args in ({"since": now - timedelta(days=91)}, {"query": "bees", "since": now - timedelta(days=200)}):
        with pytest.raises(ValidationError, match="within the last 90 days"):
            CuriosityArguments(**args)
    # Open self-questions are not windowed.
    assert CuriosityArguments(kind="self_question", since=now - timedelta(days=400)).kind == "self_question"

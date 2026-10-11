"""The `orion_day` introspect tool: argument contract, transport, registry, brief line."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone

import pytest
import yaml
from pydantic import ValidationError

from orion.introspect.brief import ORION_DAY, introspect_brief_lines
from orion.introspect.tests.test_introspect_tools import BINDING, DreamBus, ReplyBus, _dream_ok
from orion.introspect.tools import RPC_TIMEOUT_SEC, IntrospectTools, IntrospectUnknownError
from orion.introspect.transport import ORION_DAY_REQUEST_CHANNEL, REQUEST_KIND, RESULT_PREFIX
from orion.schemas.introspect import (
    MAX_ITEMS,
    IntrospectRequestV1,
    IntrospectResultV1,
    OrionDayArguments,
)

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)


def _ok(**kw):
    return IntrospectResultV1(ok=True, operation="orion_day", as_of=NOW, total_available=0, **kw).model_dump(mode="json")


def _invoke(bus, args):
    return asyncio.run(IntrospectTools(bus, BINDING).invoke("orion_day", args))


# --- contract --------------------------------------------------------------------------------


def test_defaults_are_the_most_recent_letters_outline():
    args = OrionDayArguments()
    assert (args.letter_date, args.part, args.index, args.section, args.query, args.limit) == (
        None, "list", None, None, None, 5)


@pytest.mark.parametrize("fields", [
    {"letter_date": "2026-10-09", "part": "note", "index": 3},
    {"part": "carry_forward", "index": 5},
    {"part": "carry_forward"},
    {"part": "section", "section": "dreams", "letter_date": date(2026, 10, 9)},
    {"query": "  pool composition  "},
    {"query": "dream organ", "letter_date": "2026-10-09", "part": "note", "limit": 2},
])
def test_valid_selections(fields):
    args = OrionDayArguments.model_validate(fields)
    if args.query is not None:
        assert args.query == args.query.strip()


@pytest.mark.parametrize("fields", [
    {"index": 3},                                         # index needs note / carry_forward
    {"part": "section"},                                  # section needs its name
    {"part": "section", "section": "weather"},            # not a section
    {"section": "dreams"},                                # section only with part=section
    {"part": "note", "index": 0},                         # 1-based
    {"query": "x", "part": "note", "index": 2},           # query excludes index
    {"query": "x", "part": "section", "section": "dreams"},
    {"query": "   "},                                     # blank is an error, not "recent"
    {"limit": MAX_ITEMS + 1},
    {"letter_date": "2026-10-09T06:00:00Z"},              # a date, not a timestamp
    {"letter_date": "yesterday"},
    {"memory_allowed": True},                             # extra fields rejected
])
def test_invalid_selections_are_rejected(fields):
    with pytest.raises(ValidationError):
        OrionDayArguments.model_validate(fields)


def test_registry_channel_and_operation_know_orion_day():
    from orion.schemas.registry import resolve

    assert resolve("OrionDayArguments") is OrionDayArguments
    assert ORION_DAY_REQUEST_CHANNEL == "orion:introspect:orion_day:request"
    binding = BINDING
    assert IntrospectRequestV1(operation="orion_day", binding=binding, args={}).operation == "orion_day"
    with open("orion/bus/channels.yaml", encoding="utf-8") as fh:
        channels = {c["name"]: c for c in yaml.safe_load(fh)["channels"]}
    req = channels[ORION_DAY_REQUEST_CHANNEL]
    assert req["schema_id"] == "IntrospectRequestV1" and req["consumer_services"] == ["orion-hub"]
    assert "orion-hub" in channels["orion:introspect:result:*"]["producer_services"]


# --- transport -------------------------------------------------------------------------------


def test_orion_day_uses_its_channel_with_binding_and_clean_args():
    bus = DreamBus(_ok())
    out = _invoke(bus, {"letter_date": "2026-10-09", "part": "note", "index": 3})
    assert out["ok"] is True and out["operation"] == "orion_day"
    [(channel, envelope, reply_channel, timeout)] = bus.sent
    assert channel == ORION_DAY_REQUEST_CHANNEL and envelope.kind == REQUEST_KIND
    assert envelope.reply_to == reply_channel == f"{RESULT_PREFIX}{envelope.correlation_id}"
    assert timeout == RPC_TIMEOUT_SEC
    assert envelope.payload["operation"] == "orion_day"
    assert envelope.payload["args"] == {"letter_date": "2026-10-09", "part": "note", "index": 3, "limit": 5}


def test_bad_args_never_reach_the_bus():
    bus = DreamBus(_ok())
    with pytest.raises(ValidationError):
        _invoke(bus, {"index": 2})
    assert bus.sent == []


@pytest.mark.parametrize("bus", [
    DreamBus(raise_exc=TimeoutError()),
    DreamBus(_ok(), wrong_correlation=True),
    DreamBus(IntrospectResultV1(ok=False, operation="orion_day", as_of=NOW,
                                error="unknown letter_date 2030-01-01: no Orion's Day letter for that date (not found)"
                                ).model_dump(mode="json")),
    DreamBus(_dream_ok()),
])
def test_failures_and_missing_letters_are_tool_errors_never_empty(bus):
    with pytest.raises(IntrospectUnknownError, match="orion_day: answer unknown"):
        _invoke(bus, {})


def test_a_missing_letter_error_keeps_its_not_found_words():
    bus = DreamBus(IntrospectResultV1(ok=False, operation="orion_day", as_of=NOW,
                                      error="unknown letter_date 2030-01-01: no Orion's Day letter for that date (not found)"
                                      ).model_dump(mode="json"))
    with pytest.raises(IntrospectUnknownError, match="unknown letter_date 2030-01-01.*not found"):
        _invoke(bus, {"letter_date": "2030-01-01"})


# --- what Orion is told ----------------------------------------------------------------------


def test_description_is_honest_about_text_refs_claim_check_and_errors():
    [spec] = [s for s in IntrospectTools(ReplyBus(), BINDING).tool_specs() if s.name == "orion_day"]
    for phrase in (
        "not established fact", "epistemic_status=unsettled", "'2026-10-09 ¶3'", "'2026-10-09 carry 5'",
        "same numbers Juniper sees", "string evidence only, not a verdict",
        "not in that day's records verbatim", "resolved=false", "items=[]", "not found",
        "never that you wrote nothing",
    ):
        assert phrase in spec.description, phrase


def test_brief_names_the_tool_exactly_and_carries_the_guide_line():
    text = " ".join(introspect_brief_lines(BINDING))
    assert ORION_DAY == "mcp__orion-introspect__orion_day"
    assert f"{ORION_DAY}." in text  # in the list of exact names
    assert (
        "When Juniper refers to an Orion's Day letter or a part of it (a date, ¶N, carry N, a section "
        f"name), call {ORION_DAY} before answering. Quote what you actually wrote. Separate what the "
        "records support from what they do not, using the claim check. If you were wrong, say so "
        "plainly; if you still stand by something the records don't show, say what it rests on."
    ) in text

"""The `record_correction` introspect tool: argument contract, transport, registry, brief line."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orion.introspect.brief import RECORD_CORRECTION, introspect_brief_lines
from orion.introspect.tests.test_introspect_tools import BINDING, DreamBus
from orion.introspect.tools import RPC_TIMEOUT_SEC, IntrospectTools, IntrospectUnknownError
from orion.introspect.transport import RECORD_CORRECTION_REQUEST_CHANNEL, REQUEST_KIND, RESULT_PREFIX
from orion.schemas.introspect import (
    CORRECTION_EVIDENCE_MAX,
    CORRECTION_PRIORS_MAX,
    IntrospectRequestV1,
    IntrospectResultV1,
    RecordCorrectionArguments,
)

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[3]
VALID = {
    "ref": "2026-10-09 ¶3",
    "claim": "the dream organ was silent for 260.6 hours",
    "correction": "It was silent for 26.6 hours; 260.6 was a typo in a log line.",
    "evidence": ["curiosity:e7d03d2ecdbb", "Juniper said so in this conversation"],
    "prior_ids": ["self:dream_organ_goes_quiet_for_weeks"],
}


def _ok():
    return IntrospectResultV1(ok=True, operation="record_correction", as_of=NOW, total_available=0).model_dump(mode="json")


def _invoke(bus, args):
    return asyncio.run(IntrospectTools(bus, BINDING).invoke("record_correction", args))


# --- contract --------------------------------------------------------------------------------


@pytest.mark.parametrize("fields", [
    {},
    {"ref": "2026-10-09 carry 5"},
    {"prior_ids": []},
    {"ref": "  2026-10-09 ¶12  ", "claim": "  padded  "},
])
def test_valid_corrections(fields):
    args = RecordCorrectionArguments.model_validate({**VALID, **fields})
    assert args.ref == args.ref.strip() and args.claim == args.claim.strip()


@pytest.mark.parametrize("fields", [
    {"ref": "Oct 9 ¶3"},                                   # not the printed ref
    {"ref": "2026-10-09 Dreams"},                          # a section is not a numbered part
    {"ref": "2026-10-09 ¶3; DROP TABLE"},
    {"claim": "   "},
    {"correction": ""},
    {"evidence": []},                                      # a correction rests on something
    {"evidence": ["   "]},
    {"evidence": ["x" * 501]},
    {"evidence": ["e"] * (CORRECTION_EVIDENCE_MAX + 1)},
    {"prior_ids": ["a", "a"]},
    {"prior_ids": ['self:x"}) DETACH DELETE p //']},        # not a prior id
    {"prior_ids": [f"p{i}" for i in range(CORRECTION_PRIORS_MAX + 1)]},
    {"claim": "c" * 2001},
    {"confidence": 0.1},                                   # extra fields are rejected, never ignored
])
def test_invalid_corrections_are_rejected(fields):
    with pytest.raises(ValidationError):
        RecordCorrectionArguments.model_validate({**VALID, **fields})


def test_registry_channel_operation_and_skew_writer_know_record_correction():
    from orion.schema_skew_discovery import DECLARED_WRITERS
    from orion.schemas.registry import resolve

    assert resolve("RecordCorrectionArguments") is RecordCorrectionArguments
    assert RECORD_CORRECTION_REQUEST_CHANNEL == "orion:introspect:record_correction:request"
    assert IntrospectRequestV1(operation="record_correction", binding=BINDING, args={}).operation == "record_correction"
    assert "orion.schemas.introspect:RecordCorrectionArguments" in DECLARED_WRITERS
    channels = {c["name"]: c for c in yaml.safe_load((REPO / "orion/bus/channels.yaml").read_text())["channels"]}
    req = channels[RECORD_CORRECTION_REQUEST_CHANNEL]
    assert req["schema_id"] == "IntrospectRequestV1" and req["consumer_services"] == ["orion-hub"]


# --- transport -------------------------------------------------------------------------------


def test_record_correction_uses_its_channel_with_binding_and_clean_args():
    bus = DreamBus(_ok())
    out = _invoke(bus, {**VALID, "ref": " 2026-10-09 ¶3 "})
    assert out["ok"] is True and out["operation"] == "record_correction"
    [(channel, envelope, reply_channel, timeout)] = bus.sent
    assert channel == RECORD_CORRECTION_REQUEST_CHANNEL and timeout == RPC_TIMEOUT_SEC
    assert envelope.kind == REQUEST_KIND and reply_channel == f"{RESULT_PREFIX}{envelope.correlation_id}"
    request = IntrospectRequestV1.model_validate(envelope.payload)
    assert request.binding == BINDING
    assert request.args == {**VALID, "ref": "2026-10-09 ¶3"}


def test_bad_args_never_reach_the_bus():
    bus = DreamBus(_ok())
    with pytest.raises(ValidationError):
        _invoke(bus, {**VALID, "ref": "yesterday"})
    assert bus.sent == []


def test_a_refusal_says_not_recorded_and_no_reply_says_unknown():
    refused = IntrospectResultV1(ok=False, operation="record_correction", as_of=NOW,
                                 error="unknown prior_id self:x: no :Prior with that id in your graph (not found)")
    with pytest.raises(IntrospectUnknownError, match=r"record_correction: not recorded \(unknown prior_id self:x"):
        _invoke(DreamBus(refused.model_dump(mode="json")), VALID)
    with pytest.raises(IntrospectUnknownError, match="answer unknown"):
        _invoke(DreamBus(_ok(), raise_exc=TimeoutError()), VALID)


def test_description_is_honest_that_it_changes_no_belief():
    tools = IntrospectTools(DreamBus(_ok()), BINDING)
    spec = next(s for s in tools.tool_specs() if s.name == "record_correction")
    text = spec.description
    for phrase in ("edits nothing", "not any belief", "re-test", "must name a part that exists",
                   "every one must exist", "Only from a conversation with Juniper", "correction_id"):
        assert phrase in text, phrase


def test_brief_names_the_tool_exactly_and_carries_the_guide_line():
    text = " ".join(introspect_brief_lines(BINDING))
    assert RECORD_CORRECTION == "mcp__orion-introspect__record_correction"
    assert f"{RECORD_CORRECTION}." in text  # last in the list of exact names
    assert f"call {RECORD_CORRECTION} with the part's ref, the claim's own words" in text
    assert "Do not record a correction you do not actually accept" in text

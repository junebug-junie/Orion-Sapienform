"""check_naming: one DispositionNamingV1 per naming run, every refusal reason reachable, ids never
repaired, nothing written -> no_answer."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

import pytest

from orion.personality.naming_check import check_naming, naming_id_for
from orion.schemas.personality import DispositionNamingPacketV1

T = datetime(2026, 10, 20, 18, tzinfo=timezone.utc)
RUN = "abcdef123456"


def _packet() -> DispositionNamingPacketV1:
    return DispositionNamingPacketV1(
        packet_id="0123456789abcdef",
        disposition_id="disp-1",
        label_excerpt="keeps returning to prediction error",
        origin="recurrence",
        evidence_count=10,
        lane_counts={"reverie": 8, "chat_stance": 2},
        first_seen_at=T,
        last_reinforced_at=T,
        evidence=[
            {"evidence_id": "ev-1", "lane": "reverie", "occurred_at": T, "excerpt": "one"},
            {"evidence_id": "ev-2", "lane": "chat_stance", "occurred_at": T, "excerpt": "two"},
        ],
        situation_mix={"arousal": {"strained": 10}},
        situation_coverage=0.8,
        neighbors=[{"disposition_id": "disp-2", "label": "a neighbor", "cosine": 0.81}],
        generated_at=T,
    )


SENTENCE = "It is the part of me that keeps checking where I was wrong."


def _raw(**over):
    raw = {
        "run_id": RUN,
        "packet_id": "0123456789abcdef",
        "disposition_id": "disp-1",
        "verdict": "name",
        "name": "the error-checker",
        "sentence": SENTENCE,
        "merge_with": "",
        "evidence": ["ev-1"],
        "written_at": 1_792_000_000_000,
    }
    raw.update(over)
    return raw


def test_accepted_name() -> None:
    out = check_naming(_raw(), _packet(), RUN)
    assert out.hub_check == "accepted" and out.refused_reasons == []
    assert out.verdict == "name" and out.name == "the error-checker"
    assert out.naming_id == str(uuid5(NAMESPACE_URL, f"{RUN}:0123456789abcdef")) == naming_id_for(RUN, "0123456789abcdef")
    assert out.shown.packet_id == "0123456789abcdef"
    assert out.written_at.tzinfo is not None


def test_accepted_reject_and_merge() -> None:
    reject = check_naming(_raw(verdict="reject", name=""), _packet(), RUN)
    assert (reject.verdict, reject.hub_check) == ("reject", "accepted")
    merge = check_naming(_raw(verdict="merge", name="", sentence="", merge_with="disp-2"), _packet(), RUN)
    assert (merge.verdict, merge.hub_check, merge.merge_with) == ("merge", "accepted", "disp-2")


@pytest.mark.parametrize(
    "over, reason",
    [
        ({"disposition_id": "disp-99"}, "unknown_disposition"),
        ({"packet_id": "ffffffffffffffff"}, "packet_mismatch"),
        ({"verdict": "merge", "merge_with": "disp-1"}, "merge_target_not_shown"),
        ({"verdict": "merge", "merge_with": ""}, "missing_merge_target"),
        ({"evidence": ["ev-1", "ev-77"]}, "evidence_not_shown"),
        ({"evidence": []}, "no_evidence_refs"),
        ({"name": "x" * 41}, "name_too_long"),
        ({"name": "one two three four five six seven"}, "name_too_long"),
        ({"name": "two\nlines"}, "name_too_long"),
        ({"name": "Keeps returning to prediction error"}, "name_equals_label"),
        ({"name": ""}, "missing_name"),
        ({"name": "x"}, "missing_name"),
        ({"sentence": "too short"}, "empty_sentence"),
        ({"verdict": "reject", "sentence": ""}, "empty_sentence"),
        ({"sentence": "s" * 241}, "sentence_too_long"),
    ],
)
def test_each_refusal_reason(over, reason) -> None:
    out = check_naming(_raw(**over), _packet(), RUN)
    assert out.hub_check == "refused"
    assert reason in out.refused_reasons


def test_unknown_id_is_recorded_not_repaired() -> None:
    out = check_naming(_raw(disposition_id="disp-99"), _packet(), RUN)
    assert out.disposition_id == "disp-99"
    # The row stays keyed on the packet Hub fetched for this run.
    assert out.packet_id == "0123456789abcdef"


def test_evidence_as_json_string_is_read() -> None:
    out = check_naming(_raw(evidence='["ev-1", "ev-2"]'), _packet(), RUN)
    assert out.evidence_refs == ["ev-1", "ev-2"] and out.hub_check == "accepted"


@pytest.mark.parametrize("raw", [None, {}, {"verdict": "<name | reject | merge>"}, {"verdict": ""}])
def test_nothing_written_is_no_answer(raw) -> None:
    out = check_naming(raw, _packet(), RUN, now=T)
    assert out.verdict == "no_answer"
    assert out.disposition_id == "disp-1" and out.run_id == RUN
    assert out.hub_check == "accepted" and out.refused_reasons == []
    assert out.written_at == T


def test_overlong_raw_text_never_breaks_construction() -> None:
    out = check_naming(_raw(name="n" * 500, sentence="s" * 5000), _packet(), RUN)
    assert out.hub_check == "refused"
    assert {"name_too_long", "sentence_too_long"} <= set(out.refused_reasons)

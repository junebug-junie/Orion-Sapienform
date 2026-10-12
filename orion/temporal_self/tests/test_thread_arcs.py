"""Open threads carry their concern arc id; every reader cites every thread it renders.

Live 2026-10-12 00:59Z: the frame had 2 open threads and the cue said "2 open threads from
conversation, 1 with its record", because one thread's concern arc (684d45c2010540ed, carried in at
local midnight) had fallen off ``arcs_today``'s 64-arc cap and the thread had no id of its own.
"""
from __future__ import annotations

import copy
import json
import re
from datetime import timedelta

from orion.schemas.temporal_self import OpenThreadV1, TemporalSelfDayV1, TemporalSelfFrameV1, TemporalSelfStanceCueV1
from orion.schemas.thought import HubAssociationBundleV1, StanceHarnessSliceV1, StanceReactRequestV1, ThoughtEventV1
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1
from orion.temporal_self.day_block import render_closed_day_block
from orion.temporal_self.stance_cue import MAX_THREAD_REFS, project_stance_cue
from orion.temporal_self.tests.patch4_fixtures import TALKING_NOW, day_1009, frame
from orion.thought.stance_react import apply_stance_react_pipeline

# The real concern arcs behind the two threads in the 10-11 frame (temporal_self_arc, live).
THREAD_ARCS = {"open-loop-5cb6cd032923": "684d45c2010540ed", "open-loop-496dd47e676b": "fd63c7ab0ad3ecbe"}
TALKING_WITH_THREAD_ARCS = (
    "Afternoon, Sun 17:58 local (day record as of 17:53). On now: talking with Juniper, last turn 42 min ago "
    "[arc:571bce17764b0a9e]. Before that: attention on Juniper is talking (1 message in the last 15 min), set "
    "aside 28 min ago [arc:4f06f3b115f9a3ac]. 2 open threads from conversation [arc:684d45c2010540ed] "
    "[arc:fd63c7ab0ad3ecbe]. Last slept 14 h 52 min ago [arc:460c1b680514c3e8]."
)


def frame_with_thread_arcs() -> dict:
    f = frame("talking")
    for t in f["open_threads"]:
        t["arc_id"] = THREAD_ARCS[t["subject_ref"]]
    return f


# ---------------------------------------------------------------- schema: old rows still load


def test_old_thread_without_arc_id_still_validates():
    old = frame("talking")["open_threads"][0]
    assert "arc_id" not in old
    assert OpenThreadV1.model_validate(old).arc_id is None


def test_old_frame_and_old_closed_day_still_load():
    assert all(t.arc_id is None for t in TemporalSelfFrameV1.model_validate(frame("talking")).open_threads)
    assert all(t.arc_id is None for t in day_1009().frame.open_threads)


def test_new_frame_round_trips_through_json():
    f = TemporalSelfFrameV1.model_validate(frame_with_thread_arcs())
    again = TemporalSelfFrameV1.model_validate_json(f.model_dump_json())
    assert again == f
    assert [t.arc_id for t in again.open_threads] == ["684d45c2010540ed", "fd63c7ab0ad3ecbe"]


def test_new_closed_day_round_trips_through_json():
    raw = json.loads(day_1009().model_dump_json())
    for t in raw["frame"]["open_threads"]:
        t["arc_id"] = "1e63167741582226" if t["subject_ref"] == "open-loop-5cb6cd032923" else "5afab5a718d5b81c"
    day = TemporalSelfDayV1.model_validate(raw)
    assert TemporalSelfDayV1.model_validate_json(day.model_dump_json()) == day


# ---------------------------------------------------------------- stance cue


def test_every_thread_is_cited_in_the_cue():
    cue = project_stance_cue(frame_with_thread_arcs(), TALKING_NOW)
    assert cue.text == TALKING_WITH_THREAD_ARCS
    assert "with its record" not in cue.text
    assert set(THREAD_ARCS.values()) <= set(cue.arc_ids)


def test_old_frame_keeps_the_partial_wording():
    cue = project_stance_cue(frame("talking"), TALKING_NOW)
    assert "2 open threads from conversation, 1 with its record [arc:fd63c7ab0ad3ecbe]." in cue.text


def _many_threads(n: int) -> dict:
    f = frame_with_thread_arcs()
    base = f["open_threads"][0]
    f["open_threads"] = []
    for i in range(n):
        t = copy.deepcopy(base)
        t.update(subject_ref=f"open-loop-{i:012x}", arc_id=f"{i:016x}",
                 last_returned=(TALKING_NOW - timedelta(minutes=i)).isoformat())
        f["open_threads"].append(t)
    # Room for the clause: keep only the threads clause and the head.
    f.update(active_arc=None, previous_arc=None, arcs_today=[a for a in f["arcs_today"] if a["kind"] == "attention"][:1])
    return f


def test_over_the_cap_the_cue_says_how_many_it_lists_and_cites_each():
    cue = project_stance_cue(_many_threads(MAX_THREAD_REFS + 2), TALKING_NOW)
    assert (f"{MAX_THREAD_REFS + 2} open threads from conversation, the {MAX_THREAD_REFS} most recent "
            + " ".join(f"[arc:{i:016x}]" for i in range(MAX_THREAD_REFS)) + ".") in cue.text


# ---------------------------------------------------------------- citation gate shown set


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="c-1", session_id="s-1", user_message="how is your day?",
        association=HubAssociationBundleV1(
            correlation_id="c-1", read_source="felt_state_reader", broadcast_stale=False,
            broadcast=AttentionBroadcastProjectionV1(frame=AttentionFrameV1(), attended_node_ids=["node-a"]),
        ),
        repair_bundle=None, stance_inputs={"user_message": "how is your day?"},
    )


def _thought(**overrides) -> ThoughtEventV1:
    base = dict(
        event_id="t-1", correlation_id="c-1", session_id="s-1", created_at=TALKING_NOW,
        imperative="Answer warmly.", tone="present", strain_refs=[], evidence_refs=["node-a"],
        stance_harness_slice=StanceHarnessSliceV1(task_mode="reflective_dialogue", conversation_frame="reflective",
                                                  answer_strategy="share"),
    )
    base.update(overrides)
    return ThoughtEventV1.model_validate(base)


def test_gate_accepts_a_thread_arc_only_the_thread_carried():
    carried = THREAD_ARCS["open-loop-5cb6cd032923"]  # not in arcs_today: only the thread names it
    assert carried not in {a["arc_id"] for a in frame("talking")["arcs_today"]}
    new_cue = project_stance_cue(frame_with_thread_arcs(), TALKING_NOW)
    out = apply_stance_react_pipeline(
        _thought(evidence_refs=["node-a", f"arc:{carried}"], imperative=f"Mention the open thread [arc:{carried}]."),
        _request(), day_cue=new_cue)
    assert out.day_citations.label == "verified"
    assert f"arc:{carried}" in out.day_citations.cited_valid and out.day_citations.rejected_unknown == []
    assert f"arc:{carried}" in out.evidence_refs

    # The same stance against the old cue: the id was never shown, so it is removed and counted.
    old_cue = project_stance_cue(frame("talking"), TALKING_NOW)
    old = apply_stance_react_pipeline(
        _thought(evidence_refs=["node-a", f"arc:{carried}"]), _request(), day_cue=old_cue)
    assert old.day_citations.rejected_unknown == [f"arc:{carried}"]


# ---------------------------------------------------------------- closed-day block


def test_day_block_cites_every_thread_open_at_close():
    day = day_1009()
    block = render_closed_day_block(day)
    for t in day.frame.open_threads:
        arc = next(a for a in day.arcs if a.kind == "concern" and a.subject_ref == t.subject_ref)
        assert f"[arc:{arc.arc_id}]" in block.text


def test_day_block_names_a_thread_whose_arc_row_is_missing():
    raw = json.loads(day_1009().model_dump_json())
    raw["frame"]["open_threads"].append({
        "arc_id": "00000000000000aa", "subject_ref": "open-loop-aaaaaaaaaaaa", "subject_label": "",
        "first_seen_today": "2026-10-09T23:00:00Z", "last_returned": "2026-10-09T23:00:00Z",
        "returns_today": 0, "carried_from_previous_day": False,
    })
    block = render_closed_day_block(TemporalSelfDayV1.model_validate(raw))
    line = next(x for x in block.text.splitlines() if x.startswith("Conversation threads:"))
    assert line.endswith(
        "open-loop-aaaaaaaaaaaa, open at close, arc 00000000000000aa not in the day record [day:2026-10-09].")
    # Not cited as an arc (no row to resolve), so it is not in the gate's shown set.
    assert "arc:00000000000000aa" not in block.records
    assert not re.search(r"\[arc:00000000000000aa\]", block.text)


def test_unchanged_day_renders_unchanged():
    from orion.temporal_self.tests.test_day_block import BLOCK_1009

    assert render_closed_day_block(day_1009()).text == BLOCK_1009


def test_cue_model_lists_thread_arcs_in_order():
    cue: TemporalSelfStanceCueV1 = project_stance_cue(frame_with_thread_arcs(), TALKING_NOW)
    assert cue.arc_ids == re.findall(r"\[arc:([0-9a-f]{16})\]", cue.text)


def test_many_threads_with_long_labels_shrink_the_refs_not_the_clause():
    """Over the 400-char cap the sleep line goes first, then the thread refs shrink (the count stays
    true); the thread clause itself is the last to go."""
    f = _many_threads(7)
    f.update(active_arc=frame("busy")["active_arc"], previous_arc=frame("busy")["previous_arc"])
    f["arcs_today"] = frame("talking")["arcs_today"]  # has a sleep arc and a conversation arc
    cue = project_stance_cue(f, TALKING_NOW)
    assert len(cue.text) <= 400
    m = re.search(r"7 open threads from conversation, the (?:(\d) )?most recent((?: \[arc:[0-9a-f]{16}\])+)\.", cue.text)
    assert m is not None, cue.text
    shown = re.findall(r"\[arc:([0-9a-f]{16})\]", m.group(2))
    assert shown == [f"{i:016x}" for i in range(len(shown))]  # most recent first, each cited
    assert "Last slept" not in cue.text and cue.clauses_dropped >= 1


def test_two_threads_keep_the_sleep_line():
    assert TALKING_WITH_THREAD_ARCS.endswith("Last slept 14 h 52 min ago [arc:460c1b680514c3e8].")

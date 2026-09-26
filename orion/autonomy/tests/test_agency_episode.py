from copy import deepcopy
from datetime import datetime
import json

import pytest

from orion.autonomy.agency_episode import reconstruct, stamp


def snapshot():
    data = {
        "asks": [{"help_id": "h1", "run_id": "run1", "written_at": "2026-09-01T00:00:00Z"}],
        "graph_briefs": [{"brief_id": "b1", "help_id": "h1", "run_id": "run1", "answers_help_id": "h1", "peer": "cursor_auto", "status": "ok", "consumed": "true", "written_at": "2026-09-01T00:01:00Z"}],
        "sql_briefs": [{"brief_id": "b1", "help_id": "h1", "run_id": "run1", "peer": "cursor_auto", "status": "ok"}],
        "results": [{"result_id": "r1", "dispatch_id": "d1", "frame_id": "f1", "status": "success", "created_at": "2026-09-01T00:01:00Z"}],
        "outcomes": [{"id": 1, "dispatch_id": "d1", "dispatch_frame_id": "f1", "feedback_frame_id": "fb1", "signal_id": "resource_pressure", "predicted_delta": -0.1, "arm": "dispatched", "observed_at": "2026-09-01T00:03:00Z", "frame_dispatch_count": 1}],
        "dispatch_frames": [{"frame_id": "f1", "source_proposal_frame_id": "p1", "created_at": "2026-09-01T00:02:00Z", "candidates": [{"dispatch_id": "d1", "source_proposal_id": "p1a", "dispatched_at": "2026-09-01T00:00:00Z", "expected_effect": {"signal_id": "resource_pressure", "predicted_delta": -0.1}}]}],
        "proposal_frames": [{"frame_id": "p1", "proposal_ids": ["p1a", "p1b"]}],
        "feedback_frames": [{"frame_id": "fb1", "source_execution_dispatch_frame_id": "f1"}],
        "result_sample": [], "outcome_sample": [],
    }
    return {"captured_at": "2026-09-26T00:00:00Z", **{key: {"status": "ok", "rows": rows} for key, rows in data.items()}}


def episode(data, lane):
    return next(e for e in reconstruct(data)["episodes"] if e["lane"] == lane)


def test_returned_consumed_brief_does_not_prove_decision_or_delivery():
    ask = episode(snapshot(), "contractor_ask")
    assert ask["links"]["response"]["status"] == "observed"
    assert ask["links"]["consumption_marker"]["status"] == "observed"
    assert ask["links"]["later_choice"]["status"] == "unverified"
    assert ask["links"]["intervention"]["status"] == "unverified"
    assert ask["verdict"] == "UNVERIFIED"


def test_motor_checks_actual_persistence_not_generated_time():
    data = snapshot()
    data["dispatch_frames"]["rows"][0]["generated_at"] = "2026-08-01T00:00:00Z"
    motor = episode(data, "motor")
    assert "expectation_frame_inserted_after_result" in motor["issues"]
    assert motor["links"]["expectation_recorded"]["status"] == "observed"
    assert motor["links"]["expectation_precommitted"]["status"] == "unverified"
    assert motor["links"]["later_choice"]["status"] == "unverified"


def test_result_without_scoring_remains_visible():
    data = snapshot()
    data["outcomes"]["rows"] = []
    motor = episode(data, "motor")
    assert motor["links"]["intervention"]["status"] == "observed"
    assert motor["links"]["field_scored_outcome"]["status"] == "missing"


def test_successful_visual_rpc_can_report_deferral_not_production():
    data = snapshot()
    data["outcomes"]["rows"] = []
    data["results"]["rows"][0]["visual_outcome"] = "deferred_busy"
    data["dispatch_frames"]["rows"][0]["candidates"][0].update(
        cortex_verb="skills.imagination.render_scene.v1", expected_effect=None)
    motor = episode(data, "motor")
    assert motor["outcome_path"] == "visual"
    assert motor["links"]["visual_outcome"]["status"] == "observed"
    assert motor["results"][0]["visual_outcome"] == "deferred_busy"
    assert motor["verdict"] == "UNVERIFIED"


@pytest.mark.parametrize("status,expected", [("ok", "missing"), ("unavailable", "unverified"), ("truncated", "unverified")])
def test_absent_response_is_not_failed_or_resolved(status, expected):
    data = snapshot()
    data["graph_briefs"] = {"status": status, "rows": []}
    ask = episode(data, "contractor_ask")
    assert ask["links"]["response"]["status"] == expected
    assert ask["verdict"] == "UNVERIFIED"


def test_unrelated_reply_cannot_join_by_time():
    data = snapshot()
    data["graph_briefs"]["rows"][0]["help_id"] = "other"
    assert episode(data, "contractor_ask")["links"]["response"]["status"] == "missing"


def test_conflicting_duplicate_is_not_arbitrarily_selected():
    data = snapshot()
    other = {**data["graph_briefs"]["rows"][0], "status": "failed"}
    data["graph_briefs"]["rows"].append(other)
    report = reconstruct(data)
    assert report["conflicts"] == {"graph_briefs": ["b1"]}
    assert report["sources"]["graph_briefs"]["status"] == "conflict"
    assert report["episodes"][0]["links"]["response"]["status"] == "unverified"


def test_conflicting_candidates_do_not_report_prediction_missing():
    data = snapshot()
    candidates = data["dispatch_frames"]["rows"][0]["candidates"]
    other = deepcopy(candidates[0])
    other["expected_effect"]["predicted_delta"] = 0.5
    candidates.append(other)
    motor = episode(data, "motor")
    assert "conflicting_dispatch_candidate" in motor["issues"]
    assert motor["links"]["expectation_recorded"]["status"] == "unverified"
    assert motor["links"]["selection"]["status"] == "unverified"


def test_replay_is_idempotent_order_independent_and_does_not_mutate():
    data = snapshot()
    original = deepcopy(data)
    first = reconstruct(data)
    assert data == original
    for source in data.values():
        if isinstance(source, dict):
            source["rows"] = list(reversed(source["rows"] * 2))
    assert reconstruct(data) == first


def test_source_disagreements_and_invalid_chronology_are_explicit():
    data = snapshot()
    data["graph_briefs"]["rows"][0].update(written_at="2026-08-01T00:00:00Z", answers_help_id=None)
    data["sql_briefs"]["rows"][0]["status"] = "failed"
    issues = episode(data, "contractor_ask")["issues"]
    assert "brief_precedes_ask:b1" in issues
    assert "answers_edge_missing:b1" in issues
    assert "brief_store_disagreement:b1" in issues
    data["outcomes"]["rows"][0].update(predicted_delta=0.9, observed_at="2026-08-01T00:00:00Z")
    data["feedback_frames"]["rows"][0]["source_execution_dispatch_frame_id"] = "other"
    issues = episode(data, "motor")["issues"]
    assert "outcome_prediction_mismatch:1" in issues
    assert "outcome_precedes_dispatch:1" in issues
    assert "feedback_dispatch_mismatch:1" in issues


def test_prose_and_request_envelopes_are_not_exported():
    data = snapshot()
    for source in data.values():
        if isinstance(source, dict):
            for row in source["rows"]:
                row.update(summary="PRIVATE_SENTINEL", question="PRIVATE_SENTINEL", request_envelope={"text": "PRIVATE_SENTINEL"})
    assert "PRIVATE_SENTINEL" not in json.dumps(reconstruct(data))


@pytest.mark.parametrize("value", [None, "garbage", "2026-09-01T00:00:00", datetime(2026, 9, 1), True, float("inf")])
def test_missing_invalid_and_naive_timestamps_are_unknown(value):
    assert stamp(value) is None


def test_numeric_graph_timestamp_preserved():
    assert stamp(1790386258611) == "2026-09-26T01:30:58.611000+00:00"

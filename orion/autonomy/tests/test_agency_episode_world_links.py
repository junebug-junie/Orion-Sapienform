"""The motor lane's two attend-to-act links (design D4): attention_winner and loop_outcome."""
from __future__ import annotations

from copy import deepcopy

from orion.autonomy.agency_episode import reconstruct


def ok(rows):
    return {"status": "ok", "rows": rows}


BASE = {
    "captured_at": "2026-10-01T13:00:00+00:00",
    "asks": ok([]), "graph_briefs": ok([]), "sql_briefs": ok([]), "feedback_frames": ok([]),
    "results": ok([{"result_id": "result:d", "dispatch_id": "d", "frame_id": "f", "status": "success",
                    "created_at": "2026-10-01T12:00:02+00:00", "visual_outcome": None}]),
    "outcomes": ok([]),
    "dispatch_frames": ok([{"frame_id": "f", "created_at": "2026-10-01T12:00:03+00:00", "source_proposal_frame_id": "p",
                            "candidates": [{"dispatch_id": "d", "source_proposal_id": "x", "source_decision_id": "y",
                                            "dispatched_at": "2026-10-01T12:00:01+00:00", "cortex_verb": "orion.gpu_pool.shed.v1",
                                            "expected_effect": {"signal_id": "cabinet_heat_pressure", "predicted_delta": 0.0}}]}]),
    "proposal_frames": ok([{"frame_id": "p", "proposal_ids": ["x"]}]),
    "result_sample": ok([{"result_id": "result:d", "dispatch_id": "d", "frame_id": "f", "status": "success",
                          "created_at": "2026-10-01T12:00:02+00:00"}]),
    "outcome_sample": ok([]),
    "world_episodes": ok([{"episode_id": "d", "arm": "treated", "template": "shed_background_gpu",
                           "decided_at": "2026-10-01T12:00:00+00:00", "open_loop_id": "open-loop-c",
                           "broadcast_log_id": "b1", "expected_effect_recorded": True, "settlement_state": "expired",
                           "posterior_from_reflex_overlap": False}]),
    "broadcast_rows": ok([{"log_id": "b1", "generated_at": "2026-10-01T11:59:30+00:00", "selected_open_loop_id": "open-loop-c"}]),
    "loop_outcomes": ok([{"outcome_id": "o1", "loop_id": "open-loop-c", "verdict": "acted", "actor": "orion",
                          "created_at": "2026-10-01T12:21:00+00:00", "episode_id": "d"}]),
}


def _motor(bundle):
    return next(e for e in reconstruct(bundle)["episodes"] if e["lane"] == "motor")


def test_closed_world_chain_observes_both_links_and_the_precommit():
    e = _motor(BASE)
    assert e["links"]["attention_winner"]["status"] == "observed"
    assert e["links"]["loop_outcome"]["status"] == "observed"
    assert e["links"]["expectation_precommitted"]["status"] == "observed"
    assert e["verdict"] == "UNVERIFIED" and e["world"]["arm"] == "treated" and not e["issues"]


def test_stale_winner_and_orion_resolved_are_issues():
    b = deepcopy(BASE)
    b["broadcast_rows"]["rows"][0]["generated_at"] = "2026-10-01T11:58:00+00:00"
    b["loop_outcomes"]["rows"].append({"outcome_id": "o2", "loop_id": "open-loop-c", "verdict": "resolved", "actor": "orion"})
    e = _motor(b)
    assert e["links"]["attention_winner"]["status"] == "missing"
    assert {"winner_stale_at_bind", "orion_wrote_terminal_verdict"} <= set(e["issues"])


def test_missing_verdict_is_missing_and_older_captures_keep_their_shape():
    b = deepcopy(BASE)
    b["loop_outcomes"]["rows"] = []
    assert _motor(b)["links"]["loop_outcome"]["status"] == "missing"
    old = {k: v for k, v in BASE.items() if k not in ("world_episodes", "broadcast_rows", "loop_outcomes")}
    e = _motor(old)
    assert "attention_winner" not in e["links"] and e["links"]["expectation_precommitted"]["status"] == "unverified"

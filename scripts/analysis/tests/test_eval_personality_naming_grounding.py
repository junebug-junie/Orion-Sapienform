"""Pure parts of the P0 naming eval: packet building, grounding rank math, answer checks, report."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
_SPEC = importlib.util.spec_from_file_location(
    "eval_personality_naming_grounding",
    ROOT / "scripts/analysis/eval_personality_naming_grounding.py",
)
m = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m)

T0 = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def _ev(eid, lane, sim, *, hours=0, excerpt=None, situation=None):
    meta = {"situation": situation} if situation is not None else {}
    return {"evidence_id": eid, "disposition_id": "disp-a", "lane": lane, "similarity": sim,
            "occurred_at": T0 - timedelta(hours=hours), "excerpt": excerpt or f"text {eid}",
            "meta": meta}


def _node(nid="disp-a", status="active", weight=1.0):
    return {"disposition_id": nid, "origin": "recurrence", "status": status, "label": f"label {nid}",
            "weight": weight, "evidence_count": 7, "lane_counts": {"reverie": 5, "journal": 2},
            "first_seen_at": T0 - timedelta(days=3), "last_reinforced_at": T0}


CENTROIDS = {"disp-a": [1.0, 0.0, 0.0], "disp-b": [0.9, 0.1, 0.0], "disp-c": [0.0, 1.0, 0.0],
             "disp-d": [0.0, 0.0, 1.0], "disp-e": [0.5, 0.5, 0.0]}
LABELS = {k: f"label {k}" for k in CENTROIDS}


def test_select_evidence_one_per_lane_first_then_similarity():
    rows = [_ev("r1", "reverie", 0.99), _ev("r2", "reverie", 0.98), _ev("r3", "reverie", 0.97),
            _ev("j1", "journal", 0.60), _ev("c1", "chat", 0.50), _ev("r4", "reverie", 0.96),
            _ev("r5", "reverie", 0.95), _ev("empty", "dream", 1.0, excerpt="   ")]
    picked = [r["evidence_id"] for r in m.select_evidence(rows)]
    assert picked == ["r1", "j1", "c1", "r2", "r3"]  # empty excerpt skipped, lanes first, cap 5


def test_select_evidence_tie_breaks_on_newer_row():
    rows = [_ev("old", "reverie", 0.9, hours=5), _ev("new", "reverie", 0.9, hours=1)]
    assert [r["evidence_id"] for r in m.select_evidence(rows)] == ["new", "old"]


def test_situation_mix_counts_and_coverage():
    rows = [_ev("a", "reverie", 0.9, situation={"arousal": "strained", "company": True, "day_phase": "dusk"}),
            _ev("b", "reverie", 0.8, situation={"arousal": "strained", "company": False, "day_phase": "dusk"}),
            _ev("c", "reverie", 0.7), _ev("d", "reverie", 0.6)]
    mix, cov = m.situation_mix(rows)
    assert mix == {"arousal": {"strained": 2}, "company": {"false": 1, "true": 1}, "day_phase": {"dusk": 2}}
    assert cov == 0.5
    assert m.situation_mix([]) == ({}, 0.0)


def test_build_packet_validates_contract_and_hides_weight():
    rows = [_ev("r1", "reverie", 0.9, situation={"arousal": "idle"}), _ev("j1", "journal", 0.8)]
    p = m.build_packet(_node(), rows, CENTROIDS, LABELS, T0)
    assert p.disposition_id == "disp-a" and len(p.packet_id) == 16
    assert [e.evidence_id for e in p.evidence] == ["r1", "j1"]
    assert [n.disposition_id for n in p.neighbors] == ["disp-b", "disp-e", "disp-c"]
    assert p.neighbors[0].cosine > 0.99
    assert p.situation_coverage == 0.5
    assert "weight" not in p.model_dump()
    # same inputs -> same id; a different evidence set -> a different id
    assert m.build_packet(_node(), rows, CENTROIDS, LABELS, T0).packet_id == p.packet_id
    assert m.build_packet(_node(), rows[:1], CENTROIDS, LABELS, T0).packet_id != p.packet_id
    assert m.build_packet(_node(), [], CENTROIDS, LABELS, T0) is None


def test_build_packet_clips_long_text_to_contract_limit():
    long = "word " * 100
    node = _node()
    node["label"] = long
    p = m.build_packet(node, [_ev("r1", "reverie", 0.9, excerpt=long)], CENTROIDS, LABELS, T0)
    assert len(p.label_excerpt) <= 160 and len(p.evidence[0].excerpt) <= 160


def test_select_nodes_active_then_top_candidates_then_extras():
    nodes = [_node("a1", "active", 5), _node("c1", "candidate", 4), _node("c2", "candidate", 3),
             _node("c3", "candidate", 2), _node("a2", "active", 1)]
    got = [n["disposition_id"] for n in m.select_nodes(nodes, 2, ["c3", "c1", "missing"])]
    assert got == ["a1", "a2", "c1", "c2", "c3"]


def test_grounding_rank_and_margin():
    g = m.grounding([1.0, 0.05, 0.0], CENTROIDS, "disp-a")
    assert g["rank"] == 1 and g["best_other"] == "disp-b" and g["margin"] > 0
    g = m.grounding([0.0, 1.0, 0.0], CENTROIDS, "disp-a")
    assert g["rank"] == 4 and g["best_other"] == "disp-c"
    assert g["margin"] == round(0.0 - 1.0, 4)
    assert m.grounding([1.0, 0.0, 0.0], CENTROIDS, "disp-zzz")["rank"] is None


def test_pairwise_and_distribution():
    pairs = m.pairwise_cosines({"a": [1, 0], "b": [1, 0], "c": [0, 1]})
    assert len(pairs) == 3 and pairs[0][1:] == ("a", "b") and abs(pairs[0][0] - 1.0) < 1e-9
    d = m.distribution([c for c, _, _ in pairs])
    assert d["n"] == 3 and d["min"] == 0.0 and d["max"] == 1.0 and d["median"] == 0.0
    assert d["p90"] == 0.8
    assert m.distribution([]) == {"n": 0}


def _packet():
    rows = [_ev("r1", "reverie", 0.9), _ev("j1", "journal", 0.8)]
    return m.build_packet(_node(), rows, CENTROIDS, LABELS, T0)


def test_parse_answer_tolerates_fence_and_refuses_prose():
    assert m.parse_answer('```json\n{"verdict": "reject"}\n```') == {"verdict": "reject"}
    assert m.parse_answer("I think this is me") is None
    assert m.parse_answer("") is None


def test_check_answer_reasons():
    p = _packet()
    ok = {"verdict": "name", "name": "the watcher", "sentence": "I keep checking the same numbers.",
          "evidence_refs": ["r1"]}
    assert m.check_answer(ok, p) == []
    assert m.check_answer(None, p) == ["no_answer"]
    assert m.check_answer({"verdict": "maybe"}, p) == ["no_answer"]
    assert "name_too_long" in m.check_answer({**ok, "name": "one two three four five six seven"}, p)
    assert "name_equals_label" in m.check_answer({**ok, "name": "LABEL disp-a"}, p)
    assert "empty_sentence" in m.check_answer({**ok, "sentence": "short"}, p)
    assert "evidence_not_shown" in m.check_answer({**ok, "evidence_refs": ["zzz"]}, p)
    assert "no_evidence_refs" in m.check_answer({**ok, "evidence_refs": []}, p)
    merge = {"verdict": "merge", "merge_with": "disp-d", "evidence_refs": ["r1"]}
    assert m.check_answer(merge, p) == ["merge_target_not_shown"]
    assert m.check_answer({**merge, "merge_with": "disp-b"}, p) == []


def test_prompt_has_question_fixed_options_ids_and_no_weight():
    text = m.render_prompt(_packet())
    assert m.QUESTION in text
    assert text.index("1. name") < text.index("2. reject") < text.index("3. merge")
    assert "[r1]" in text and "[disp-b]" in text
    assert "weight" not in text.lower()


def test_summary_and_report_render():
    p = _packet()
    rows = [
        {"disposition_id": "disp-a", "status": "active", "label_excerpt": p.label_excerpt,
         "lane_counts": p.lane_counts, "verdict": "name", "name": "the watcher", "sentence": "s" * 30,
         "grounding": {"rank": 1, "margin": 0.12, "n_nodes": 5}, "refused_reasons": []},
        {"disposition_id": "disp-c", "status": "candidate", "label_excerpt": "pe | noise",
         "lane_counts": {"reverie": 3}, "verdict": "reject", "sentence": "not me at all, it is a meter",
         "refused_reasons": []},
        {"disposition_id": "disp-d", "status": "candidate", "label_excerpt": "x",
         "lane_counts": {}, "verdict": "no_answer", "refused_reasons": ["no_answer"]},
    ]
    pairs = m.pairwise_cosines(CENTROIDS)
    s = m.summarize(rows, pairs, [0.9, 0.5], ["disp-c"])
    assert s["verdicts"] == {"name": 1, "no_answer": 1, "reject": 1}
    assert s["grounding_rank1_rate"] == 1.0 and s["names"] == ["the watcher"]
    assert s["machine_status"][0]["verdict"] == "reject"
    assert s["refused"] == 1
    md = m.render_report(rows, s, {"model": "chat"})
    assert "| disp-a | active |" in md and "1/5 (+0.120)" in md
    assert "pe \\| noise" in md  # pipes escaped
    assert "disp-c: **reject**" in md

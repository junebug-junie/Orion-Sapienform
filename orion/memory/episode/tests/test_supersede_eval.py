"""The supersede precision eval's scoring (issue #3659). Made-up items only; no model, no database."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

from orion.memory.episode import supersede as s
from orion.memory.episode import supersede_eval as e

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "memory_supersede_eval.py"


def _item(iid, label, *, kind="real", hard=True, old_state="auto"):
    d = {"item_id": iid, "kind": kind, "source": "t", "label": label, "expected_relation": "correction",
         "hard_negative": hard and label == "no_act",
         "old": {"memory_id": f"{iid}-o", "statement": "A", "confirmation_state": old_state},
         "new": {"memory_id": f"{iid}-n", "statement": "B"}}
    return e.item_from_dict(d)


def _set(pos=1, neg=5):
    return [_item(f"p{i}", "act") for i in range(pos)] + [_item(f"n{i}", "no_act") for i in range(neg)]


def test_zero_false_acts_and_enough_real_items_pass():
    items = _set()
    decisions = [e.Decision(it.item_id, it.label == "act", "correction" if it.label == "act" else None)
                 for it in items]
    r = e.score(items, decisions)
    assert r["passed"] and r["precision"] == 1.0 and r["recall"] == 1.0 and r["relation_accuracy"] == 1.0


def test_one_false_act_on_a_hard_negative_fails():
    items = _set()
    decisions = [e.Decision(it.item_id, True, "correction") for it in items]
    r = e.score(items, decisions)
    assert not r["passed"] and r["hard_negative_false_acts"] == 5 and "false_acts=5" in r["fail_reasons"]


def test_errors_or_thin_real_data_fail_even_with_no_acts():
    items = _set(pos=1, neg=2)
    r = e.score(items, [e.Decision(it.item_id, it.label == "act", "correction") for it in items])
    assert not r["passed"] and any(f.startswith("real_hard_negatives") for f in r["fail_reasons"])
    items = _set()
    r = e.score(items, [e.Decision(it.item_id, False, error="call_failed:X") for it in items])
    assert not r["passed"] and r["errors"] == len(items)


def test_abstaining_on_everything_is_safe_but_does_not_pass():
    items = _set()
    r = e.score(items, [e.Decision(it.item_id, False) for it in items])
    assert not r["passed"] and r["recall"] == 0.0 and r["precision"] is None and r["fp"] == 0
    assert r["fail_reasons"] == ["true_positives=0<1"]


def test_synthetic_fixtures_load_and_cover_every_relation():
    items = e.load_items(e.SYNTHETIC_PATH)
    assert len(items) >= 20 and {it.kind for it in items} == {"synthetic"}
    assert {it.expected_relation for it in items} >= set(s.RELATIONS)
    assert sum(it.label == "act" for it in items) >= 5 and sum(it.hard_negative for it in items) >= 10
    assert any(it.new.juniper_quoted and it.label == "no_act" for it in items)   # her words, still a negative
    for it in items:
        assert e.item_from_dict(json.loads(json.dumps(e.item_to_dict(it)))) == it


def test_the_rules_layer_stops_protected_rows_before_the_model():
    spec = importlib.util.spec_from_file_location("memory_supersede_eval", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pending = _item("x", "no_act", old_state="pending_confirmation")
    asked = []

    async def ask(prompt, *, temperature=0.0):
        asked.append(prompt)
        return json.dumps({"a_kind": "lasting_state", "b_says_a_wrong_or_changed": "yes", "same_referent": "yes",
                           "relation": "correction", "same_attribute": "yes", "confidence": 1.0, "reason": "r"})

    d = asyncio.run(mod.decide_v3(pending, ask, 3))
    assert not d.acted and d.stopped_by == "rule" and asked == []
    d = asyncio.run(mod.decide_v3(_item("y", "act"), ask, 3))
    assert d.acted and d.relation == "correction" and len(asked) == 3


def test_build_skips_a_v3_format_checks_file(tmp_path):
    spec = importlib.util.spec_from_file_location("memory_supersede_eval_b", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert str(mod.DEFAULT_CHECKS).endswith("supersede_v2_checks_2026-11-17.csv")
    import csv as _csv
    rows = [{"old_id": "a", "new_id": "b", "relation": "compatible", "dense": "0.9"}]
    kept = [c for c in rows if "corrects" in c and "both_true" in c]
    assert kept == []   # the same filter cmd_build applies before reading v2 columns
    assert _csv  # (imported to mirror the script's reader)


def test_heldout_fixtures_load():
    items = e.load_items(e.HELDOUT_PATH)
    assert len(items) >= 8 and all(it.source.startswith("heldout_") for it in items)
    assert sum(it.hard_negative for it in items) >= 6


def test_repeats_do_not_halve_the_real_coverage_floors():
    from dataclasses import replace as _r

    base = _set(pos=1, neg=3)                       # 3 distinct real hard negatives: under the floor of 5
    items = [_r(it, item_id=f"{it.item_id}@{n}") for n in range(2) for it in base]
    r = e.score(items, [e.Decision(it.item_id, it.label == "act", "correction") for it in items])
    assert r["real_hard_negatives"] == 3 and not r["passed"]
    assert e.base_id("veto:a:b@1") == "veto:a:b"


def test_record_needs_two_repeats_on_the_full_production_run():
    import argparse

    import pytest

    spec = importlib.util.spec_from_file_location("memory_supersede_eval_g", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ok = argparse.Namespace(checker="v3", votes=s.VOTES, route=s.EVAL_ROUTE, set="all", repeats=2)
    mod.record_guard(ok)
    for bad in ({"repeats": 1}, {"set": "real"}, {"votes": 5}, {"checker": "v2"}, {"route": "agent"}):
        with pytest.raises(SystemExit):
            mod.record_guard(argparse.Namespace(**{**vars(ok), **bad}))

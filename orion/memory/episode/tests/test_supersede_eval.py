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
    r = e.score(items, [e.Decision(it.item_id, False) for it in items])
    assert not r["passed"] and any(f.startswith("real_hard_negatives") for f in r["fail_reasons"])
    items = _set()
    r = e.score(items, [e.Decision(it.item_id, False, error="call_failed:X") for it in items])
    assert not r["passed"] and r["errors"] == len(items)


def test_abstaining_on_everything_passes_safety_with_zero_recall():
    items = _set()
    r = e.score(items, [e.Decision(it.item_id, False) for it in items])
    assert r["passed"] and r["recall"] == 0.0 and r["precision"] is None


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
        return json.dumps({"relation": "correction", "same_attribute": "yes", "confidence": 1.0, "reason": "r"})

    d = asyncio.run(mod.decide_v3(pending, ask, 3))
    assert not d.acted and d.stopped_by == "rule" and asked == []
    d = asyncio.run(mod.decide_v3(_item("y", "act"), ask, 3))
    assert d.acted and d.relation == "correction" and len(asked) == 3

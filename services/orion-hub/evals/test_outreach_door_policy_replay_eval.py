"""Offline eval of the door-policy replay over a planted week shaped like the live one."""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

T0 = datetime(2026, 11, 9, 15, 5, tzinfo=timezone.utc)


def _load():
    path = Path(__file__).with_name("run_outreach_door_policy_replay.py")
    spec = importlib.util.spec_from_file_location("run_outreach_door_policy_replay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_11_09_triple_and_every_door():
    mod = _load()
    sends = [
        # The 2026-11-09 triple: 15:05 and 16:03 read the same daydream; 16:53 a new one.
        {"decided_at": T0, "source": "tick", "daydream_id": "daydream:a"},
        {"decided_at": T0 + timedelta(minutes=58), "source": "tick", "daydream_id": "daydream:a"},
        {"decided_at": T0 + timedelta(minutes=108), "source": "tick", "daydream_id": "daydream:b"},
        {"decided_at": T0, "source": "tick", "daydream_id": "", "tension": False},
        {"decided_at": T0, "source": "tick", "daydream_id": "", "tension": True},
        {"decided_at": T0, "source": "curiosity_outreach", "line": "self_inquiry", "run_id": "s1"},
        {"decided_at": T0, "source": "curiosity_outreach", "line": "investigate", "run_id": "i1"},
        {"decided_at": T0, "source": "curiosity_outreach", "line": "investigate", "run_id": "i2"},
        {"decided_at": T0, "source": "self_finding_outreach"},
        {"decided_at": T0, "source": "reading_finding_outreach"},
        {"decided_at": T0, "source": "office_reach_out"},
    ]
    focus = {"i2": {"origin": "juniper", "chat_correlation_id": "chat-1"},
             "i1": {"origin": "self", "chat_correlation_id": "prime:x"}}
    out = mod.decide(sends, focus)
    assert [r["verdict"] for r in out] == [
        "kept", "content_already_used", "kept", "no_material", "kept",
        "kept", "not_for_juniper", "kept", "retired", "kept", "kept",
    ]
    assert out[7]["why"] == "asked_by_juniper"
    assert mod.per_day(out, kept_only=False, days=1)["tick"] == 5
    assert mod.per_day(out, kept_only=True, days=1)["tick"] == 3


def test_a_daydream_is_fresh_again_after_the_ledger_window():
    mod = _load()
    sends = [{"decided_at": T0, "source": "tick", "daydream_id": "daydream:a"},
             {"decided_at": T0 + timedelta(days=8), "source": "tick", "daydream_id": "daydream:a"}]
    assert [r["verdict"] for r in mod.decide(sends, {})] == ["kept", "kept"]

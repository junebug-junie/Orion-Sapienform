"""Offline eval of the outreach choice readout over planted tick rows."""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

T0 = datetime(2026, 11, 14, 15, 0, tzinfo=timezone.utc)


def _load():
    path = Path(__file__).with_name("run_outreach_choice_readout.py")
    spec = importlib.util.spec_from_file_location("run_outreach_choice_readout", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(hours, reason, pick, *, rank=None, subject=None, shown=2, published=True):
    return {"decided_at": T0 + timedelta(hours=hours), "reason": reason,
            "choice": {"pick": pick, "chosen_rank": rank, "chosen_subject": subject, "published": published,
                       "shown": [{"subject": f"s{i}"} for i in range(shown)]}}


def test_readout_counts_picks_declines_ranks_and_repeats():
    mod = _load()
    rows = [
        _row(0, "sent", "chosen", rank=2, subject="reading:a"),
        _row(5, "orion_declined", "declined", shown=3),
        _row(10, "sent", "chosen", rank=1, subject="daydream:x", shown=1),
        _row(20, "no_pick", "missing", published=False),
        _row(30, "sent", "chosen", rank=1, subject="reading:a"),  # the repeat the ledger should prevent
        _row(30 + 24 * 8, "sent", "chosen", rank=1, subject="reading:a"),  # outside 7 days: fine
    ]
    out = mod.summarize(rows, days=7)
    assert out["choices"] == 6 and out["published"] == 5
    assert out["picks"] == {"chosen": 4, "declined": 1, "missing": 1}
    assert out["chosen_rank"] == {"1": 3, "2": 1}
    assert out["departed_from_rank_1"] == 0.25
    assert out["options_shown"] == {"1": 1, "2": 4, "3": 1}
    assert [r["subject"] for r in out["repeated_subjects_7d"]] == ["reading:a"]


def test_readout_accepts_json_text_and_no_rows():
    mod = _load()
    assert mod.summarize([], days=7)["departed_from_rank_1"] is None
    out = mod.summarize([{"decided_at": T0, "reason": "orion_declined",
                          "choice": '{"pick": "declined", "shown": [{"subject": "a"}]}'}], days=1)
    assert out["picks"] == {"declined": 1}

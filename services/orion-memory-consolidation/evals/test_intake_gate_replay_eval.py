"""Gate over the checked-in 30-day replay (see run_intake_gate_replay_eval.py).

Thresholds are the spec's Stage 0 acceptance checks, applied to replayed real
windows instead of waiting 48 h on live traffic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_intake_gate_replay_eval as ev  # noqa: E402


def _report():
    fixture = json.loads(ev.FIXTURE.read_text(encoding="utf-8"))
    return ev.replay(fixture["windows"])


def test_real_memories_are_still_kept():
    assert _report()["named_keepers"] == {
        "austin": "kept",
        "offsite": "kept",
        "labs": "kept",
        "family": "kept",
    }


def test_no_kept_row_is_summarised_by_a_greeting_or_command():
    report = _report()
    assert report["old_rows_with_junk_summary"] > 0  # the defect is in the data
    assert report["kept_with_junk_summary"] == 0


def test_command_and_greeting_windows_are_dropped():
    dropped_prompts = {p for d in _report()["dropped_rows"] for p in d["prompts"]}
    for junk in ("Run github compactor.", "Do a journal pass.", "sup", "ty!",
                 "Compact the last 24 hours of chat into a memory digest."):
        assert junk in dropped_prompts, junk


def test_repair_signal_share_falls_below_twenty_percent():
    report = _report()
    assert report["repair_signal_share_old"] > 0.5
    assert report["repair_signal_share_new"] < 0.20


def test_the_gate_still_keeps_most_windows():
    # Lean toward remembering: this stage removes junk, it does not thin real
    # conversation. A collapse here means the filter is eating content.
    report = _report()
    assert report["kept"] >= 0.8 * report["windows"]


def test_synthetic_keepers_are_all_kept():
    # Labs and family are redacted in the fixture and cannot be judged junk by
    # construction; these synthetic messages are the real over-drop guard.
    synth = _report()["synthetic_keepers"]
    assert len(synth) == len(ev.SYNTHETIC_KEEPERS)
    assert all(v == "kept" for v in synth.values()), synth


def test_review_change_is_the_only_new_keep():
    # The only window the review fixes newly keep is "hi | hey, which queue?":
    # a short question with a real content word is now kept (finding 4).
    report = _report()
    assert report["dropped"] == 6
    kept_prompts = [k["prompts"] for k in report["kept_rows"]]
    assert ["hi", "hey, which queue?"] in kept_prompts

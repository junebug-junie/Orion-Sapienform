"""Offline eval of the PASS / spacing replay on planted rows shaped like the live 2026-10-28 morning."""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

T0 = datetime(2026, 10, 28, 14, 11, tzinfo=timezone.utc)
APPENDIX = "\n\nWhat the piece says -- Some piece (https://example.org): It says things."


def _load():
    path = Path(__file__).with_name("run_outreach_pass_spacing_replay.py")
    spec = importlib.util.spec_from_file_location("run_outreach_pass_spacing_replay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_reasoned_passes_are_caught_and_the_morning_pile_is_spread():
    mod = _load()
    rows = []
    # 14 messages in 57 minutes, 12 of them reasoned declines (live shape), 2 real.
    for i in range(14):
        real = i in (0, 7)
        words = "One link changed how I read this." if real else "Let me assess this.\n\nNot worth it.\n\nPASS"
        rows.append({"decided_at": T0 + timedelta(minutes=4 * i), "source": "reading_finding_outreach",
                     "content": words + APPENDIX})
    rows.append({"decided_at": T0, "source": "endogenous", "content": "I'll pass along the note."})
    out = mod.replay(rows, 5400.0)
    assert out["by_source"]["reading_finding_outreach"] == {"delivered": 14, "now_pass": 12}
    assert out["by_source"]["endogenous"] == {"delivered": 1, "now_pass": 0}
    assert out["reading_real_messages"] == 2
    assert out["reading_max_per_hour_before"] == 2 and out["reading_max_per_hour_after"] == 1
    assert out["reading_max_delay_min"] == 90.0 - 28.0


def test_the_appendix_never_decides_pass():
    mod = _load()
    assert mod.orion_words("Real words." + "\n\nSource: x. PASS") == "Real words."

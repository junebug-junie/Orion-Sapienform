"""The arc kind-mix replay's fixture lane, under pytest (CI). The live lane is
``python services/orion-dream/evals/dream_arc_kind_mix_replay.py --live --days 14`` (read-only, needs the live DB)."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def test_dream_arc_kind_mix_replay_fixture_passes(capsys):
    path = Path(__file__).resolve().parent / "dream_arc_kind_mix_replay.py"
    spec = importlib.util.spec_from_file_location("dream_arc_kind_mix_replay", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main([]) == 0, capsys.readouterr().out
    assert "fixture: PASS" in capsys.readouterr().out

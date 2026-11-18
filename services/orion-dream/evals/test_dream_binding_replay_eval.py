"""The dream binding replay's fixture lane, under pytest (CI). The live lane is
``python services/orion-dream/evals/dream_binding_replay.py --live --days 7`` (read-only, needs the live DB)."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def test_dream_binding_replay_fixture_passes(capsys):
    path = Path(__file__).resolve().parent / "dream_binding_replay.py"
    spec = importlib.util.spec_from_file_location("dream_binding_replay", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main([]) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "then funded by its own hunger" in out and "overdue (48 h): floor:overdue" in out

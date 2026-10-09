"""The deploy-order note on the memory.episode_distill workflow value is consumer-first (review 2026-10-02)."""

from pathlib import Path

SRC = (Path(__file__).resolve().parents[3] / "schemas" / "durable_run.py").read_text()


def test_note_puts_consumers_before_durable_runs():
    note = SRC[: SRC.index('"memory.episode_distill",')].rsplit('"orion_day.letter",', 1)[1]
    assert note.index("orion-sql-writer") < note.index("BEFORE orion-durable-runs")
    assert note.index("orion-llm-gateway") < note.index("BEFORE orion-durable-runs")

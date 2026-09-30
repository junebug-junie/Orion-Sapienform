"""The Orion's Day eval must pass on the fixture day (the live mode runs from the CLI)."""

from __future__ import annotations

import asyncio

from orion.orion_day.evals.run_orion_day_eval import _fixture_material, evaluate


def test_fixture_day_passes_every_eval_check():
    results = evaluate(asyncio.run(_fixture_material()))
    assert [name for name, _, _ in results] == [
        "separation", "grounding", "blind", "full_text", "condensation", "budget", "determinism"]
    failed = [(name, detail) for name, ok, detail in results if not ok]
    assert not failed, failed

"""See services/orion-thought/tests/conftest.py's own `_gpu2_capacity_off_
by_default` for the full story -- this is the same fixture, duplicated
rather than shared, because `evals/` is a sibling of `tests/`, not a child:
pytest conftest discovery walks up from each collected file's own
directory, so `tests/conftest.py` never applies here. Confirmed live in CI
(.github/workflows/visual-baseline.yml's "Image degradation eval" step,
`services/orion-thought/evals/test_visual_chain_honesty_eval.py`): this
eval calls `run_visual_chain_once` with `call_diffusion_generate` mocked
but no capacity mocking, and with `visual_chain_gpu2_capacity_enabled`
defaulting True in production, it hung the whole CI job for 20+ minutes
polling a real, unreachable durable-runs address up to its 180s budget,
repeated across the eval's scenario matrix, before this fixture existed.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _gpu2_capacity_off_by_default(monkeypatch):
    monkeypatch.setenv("ORION_VISUAL_CHAIN_GPU2_CAPACITY_ENABLED", "false")
    try:
        from app import visual_chain

        settings_obj = getattr(visual_chain, "settings", None)
        if settings_obj is not None:
            monkeypatch.setattr(settings_obj, "visual_chain_gpu2_capacity_enabled", False)
    except ImportError:
        pass

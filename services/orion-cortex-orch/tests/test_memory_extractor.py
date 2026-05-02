from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_ORCH = _REPO / "services" / "orion-cortex-orch"
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class _Env:
    kind = "chat.history.turn"


@pytest.mark.asyncio
async def test_extractor_disabled_noop(monkeypatch) -> None:
    import app.memory_extractor as mod

    class S:
        orion_auto_extractor_enabled = False
        orion_auto_extractor_stage2_enabled = False

    monkeypatch.setattr(mod, "get_settings", lambda: S())
    await mod.handle_memory_history_turn(_Env())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_stage2_raises(monkeypatch) -> None:
    import app.memory_extractor as mod

    class S:
        orion_auto_extractor_enabled = True
        orion_auto_extractor_stage2_enabled = True

    monkeypatch.setattr(mod, "get_settings", lambda: S())
    with pytest.raises(NotImplementedError):
        await mod.handle_memory_history_turn(_Env())  # type: ignore[arg-type]

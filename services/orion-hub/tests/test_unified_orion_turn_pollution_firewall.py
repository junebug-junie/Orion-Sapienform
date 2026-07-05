from __future__ import annotations

import inspect
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

_FORBIDDEN_TOKENS = (
    "chat_general",
    "compile_speech_contract",
    "build_chat_request",
    "llm_chat_general",
)

_UNIFIED_PATH_MODULES = (
    "orion/hub/turn_orchestrator.py",
    "orion/hub/turn_request.py",
    "services/orion-hub/scripts/unified_turn_stub.py",
)


def _module_source(relative_path: str) -> str:
    path = _REPO_ROOT / relative_path
    assert path.is_file(), f"missing unified-path module: {relative_path}"
    return path.read_text(encoding="utf-8")


def test_unified_orion_turn_pollution_firewall() -> None:
    """Unified Orion path must not import Brain speech-shim symbols."""
    from orion.hub import turn_orchestrator

    sources = [inspect.getsource(turn_orchestrator)]
    sources.extend(_module_source(rel) for rel in _UNIFIED_PATH_MODULES[1:])

    for relative_path, src in zip(_UNIFIED_PATH_MODULES, sources, strict=True):
        for token in _FORBIDDEN_TOKENS:
            assert token not in src, (
                f"pollution firewall violation in {relative_path}: forbidden token {token!r}"
            )

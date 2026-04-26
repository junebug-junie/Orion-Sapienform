from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
_hub = str(HUB_ROOT.resolve())
_repo = str(REPO_ROOT.resolve())
for entry in (_hub, _repo):
    while entry in sys.path:
        sys.path.remove(entry)
sys.path.insert(0, _hub)
sys.path.insert(1, _repo)

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")


def test_recall_strategy_readiness_get_returns_payload() -> None:
    import importlib

    # Pytest prepends the repo root to sys.path after this module loads; hub `scripts` must win over repo `scripts/`.
    while _hub in sys.path:
        sys.path.remove(_hub)
    sys.path.insert(0, _hub)
    for _mod in list(sys.modules):
        if _mod == "scripts" or _mod.startswith("scripts."):
            del sys.modules[_mod]
    api_routes = importlib.import_module("scripts.api_routes")

    body = api_routes.api_substrate_mutation_runtime_recall_strategy_readiness(telemetry_limit=50)
    assert "data" in body
    readiness = body["data"]["readiness"]
    assert readiness["recommendation"] in {
        "not_ready",
        "review_candidate",
        "ready_for_shadow_expansion",
        "ready_for_operator_promotion",
    }
    assert "gates_blocked" in readiness
    assert "corpus_coverage" in readiness

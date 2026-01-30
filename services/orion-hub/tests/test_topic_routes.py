import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HUB_ROOT))
sys.path.insert(1, str(REPO_ROOT))

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

from scripts import api_routes  # noqa: E402


def _build_client():
    app = FastAPI()
    app.include_router(api_routes.router)
    return TestClient(app)


def test_topics_summary_defaults(monkeypatch):
    async def fake_fetch(path, params):
        assert path == "/api/topics/summary"
        assert params == {"window_minutes": 1440, "max_topics": 20}
        return {"model_version": "v1", "topics": []}

    monkeypatch.setattr(api_routes, "_fetch_landing_pad", fake_fetch)

    client = _build_client()
    response = client.get("/api/topics/summary")

    assert response.status_code == 200
    assert response.json()["model_version"] == "v1"


def test_topics_drift_params(monkeypatch):
    async def fake_fetch(path, params):
        assert path == "/api/topics/drift"
        assert params == {
            "window_minutes": 60,
            "min_turns": 5,
            "max_sessions": 10,
            "model_version": "v5",
        }
        return {"model_version": "v5", "sessions": []}

    monkeypatch.setattr(api_routes, "_fetch_landing_pad", fake_fetch)

    client = _build_client()
    response = client.get(
        "/api/topics/drift",
        params={
            "window_minutes": 60,
            "min_turns": 5,
            "max_sessions": 10,
            "model_version": "v5",
        },
    )

    assert response.status_code == 200
    assert response.json()["model_version"] == "v5"

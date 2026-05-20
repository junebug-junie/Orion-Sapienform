from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "knowledge_forge"

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    dest = tmp_path / "corpus"
    shutil.copytree(FIXTURE_ROOT, dest)
    return dest


@pytest.fixture
def client(corpus_root: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("KNOWLEDGE_FORGE_REPO_ROOT", str(corpus_root))
    monkeypatch.setenv("KNOWLEDGE_FORGE_WRITE_ENABLED", "false")
    monkeypatch.setenv("KNOWLEDGE_FORGE_ENABLED", "true")

    from app.settings import Settings
    from app.service import KnowledgeForgeService
    from app.routers import v1

    cfg = Settings()
    v1.reset_service(KnowledgeForgeService(cfg))

    from app.main import app

    return TestClient(app)

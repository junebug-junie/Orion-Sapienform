from __future__ import annotations

import os
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _collapse_mirror_service_cwd():
    prev = os.getcwd()
    os.chdir(SERVICE_DIR)
    yield
    os.chdir(prev)

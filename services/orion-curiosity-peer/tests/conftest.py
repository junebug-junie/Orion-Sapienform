"""Make the service package importable under the repo-standard test command."""

from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))


import pytest


@pytest.fixture(autouse=True)
def _reset_claude_limit_memory():
    """The limit back-off lives in process memory; keep tests independent."""
    from app.claude_limit_backoff import MEMORY

    MEMORY.reset()
    yield
    MEMORY.reset()

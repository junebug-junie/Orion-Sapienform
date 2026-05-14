"""Pytest path setup so `app` and `orion` import from repo root without manual PYTHONPATH."""

from __future__ import annotations

import sys
from pathlib import Path

_orch_root = Path(__file__).resolve().parents[1]
_repo_root = Path(__file__).resolve().parents[3]
for _p in (_repo_root, _orch_root):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

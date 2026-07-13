"""Ensure repo-root imports resolve for orion-cortex-orch evals."""
from __future__ import annotations

import sys
from pathlib import Path

_ORCH_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _ORCH_ROOT.parents[1]

for p in (str(_REPO_ROOT), str(_ORCH_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

"""Path setup for Hub evals — the import half of `tests/conftest.py`.

Same problem that file solves: `scripts` must resolve to
`services/orion-hub/scripts`, not the repo-root `scripts/` package, and the
repo root must still be importable for `orion.*`. Hub goes on `sys.path`
*before* the repo root, and any already-cached `scripts.*` import is dropped
so the corrected path wins.

Deliberately NOT importing `tests/conftest.py`: that file's autouse fixture
re-runs this before every test to keep unit tests isolated, which evals do
not want -- these read the live database through the normal import path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HUB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _HUB_ROOT.parents[1]


def pytest_configure() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for path in (str(_REPO_ROOT), str(_HUB_ROOT)):
        try:
            sys.path.remove(path)
        except ValueError:
            pass
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_HUB_ROOT))

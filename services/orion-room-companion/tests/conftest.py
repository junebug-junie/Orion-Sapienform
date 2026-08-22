"""Make the service package importable under the repo-standard test command.

Without this, `PYTHONPATH=. pytest services/orion-room-companion/tests -q`
from the repo root fails collection with `ModuleNotFoundError: No module
named 'app'` -- the tests only ran under a bespoke incantation, which is how a
suite quietly stops being run at all. Same fix the sibling service applies
inline in its own test module.
"""

from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

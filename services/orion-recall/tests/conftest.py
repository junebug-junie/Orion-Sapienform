from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECALL_SERVICE_ROOT = ROOT / "services" / "orion-recall"
if str(RECALL_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(RECALL_SERVICE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from __future__ import annotations

import sys
from pathlib import Path

SERVICE_DIR = str(Path(__file__).resolve().parents[1])
REPO_ROOT = str(Path(__file__).resolve().parents[3])

if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

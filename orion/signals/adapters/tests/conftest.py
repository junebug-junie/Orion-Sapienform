import sys
from pathlib import Path

# Repo root: orion/signals/adapters/tests -> parents[4]
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

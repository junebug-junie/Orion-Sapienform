from __future__ import annotations

import sys
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent
repo_root = scripts_dir.parent
if str(scripts_dir) in sys.path:
    sys.path.remove(str(scripts_dir))
sys.path.insert(0, str(repo_root))

from orion.schemas.collapse_mirror import should_route_to_triage


def main() -> None:
    assert should_route_to_triage({"observer": "juniper"}) is True
    assert should_route_to_triage({"observer": "orion"}) is False
    print("ok")


if __name__ == "__main__":
    main()

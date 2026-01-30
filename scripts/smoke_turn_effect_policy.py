from __future__ import annotations

import sys
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent
repo_root = scripts_dir.parent
if str(scripts_dir) in sys.path:
    sys.path.remove(str(scripts_dir))
sys.path.insert(0, str(repo_root))

from orion.schemas.telemetry.turn_effect_policy import (
    recommend_actions_from_alerts,
    summarize_recommended_actions,
)


def main() -> None:
    alerts = [
        {"rule": "coherence_drop", "severity": "error"},
        {"rule": "novelty_spike", "severity": "warn"},
    ]
    policy = recommend_actions_from_alerts(alerts)
    assert "stabilize_mode" in policy["actions"]
    assert "capture_insight_candidate" in policy["actions"]
    print(summarize_recommended_actions(policy))
    print("ok")


if __name__ == "__main__":
    main()

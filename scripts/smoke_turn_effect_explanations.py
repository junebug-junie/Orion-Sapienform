from __future__ import annotations

import sys
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent
repo_root = scripts_dir.parent
if str(scripts_dir) in sys.path:
    sys.path.remove(str(scripts_dir))
sys.path.insert(0, str(repo_root))

from orion.schemas.telemetry.turn_effect_explanations import (
    explain_alerts,
    summarize_explanations,
)


def main() -> None:
    alerts = [
        {"rule": "coherence_drop", "severity": "error"},
        {"rule": "novelty_spike", "severity": "warn"},
    ]
    expl = explain_alerts(alerts)
    assert "context_fragmentation" in expl["likely_causes"]
    assert "Should we capture this as an insight?" in expl["suggested_questions"]
    print(summarize_explanations(expl))
    print("ok")


if __name__ == "__main__":
    main()

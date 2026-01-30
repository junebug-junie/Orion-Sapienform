from __future__ import annotations

import sys
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent
repo_root = scripts_dir.parent
if str(scripts_dir) in sys.path:
    sys.path.remove(str(scripts_dir))
sys.path.insert(0, str(repo_root))

from orion.schemas.telemetry.turn_effect import evaluate_turn_effect_alert, should_emit_turn_effect_alert


def main() -> None:
    effect = {"turn": {"coherence": -0.3, "valence": -0.1, "novelty": 0.1}}
    alert = evaluate_turn_effect_alert(
        effect,
        coherence_drop=0.25,
        valence_drop=0.25,
        novelty_spike=0.35,
    )
    assert alert is not None and alert["metric"] == "coherence_drop"

    now = 100.0
    if should_emit_turn_effect_alert(None, now, 120) is True:
        print("fired")
    if should_emit_turn_effect_alert(now, now + 10, 120) is False:
        print("suppressed")
    assert should_emit_turn_effect_alert(now, now + 120, 120) is True


if __name__ == "__main__":
    main()

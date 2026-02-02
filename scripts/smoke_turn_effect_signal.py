from __future__ import annotations

import sys
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent
repo_root = scripts_dir.parent
if str(scripts_dir) in sys.path:
    sys.path.remove(str(scripts_dir))
sys.path.insert(0, str(repo_root))

from orion.schemas.telemetry.turn_effect import turn_effect_from_spark_meta


def main() -> None:
    spark_meta = {
        "phi_before": {"valence": 0.2, "energy": 0.4, "coherence": 0.6, "novelty": 0.1},
        "phi_after": {"valence": 0.3, "energy": 0.5, "coherence": 0.4, "novelty": 0.2},
        "phi_post_before": {"valence": 0.3, "energy": 0.5, "coherence": 0.4, "novelty": 0.2},
        "phi_post_after": {"valence": 0.1, "energy": 0.7, "coherence": 0.5, "novelty": 0.4},
    }
    effect = turn_effect_from_spark_meta(spark_meta)
    assert effect is not None
    assert abs(effect["user"]["valence"] - 0.1) < 1e-6
    assert abs(effect["assistant"]["energy"] - 0.2) < 1e-6
    assert abs(effect["turn"]["coherence"] - -0.1) < 1e-6
    print("ok")


if __name__ == "__main__":
    main()

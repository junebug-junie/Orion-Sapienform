import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TURN_EFFECT_PATH = REPO_ROOT / "orion" / "schemas" / "telemetry" / "turn_effect.py"


def _load_turn_effect_module():
    spec = importlib.util.spec_from_file_location("turn_effect", TURN_EFFECT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    mod = _load_turn_effect_module()
    spark_meta = {
        "phi_before": {"valence": 0.2, "energy": 0.4, "coherence": 0.6, "novelty": 0.1},
        "phi_after": {"valence": 0.1, "energy": 0.5, "coherence": 0.3, "novelty": 0.2},
        "phi_post_before": {"valence": 0.1, "energy": 0.5, "coherence": 0.3, "novelty": 0.2},
        "phi_post_after": {"valence": 0.0, "energy": 0.6, "coherence": 0.2, "novelty": 0.4},
    }
    effect = mod.turn_effect_from_spark_meta(spark_meta) or {}
    evidence = effect.get("evidence") or {}
    print("evidence", evidence)
    assert "phi_before" in evidence
    assert set(evidence["phi_before"].keys()) == {"valence", "energy", "coherence", "novelty"}


if __name__ == "__main__":
    main()

import sys
import types
import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    module_path = scripts_dir / "print_recent_turn_effects.py"
    package_name = "orion_scripts"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(scripts_dir)]
        sys.modules[package_name] = pkg
    spec = importlib.util.spec_from_file_location(f"{package_name}.print_recent_turn_effects", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_flatten_delta_fields():
    module = _load_module()
    delta = {"coherence": -0.2, "valence": 0.1, "energy": 0.3, "novelty": 0.0}
    flattened = module._flatten_delta(delta, "delta_turn")
    assert flattened["delta_turn_coherence"] == -0.2
    assert flattened["delta_turn_valence"] == 0.1
    assert flattened["delta_turn_energy"] == 0.3
    assert flattened["delta_turn_novelty"] == 0.0

import yaml
from pathlib import Path

def test_yaml_validity():
    base = Path(__file__).resolve().parents[1]
    verbs_dir = base / "verbs"

    for path in verbs_dir.glob("*.yaml"):
        with path.open() as f:
            data = yaml.safe_load(f)

        assert "name" in data
        assert "plan" in data

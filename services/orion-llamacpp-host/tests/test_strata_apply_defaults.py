"""strata_apply_defaults.py: Orion's chat defaults merge into Strata's run config and nothing else moves."""
import json
import subprocess
import sys
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
SCRIPT = SERVICE / "scripts" / "strata_apply_defaults.py"
DEFAULTS = SERVICE / "strata" / "chat-defaults.json"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_merges_defaults_and_keeps_setup_keys(tmp_path):
    cfg = tmp_path / "strata-iq3_s.json"
    cfg.write_text(json.dumps({"args": ["--pack", "/data/p"], "model_name": "m", "sampling": {"temperature": 1.0}}))
    first = run(cfg, DEFAULTS)
    assert first.returncode == 0 and "set reasoning_effort, sampling" in first.stdout
    got = json.loads(cfg.read_text())
    assert got["args"] == ["--pack", "/data/p"] and got["model_name"] == "m"
    assert got["reasoning_effort"] == "none" and got["sampling"]["temperature"] == 0.48
    second = run(cfg, DEFAULTS)
    assert second.returncode == 0 and "already current" in second.stdout


def test_missing_config_is_refused(tmp_path):
    result = run(tmp_path / "absent.json", DEFAULTS)
    assert result.returncode == 1 and "start Strata once" in result.stderr

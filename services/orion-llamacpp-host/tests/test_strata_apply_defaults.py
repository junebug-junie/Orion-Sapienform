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


def test_agent_defaults_keep_the_27b_lanes_thinking_and_sampling(tmp_path):
    cfg = tmp_path / "strata-iq3_s.json"
    cfg.write_text(json.dumps({"args": []}))
    assert run(cfg, SERVICE / "strata" / "agent-defaults.json").returncode == 0
    got = json.loads(cfg.read_text())
    assert got["reasoning_effort"] == "high" and got["sampling"]["temperature"] == 1.0


def test_context_sets_max_context_so_a_seeded_config_needs_no_new_setup(tmp_path):
    """A config copied from the 262144 chat lane must read 131072 for the agent launch (desc_matches)."""
    cfg = tmp_path / "strata-iq3_s.json"
    cfg.write_text(json.dumps({"args": ["--pack", "/data/p", "--max-context", "262144", "--kv", "int8"]}))
    assert run(cfg, SERVICE / "strata" / "agent-defaults.json", "--context", "131072").returncode == 0
    assert json.loads(cfg.read_text())["args"] == ["--pack", "/data/p", "--max-context", "131072", "--kv", "int8"]
    cfg.write_text(json.dumps({"args": ["--pack", "/data/p"]}))
    assert run(cfg, DEFAULTS, "--context", "131072").returncode == 1


def test_each_strata_lane_launches_with_its_profiles_context():
    """The compose STRATA_CONTEXT is what the engine serves; the profile's ctx_size is what Orion budgets."""
    import yaml

    repo = SERVICE.parents[1]
    compose = yaml.safe_load((SERVICE / "docker-compose.atlas-workers.yml").read_text())["services"]
    profiles = yaml.safe_load((repo / "config" / "llm_profiles.yaml").read_text())["profiles"]
    for service, profile in (("atlas-chat", "qwen38-flash-next-iq3s-strata-1xv100-circe-chat"),
                             ("atlas-agent", "qwen38-flash-next-iq3s-strata-1xv100-circe-agent")):
        env = dict(e.split("=", 1) for e in compose[service]["environment"])
        assert int(env["STRATA_CONTEXT"]) == profiles[profile]["strata"]["ctx_size"] \
            == profiles[profile]["gpu"]["max_model_len"], service

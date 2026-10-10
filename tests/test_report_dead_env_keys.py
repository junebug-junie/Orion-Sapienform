"""Tests for scripts/report_dead_env_keys.py (GPU pool stage 6.6)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
import report_dead_env_keys as tool  # noqa: E402

_SETTINGS = '''\
"""Settings. OLD_DOCSTRING_KEY was removed (a docstring mention must not keep it alive)."""
import os
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # COMMENTED_KEY was removed (a comment must not keep it alive)
    hub_timeout_sec: float = 3.0                      # field name, case-insensitive
    aliased: str = Field("x", alias="ALIASED_KEY")
    model_config = {"extra": "ignore"}


class Prefixed(BaseSettings):
    port: int = 1

    class Config:
        env_prefix = "VOIP_"


EXTRA = os.getenv("GETENV_KEY", "")
SCAN = [k for k in os.environ if k.startswith("DYN_SCAN_")]
'''

_ENV = """\
# a comment line
HUB_TIMEOUT_SEC=3
ALIASED_KEY=y
VOIP_PORT=5060
GETENV_KEY=1
DYN_SCAN_ONE=1
IN_EXAMPLE_ONLY=1
COMPOSE_READ_KEY=1
OLD_DOCSTRING_KEY=1
COMMENTED_KEY=1
GONE_PLAIN_KEY=1
GONE_API_TOKEN=secret
ORION_BUS_URL=redis://100.92.216.81:6379/0
GPU_LANE_CONTROLLER_TOKEN=secret
HUB_LLM_GATEWAY_URL=http://x
LLM_LANE_DEFAULT=quick
export EXPORTED_GONE=1
"""


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _tree(tmp_path: Path) -> Path:
    _write(tmp_path, "orion/__init__.py", "")
    _write(tmp_path, "config/x.yaml", "a: 1\n")
    _write(tmp_path, "services/orion-x/app/settings.py", _SETTINGS)
    _write(tmp_path, "services/orion-x/.env_example", "IN_EXAMPLE_ONLY=1\n# GONE_PLAIN_KEY=commented\n")
    _write(tmp_path, "services/orion-x/docker-compose.yml",
           "services:\n  x:\n    environment:\n      - A=${COMPOSE_READ_KEY:-1}\n    # - B=${GONE_PLAIN_KEY}\n")
    _write(tmp_path, "services/orion-x/tests/test_x.py", 'X = "GONE_PLAIN_KEY"\n')  # tests don't count
    _write(tmp_path, "services/orion-x/.env", _ENV)
    return tmp_path


def _report(root: Path) -> tool.ServiceReport:
    (rep,) = [r for r in tool.build_reports(root, root, None) if r.service == "orion-x"]
    return rep


def test_classification(tmp_path: Path) -> None:
    rep = _report(_tree(tmp_path))
    assert sorted(rep.dead) == sorted([
        "OLD_DOCSTRING_KEY", "COMMENTED_KEY", "GONE_PLAIN_KEY", "EXPORTED_GONE",
        "GPU_LANE_CONTROLLER_TOKEN",   # secret-named, but KNOWN_DEAD wins
        "HUB_LLM_GATEWAY_URL",
    ])
    # secret-named / NEVER_SYNC and unread here: listed, never auto-removed
    assert sorted(rep.protected) == ["GONE_API_TOKEN", "ORION_BUS_URL"]
    assert rep.conflicts == []
    for live in ("HUB_TIMEOUT_SEC", "ALIASED_KEY", "VOIP_PORT", "GETENV_KEY", "DYN_SCAN_ONE",
                 "IN_EXAMPLE_ONLY", "COMPOSE_READ_KEY", "LLM_LANE_DEFAULT"):
        assert live not in rep.dead and live not in rep.protected, live


def test_known_dead_but_still_read_is_a_conflict(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _write(root, "services/orion-x/app/more.py", 'import os\nU = os.getenv("HUB_LLM_GATEWAY_URL")\n')
    rep = _report(root)
    assert "HUB_LLM_GATEWAY_URL" in rep.conflicts and "HUB_LLM_GATEWAY_URL" not in rep.dead


def test_report_is_read_only(tmp_path: Path, capsys) -> None:
    root = _tree(tmp_path)
    before = (root / "services/orion-x/.env").read_text()
    assert tool.main(["--root", str(root), "--env-root", str(root)]) == 0
    assert (root / "services/orion-x/.env").read_text() == before
    assert not list((root / "services/orion-x").glob(".env.bak.*"))
    assert "GONE_PLAIN_KEY" in capsys.readouterr().out


def test_apply_backs_up_and_removes_only_dead_lines(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    env = root / "services/orion-x/.env"
    before = env.read_text()
    assert tool.main(["--root", str(root), "--env-root", str(root), "--apply", "--include-heuristic"]) == 0
    (backup,) = list(env.parent.glob(".env.bak.*"))
    assert backup.read_text() == before
    after = env.read_text()
    for gone in ("GONE_PLAIN_KEY=", "COMMENTED_KEY=", "EXPORTED_GONE=", "GPU_LANE_CONTROLLER_TOKEN="):
        assert gone not in after
    for kept in ("# a comment line", "HUB_TIMEOUT_SEC=3", "GONE_API_TOKEN=secret", "ORION_BUS_URL=",
                 "LLM_LANE_DEFAULT=quick", "IN_EXAMPLE_ONLY=1"):
        assert kept in after
    # second run: nothing left to remove, no second backup
    assert tool.main(["--root", str(root), "--env-root", str(root), "--apply", "--include-heuristic"]) == 0
    assert len(list(env.parent.glob(".env.bak.*"))) == 1


def test_apply_defaults_to_known_dead_only(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    env = root / "services/orion-x/.env"
    assert tool.main(["--root", str(root), "--env-root", str(root), "--apply"]) == 0
    after = env.read_text()
    assert "GPU_LANE_CONTROLLER_TOKEN=" not in after and "HUB_LLM_GATEWAY_URL=" not in after
    assert "GONE_PLAIN_KEY=1" in after


def test_orphan_env_is_reported_never_edited(tmp_path: Path, capsys) -> None:
    root = _tree(tmp_path)
    orphan = _write(root, "services/orion-deleted/.env", "ANYTHING=1\n")
    assert tool.main(["--root", str(root), "--env-root", str(root), "--apply"]) == 0
    assert orphan.read_text() == "ANYTHING=1\n"
    assert not list(orphan.parent.glob(".env.bak.*"))
    assert "orphan" in capsys.readouterr().out


def test_json_output(tmp_path: Path, capsys) -> None:
    root = _tree(tmp_path)
    assert tool.main(["--root", str(root), "--env-root", str(root), "--json", "--service", "orion-x"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == "report" and data["dead_total"] == 6


# The keys the GPU pool PR reports listed by hand as still sitting in live .env files (athena and
# circe, 2026-09-29/30), by the service whose .env holds them.
_LISTED_IN_PR_REPORTS = {
    "orion-gpu-lane-controller": ["GPU2_ENABLED", "GPU2_DIFFUSION_URL", "GPU2_AGENT_URL", "GPU2_AUTHORITY",
                                  "GPU2_AUTHORITY_URL", "GPU2_DRAIN_TIMEOUT_SEC", "GPU2_MODEL_READY_TIMEOUT_SEC",
                                  "GPU2_POOL_FENCE_STATE_PATH", "GPU_LANE_CONTROLLER_TOKEN"],
    "orion-world-model": ["WM_GPU2_CAPACITY_ENABLED", "WM_GPU2_CAPACITY_URL", "WM_GPU2_CAPACITY_BACKEND_KEY"],
    "orion-thought": ["ORION_VISUAL_CHAIN_GPU2_CAPACITY_ENABLED", "ORION_VISUAL_ELASTIC_STATUS_ENABLED",
                      "ORION_VISUAL_ELASTIC_CONTROLLER_URL"],
    "orion-gpu-pool": ["GPU_POOL_VISUAL_ACTIVITY_URL", "GPU_POOL_ACTUATE_ROLES"],
    "orion-hub": ["GPU_LANE_MAP_ATHENA_JSON", "GPU_LANE_MAP_CIRCE_JSON", "HUB_CURIOSITY_LEASE_VALIDATION_URL",
                  "HUB_CURIOSITY_ELASTIC_ACTIVATION_ENABLED", "HUB_LLM_GATEWAY_URL",
                  # orion-context-exec retirement, 2026-10-10:
                  "HUB_PROPOSAL_REVIEW_ENABLED", "HUB_PROPOSAL_REVIEW_API_URL", "HUB_PROPOSAL_REVIEW_TIMEOUT_SEC",
                  "HUB_AGENT_CONTEXT_EXEC_ENABLED", "HUB_CONTEXT_EXEC_API_URL", "HUB_CONTEXT_EXEC_TIMEOUT_SEC",
                  "HUB_CONTEXT_EXEC_EVENT_CHANNEL", "CONTEXT_EXEC_INVESTIGATION_V2_ENABLED",
                  "HUB_AGENT_REPL_ENABLED", "HUB_AGENT_CURIOSITY_HINT_ENABLED"],
    "orion-diffusion-host": ["DIFFUSION_POWER_INTENT_GPU_INDEX"],
    "orion-durable-runs": ["DURABLE_RUNS_CAPACITY_ENABLED", "DURABLE_RUNS_LEASE_SECONDS",
                           "DURABLE_RUNS_ELASTIC_ENABLED", "DURABLE_RUNS_ELASTIC_BACKEND"],
    "orion-llm-gateway": ["LLM_GATEWAY_LEASE_VALIDATION_URL", "LLM_GATEWAY_CAPACITY_ENABLED"],
    "orion-cortex-exec": ["CORTEX_EXEC_LLM_GATEWAY_URL", "CHANNEL_CONTEXT_EXEC_INTAKE",
                          "CHANNEL_CONTEXT_EXEC_REPLY_PREFIX", "CONTEXT_EXEC_ENABLED", "CONTEXT_EXEC_TIMEOUT_SEC",
                          "CONTEXT_EXEC_DEPTH2_DEFAULT", "CONTEXT_EXEC_LEGACY_FALLBACK"],
    # orion-context-exec retirement, 2026-10-10 (the service dir itself is gone).
    "orion-self-experiments": ["SELF_EXPERIMENTS_DISPATCH_ENABLED", "SELF_EXPERIMENTS_CONTEXT_EXEC_DISPATCH_TRANSPORT",
                               "SELF_EXPERIMENTS_CONTEXT_EXEC_URL", "SELF_EXPERIMENTS_CONTEXT_EXEC_REQUEST_CHANNEL",
                               "SELF_EXPERIMENTS_CONTEXT_EXEC_TIMEOUT_SECONDS", "SELF_EXPERIMENTS_MAX_DISPATCH_ATTEMPTS"],
}


def test_known_dead_keys_are_really_unread_in_this_tree() -> None:
    """Each hand-listed key matches KNOWN_DEAD, is not in its .env_example, and no code reads it.

    Keeps KNOWN_DEAD honest in CI: if code starts reading one again, the tool would report a
    CONFLICT on a host; this fails first.
    """
    shared = tool.ReadSet()
    for top in ("orion", "config"):
        shared.update(tool.reads_under(_REPO / top, _REPO))
    for service, keys in _LISTED_IN_PR_REPORTS.items():
        reads = tool.ReadSet()
        reads.update(shared)
        reads.update(tool.reads_under(_REPO / "services" / service, _REPO))
        example = {k for _, k in tool.env_keys(_REPO / "services" / service / ".env_example")}
        for key in keys:
            assert tool._match(key, tool.KNOWN_DEAD), f"{key} not covered by KNOWN_DEAD"
            assert key not in example, f"{service}: {key} is back in .env_example"
            assert not reads.reads(key), f"{service}: code reads {key} again"


def test_double_dash_flag_lines_read_keys_outside_sql(tmp_path: Path) -> None:
    rs = tool._text_reads("command: >\n  --max-rows ${ONLY_HERE_KEY:-1}\n")
    assert rs.reads("ONLY_HERE_KEY")
    assert not tool._text_reads("-- ONLY_HERE_KEY=true\n", sql=True).reads("ONLY_HERE_KEY")


def test_image_and_library_keys_are_protected(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    env = root / "services/orion-x/.env"
    env.write_text(env.read_text() + "LLAMA_ARG_N_GPU_LAYERS=99\nHF_HOME=/x\nCUDA_VISIBLE_DEVICES=0\nTZ=UTC\n")
    rep = _report(root)
    for key in ("LLAMA_ARG_N_GPU_LAYERS", "HF_HOME", "CUDA_VISIBLE_DEVICES", "TZ"):
        assert key not in rep.dead and key not in rep.protected


def test_multiline_value_is_never_half_removed(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    env = root / "services/orion-x/.env"
    env.write_text('MULTI_GONE="line one\nGONE_PLAIN_KEY=inside the value\nend"\n' + env.read_text())
    rep = _report(root)
    assert "MULTI_GONE" in rep.protected and "MULTI_GONE" not in rep.dead
    assert tool.main(["--root", str(root), "--env-root", str(root), "--apply", "--include-heuristic"]) == 0
    after = env.read_text()
    assert after.startswith('MULTI_GONE="line one\nGONE_PLAIN_KEY=inside the value\nend"\n')
    assert "GONE_PLAIN_KEY=1" not in after   # the real key line further down is still removed


def test_apply_refuses_when_code_tree_is_on_another_commit(tmp_path: Path, monkeypatch, capsys) -> None:
    code = _tree(tmp_path / "code")
    envs = _tree(tmp_path / "envs")
    monkeypatch.setattr(tool, "_head", lambda p: "aaa" if p == code else "bbb")
    before = (envs / "services/orion-x/.env").read_text()
    assert tool.main(["--root", str(code), "--env-root", str(envs), "--apply"]) == 3
    assert (envs / "services/orion-x/.env").read_text() == before
    assert "refusing --apply" in capsys.readouterr().err
    assert tool.main(["--root", str(code), "--env-root", str(envs), "--apply", "--allow-tree-mismatch"]) == 0

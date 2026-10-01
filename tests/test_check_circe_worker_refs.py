"""Tests for scripts/check_circe_worker_refs.py (GPU pool stage 6.6 port gate, CI layer).

The acceptance check from the stage 6 spec: the gate exits 0 on the real tree and 1 on a planted
`http://100.112.254.99:8011` in a service settings.py.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
import check_circe_worker_refs as gate  # noqa: E402

_POOL_YAML = """\
version: 1
host: {name: circe, address: 100.112.254.99}
roles:
  chat:  {kind: llm, cards: [gpu0], owner: chat, port: 8011}
  agent: {kind: llm, cards: [gpu1], owner: agent, port: 8015}
  world: {kind: service, cards: [gpu2], owner: world, port: 6613, slots: 2, vram_gb: 1}
  diffusion: {kind: service, cards: [gpu2], owner: diffusion, port: 8014, slots: 1, vram_gb: 24,
              launch: {actuator: circe, compose: x.yml, service: d}}
"""


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _tree(tmp_path: Path) -> Path:
    _write(tmp_path, "config/gpu_pool.yaml", _POOL_YAML)
    _write(tmp_path, "services/orion-gpu-lane-controller/.env_example", "GPU_LANE_CONTROLLER_HOST_PORT=8090\n")
    _write(tmp_path, "services/orion-llamacpp-host/.env_example",
           "ATLAS_FAST_HOST_PORT=8013\nLLAMACPP_HOST_PORT=7005\n")
    _write(tmp_path, "services/orion-llm-gateway/app/pool_placement.py", 'URL = "http://100.112.254.99:8011"\n')
    _write(tmp_path, "services/orion-hub/app/settings.py", "TIMEOUT = 3\n")
    return tmp_path


def test_identity_reads_pool_config_and_host_ports(tmp_path: Path) -> None:
    ident = gate.load_identity(_tree(tmp_path))
    assert ident.address == "100.112.254.99" and ident.name == "circe"
    # llm seats + the launch-backed diffusion seat + worker/controller host ports
    assert ident.ports == frozenset({8011, 8015, 8014, 8013, 8090})
    assert 6613 not in ident.ports  # world is not a worker
    assert 7005 not in ident.ports  # athena-side single-model server, not a circe seat


def test_planted_direct_call_in_settings_fails(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _write(root, "services/orion-hub/app/settings.py", 'llm_url: str = "http://100.112.254.99:8011"\n')
    hits = gate.scan_tree(root)
    unallowed, _, _ = gate.classify(hits, {})
    assert [(h.path, h.host, h.port) for h in unallowed] == [
        ("services/orion-hub/app/settings.py", "100.112.254.99", 8011)
    ]


@pytest.mark.parametrize(
    "line",
    [
        'BASE = "http://circe:8015/v1/chat/completions"',
        'BASE = "http://circe.tail1234.ts.net:8090/actuate"',
        'BASE = "http://orion-circe-atlas-llamacpp-chat:8080"',
        'BASE = "http://atlas-metacog:8080"',
        'BASE = "http://orion-atlas-llamacpp-chat:8080"',       # compose default container name
        'BASE = "http://${PROJECT}-atlas-llamacpp-chat:8080"',
        'BASE = "http://orion-circe-bonsai-worker:8080"',
        'BASE = "http://bonsai-worker:8080"',
        'BASE = "http://dsv41-flash:8080"',
        'BASE = "http://diffusion-host:6700"',
        'BASE = "http://192.168.1.22:8011"',                   # circe LAN address
        'BASE = "http://192.168.1.24:8090"',
    ],
)
@pytest.mark.parametrize("where", ["orion/thing/client.py", "deploy/x/compose.yml", "Makefile",
                                   "services/orion-hub/templates/x.html", "mesh-utilities/a.service"])
def test_other_address_shapes_are_caught(tmp_path: Path, line: str, where: str) -> None:
    root = _tree(tmp_path)
    _write(root, where, line + "\n")
    assert gate.scan_tree(root), (line, where)


def test_worker_names_come_from_the_seat_compose_files(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _write(root, "services/orion-llamacpp-bonsai-host/docker-compose.yml",
           "services:\n  newseat-worker:\n    container_name: ${PROJECT:-orion}-newseat-llamacpp\n")
    ident = gate.load_identity(root)
    assert {"newseat-worker", "newseat-llamacpp"} <= set(ident.containers)
    _write(root, "orion/c.py", 'U = "http://orion-newseat-llamacpp:8080"\n')
    assert [(h.host, h.port) for h in gate.scan_tree(root)] == [("newseat-llamacpp", 8080)]


def test_zones_comments_tests_and_non_worker_ports_are_ignored(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _write(root, "services/orion-hub/app/a.py", "# see http://100.112.254.99:8011 (comment)\n")
    _write(root, "services/orion-hub/static/js/a.js", "// http://circe:8011\n")
    _write(root, "services/orion-hub/tests/test_a.py", 'X = "http://100.112.254.99:8011"\n')
    _write(root, "services/orion-hub/static/js/a.test.js", 'const x = "http://circe:8011";\n')
    _write(root, "services/orion-hub/.env_example", "CIRCE_BIOMETRICS_BASE_URL=http://100.112.254.99:8100\n")
    _write(root, "services/orion-hub/app/b.py", 'X = "http://100.112.254.99:80111"\n')  # not 8011
    _write(root, ".orion-smoke-logs/run.txt", "http://100.112.254.99:8011\n")  # hidden dir: tooling state
    _write(root, "services/orion-hub/.env", "LIVE=http://100.112.254.99:8011\n")  # live env: --live-env only
    assert gate.scan_tree(root) == []  # the gateway zone's literal is not scanned either


def test_main_fails_on_stale_allow_entry(tmp_path: Path, monkeypatch, capsys) -> None:
    root = _tree(tmp_path)
    monkeypatch.setattr(gate, "ZONES", {"services/orion-llm-gateway/*": "dispatch"})
    monkeypatch.setattr(gate, "ALLOW", {"services/gone/app.py:circe:8011": "old"})
    assert gate.main(["--root", str(root)]) == 1
    assert "STALE allow" in capsys.readouterr().out


def test_main_fails_on_stale_zone(tmp_path: Path, monkeypatch, capsys) -> None:
    root = _tree(tmp_path)
    monkeypatch.setattr(gate, "ZONES", {"services/orion-renamed/*": "gone"})
    monkeypatch.setattr(gate, "ALLOW", {"services/orion-llm-gateway/app/pool_placement.py:*": "x"})
    assert gate.main(["--root", str(root)]) == 1
    assert "STALE zones" in capsys.readouterr().out


def test_main_passes_with_allowlisted_hit(tmp_path: Path, monkeypatch) -> None:
    root = _tree(tmp_path)
    _write(root, "services/orion-thought/app/settings.py", 'url = "http://100.112.254.99:8014"\n')
    monkeypatch.setattr(gate, "ZONES", {"services/orion-llm-gateway/*": "dispatch"})
    monkeypatch.setattr(gate, "ALLOW", {"services/orion-thought/app/settings.py:100.112.254.99:8014": "leased"})
    assert gate.main(["--root", str(root)]) == 0


def test_live_env_report_never_prints_values(tmp_path: Path, capsys) -> None:
    root = _tree(tmp_path)
    _write(root, "services/orion-thought/.env", "SECRETISH_URL=http://user:pw@100.112.254.99:8014\n")
    hits = gate.scan_live_env(root)
    assert [(h.line, h.port) for h in hits] == [("SECRETISH_URL", 8014)]


def test_real_repo_passes() -> None:
    """Acceptance: exit 0 on main."""
    assert gate.main(["--root", str(_REPO)]) == 0


def test_real_repo_fails_on_planted_settings_url(tmp_path: Path) -> None:
    """Acceptance: exit 1 on a planted http://100.112.254.99:8011 in a service settings.py."""
    root = tmp_path / "repo"
    # Copy only what the gate reads: every scanned file, the identity sources, one file per zone.
    wanted = set(gate.iter_scan_files(_REPO))
    wanted |= {_REPO / "config" / "gpu_pool.yaml"} | {_REPO / rel for rel in gate._HOST_PORT_SOURCES}
    wanted |= {_REPO / rel for rel in gate._WORKER_COMPOSE_FILES}
    for zone in gate.ZONES:
        wanted.add(next(p for p in gate._walk(_REPO)
                        if gate.fnmatch.fnmatchcase(p.relative_to(_REPO).as_posix(), zone)))
    for src in wanted:
        dst = root / src.relative_to(_REPO)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    services = root / "services"
    assert gate.main(["--root", str(root)]) == 0
    settings = services / "orion-hub" / "app" / "settings.py"
    settings.write_text(settings.read_text() + '\nPLANTED = "http://100.112.254.99:8011"\n')
    assert gate.main(["--root", str(root)]) == 1

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import check_service_env_compose_parity as parity  # noqa: E402


def _make_service(tmp_path: Path, *, env_example: str, compose: str) -> Path:
    services_dir = tmp_path / "services"
    svc = services_dir / "orion-fake"
    svc.mkdir(parents=True)
    (svc / ".env_example").write_text(env_example, encoding="utf-8")
    (svc / "docker-compose.yml").write_text(compose, encoding="utf-8")
    return services_dir


def test_read_env_example_keys_ignores_comments_and_blanks():
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".env_example", delete=False) as f:
        f.write("# comment\n\nFOO=bar\nBAZ=\n# another comment\nQUX=1\n")
        path = Path(f.name)
    try:
        assert parity._read_env_example_keys(path) == ["FOO", "BAZ", "QUX"]
    finally:
        path.unlink()


def test_compose_env_keys_matches_list_items():
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        f.write(
            "services:\n"
            "  fake:\n"
            "    environment:\n"
            "      - FOO=${FOO}\n"
            "      - BAZ=${BAZ:-}\n"
        )
        path = Path(f.name)
    try:
        keys, has_env_file = parity._read_compose_env_keys(path)
        assert keys == {"FOO", "BAZ"}
        assert has_env_file is False
    finally:
        path.unlink()


def test_compose_env_file_directive_detected_inline():
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        f.write("services:\n  fake:\n    env_file: .env\n")
        path = Path(f.name)
    try:
        _, has_env_file = parity._read_compose_env_keys(path)
        assert has_env_file is True
    finally:
        path.unlink()


def test_compose_env_file_directive_detected_list_form():
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        f.write("services:\n  fake:\n    env_file:\n      - .env\n      - .env.local\n")
        path = Path(f.name)
    try:
        _, has_env_file = parity._read_compose_env_keys(path)
        assert has_env_file is True
    finally:
        path.unlink()


def test_main_reports_missing_keys_and_fails(tmp_path, monkeypatch, capsys):
    services_dir = _make_service(
        tmp_path,
        env_example="FOO=bar\nBAZ=\nQUX=1\n",
        compose="services:\n  fake:\n    environment:\n      - FOO=${FOO}\n",
    )
    monkeypatch.setattr(parity, "_REPO_ROOT", tmp_path)
    exit_code = parity.main(["orion-fake"])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "BAZ" in out
    assert "QUX" in out
    # FOO is already covered by the compose file -- it must not appear in the
    # reported missing-keys list, only in the summary line.
    missing_section = out.split("environment: list:", 1)[1]
    assert "FOO" not in missing_section


def test_main_passes_when_all_keys_covered(tmp_path, monkeypatch):
    services_dir = _make_service(
        tmp_path,
        env_example="FOO=bar\nBAZ=\n",
        compose="services:\n  fake:\n    environment:\n      - FOO=${FOO}\n      - BAZ=${BAZ:-}\n",
    )
    monkeypatch.setattr(parity, "_REPO_ROOT", tmp_path)
    exit_code = parity.main(["orion-fake"])
    assert exit_code == 0


def test_main_passes_when_env_file_directive_present(tmp_path, monkeypatch):
    services_dir = _make_service(
        tmp_path,
        env_example="FOO=bar\nBAZ=\nQUX=1\n",
        compose="services:\n  fake:\n    env_file: .env\n",
    )
    monkeypatch.setattr(parity, "_REPO_ROOT", tmp_path)
    exit_code = parity.main(["orion-fake"])
    assert exit_code == 0


def test_main_report_only_never_fails(tmp_path, monkeypatch):
    services_dir = _make_service(
        tmp_path,
        env_example="FOO=bar\nBAZ=\n",
        compose="services:\n  fake:\n    environment:\n      - FOO=${FOO}\n",
    )
    monkeypatch.setattr(parity, "_REPO_ROOT", tmp_path)
    exit_code = parity.main(["orion-fake", "--report-only"])
    assert exit_code == 0


def test_main_missing_service_exits_two(tmp_path, monkeypatch):
    (tmp_path / "services").mkdir()
    monkeypatch.setattr(parity, "_REPO_ROOT", tmp_path)
    exit_code = parity.main(["does-not-exist"])
    assert exit_code == 2


def test_main_json_output_shape(tmp_path, monkeypatch, capsys):
    _make_service(
        tmp_path,
        env_example="FOO=bar\nBAZ=\n",
        compose="services:\n  fake:\n    environment:\n      - FOO=${FOO}\n",
    )
    monkeypatch.setattr(parity, "_REPO_ROOT", tmp_path)
    exit_code = parity.main(["orion-fake", "--json"])
    assert exit_code == 1
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["service"] == "orion-fake"
    assert payload["missing_from_compose"] == ["BAZ"]
    assert payload["has_env_file_directive"] is False


def test_real_orion_recall_service_has_no_missing_keys():
    """Regression test: the actual orion-recall docker-compose.yml, checked against the
    actual .env_example. Fails if either file regresses to missing a key again -- this
    is the real gap this whole patch closed (30 keys, including RECALL_GRAPHITI_IN_CHAT
    /RECALL_GRAPHITI_ADAPTER_URL, were missing from environment: before this patch)."""
    exit_code = parity.main(["orion-recall", "--report-only"])
    assert exit_code == 0

import importlib.util
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/init_local_env.py"
SPEC = importlib.util.spec_from_file_location("init_local_env", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
init_local_env = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(init_local_env)


def _example() -> str:
    return """# preserved comment
ORION_ANALYTICS_DBT_PASSWORD=
ORION_ANALYTICS_READER_PASSWORD=
LIGHTDASH_SECRET=
LIGHTDASH_METADATA_PASSWORD=
LIGHTDASH_S3_SECRET_KEY=
UNCHANGED=value
"""


def test_initializer_fills_only_required_blanks_and_is_idempotent(tmp_path: Path) -> None:
    example_file = tmp_path / ".env_example"
    env_file = tmp_path / ".env"
    example_file.write_text(_example())
    env_file.write_text(_example().replace("LIGHTDASH_SECRET=", "LIGHTDASH_SECRET=operator-value"))

    tokens = iter(f"generated-{index}" for index in range(10))
    generated = init_local_env.initialize_env(
        example_file,
        env_file,
        token_factory=lambda: next(tokens),
    )

    assert generated == (
        "ORION_ANALYTICS_DBT_PASSWORD",
        "ORION_ANALYTICS_READER_PASSWORD",
        "LIGHTDASH_METADATA_PASSWORD",
        "LIGHTDASH_S3_SECRET_KEY",
    )
    content = env_file.read_text()
    assert "# preserved comment" in content
    assert "UNCHANGED=value" in content
    assert "LIGHTDASH_SECRET=operator-value" in content
    assert "ORION_ANALYTICS_DBT_PASSWORD=generated-0" in content
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600

    before = env_file.read_bytes()
    assert init_local_env.initialize_env(example_file, env_file) == ()
    assert env_file.read_bytes() == before


def test_initializer_creates_env_from_example_and_never_prints_values(
    tmp_path: Path, capsys
) -> None:
    example_file = tmp_path / ".env_example"
    env_file = tmp_path / ".env"
    example_file.write_text(_example())

    assert init_local_env.main(
        ["--example-file", str(example_file), "--env-file", str(env_file)]
    ) == 0

    output = capsys.readouterr().out
    assert env_file.is_file()
    assert "LIGHTDASH_METADATA_PASSWORD" in output
    for line in env_file.read_text().splitlines():
        if line.split("=", 1)[0] in init_local_env.REQUIRED_SECRET_KEYS:
            assert line.split("=", 1)[1]
            assert line.split("=", 1)[1] not in output


def test_initializer_rejects_duplicate_required_keys(tmp_path: Path) -> None:
    example_file = tmp_path / ".env_example"
    env_file = tmp_path / ".env"
    example_file.write_text(_example())
    env_file.write_text(_example() + "LIGHTDASH_SECRET=second-value\n")

    try:
        init_local_env.initialize_env(example_file, env_file)
    except ValueError as exc:
        assert "duplicate required key" in str(exc)
        assert "LIGHTDASH_SECRET" in str(exc)
    else:
        raise AssertionError("duplicate required secret key was accepted")

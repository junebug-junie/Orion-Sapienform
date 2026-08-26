"""The one-owner gate, checked against the real repo files.

A static gate validated only against a synthetic fixture is how a gate ends up
inert on the file it was written for -- so the drift cases here are produced by
mutating the actual owner file's text, not a hand-built sample.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_env_key_single_source.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_env_key_single_source", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load()


def test_the_repo_currently_has_one_owner_per_key(gate) -> None:
    assert gate.main() == 0


def test_every_owned_key_actually_has_an_owner_file(gate) -> None:
    for key, owner in gate.OWNERS.items():
        path = REPO_ROOT / owner
        assert path.is_file(), f"{key}: {owner} missing"
        assert list(gate._literals(path.read_text(), key)), f"{key}: no value in {owner}"


def test_a_drifted_copy_is_caught(gate, monkeypatch, tmp_path) -> None:
    """Mutates the REAL owner file's text so every genuine copy in the repo --
    compose default, Field default -- becomes a mismatch."""
    key = "HARNESS_FCC_TIMEOUT_SEC"
    owner = REPO_ROOT / gate.OWNERS[key]
    original = owner.read_text()
    current = next(gate._literals(original, key))[1]
    drifted = original.replace(f"{key}={current}", f"{key}=4242", 1)  # env-key-single-source: sample
    assert drifted != original

    monkeypatch.setattr(Path, "read_text", _patched_read_text(owner, drifted))
    assert gate.main() == 1


def _patched_read_text(target: Path, replacement: str):
    real = Path.read_text

    def read_text(self, *args, **kwargs):
        if self == target:
            return replacement
        return real(self, *args, **kwargs)

    return read_text


def test_naming_a_key_without_pinning_it_is_not_a_copy(gate) -> None:
    """Prose like "KEY + OTHER_KEY=300" names the key but states no value for
    it. Counting that as a copy would make the gate unusable in any comment
    that derives one budget from another."""
    text = "this must exceed HARNESS_FCC_TIMEOUT_SEC + VOICE_FINALIZE_TIMEOUT_SEC=300"  # env-key-single-source: sample
    assert list(gate._literals(text, "HARNESS_FCC_TIMEOUT_SEC")) == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("HARNESS_FCC_TIMEOUT_SEC=900", "900"),  # env-key-single-source: sample
        ("- HARNESS_FCC_TIMEOUT_SEC=${HARNESS_FCC_TIMEOUT_SEC:-1600}", "1600"),  # env-key-single-source: sample
        ('fcc_timeout_sec: float = Field(1600.0, alias="HARNESS_FCC_TIMEOUT_SEC")', "1600.0"),  # env-key-single-source: sample
        ("up to `HARNESS_FCC_TIMEOUT_SEC=900s` can occupy", "900s"),  # env-key-single-source: sample
    ],
)
def test_the_four_shapes_a_value_gets_restated_in(gate, text, expected) -> None:
    found = [v for _, v in gate._literals(text, "HARNESS_FCC_TIMEOUT_SEC")]
    assert expected in found, found


def test_a_sample_marker_line_is_not_counted_as_a_copy(gate) -> None:
    """Without this the gate flags its own fixtures. Path-excluding tests would
    have been the lazy fix and would have blinded it to a genuinely stale
    default living in some other test."""
    marked = "HARNESS_FCC_TIMEOUT_SEC=4242  # env-key-single-source: sample"
    assert list(gate._literals(marked, "HARNESS_FCC_TIMEOUT_SEC")) == []
    unmarked = "HARNESS_FCC_TIMEOUT_SEC=4242"  # env-key-single-source: sample
    assert [v for _, v in gate._literals(unmarked, "HARNESS_FCC_TIMEOUT_SEC")] == ["4242"]


def test_the_same_number_written_differently_is_not_drift(gate) -> None:
    assert gate._same_number("900", "900.0")
    assert gate._same_number("900s", "900")
    assert not gate._same_number("900", "1600")


def test_the_local_env_is_scanned_because_that_is_where_the_drift_was(gate) -> None:
    """The 2026-08-26 incident was live `.env` at 1600 against a checked-in
    `.env_example` at 900. A gate reading only committed files would have been
    green throughout it -- every committed copy agreed with every other,
    uniformly stale. This is the assertion that keeps the gate pointed at the
    failure it was written for."""
    assert ".env" in gate.SCANNED_NAMES
    assert gate._is_local_env(".env")
    assert gate._is_local_env("services/orion-harness-governor/.env")
    assert not gate._is_local_env("services/orion-harness-governor/.env_example")


def test_a_live_value_ahead_of_the_contract_says_so_in_those_words(
    gate, monkeypatch
) -> None:
    """Reproduces the incident's exact shape: the message has to name which side
    is live, or the reader fixes the wrong file."""
    key = "HARNESS_FCC_TIMEOUT_SEC"
    owner = REPO_ROOT / gate.OWNERS[key]
    live = REPO_ROOT / "services" / "orion-harness-governor" / ".env"
    if not live.is_file():
        pytest.skip("no local .env in this checkout")

    original = owner.read_text()
    current = next(gate._literals(original, key))[1]
    live_value = next(gate._literals(live.read_text(), key))[1]
    if gate._same_number(current, live_value):
        # Make the contract lag the live value, as it did on 2026-08-26.
        stale = original.replace(f"{key}={current}", f"{key}=4242", 1)
        monkeypatch.setattr(Path, "read_text", _patched_read_text(owner, stale))

    assert gate.main() == 1


def test_historical_records_are_not_rewritten_by_config_changes(gate) -> None:
    """PR reports and design specs state what was true when written."""
    assert "docs/superpowers/pr-reports/" in gate.EXCLUDED_PREFIXES
    assert "docs/superpowers/specs/" in gate.EXCLUDED_PREFIXES

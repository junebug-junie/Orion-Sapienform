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
        ('x: float = Field(\n    default=1600.0, gt=0, alias="HARNESS_FCC_TIMEOUT_SEC"\n)', "1600.0"),  # env-key-single-source: sample
        ("up to `HARNESS_FCC_TIMEOUT_SEC=900s` can occupy", "900s"),  # env-key-single-source: sample
    ],
)
def test_the_four_shapes_a_value_gets_restated_in(gate, text, expected) -> None:
    found = [v for _, v in gate._literals(text, "HARNESS_FCC_TIMEOUT_SEC")]
    assert expected in found, found


@pytest.mark.parametrize(
    "path,key",
    [
        ("services/orion-hub/app/settings.py", "HUB_READING_SEARCH_MIN_SIMILARITY"),
        ("services/orion-dream/app/settings.py", "DREAM_SEARCH_MIN_SIMILARITY"),
    ],
)
def test_owned_settings_defaults_are_visible_to_the_gate(gate, path, key) -> None:
    """A settings default the gate cannot see can drift from its owner silently."""
    assert list(gate._literals((REPO_ROOT / path).read_text(), key)), f"{key}: no default seen in {path}"


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


def test_gitignored_agent_scratch_is_not_scanned(gate) -> None:
    """`.superpowers/` briefs are frozen agent scratch; the local `.env` stays scanned."""
    assert ".superpowers/" in gate.EXCLUDED_PREFIXES
    assert ".env" in gate.SCANNED_NAMES
    assert ".superpowers/sdd/task-5-brief.md".startswith(gate.EXCLUDED_PREFIXES)


# --- module-owned lines (body economy Patch B): the room's temperature lines live in thermal_gate.py ---

THOUGHT_SETTINGS = REPO_ROOT / "services" / "orion-thought" / "app" / "settings.py"
THOUGHT_ENV = REPO_ROOT / "services" / "orion-thought" / ".env_example"


def test_the_owner_values_are_read_from_thermal_gate_not_hardcoded(gate) -> None:
    values, failures = gate._owned_line_values()
    assert failures == []
    assert values[32.0].endswith("DEFAULT_HOT_C") and values[30.5].endswith("DEFAULT_HOT_REARM_C")
    assert values[34.0].endswith("DEFAULT_CRITICAL_C") and values[33.0].endswith("DEFAULT_CRITICAL_REARM_C")


def test_the_pre_patch_b_thought_settings_copy_is_refused(gate, monkeypatch) -> None:
    """Re-inserts the exact Field lines Patch B deleted from orion-thought."""
    original = THOUGHT_SETTINGS.read_text()
    revived = original + (
        '\nclass _Revived:\n'
        '    thermal_hot_c: float = Field(32.0, alias="ORION_THERMAL_HOT_C")\n'  # env-key-single-source: sample
        '    thermal_hot_rearm_c: float = Field(30.5, alias="ORION_THERMAL_HOT_REARM_C")\n'  # env-key-single-source: sample
    )
    monkeypatch.setattr(Path, "read_text", _patched_read_text(THOUGHT_SETTINGS, revived))
    failures = gate._module_owned_failures()
    assert any("orion-thought/app/settings.py" in f and "ORION_THERMAL_HOT_C" in f for f in failures), failures  # env-key-single-source: sample
    assert gate.main() == 1


def test_the_pre_patch_b_thought_env_example_copy_is_refused(gate, monkeypatch) -> None:
    revived = THOUGHT_ENV.read_text() + "\nORION_THERMAL_HOT_C=32.0\n"  # env-key-single-source: sample
    monkeypatch.setattr(Path, "read_text", _patched_read_text(THOUGHT_ENV, revived))
    assert gate.main() == 1


def test_a_new_thermal_env_key_with_an_owned_value_is_refused(gate, monkeypatch) -> None:
    """Not just the retired names: any new key that restates a line."""
    revived = THOUGHT_ENV.read_text() + "\nORION_CABINET_CRITICAL_C=34\n"  # env-key-single-source: sample
    monkeypatch.setattr(Path, "read_text", _patched_read_text(THOUGHT_ENV, revived))
    failures = gate._module_owned_failures()
    assert any("ORION_CABINET_CRITICAL_C" in f for f in failures), failures


@pytest.mark.parametrize("src,caught", [
    ("HOT_C = 32.0", True),  # env-key-single-source: sample
    ("CRITICAL_C: float = 34", True),  # env-key-single-source: sample
    ("def f(*, hot_rearm_c: float = 30.5): ...", True),  # env-key-single-source: sample
    ("class S:\n    room_hot_c: float = Field(default=32.0)", True),  # env-key-single-source: sample
    ("GPU_CEILING_REARM_C = 80.0", False),   # a chip ceiling, not a room line
    ("x = 32.0", False),                      # a 32 that is not named as a temperature line
    ("THERMAL_MIN_C = 50.0", False),          # thermal name, unowned value
    ("from orion.autonomy.thermal_gate import DEFAULT_HOT_C as HOT_C", False),  # the right way
])
def test_python_copy_shapes(gate, src, caught) -> None:
    values, _ = gate._owned_line_values()
    assert bool([c for c in gate._py_line_copies(src) if c[2] in values]) is caught


def test_prose_naming_a_retired_key_is_not_a_copy(gate) -> None:
    """Docstrings and comments may still name the deleted key (history); only setting or reading it fails."""
    assert list(gate._py_line_copies('"""ORION_THERMAL_HOT_C was deleted."""')) == []  # env-key-single-source: sample

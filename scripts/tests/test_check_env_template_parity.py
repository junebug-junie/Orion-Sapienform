"""The failure this gate exists for is a key that is PRESENT but short.

2026-09-05: `orion-sql-writer`'s `.env` carried SQL_WRITER_SUBSCRIBE_CHANNELS,
so every key-presence check passed -- but the JSON list inside it was four
channels short. `self_concept_history` and `self_knowledge_items` sat at 0 rows
through two merged PRs and two deploys that both reported success. No tool in
the repo could see it: CI cannot read a gitignored file, and
sync_local_env_from_example.py adds whole keys without looking inside a value.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MOD = Path(__file__).resolve().parents[1] / "check_env_template_parity.py"
_spec = importlib.util.spec_from_file_location("check_env_template_parity", _MOD)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _svc(tmp_path: Path, example: str, local: str) -> Path:
    d = tmp_path / "svc"
    d.mkdir(exist_ok=True)
    (d / ".env_example").write_text(example, encoding="utf-8")
    (d / ".env").write_text(local, encoding="utf-8")
    return d


def test_short_json_list_blocks(tmp_path):
    """The motivating case, reproduced from the real values."""
    ex = 'SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:a","orion:b","orion:self_concept:history:write"]\n'
    lo = 'SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:a","orion:b"]\n'
    blocking, warnings = mod.check_service(_svc(tmp_path, ex, lo))
    assert warnings == [], "the key is present, so it is not a missing-key warning"
    assert len(blocking) == 1
    assert "orion:self_concept:history:write" in blocking[0]


def test_short_json_object_blocks(tmp_path):
    ex = 'ROUTE_MAP={"a.v1":"A","b.v1":"B"}\n'
    lo = 'ROUTE_MAP={"a.v1":"A"}\n'
    blocking, _ = mod.check_service(_svc(tmp_path, ex, lo))
    assert len(blocking) == 1 and "b.v1" in blocking[0]


def test_missing_key_only_warns(tmp_path):
    """29 services were in this state. Blocking here would make the gate noise,
    and sync_local_env_from_example.py already fixes it."""
    blocking, warnings = mod.check_service(
        _svc(tmp_path, "A=1\nHEARTBEAT_INTERVAL_SEC=30\n", "A=1\n")
    )
    assert blocking == []
    assert len(warnings) == 1 and "HEARTBEAT_INTERVAL_SEC" in warnings[0]


def test_differing_scalar_is_allowed(tmp_path):
    """Local overrides are legitimate: secrets, host URLs, tuned thresholds.
    A gate that fires on these trains everyone to reach for the escape hatch."""
    blocking, warnings = mod.check_service(
        _svc(tmp_path, "TOKEN=placeholder\nGATE=0.2\n", "TOKEN=real-secret\nGATE=0.05\n")
    )
    assert blocking == [] and warnings == []


def test_extra_local_entries_are_allowed(tmp_path):
    """A local list may be a SUPERSET -- an operator adding a channel early is
    not a contract violation. Only entries the contract has and the live file
    lacks are a defect."""
    ex = 'C=["a","b"]\n'
    lo = 'C=["a","b","c"]\n'
    blocking, warnings = mod.check_service(_svc(tmp_path, ex, lo))
    assert blocking == [] and warnings == []


def test_single_quoted_json_is_parsed(tmp_path):
    """The templates document wrapping these in single quotes for shell safety,
    so the quoting must not make the check silently vacuous."""
    ex = """C='["a","b"]'\n"""
    lo = """C='["a"]'\n"""
    blocking, _ = mod.check_service(_svc(tmp_path, ex, lo))
    assert len(blocking) == 1 and '"b"' in blocking[0] or "b" in blocking[0]


def test_unparseable_value_does_not_block(tmp_path):
    """Fail open on garbage rather than blocking every deploy on a quoting
    quirk. A false block on a value we cannot parse would get the whole gate
    bypassed."""
    ex = "C=[not json\n"
    lo = "C=[also not json\n"
    blocking, warnings = mod.check_service(_svc(tmp_path, ex, lo))
    assert blocking == [] and warnings == []


def test_type_change_does_not_block(tmp_path):
    """list -> object is a redefinition, not a missing entry. Out of scope."""
    blocking, _ = mod.check_service(_svc(tmp_path, 'C=["a"]\n', 'C={"a":1}\n'))
    assert blocking == []


def test_absent_env_is_skipped_not_failed(tmp_path):
    """A service with no local .env is not deployable here; do not invent a
    failure for it."""
    d = tmp_path / "svc"
    d.mkdir()
    (d / ".env_example").write_text("A=1\n", encoding="utf-8")
    assert mod.check_service(d) == ([], [])


@pytest.mark.parametrize("raw,expected", [
    ('["a"]', ["a"]),
    ("'[\"a\"]'", ["a"]),
    ('{"a":1}', {"a": 1}),
    ("plain-value", None),
    ("", None),
    ("[broken", None),
    ('"a string"', None),
    ("42", None),
])
def test_structured_parser(raw, expected):
    assert mod._structured(raw) == expected

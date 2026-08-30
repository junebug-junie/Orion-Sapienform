"""Tests for the substrate control-surface parity gate.

The gate's first version reported "2 service(s) checked, all configured" while a
real unconfigured consumer existed and while the key it credited orion-hub for was
provably empty. These pin the three holes that caused that.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "scripts" / "check_control_surface_store_parity.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("control_surface_gate", GATE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


class TestNonemptyAssignment:
    """`_resolve_postgres_url()` strips values and falls through on empty, so a
    key that is present but empty is 100% fail-open and must not count."""

    @pytest.mark.parametrize(
        "line",
        [
            "SUBSTRATE_CONTROL_PLANE_POSTGRES_URL=",
            "      - SUBSTRATE_CONTROL_PLANE_POSTGRES_URL=${SUBSTRATE_CONTROL_PLANE_POSTGRES_URL}",
            "      SUBSTRATE_CONTROL_PLANE_POSTGRES_URL: ${SUBSTRATE_CONTROL_PLANE_POSTGRES_URL:-}",
            "SUBSTRATE_CONTROL_PLANE_POSTGRES_URL:    ",
        ],
    )
    def test_empty_or_bare_passthrough_is_not_configured(self, line: str) -> None:
        assert gate._nonempty_assignment(line + "\n", "SUBSTRATE_CONTROL_PLANE_POSTGRES_URL") is False

    @pytest.mark.parametrize(
        "line",
        [
            "SUBSTRATE_CONTROL_PLANE_POSTGRES_URL=postgresql://u:p@host:5432/db",
            "      - SUBSTRATE_CONTROL_PLANE_POSTGRES_URL=${SUBSTRATE_CONTROL_PLANE_POSTGRES_URL:-postgresql://u:p@host:5432/db}",
            "      SUBSTRATE_CONTROL_PLANE_POSTGRES_URL: ${VAR:-postgresql://u:p@host:5432/db}",
        ],
    )
    def test_a_real_value_is_configured(self, line: str) -> None:
        assert gate._nonempty_assignment(line + "\n", "SUBSTRATE_CONTROL_PLANE_POSTGRES_URL") is True

    def test_an_empty_key_does_not_borrow_the_next_lines_value(self) -> None:
        """The original regex used `\\s*` after the separator, which matches
        newlines, so `KEY=` followed by `OTHER=` captured the next line as this
        key's value. That is exactly how orion-hub -- whose key IS empty -- was
        reported as configured."""
        text = "SUBSTRATE_CONTROL_PLANE_POSTGRES_URL=\nSUBSTRATE_POLICY_POSTGRES_URL=\n"
        assert gate._nonempty_assignment(text, "SUBSTRATE_CONTROL_PLANE_POSTGRES_URL") is False

    def test_a_commented_out_key_is_not_configured(self) -> None:
        text = "# SUBSTRATE_CONTROL_PLANE_POSTGRES_URL=postgresql://u:p@host:5432/db\n"
        assert gate._nonempty_assignment(text, "SUBSTRATE_CONTROL_PLANE_POSTGRES_URL") is False


class TestImportDiscovery:
    def test_both_import_spellings_are_seen(self, tmp_path: Path) -> None:
        """`from orion.substrate import mutation_control_surface` does not contain
        the dotted module name as a substring, and real code in this repo uses it."""
        dotted = tmp_path / "dotted.py"
        dotted.write_text("from orion.substrate.mutation_control_surface import get_chat_reflective_lane_threshold\n")
        submodule = tmp_path / "submodule.py"
        submodule.write_text("from orion.substrate import mutation_control_surface\n")
        plain = tmp_path / "plain.py"
        plain.write_text("import orion.substrate.mutation_control_surface\n")

        for path in (dotted, submodule, plain):
            assert gate.CONTROL_SURFACE_MODULE in gate._imported_orion_modules(path), path.name

    def test_unrelated_imports_are_not_seen(self, tmp_path: Path) -> None:
        path = tmp_path / "unrelated.py"
        path.write_text("import os\nfrom orion.bus import channels\n")
        assert gate.CONTROL_SURFACE_MODULE not in gate._imported_orion_modules(path)

    def test_transitive_reach_is_followed(self) -> None:
        """orion-field-digester reaches the control surface only through
        worker -> causal_geometry_producer -> mutation_trials. A direct-import
        check reports it clean; it was in the fail-open state the gate exists to
        catch."""
        worker = REPO_ROOT / "services" / "orion-field-digester" / "app" / "worker.py"
        if not worker.exists():
            pytest.skip("orion-field-digester/app/worker.py not present")
        assert gate._reaches_control_surface(worker, {}, set()) is True


def test_the_repo_currently_passes_its_own_gate() -> None:
    assert gate.main() == 0

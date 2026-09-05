"""Tests for self_study.py's Layer 1 broadening (self-model rebuild arc,
2026-09-05): _hardware_items/_hardware_concepts (field-topology facts) and
_behavioral_items/_behavioral_concepts (chat_stance_belief_log facts)."""
import importlib.util
import sys
import types
from pathlib import Path
from uuid import uuid4

SERVICE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = SERVICE_DIR / "app"
PACKAGE_NAME = "orion_cortex_exec"
APP_PACKAGE_NAME = f"{PACKAGE_NAME}.app"
if PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(SERVICE_DIR)]
    sys.modules[PACKAGE_NAME] = pkg
if APP_PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(APP_PACKAGE_NAME)
    pkg.__path__ = [str(APP_DIR)]
    sys.modules[APP_PACKAGE_NAME] = pkg

_self_study_key = f"{APP_PACKAGE_NAME}.self_study"
if _self_study_key in sys.modules:
    self_study = sys.modules[_self_study_key]
else:
    spec = importlib.util.spec_from_file_location(_self_study_key, APP_DIR / "self_study.py")
    self_study = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = self_study
    spec.loader.exec_module(self_study)

REPO_ROOT = SERVICE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FIELD_TOPOLOGY_FIXTURE = """
schema_version: field_lattice.v1
nodes:
  - node_id: athena
  - node_id: circe
capabilities:
  - capability_id: llm_inference
edges:
  - source_id: athena
    target_id: circe
    edge_type: node_capability
    weight: 1.0
"""


class TestHardwareItems:
    def test_parses_nodes_capabilities_and_edges(self, tmp_path, monkeypatch):
        topo_dir = tmp_path / "config" / "field"
        topo_dir.mkdir(parents=True)
        (topo_dir / "orion_field_topology.v1.yaml").write_text(FIELD_TOPOLOGY_FIXTURE, encoding="utf-8")
        monkeypatch.setattr(self_study, "REPO_ROOT", tmp_path)

        items = self_study._hardware_items(run_id="run-1", observed_at="2026-09-05T00:00:00Z")

        names = {item.name for item in items}
        assert names == {"node:athena", "node:circe", "capability:llm_inference", "edge:athena->circe"}
        assert all(item.category == "hardware" for item in items)

    def test_missing_topology_file_returns_empty_not_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(self_study, "REPO_ROOT", tmp_path)

        items = self_study._hardware_items(run_id="run-1", observed_at="2026-09-05T00:00:00Z")

        assert items == []

    def test_malformed_yaml_returns_empty_not_error(self, tmp_path, monkeypatch):
        topo_dir = tmp_path / "config" / "field"
        topo_dir.mkdir(parents=True)
        (topo_dir / "orion_field_topology.v1.yaml").write_text("not: [valid, yaml,", encoding="utf-8")
        monkeypatch.setattr(self_study, "REPO_ROOT", tmp_path)

        items = self_study._hardware_items(run_id="run-1", observed_at="2026-09-05T00:00:00Z")

        assert items == []


class _FakeRow(dict):
    """Mimics a SQLAlchemy RowMapping (dict-like, supports both [] and .get())."""


class _FakeResult:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return [_FakeRow(r) for r in self._rows]


class _FakeConn:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, _stmt, _params=None):
        return _FakeResult(self._rows)


class _FakeEngine:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def connect(self):
        return _FakeConn(self._rows)


class TestBehavioralItems:
    def test_reads_recent_rows_as_real_facts(self, monkeypatch):
        fake_rows = [
            {
                "entry_id": str(uuid4()),
                "created_at": "2026-09-05T01:00:00+00:00",
                "shift_kind": "REPAIR",
                "anchor_summary": "anchors this turn: orion, relationship(degraded)",
                "lineage_summary": '{"orion": "producer_x"}',
            }
        ]
        import app.self_study_analysis as self_study_analysis_module

        monkeypatch.setattr(self_study_analysis_module, "_get_engine", lambda: _FakeEngine(fake_rows))

        items = self_study._behavioral_items(run_id="run-1", observed_at="2026-09-05T00:00:00Z")

        assert len(items) == 1
        assert items[0].category == "behavioral"
        assert "REPAIR" in items[0].name
        assert items[0].metadata["anchor_summary"] == fake_rows[0]["anchor_summary"]

    def test_no_engine_returns_empty_not_error(self, monkeypatch):
        import app.self_study_analysis as self_study_analysis_module

        monkeypatch.setattr(self_study_analysis_module, "_get_engine", lambda: None)

        items = self_study._behavioral_items(run_id="run-1", observed_at="2026-09-05T00:00:00Z")

        assert items == []

    def test_query_failure_returns_empty_not_error(self, monkeypatch):
        import app.self_study_analysis as self_study_analysis_module

        class _BrokenEngine:
            def connect(self):
                raise RuntimeError("relation \"chat_stance_belief_log\" does not exist")

        monkeypatch.setattr(self_study_analysis_module, "_get_engine", lambda: _BrokenEngine())

        items = self_study._behavioral_items(run_id="run-1", observed_at="2026-09-05T00:00:00Z")

        assert items == []


class TestHardwareAndBehavioralConcepts:
    def _snapshot_with(self, *, hardware=None, behavioral=None):
        from orion.schemas.self_study import SelfKnowledgeSectionCountsV1, SelfSnapshotV1

        hardware = hardware or []
        behavioral = behavioral or []
        sections = {
            "services": [], "modules": [], "channels": [], "verbs": [], "schemas": [],
            "touchpoints": [], "env_surfaces": [], "hardware": hardware, "behavioral": behavioral,
        }
        return SelfSnapshotV1(
            snapshot_id="self-snapshot-test",
            run_id="run-1",
            observed_at="2026-09-05T00:00:00Z",
            repo_root="/tmp",
            counts=SelfKnowledgeSectionCountsV1(**{k: len(v) for k, v in sections.items()}),
            **sections,
        )

    def test_hardware_concepts_empty_when_no_hardware_items(self):
        snapshot = self._snapshot_with()
        assert self_study._hardware_concepts(snapshot) == []

    def test_hardware_concepts_one_concept_from_items(self):
        item = self_study._item(
            run_id="run-1", observed_at="2026-09-05T00:00:00Z", category="hardware",
            name="node:athena", source_path="config/field/orion_field_topology.v1.yaml",
        )
        snapshot = self._snapshot_with(hardware=[item])

        concepts = self_study._hardware_concepts(snapshot)

        assert len(concepts) == 1
        assert concepts[0].concept_kind == "physical_topology"
        assert "node:athena" in concepts[0].description

    def test_behavioral_concepts_empty_when_no_behavioral_items(self):
        snapshot = self._snapshot_with()
        assert self_study._behavioral_concepts(snapshot) == []

    def test_behavioral_concepts_summarizes_shift_kinds(self):
        item = self_study._item(
            run_id="run-1", observed_at="2026-09-05T00:00:00Z", category="behavioral",
            name="turn:2026-09-05T01:00:00Z:REPAIR", source_path="chat_stance_belief_log",
            metadata={"shift_kind": "REPAIR"},
        )
        snapshot = self._snapshot_with(behavioral=[item])

        concepts = self_study._behavioral_concepts(snapshot)

        assert len(concepts) == 1
        assert concepts[0].concept_kind == "behavioral_pattern"
        assert "REPAIR" in concepts[0].description

    def test_induce_self_concepts_includes_new_kinds_when_present(self):
        hw_item = self_study._item(
            run_id="run-1", observed_at="2026-09-05T00:00:00Z", category="hardware",
            name="node:athena", source_path="config/field/orion_field_topology.v1.yaml",
        )
        beh_item = self_study._item(
            run_id="run-1", observed_at="2026-09-05T00:00:00Z", category="behavioral",
            name="turn:1", source_path="chat_stance_belief_log", metadata={"shift_kind": "TOPIC"},
        )
        snapshot = self._snapshot_with(hardware=[hw_item], behavioral=[beh_item])

        concepts = self_study.induce_self_concepts(snapshot)

        kinds = {c.concept_kind for c in concepts}
        assert "physical_topology" in kinds
        assert "behavioral_pattern" in kinds

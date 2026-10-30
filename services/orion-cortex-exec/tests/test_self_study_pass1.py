import asyncio
import importlib.util
import sys
import types
from pathlib import Path

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

spec = importlib.util.spec_from_file_location(f"{APP_PACKAGE_NAME}.self_study", APP_DIR / "self_study.py")
self_study = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = self_study
spec.loader.exec_module(self_study)

spec_va = importlib.util.spec_from_file_location(f"{APP_PACKAGE_NAME}.verb_adapters", APP_DIR / "verb_adapters.py")
verb_adapters = importlib.util.module_from_spec(spec_va)
assert spec_va and spec_va.loader
sys.modules[spec_va.name] = verb_adapters
spec_va.loader.exec_module(verb_adapters)

REPO_ROOT = SERVICE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402


def test_resolve_repo_root_prefers_explicit_env_override(monkeypatch, tmp_path):
    forced_root = tmp_path / "forced-root"
    forced_root.mkdir()
    monkeypatch.setenv("ORION_REPO_ROOT", str(forced_root))

    resolved = self_study._resolve_repo_root(tmp_path / "nested" / "app" / "self_study.py")

    assert resolved == forced_root.resolve()


def test_resolve_repo_root_detects_normal_repo_layout_from_markers(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_REPO_ROOT", raising=False)
    repo_root = tmp_path / "Orion-Sapienform"
    service_dir = repo_root / "services" / "orion-cortex-exec" / "app"
    service_dir.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text('[tool.poetry]\nname = "orion"\n', encoding="utf-8")
    (repo_root / "orion").mkdir()
    (repo_root / "config").mkdir()

    resolved = self_study._resolve_repo_root(service_dir / "self_study.py")

    assert resolved == repo_root.resolve()


def test_resolve_repo_root_handles_flattened_container_layout_without_crashing(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_REPO_ROOT", raising=False)
    module_path = tmp_path / "app" / "app" / "self_study.py"

    resolved = self_study._resolve_repo_root(module_path)

    assert resolved == (tmp_path / "app").resolve()


def test_resolve_repo_root_fallback_is_deterministic_when_markers_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_REPO_ROOT", raising=False)
    module_path = tmp_path / "sandbox" / "nested" / "self_study.py"

    first = self_study._resolve_repo_root(module_path)
    second = self_study._resolve_repo_root(module_path)

    assert first == second == (tmp_path / "sandbox").resolve()


class _FakeBus:
    def __init__(self, *, fail_channel: str | None = None) -> None:
        self.fail_channel = fail_channel
        self.published: list[tuple[str, BaseEnvelope]] = []

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        if self.fail_channel == channel:
            raise RuntimeError(f"publish_failed:{channel}")
        self.published.append((channel, envelope))


def test_build_self_snapshot_returns_typed_authoritative_shape():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    assert snapshot.trust_tier == "authoritative"
    assert snapshot.snapshot_id.startswith("self-snapshot-")
    assert snapshot.run_id.startswith("self-run-")
    assert snapshot.counts.services > 0
    assert snapshot.counts.channels > 0
    assert snapshot.counts.verbs > 0
    assert all(item.trust_tier == "authoritative" for item in snapshot.services)
    assert all(item.item_id.startswith("self-item-") for item in snapshot.services)
    assert all(item.run_id == snapshot.run_id for item in snapshot.services)
    assert any(item.name == "orion:self_study:items:write" for item in snapshot.channels)
    assert any(item.name == "skills.self_study.analyze.v1" for item in snapshot.verbs)


def test_repeated_snapshot_keeps_stable_snapshot_and_item_ids():
    first = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    second = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    assert first.snapshot_id == second.snapshot_id
    assert first.run_id != second.run_id
    assert [item.item_id for item in first.services] == [item.item_id for item in second.services]


def test_journal_entry_is_concise_and_marked_non_authoritative():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    journal = self_study.build_self_study_journal_entry(snapshot, correlation_id="corr-1")

    assert journal.source_kind == "self_study"
    assert journal.mode == "manual"
    assert journal.title == "Self-study factual snapshot"
    assert "Journal is summary-only and not authoritative storage." in journal.body


def test_publish_self_study_artifacts_gracefully_skips_without_bus():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    journal_status, journal_entry = asyncio.run(
        self_study.publish_self_study_artifacts(
            bus=None,
            source=ServiceRef(name="orion-cortex-exec"),
            snapshot=snapshot,
            correlation_id="corr-2",
        )
    )

    assert journal_status.status == "skipped"
    assert journal_status.authoritative is False
    assert journal_status.append_only is True
    assert journal_entry.source_kind == "self_study"


def test_run_self_repo_inspect_writes_journal_then_items_and_no_graph():
    """The Layer 1 scan that feeds Self Atlas: one journal summary, then one
    items-write per fact. No RDF graph write and no graph status (retired 2026-10-30)."""
    bus = _FakeBus()
    result = asyncio.run(
        self_study.run_self_repo_inspect(
            bus=bus,
            source=ServiceRef(name="orion-cortex-exec"),
            correlation_id="corr-3",
        )
    )

    data = result.model_dump(mode="json")
    assert "graph_write" not in data
    assert data["snapshot"]["trust_tier"] == "authoritative"
    assert data["journal_write"]["status"] == "written"
    assert data["self_knowledge_items_write"]["status"] == "written"
    channels = [channel for channel, _ in bus.published]
    assert channels[0] == "orion:journal:write"
    assert bus.published[0][1].kind == "journal.entry.write.v1"
    assert set(channels[1:]) == {"orion:self_study:items:write"}
    assert all(env.kind == "self_study.items.write.v1" for _, env in bus.published[1:])
    assert len(channels) > 1


def test_self_repo_inspect_reports_partial_backend_failure():
    bus = _FakeBus(fail_channel="orion:journal:write")
    result = asyncio.run(
        self_study.run_self_repo_inspect(
            bus=bus,
            source=ServiceRef(name="orion-cortex-exec"),
            correlation_id="corr-4",
        )
    )

    assert result.journal_write.status == "failed"
    assert result.journal_write.authoritative is False
    assert "publish_failed:orion:journal:write" in (result.journal_write.detail or "")


def test_repeat_publish_keeps_journal_append_intent():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    statuses = []
    for corr in ("corr-5", "corr-6"):
        bus = _FakeBus()
        journal_status, _ = asyncio.run(
            self_study.publish_self_study_artifacts(
                bus=bus,
                source=ServiceRef(name="orion-cortex-exec"),
                snapshot=snapshot,
                correlation_id=corr,
            )
        )
        statuses.append(journal_status)
        assert [channel for channel, _ in bus.published] == ["orion:journal:write"]
    assert all(s.status == "written" and s.append_only for s in statuses)


def test_retired_self_study_entry_points_are_gone():
    """Kill means kill (2026-10-30 fold into curiosity): no reflect, induce,
    retrieve or RDF entry point survives in the module, and no self-study verb
    except the analysis skill is registered."""
    for name in (
        "run_self_concept_reflect",
        "run_self_concept_induce",
        "run_self_retrieve",
        "reflect_self_concepts",
        "induce_self_concepts",
        "retrieve_self_study",
        "build_self_study_rdf_request",
    ):
        assert not hasattr(self_study, name), name
    for cls in ("SelfRepoInspectVerb", "SelfConceptInduceVerb", "SelfConceptReflectVerb", "SelfRetrieveVerb"):
        assert not hasattr(verb_adapters, cls), cls
    assert hasattr(verb_adapters, "SelfStudyAnalyzeVerb")

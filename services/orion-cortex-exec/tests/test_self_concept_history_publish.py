"""Tests for self_study.py's Layer-3-to-self_concept_history producer
(self-model rebuild arc, Patch 3, 2026-09-05):
publish_self_concept_history_from_reflection() and _next_self_concept_version().
"""
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

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.self_study import (  # noqa: E402
    SelfConceptEvidenceRefV1,
    SelfConceptRefV1,
    SelfReflectiveFindingV1,
)


class _FakeBus:
    def __init__(self, *, fail: bool = False) -> None:
        self.published: list[tuple[str, BaseEnvelope]] = []
        self._fail = fail

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        if self._fail:
            raise RuntimeError("simulated publish failure")
        self.published.append((channel, envelope))


def _finding(*, concept_kinds=("physical_topology",), reflection_kind="architecture_observation") -> SelfReflectiveFindingV1:
    concept_refs = [
        SelfConceptRefV1(concept_id=f"self-concept-{kind}", concept_kind=kind, label=kind, source_snapshot_id="snap-1")
        for kind in concept_kinds
    ]
    return SelfReflectiveFindingV1(
        reflection_id="self-reflection-abc",
        reflection_kind=reflection_kind,
        title="A real finding",
        description="Real evidence-grounded description.",
        confidence=0.8,
        salience=0.7,
        source_snapshot_id="snap-1",
        evidence=[SelfConceptEvidenceRefV1(snapshot_id="snap-1", item_id="self-item-1", source_path="p")],
        concept_refs=concept_refs,
    )


def test_next_self_concept_version_falls_back_to_1_without_engine(monkeypatch):
    import app.self_study_analysis as self_study_analysis_module

    monkeypatch.setattr(self_study_analysis_module, "_get_engine", lambda: None)

    assert self_study._next_self_concept_version("physical_topology") == 1


def test_publish_skips_without_bus():
    status = asyncio.run(
        self_study.publish_self_concept_history_from_reflection(
            bus=None, source=ServiceRef(name="orion-cortex-exec"), findings=[_finding()], correlation_id="corr-1"
        )
    )
    assert status.status == "skipped"
    assert status.detail == "missing_bus"
    assert status.target == "self_concept_history"


def test_publish_skips_when_no_findings(monkeypatch):
    import app.self_study_analysis as self_study_analysis_module

    monkeypatch.setattr(self_study_analysis_module, "_get_engine", lambda: None)
    bus = _FakeBus()

    status = asyncio.run(
        self_study.publish_self_concept_history_from_reflection(
            bus=bus, source=ServiceRef(name="orion-cortex-exec"), findings=[], correlation_id="corr-1"
        )
    )
    assert status.status == "skipped"
    assert bus.published == []


def test_publish_one_row_per_concept_kind(monkeypatch):
    import app.self_study_analysis as self_study_analysis_module

    monkeypatch.setattr(self_study_analysis_module, "_get_engine", lambda: None)
    bus = _FakeBus()
    finding = _finding(concept_kinds=("physical_topology", "behavioral_pattern"))

    status = asyncio.run(
        self_study.publish_self_concept_history_from_reflection(
            bus=bus, source=ServiceRef(name="orion-cortex-exec"), findings=[finding], correlation_id="corr-1"
        )
    )

    assert status.status == "written"
    assert len(bus.published) == 2
    concept_ids = {env.payload["concept_id"] for _, env in bus.published}
    assert concept_ids == {"physical_topology", "behavioral_pattern"}
    for _, env in bus.published:
        assert env.payload["produced_by"] == "layer3_reflect"
        assert "A real finding" in env.payload["content"]
        assert env.payload["evidence_refs"] == ["self-item-1"]


def test_publish_falls_back_to_reflection_kind_when_no_concept_refs(monkeypatch):
    import app.self_study_analysis as self_study_analysis_module

    monkeypatch.setattr(self_study_analysis_module, "_get_engine", lambda: None)
    bus = _FakeBus()
    finding = _finding(concept_kinds=(), reflection_kind="blind_spot")

    asyncio.run(
        self_study.publish_self_concept_history_from_reflection(
            bus=bus, source=ServiceRef(name="orion-cortex-exec"), findings=[finding], correlation_id="corr-1"
        )
    )

    assert len(bus.published) == 1
    assert bus.published[0][1].payload["concept_id"] == "blind_spot"


def test_version_lookup_offloaded_to_a_thread_not_the_event_loop(monkeypatch):
    """Review finding: _next_self_concept_version() is a blocking SQLAlchemy
    call; calling it directly inside this async function would freeze
    cortex-exec's event loop for every concept touched by every finding."""
    import threading

    seen_thread_ids = []
    main_thread_id = None

    def fake_next_version(concept_id):
        seen_thread_ids.append(threading.get_ident())
        return 1

    monkeypatch.setattr(self_study, "_next_self_concept_version", fake_next_version)
    bus = _FakeBus()

    async def run():
        nonlocal main_thread_id
        main_thread_id = threading.get_ident()
        await self_study.publish_self_concept_history_from_reflection(
            bus=bus, source=ServiceRef(name="orion-cortex-exec"), findings=[_finding()], correlation_id="corr-1"
        )

    asyncio.run(run())

    assert len(seen_thread_ids) == 1
    assert seen_thread_ids[0] != main_thread_id


def test_publish_never_raises_when_bus_fails(monkeypatch):
    import app.self_study_analysis as self_study_analysis_module

    monkeypatch.setattr(self_study_analysis_module, "_get_engine", lambda: None)
    bus = _FakeBus(fail=True)

    status = asyncio.run(
        self_study.publish_self_concept_history_from_reflection(
            bus=bus, source=ServiceRef(name="orion-cortex-exec"), findings=[_finding()], correlation_id="corr-1"
        )
    )

    assert status.status == "failed"
    assert "failed=1" in status.detail

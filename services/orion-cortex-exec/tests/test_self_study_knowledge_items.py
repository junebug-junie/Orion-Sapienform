"""Tests for self_study.py's durable self-knowledge item log (self-model
rebuild arc, Patch 2, 2026-09-05): _flatten_metadata_text() and
publish_self_knowledge_items(), the prerequisite for pointing topic-
foundry's clustering pipeline at self-facts (Patch 3)."""
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

from orion.core.bus.bus_schemas import BaseEnvelope  # noqa: E402


class _FakeBus:
    def __init__(self, *, fail_item_ids: set[str] | None = None) -> None:
        self.published: list[tuple[str, BaseEnvelope]] = []
        self._fail_item_ids = fail_item_ids or set()

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        item_id = envelope.payload.get("item_id")
        if item_id in self._fail_item_ids:
            raise RuntimeError(f"publish_failed:{item_id}")
        self.published.append((channel, envelope))


def test_flatten_metadata_text_none_when_empty():
    item = self_study._item(run_id="r", observed_at="t", category="touchpoint", name="n", source_path="p")
    assert self_study._flatten_metadata_text(item) is None


def test_flatten_metadata_text_falls_back_to_truncating_a_single_oversized_pair():
    """Review finding: if even the first whole key=value pair alone exceeds
    the cap, the whole-pairs loop keeps nothing and used to return None for
    the entire item -- dropping all its text instead of a truncated-but-real
    string."""
    item = self_study._item(
        run_id="r", observed_at="t", category="touchpoint", name="n", source_path="p",
        metadata={"a": "x" * 3000},
    )
    result = self_study._flatten_metadata_text(item)
    assert result is not None
    assert len(result) == self_study._METADATA_TEXT_MAX_CHARS
    assert result.startswith("a=")


def test_flatten_metadata_text_real_content_sorted():
    item = self_study._item(
        run_id="r", observed_at="t", category="touchpoint", name="n", source_path="p",
        metadata={"b": 2, "a": 1},
    )
    assert self_study._flatten_metadata_text(item) == "a=1, b=2"


def test_publish_self_knowledge_items_skips_without_bus():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")

    status = asyncio.run(
        self_study.publish_self_knowledge_items(bus=None, source=None, snapshot=snapshot, correlation_id="corr-1")
    )

    assert status.status == "skipped"
    assert status.detail == "missing_bus"


def test_publish_self_knowledge_items_entry_id_is_deterministic_for_retries():
    """Review finding: a fresh random uuid4() entry_id meant a retried
    publish of the same item within the same run inserted a brand-new row
    instead of colliding on the primary key -- orion-sql-writer's
    INSERT_ONLY_MODELS duplicate-skip (idempotent-write mechanism) can only
    fire on an entry_id collision. Same item_id + same run_id must yield
    the same entry_id across two independent publish calls."""
    from orion.core.bus.bus_schemas import ServiceRef

    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    first_item = next(
        getattr(snapshot, section)[0]
        for section in self_study._SNAPSHOT_SECTION_NAMES
        if getattr(snapshot, section)
    )
    bus_a = _FakeBus()
    bus_b = _FakeBus()

    asyncio.run(
        self_study.publish_self_knowledge_items(
            bus=bus_a, source=ServiceRef(name="orion-cortex-exec"), snapshot=snapshot, correlation_id="corr-1"
        )
    )
    asyncio.run(
        self_study.publish_self_knowledge_items(
            bus=bus_b, source=ServiceRef(name="orion-cortex-exec"), snapshot=snapshot, correlation_id="corr-2"
        )
    )

    entry_id_a = next(env.payload["entry_id"] for _, env in bus_a.published if env.payload["item_id"] == first_item.item_id)
    entry_id_b = next(env.payload["entry_id"] for _, env in bus_b.published if env.payload["item_id"] == first_item.item_id)
    assert entry_id_a == entry_id_b


def test_publish_self_knowledge_items_publishes_one_message_per_item():
    from orion.core.bus.bus_schemas import ServiceRef

    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    total_items = sum(len(getattr(snapshot, section)) for section in self_study._SNAPSHOT_SECTION_NAMES)
    bus = _FakeBus()

    status = asyncio.run(
        self_study.publish_self_knowledge_items(
            bus=bus, source=ServiceRef(name="orion-cortex-exec"), snapshot=snapshot, correlation_id="corr-1"
        )
    )

    assert status.status == "written"
    assert status.channel == self_study.SELF_STUDY_ITEMS_WRITE_CHANNEL
    assert len(bus.published) == total_items
    channel, envelope = bus.published[0]
    assert channel == self_study.SELF_STUDY_ITEMS_WRITE_CHANNEL
    assert envelope.payload["run_id"] == snapshot.run_id


def test_publish_self_knowledge_items_isolates_per_item_failures():
    from orion.core.bus.bus_schemas import ServiceRef

    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    first_item = next(
        getattr(snapshot, section)[0]
        for section in self_study._SNAPSHOT_SECTION_NAMES
        if getattr(snapshot, section)
    )
    total_items = sum(len(getattr(snapshot, section)) for section in self_study._SNAPSHOT_SECTION_NAMES)
    bus = _FakeBus(fail_item_ids={first_item.item_id})

    status = asyncio.run(
        self_study.publish_self_knowledge_items(
            bus=bus, source=ServiceRef(name="orion-cortex-exec"), snapshot=snapshot, correlation_id="corr-1"
        )
    )

    # One item failed, the rest still published -- a single bad item must
    # not drop the whole run's batch (still publishes the other items),
    # but any real failure marks the overall status "failed" so a partial
    # loss is never reported as a plain success (review finding: this used
    # to read "written" as long as anything at all succeeded, so a 999-of-
    # 1000 failure would have looked identical to a clean run).
    assert status.status == "failed"
    assert len(bus.published) == total_items - 1
    assert "published=" in status.detail
    assert "failed=1" in status.detail

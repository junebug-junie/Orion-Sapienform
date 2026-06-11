"""Ensure every subscribed bus kind has a sql-writer persistence path (prevents silent drops)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

from app.settings import DEFAULT_ROUTE_MAP, settings

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_route_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)

MODEL_MAP = worker.MODEL_MAP

# Kinds handled outside DEFAULT_ROUTE_MAP -> MODEL_MAP lookup in handle_envelope.
SPECIAL_KINDS = {
    "grammar.event.v1",  # dedicated grammar ledger path
}

# Legacy / multi-kind channels where envelope kind is resolved at runtime.
LEGACY_KIND_ALIASES = {
    "chat.history",
    "chat.log",
    "collapse.mirror",
    "tags.enriched",
    "collapse.enrichment",
    "dream.log",
    "biometrics.telemetry",
    "spark.telemetry",
    "cognition.trace",
    "equilibrium.metacog.trigger",
    "metacognition.enriched.v1",
    "legacy.message",
}

# Routed via dedicated branches (not MODEL_MAP lookup).
INLINE_ROUTE_KINDS = {
    "journal.entry.index.v1": "JournalEntryIndexSQL",
}

# Evidence ingest channels project to evidence.unit.v1 via build_evidence_units().
EVIDENCE_ADAPTER_KINDS = {
    "document.markdown.spec.v1",
    "document.parsed.v1",
}


def _load_sql_writer_channel_kinds() -> set[str]:
    channels_path = REPO_ROOT / "orion" / "bus" / "channels.yaml"
    data = yaml.safe_load(channels_path.read_text(encoding="utf-8")) or {}
    kinds: set[str] = set()
    subscribed = set(settings.effective_subscribe_channels)
    for entry in data.get("channels") or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if name not in subscribed:
            continue
        consumers = entry.get("consumer_services") or []
        if "orion-sql-writer" not in consumers and "*" not in consumers:
            continue
        message_kind = entry.get("message_kind")
        if message_kind:
            kinds.add(str(message_kind))
    return kinds


def test_every_default_route_kind_maps_to_model() -> None:
    missing = []
    for kind, model_key in DEFAULT_ROUTE_MAP.items():
        if kind in INLINE_ROUTE_KINDS:
            assert INLINE_ROUTE_KINDS[kind] == model_key
            continue
        if model_key not in MODEL_MAP:
            missing.append(f"{kind} -> {model_key}")
    assert not missing, "MODEL_MAP missing route targets: " + ", ".join(missing)


def test_env_route_map_json_covers_all_default_kinds() -> None:
    merged = settings.route_map
    for kind in DEFAULT_ROUTE_MAP:
        assert kind in merged, f"route_map missing {kind}"


def test_subscribed_catalog_kinds_are_routable_or_explicitly_special() -> None:
    catalog_kinds = _load_sql_writer_channel_kinds()
    route_map = settings.route_map
    unroutable = sorted(
        kind
        for kind in catalog_kinds
        if kind not in route_map
        and kind not in SPECIAL_KINDS
        and kind not in LEGACY_KIND_ALIASES
        and kind not in EVIDENCE_ADAPTER_KINDS
        and kind not in INLINE_ROUTE_KINDS
    )
    assert not unroutable, (
        "Subscribed channel message_kind values lack sql-writer route coverage: "
        + ", ".join(unroutable)
    )


def test_env_example_route_map_json_parses_and_matches_defaults() -> None:
    env_example = (SQL_WRITER_ROOT / ".env_example").read_text(encoding="utf-8")
    marker = "SQL_WRITER_ROUTE_MAP_JSON="
    line = next(l for l in env_example.splitlines() if l.startswith(marker))
    raw = line.split("=", 1)[1].strip()
    parsed = json.loads(raw)
    for kind, model_key in DEFAULT_ROUTE_MAP.items():
        assert parsed.get(kind) == model_key, f".env_example route missing or mismatched for {kind}"

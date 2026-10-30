"""Layer 1 self-facts: the daily repo/hardware/behavior scan that feeds Self Atlas.

This is all that is left of self-study after the 2026-10-30 fold into curiosity
(docs/superpowers/specs/2026-10-29-self-study-into-curiosity-proposal.md). The daily
reflection loop (Layer 3), concept induction (Layer 2), the retrieve path, the RDF
writeback code and the Claude enrichment service were retired in the same patch.
Self-modeling now happens in curiosity (self-inquiry, the dig kit, the body atlas).

What stays, and why: ``run_self_repo_inspect`` writes one ``self_knowledge_items`` row
per fact per run, which is the table Self Atlas clusters over, plus one "Self-study
factual snapshot" journal entry. It is called only by ``self_study_refresh_loop``.
Self Atlas and this feed are the next retirement candidates.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import yaml

from orion import self_inventory
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.journaler.schemas import JournalEntryWriteV1
from orion.schemas.self_knowledge_item_log import SelfKnowledgeItemLogV1
from orion.schemas.self_study import (
    SelfKnowledgeItemV1,
    SelfKnowledgeSectionCountsV1,
    SelfRepoInspectResultV1,
    SelfSnapshotV1,
    SelfWritebackState,
    SelfWritebackStatusV1,
)

logger = logging.getLogger("orion.cortex.exec.self_study")


def _resolve_repo_root(module_path: str | Path | None = None) -> Path:
    env_root = os.getenv("ORION_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    resolved_path = Path(module_path or __file__).resolve()
    search_roots = (
        (resolved_path.parent, *resolved_path.parents)
        if resolved_path.suffix
        else (resolved_path, *resolved_path.parents)
    )
    for candidate in search_roots:
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
        if (candidate / "services").is_dir() and (
            (candidate / "orion").is_dir() or (candidate / "config").is_dir()
        ):
            return candidate

    fallback_root = resolved_path.parent.parent if resolved_path.suffix else resolved_path.parent
    logger.warning(
        "self_study_repo_root_fallback module_path=%s fallback=%s",
        resolved_path,
        fallback_root,
    )
    return fallback_root


REPO_ROOT = _resolve_repo_root()
JOURNAL_WRITE_CHANNEL = "orion:journal:write"
SELF_STUDY_ITEMS_WRITE_CHANNEL = "orion:self_study:items:write"
# Public cortex intake channel. Not used by self-study any more; verb_adapters.py
# still reads it from here for its own orch round trips.
CORTEX_ORCH_REQUEST_CHANNEL = os.getenv("CORTEX_REQUEST_CHANNEL", "orion:cortex:request")
TRUST_TIER = "authoritative"
_AUTHOR = "orion"
_SCAN_ROOTS: tuple[str, ...] = ("services", "orion")
_ENV_FIELD_RE = re.compile(r'^(?P<name>[A-Z0-9_]+)\s*:\s*[^=]+=\s*Field\([^\n]*?(?:alias|env|validation_alias)="(?P<alias>[A-Z0-9_]+)"', re.MULTILINE)
_ENV_FALLBACK_RE = re.compile(r'^(?P<name>[A-Z0-9_]+)\s*:\s*', re.MULTILINE)

# SelfSnapshotV1's list-of-items section names -- was already copy-pasted as
# a bare tuple literal at multiple pre-existing call sites (review finding:
# adding two more would have made it 5 independently-drifting copies). One
# shared constant so a future new section only needs updating here.
_SNAPSHOT_SECTION_NAMES: tuple[str, ...] = (
    "services", "modules", "channels", "verbs", "schemas", "touchpoints", "env_surfaces",
    # Added 2026-09-05 (Layer 1 broadening, self-model rebuild arc).
    "hardware", "behavioral",
)


def _all_snapshot_items(snapshot: "SelfSnapshotV1") -> list[SelfKnowledgeItemV1]:
    """Flat list of every Layer-1 item across all sections. Added for
    publish_self_knowledge_items() (review finding: a 6th near-identical
    `for section_name in _SNAPSHOT_SECTION_NAMES: for item in
    getattr(...)` loop in this file) -- the five pre-existing call sites
    each do something different with the items inline (filtering, grouping)
    and are left untouched, out of scope for this patch."""
    return [item for section_name in _SNAPSHOT_SECTION_NAMES for item in getattr(snapshot, section_name)]

# Layer 1 hardware facts, v1 scope: the static field-topology config only
# (see _hardware_items() below for why live cabinet-sensor readings are an
# explicit, disclosed fast-follow rather than in this patch).
_FIELD_TOPOLOGY_RELPATH = "config/field/orion_field_topology.v1.yaml"

# How many recent chat_stance_belief_log rows Layer 1 surfaces as behavioral
# facts each run. Not a privacy cap (see module-level note above) -- purely
# to keep one snapshot from growing unbounded as the table accumulates.
_BEHAVIORAL_ITEMS_LIMIT = 20

_ENV_TARGETS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "services/orion-cortex-exec/app/settings.py",
        "cortex_exec",
        ("ORION_BUS_URL", "CHANNEL_RECALL_INTAKE", "CHANNEL_CORE_EVENTS"),
    ),
    (
        "services/orion-recall/app/settings.py",
        "recall",
        ("RECALL_DEFAULT_PROFILE", "RECALL_ENABLE_RDF", "RECALL_RDF_ENDPOINT_URL", "GRAPHDB_URL", "GRAPHDB_REPO"),
    ),
    # orion-rdf-writer entry removed 2026-07-28: service deleted (orion-rdf-store/
    # Fuseki decommissioned, FalkorDB is the canonical graph backend). Already
    # guarded by _env_items()'s path.exists() check so this was a silent no-op,
    # not a bug -- removed for cleanliness, matching the CHANNEL_WORKER_RDF
    # precedent noted just above prior to this edit.
)

_TOUCHPOINTS: tuple[tuple[str, str, str], ...] = (
    ("journal", "orion/journaler/worker.py", "build_write_payload"),
    ("journal", "services/orion-actions/app/main.py", "_run_journal"),
    ("journal", "services/orion-sql-writer/app/worker.py", "handle_envelope"),
    # graph touchpoints (orion-rdf-writer) removed 2026-07-28: service deleted.
    # Already guarded by _touchpoint_items()'s path.exists() check.
    ("recall", "services/orion-recall/app/profiles.py", "load_profiles"),
    ("recall", "services/orion-recall/app/worker.py", "process_recall"),
    ("persistence", "services/orion-state-service/app/store.py", "StateStore"),
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _stable_digest(payload: Any, *, length: int = 16) -> str:
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:length]


def _item_id(*, category: str, name: str, source_path: str, symbol_name: str | None, origin_kind: str | None, origin_name: str | None) -> str:
    return f"self-item-{_stable_digest({'category': category, 'name': name, 'source_path': source_path, 'symbol_name': symbol_name or '', 'origin_kind': origin_kind or '', 'origin_name': origin_name or ''})}"


def _item(
    *,
    run_id: str,
    observed_at: str,
    category: str,
    name: str,
    source_path: str,
    metadata: dict[str, Any] | None = None,
    symbol_name: str | None = None,
    origin_kind: str | None = None,
    origin_name: str | None = None,
) -> SelfKnowledgeItemV1:
    return SelfKnowledgeItemV1(
        item_id=_item_id(
            category=category,
            name=name,
            source_path=source_path,
            symbol_name=symbol_name,
            origin_kind=origin_kind,
            origin_name=origin_name,
        ),
        category=category,
        name=name,
        trust_tier=TRUST_TIER,
        observed_at=observed_at,
        run_id=run_id,
        source_path=source_path,
        origin_kind=origin_kind,
        origin_name=origin_name,
        symbol_name=symbol_name,
        metadata=metadata or {},
    )


def _inventory_items(records: list[dict], *, run_id: str, observed_at: str, category: str) -> list[SelfKnowledgeItemV1]:
    """Wrap orion.self_inventory records (scanned at REPO_ROOT) as self-knowledge items."""
    return [
        _item(
            run_id=run_id,
            observed_at=observed_at,
            category=category,
            name=rec["name"],
            source_path=rec["source_path"],
            origin_kind=category,
            origin_name=rec["name"],
            metadata=rec["metadata"] or None,
        )
        for rec in records
    ]


def _service_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    return _inventory_items(self_inventory.services(REPO_ROOT, strict=True), run_id=run_id, observed_at=observed_at, category="service")


def _module_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    return _inventory_items(self_inventory.modules(REPO_ROOT, strict=True), run_id=run_id, observed_at=observed_at, category="module")


def _channel_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    return _inventory_items(self_inventory.channels(REPO_ROOT, strict=True), run_id=run_id, observed_at=observed_at, category="channel")


def _yaml_verb_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    return _inventory_items(self_inventory.yaml_verbs(REPO_ROOT), run_id=run_id, observed_at=observed_at, category="verb")


def _runtime_verb_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    return _inventory_items(self_inventory.runtime_verbs(REPO_ROOT), run_id=run_id, observed_at=observed_at, category="verb")


def _verb_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    merged: dict[tuple[str, str], SelfKnowledgeItemV1] = {}
    for item in [*_yaml_verb_items(run_id=run_id, observed_at=observed_at), *_runtime_verb_items(run_id=run_id, observed_at=observed_at)]:
        merged[(item.name, item.source_path)] = item
    return sorted(merged.values(), key=lambda item: (item.name, item.source_path))


def _schema_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    return _inventory_items(self_inventory.schemas(REPO_ROOT, strict=True), run_id=run_id, observed_at=observed_at, category="schema")


def _touchpoint_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    items: list[SelfKnowledgeItemV1] = []
    for category, rel_path, symbol_name in _TOUCHPOINTS:
        path = REPO_ROOT / rel_path
        if not path.exists():
            continue
        items.append(
            _item(
                run_id=run_id,
                observed_at=observed_at,
                category="touchpoint",
                name=f"{category}:{path.stem}",
                source_path=rel_path,
                origin_kind=category,
                origin_name=path.stem,
                symbol_name=symbol_name,
                metadata={"surface": category},
            )
        )
    return sorted(items, key=lambda item: (item.metadata.get("surface", ""), item.source_path))


def _extract_declared_env_names(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for match in _ENV_FIELD_RE.finditer(text):
        found[match.group("name")] = match.group("alias")
    for match in _ENV_FALLBACK_RE.finditer(text):
        found.setdefault(match.group("name"), match.group("name"))
    return found


def _env_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    items: list[SelfKnowledgeItemV1] = []
    for rel_path, surface, targets in _ENV_TARGETS:
        path = REPO_ROOT / rel_path
        if not path.exists():
            continue
        declared = _extract_declared_env_names(path.read_text(encoding="utf-8"))
        for name in targets:
            alias = declared.get(name, name)
            items.append(
                _item(
                    run_id=run_id,
                    observed_at=observed_at,
                    category="env_surface",
                    name=alias,
                    source_path=rel_path,
                    origin_kind="env_surface",
                    origin_name=surface,
                    metadata={"surface": surface, "field_name": name},
                )
            )
    return sorted(items, key=lambda item: (item.metadata.get("surface", ""), item.name))


def _hardware_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    """Physical/hardware self-facts, v1 scope: the static field-topology
    config (nodes/capabilities/edges) -- config/field/orion_field_topology
    .v1.yaml, parsed directly (mirrors _channel_items()'s own yaml.safe_load
    pattern; no cross-service import). Live cabinet-sensor readings
    (temp/humidity/etc) are a known, disclosed fast-follow, not silently
    dropped: cortex-exec has no /run/orion-sensors mount (only orion-hub
    does), so a live read needs either a new mount or a cross-service call
    to orion-hub's cabinet_sensors_routes.py -- out of scope for this patch,
    see the PR report."""
    path = REPO_ROOT / _FIELD_TOPOLOGY_RELPATH
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        logger.warning("self_study_field_topology_unreadable path=%s", path)
        return []

    items: list[SelfKnowledgeItemV1] = []
    for node in data.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or "")
        if not node_id:
            continue
        items.append(
            _item(
                run_id=run_id,
                observed_at=observed_at,
                category="hardware",
                name=f"node:{node_id}",
                source_path=_rel(path),
                origin_kind="hardware_node",
                origin_name=node_id,
                metadata={"kind": "node"},
            )
        )
    for capability in data.get("capabilities") or []:
        if not isinstance(capability, dict):
            continue
        capability_id = str(capability.get("capability_id") or "")
        if not capability_id:
            continue
        items.append(
            _item(
                run_id=run_id,
                observed_at=observed_at,
                category="hardware",
                name=f"capability:{capability_id}",
                source_path=_rel(path),
                origin_kind="hardware_capability",
                origin_name=capability_id,
                metadata={"kind": "capability"},
            )
        )
    for edge in data.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        source_id = str(edge.get("source_id") or "")
        target_id = str(edge.get("target_id") or "")
        if not source_id or not target_id:
            continue
        items.append(
            _item(
                run_id=run_id,
                observed_at=observed_at,
                category="hardware",
                name=f"edge:{source_id}->{target_id}",
                source_path=_rel(path),
                origin_kind="hardware_edge",
                origin_name=f"{source_id}->{target_id}",
                metadata={
                    "kind": "edge",
                    "edge_type": edge.get("edge_type"),
                    "weight": edge.get("weight"),
                },
            )
        )
    return sorted(items, key=lambda item: item.name)


def _behavioral_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    """Orion's own recent conversational behavior, read directly from
    chat_stance_belief_log (populated by chat_stance.py's real per-turn
    belief computation -- see orion/substrate/chat_stance_belief_bus.py).
    Real content, not anonymized counts: Juniper's explicit 2026-09-05 call
    is that she is the sole user and companion, so redaction/aggregation of
    her own conversations from her own self-model is moot. Fails soft (empty
    list) if the DB is unreachable or the table doesn't exist yet -- this
    table is new as of this same patch, so a fresh deploy racing this call
    is expected, not a bug."""
    from app.self_study_analysis import _get_engine
    from sqlalchemy import text

    engine = _get_engine()
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT entry_id, created_at, shift_kind, anchor_summary, lineage_summary "
                    "FROM chat_stance_belief_log ORDER BY created_at DESC LIMIT :limit"
                ),
                {"limit": _BEHAVIORAL_ITEMS_LIMIT},
            ).mappings().all()
    except Exception as exc:
        logger.debug("self_study_behavioral_items_unavailable error=%s", exc)
        return []

    items: list[SelfKnowledgeItemV1] = []
    for row in rows:
        entry_id = str(row["entry_id"])
        created_at = row["created_at"]
        created_at_str = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        shift_kind = row["shift_kind"]
        name = f"turn:{created_at_str}" + (f":{shift_kind}" if shift_kind and shift_kind != "NONE" else "")
        items.append(
            _item(
                run_id=run_id,
                observed_at=observed_at,
                category="behavioral",
                name=name,
                source_path="chat_stance_belief_log",
                origin_kind="behavioral_turn",
                origin_name=entry_id,
                metadata={
                    "shift_kind": shift_kind,
                    "anchor_summary": row["anchor_summary"],
                    "lineage_summary": row["lineage_summary"],
                    "created_at": created_at_str,
                },
            )
        )
    return items


def _counts_for_sections(sections: dict[str, Sequence[SelfKnowledgeItemV1]]) -> SelfKnowledgeSectionCountsV1:
    return SelfKnowledgeSectionCountsV1(**{key: len(value) for key, value in sections.items()})


def _canonical_snapshot_payload(sections: dict[str, Sequence[SelfKnowledgeItemV1]]) -> dict[str, Any]:
    canonical = {}
    for key, values in sections.items():
        canonical[key] = [
            {
                "item_id": item.item_id,
                "category": item.category,
                "name": item.name,
                "source_path": item.source_path,
                "origin_kind": item.origin_kind,
                "origin_name": item.origin_name,
                "symbol_name": item.symbol_name,
                "metadata": item.metadata,
            }
            for item in values
        ]
    canonical["scan_roots"] = list(_SCAN_ROOTS)
    return canonical


def build_self_snapshot(*, observed_at: str | None = None, root: Path | None = None) -> SelfSnapshotV1:
    if root is not None and root != REPO_ROOT:
        raise ValueError("custom_root_not_supported_in_pass1")
    run_id = f"self-run-{uuid4()}"
    ts = observed_at or _iso_now()
    sections = {
        "services": _service_items(run_id=run_id, observed_at=ts),
        "modules": _module_items(run_id=run_id, observed_at=ts),
        "channels": _channel_items(run_id=run_id, observed_at=ts),
        "verbs": _verb_items(run_id=run_id, observed_at=ts),
        "schemas": _schema_items(run_id=run_id, observed_at=ts),
        "touchpoints": _touchpoint_items(run_id=run_id, observed_at=ts),
        "env_surfaces": _env_items(run_id=run_id, observed_at=ts),
        "hardware": _hardware_items(run_id=run_id, observed_at=ts),
        "behavioral": _behavioral_items(run_id=run_id, observed_at=ts),
    }
    snapshot_id = f"self-snapshot-{_stable_digest(_canonical_snapshot_payload(sections))}"
    return SelfSnapshotV1(
        snapshot_id=snapshot_id,
        run_id=run_id,
        observed_at=ts,
        repo_root=REPO_ROOT.as_posix(),
        trust_tier=TRUST_TIER,
        counts=_counts_for_sections(sections),
        **sections,
    )


def _validate_authoritative_snapshot(snapshot: SelfSnapshotV1) -> None:
    if snapshot.trust_tier != TRUST_TIER:
        raise ValueError(f"snapshot_trust_tier_invalid:{snapshot.trust_tier}")
    for section_name in _SNAPSHOT_SECTION_NAMES:
        for item in getattr(snapshot, section_name):
            if item.trust_tier != TRUST_TIER:
                raise ValueError(f"non_authoritative_item:{section_name}:{item.name}:{item.trust_tier}")
            if not item.source_path:
                raise ValueError(f"missing_source_path:{section_name}:{item.name}")
            if not item.item_id:
                raise ValueError(f"missing_item_id:{section_name}:{item.name}")
            if not item.run_id or not item.observed_at:
                raise ValueError(f"missing_provenance:{section_name}:{item.name}")


def build_self_study_summary(snapshot: SelfSnapshotV1) -> str:
    touch_surfaces = sorted({str(item.metadata.get("surface") or "") for item in snapshot.touchpoints if item.metadata.get("surface")})
    return (
        f"Self-study factual snapshot captured {snapshot.counts.services} services, {snapshot.counts.modules} modules, "
        f"{snapshot.counts.channels} channels, {snapshot.counts.verbs} verbs, and {snapshot.counts.schemas} schemas. "
        f"Touchpoints present: {', '.join(touch_surfaces) or 'none'}. "
        "Authoritative write-back excludes induced and reflective content."
    )


def build_self_study_journal_entry(snapshot: SelfSnapshotV1, *, correlation_id: str, created_at: datetime | None = None) -> JournalEntryWriteV1:
    ts = created_at or datetime.now(timezone.utc)
    surfaces = sorted({str(item.metadata.get("surface") or "") for item in snapshot.touchpoints if item.metadata.get("surface")})
    body = (
        f"Snapshot {snapshot.snapshot_id} run {snapshot.run_id} at {snapshot.observed_at}.\n"
        f"Counts: services={snapshot.counts.services}, modules={snapshot.counts.modules}, channels={snapshot.counts.channels}, verbs={snapshot.counts.verbs}, schemas={snapshot.counts.schemas}.\n"
        f"Touchpoints: {', '.join(surfaces) or 'none'}.\n"
        "Trust tier: authoritative only. Journal is summary-only and not authoritative storage."
    )
    return JournalEntryWriteV1(
        created_at=ts,
        author=_AUTHOR,
        mode="manual",
        title="Self-study factual snapshot",
        body=body,
        source_kind="self_study",
        source_ref=snapshot.snapshot_id,
        correlation_id=correlation_id,
    )


def _as_envelope_correlation_id(raw: str) -> str:
    try:
        return str(UUID(str(raw)))
    except Exception:
        return str(uuid5(NAMESPACE_URL, str(raw)))


async def publish_self_study_artifacts(
    *,
    bus: Any | None,
    source: ServiceRef,
    snapshot: SelfSnapshotV1,
    correlation_id: str,
) -> tuple[SelfWritebackStatusV1, JournalEntryWriteV1]:
    """Publish the one daily "Self-study factual snapshot" journal entry.

    The RDF graph writeback this used to report (always ``skipped/channel_retired`` since
    2026-07-28) was deleted with the rest of the RDF code on 2026-10-30."""
    journal_entry = build_self_study_journal_entry(snapshot, correlation_id=correlation_id)
    if bus is None:
        logger.info("self_study_degraded snapshot_id=%s reason=missing_bus", snapshot.snapshot_id)
        return (
            SelfWritebackStatusV1(
                target="journal",
                status="skipped",
                authoritative=False,
                channel=JOURNAL_WRITE_CHANNEL,
                idempotency_key=snapshot.snapshot_id,
                append_only=True,
                detail="missing_bus",
            ),
            journal_entry,
        )

    journal_status = SelfWritebackStatusV1(
        target="journal",
        status="written",
        authoritative=False,
        channel=JOURNAL_WRITE_CHANNEL,
        idempotency_key=snapshot.snapshot_id,
        append_only=True,
        detail="append_only_by_design",
    )

    envelope_corr_id = _as_envelope_correlation_id(correlation_id)
    journal_env = BaseEnvelope(
        kind="journal.entry.write.v1",
        source=source,
        correlation_id=envelope_corr_id,
        payload=journal_entry.model_dump(mode="json"),
    )
    try:
        await bus.publish(JOURNAL_WRITE_CHANNEL, journal_env)
    except Exception as exc:
        journal_status = SelfWritebackStatusV1(
            target="journal",
            status="failed",
            authoritative=False,
            channel=JOURNAL_WRITE_CHANNEL,
            idempotency_key=snapshot.snapshot_id,
            append_only=True,
            detail=str(exc),
        )

    logger.info(
        "self_study_publish snapshot_id=%s run_id=%s journal_status=%s journal_detail=%s",
        snapshot.snapshot_id,
        snapshot.run_id,
        journal_status.status,
        journal_status.detail,
    )
    return journal_status, journal_entry


_METADATA_TEXT_MAX_CHARS = 2000


def _flatten_metadata_text(item: SelfKnowledgeItemV1) -> str | None:
    """Plain-text rendering of an item's metadata dict for topic-foundry's
    clustering text_columns -- the pipeline wants text, not a dict it would
    have to know how to flatten itself. `key=value` pairs, sorted for
    determinism, capped for storage sanity.

    Review finding: a plain `[:2000]` slice can cut a `key=value` pair in
    half, storing a syntactically broken trailing fragment as the exact
    field Patch 3's clustering pipeline is meant to consume as clean text.
    Builds up whole pairs instead and stops before the next one would
    exceed the cap, so the result never ends mid-pair."""
    if not item.metadata:
        return None
    parts = [f"{key}={value}" for key, value in sorted(item.metadata.items(), key=lambda kv: kv[0])]
    kept: list[str] = []
    length = 0
    for part in parts:
        added_length = len(part) + (2 if kept else 0)  # ", " separator
        if length + added_length > _METADATA_TEXT_MAX_CHARS:
            break
        kept.append(part)
        length += added_length
    if not kept and parts:
        # Review finding: if even the first whole pair alone exceeds the cap,
        # the loop above keeps nothing and this returned None for the whole
        # item -- dropping all its text instead of giving Patch 3's
        # clustering pipeline a truncated-but-real string. Fall back to a
        # hard truncation of just that first pair rather than nothing.
        return parts[0][:_METADATA_TEXT_MAX_CHARS]
    return ", ".join(kept) or None


_SELF_KNOWLEDGE_ITEMS_PUBLISH_CONCURRENCY = 20


async def _publish_one_self_knowledge_item(
    *,
    bus: Any,
    source: ServiceRef,
    snapshot: SelfSnapshotV1,
    item: SelfKnowledgeItemV1,
    envelope_corr_id: str,
    semaphore: "asyncio.Semaphore",
) -> bool:
    """Publish one item; returns True on success, False on any failure.
    Review finding: payload/envelope construction must live INSIDE the
    try/except too -- a pydantic validation error on one item's fields must
    not crash the whole run (it previously did, since construction happened
    before the try block, contradicting this function's own per-item-
    isolation claim)."""
    try:
        # Review finding: a fresh random uuid4() entry_id (the schema's own
        # default) meant a retried publish of the same item within the same
        # run (process crash mid-batch, verb re-invocation) inserted a
        # brand-new row instead of colliding on the primary key --
        # orion-sql-writer's INSERT_ONLY_MODELS duplicate-skip (the
        # mechanism this table relies on for idempotent writes) can only
        # fire on an entry_id collision, which a random id makes structurally
        # impossible. Deriving it from (item_id, run_id) instead makes a
        # retry of the same item in the same run collide and get skipped,
        # while a genuinely new run (a new build_self_snapshot() call, with
        # its own fresh run_id) still gets its own row -- matching the "one
        # row per item per run" design intent.
        entry_id = f"self-know-item-{_stable_digest({'item_id': item.item_id, 'run_id': item.run_id})}"
        payload = SelfKnowledgeItemLogV1(
            entry_id=entry_id,
            item_id=item.item_id,
            run_id=item.run_id,
            category=item.category,
            name=item.name,
            trust_tier=item.trust_tier,
            observed_at=item.observed_at,
            source_path=item.source_path,
            symbol_name=item.symbol_name,
            metadata_text=_flatten_metadata_text(item),
        )
        envelope = BaseEnvelope(
            kind="self_study.items.write.v1",
            source=source,
            correlation_id=envelope_corr_id,
            payload=payload.model_dump(mode="json"),
        )
        async with semaphore:
            await bus.publish(SELF_STUDY_ITEMS_WRITE_CHANNEL, envelope)
        return True
    except Exception as exc:
        logger.debug(
            "self_study_items_publish_item_failed snapshot_id=%s item_id=%s error=%s",
            snapshot.snapshot_id,
            item.item_id,
            exc,
        )
        return False


async def publish_self_knowledge_items(
    *, bus: Any | None, source: ServiceRef, snapshot: SelfSnapshotV1, correlation_id: str
) -> SelfWritebackStatusV1:
    """Durable, multi-run history of this snapshot's Layer-1 items (self-
    model rebuild arc, Patch 2) -- prerequisite for topic-foundry's Self
    Atlas, which needs a real, growing, queryable source_table to cluster
    over. One bus message per item, published concurrently (bounded by
    _SELF_KNOWLEDGE_ITEMS_PUBLISH_CONCURRENCY -- review finding: a real
    snapshot has 1000+ items, and sequential awaits would add thousands of
    Redis round trips directly to this run's critical path). Best-effort
    per item so one bad item doesn't drop the rest of a run's batch.
    Additive to the existing journal-summary write in
    publish_self_study_artifacts, not a replacement -- that one stays a
    one-off summary by design."""
    status = SelfWritebackStatusV1(
        target="self_knowledge_items",
        status="skipped",
        authoritative=False,
        channel=SELF_STUDY_ITEMS_WRITE_CHANNEL,
        idempotency_key=snapshot.snapshot_id,
        append_only=True,
        detail="missing_bus",
    )
    if bus is None:
        logger.info("self_study_items_publish_skip snapshot_id=%s reason=missing_bus", snapshot.snapshot_id)
        return status

    envelope_corr_id = _as_envelope_correlation_id(correlation_id)
    items = _all_snapshot_items(snapshot)
    semaphore = asyncio.Semaphore(_SELF_KNOWLEDGE_ITEMS_PUBLISH_CONCURRENCY)
    results = await asyncio.gather(
        *(
            _publish_one_self_knowledge_item(
                bus=bus, source=source, snapshot=snapshot, item=item,
                envelope_corr_id=envelope_corr_id, semaphore=semaphore,
            )
            for item in items
        )
    )
    published = sum(1 for ok in results if ok)
    failed = sum(1 for ok in results if not ok)

    # Review finding: "any success at all -> written" reported a run that
    # lost 999 of 1000 items the same as a fully clean run -- a monitor
    # filtering on status=="failed" or on WARNING logs would never see a
    # near-total failure. Any real failure now marks the whole batch
    # "failed" (not just an all-failed batch); `detail` still records the
    # real published count, so a partial success isn't lost, just not
    # allowed to read as a plain success.
    if failed > 0:
        overall_status: SelfWritebackState = "failed"
    elif published > 0:
        overall_status = "written"
    else:
        overall_status = "skipped"
    status = SelfWritebackStatusV1(
        target="self_knowledge_items",
        status=overall_status,
        authoritative=False,
        channel=SELF_STUDY_ITEMS_WRITE_CHANNEL,
        idempotency_key=snapshot.snapshot_id,
        append_only=True,
        detail=f"published={published} failed={failed}",
    )
    log_fn = logger.warning if failed > 0 else logger.info
    log_fn(
        "self_study_items_publish snapshot_id=%s run_id=%s status=%s published=%s failed=%s",
        snapshot.snapshot_id,
        snapshot.run_id,
        overall_status,
        published,
        failed,
    )
    return status


async def run_self_repo_inspect(*, bus: Any | None, source: ServiceRef, correlation_id: str) -> SelfRepoInspectResultV1:
    start = time.monotonic()
    # Synchronous repo walk + one Postgres SELECT; off the event loop so a
    # scheduled refresh can't stall an in-flight chat turn on the same lane.
    snapshot = await asyncio.to_thread(build_self_snapshot)
    journal_status, journal_entry = await publish_self_study_artifacts(
        bus=bus,
        source=source,
        snapshot=snapshot,
        correlation_id=correlation_id,
    )
    items_write_status = await publish_self_knowledge_items(
        bus=bus, source=source, snapshot=snapshot, correlation_id=correlation_id
    )
    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "self_study_scan snapshot_id=%s run_id=%s duration_ms=%s services=%s modules=%s channels=%s verbs=%s schemas=%s journal_status=%s items_write_status=%s",
        snapshot.snapshot_id,
        snapshot.run_id,
        duration_ms,
        snapshot.counts.services,
        snapshot.counts.modules,
        snapshot.counts.channels,
        snapshot.counts.verbs,
        snapshot.counts.schemas,
        journal_status.status,
        items_write_status.status,
    )
    return SelfRepoInspectResultV1(
        snapshot=snapshot,
        summary=build_self_study_summary(snapshot),
        journal_write=journal_status,
        journal_entry=journal_entry,
        self_knowledge_items_write=items_write_status,
    )

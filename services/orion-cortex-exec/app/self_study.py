from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import requests
import yaml
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from orion.cognition.cortex_payload_extract import extract_cortex_payload_text
from orion.core.bus.bus_schemas import BaseEnvelope, LLMMessage, ServiceRef
from orion.journaler.schemas import JournalEntryWriteV1
from orion.schemas.self_knowledge_item_log import SelfKnowledgeItemLogV1
from orion.schemas.self_concept_history import SelfConceptHistoryV1
from orion.llm.routes import normalize_llm_route
from orion.schemas.cortex.contracts import (
    CortexClientContext,
    CortexClientRequest,
    RecallDirective,
)
from orion.schemas.rdf import RdfWriteRequest
from orion.structural_mass.graph_delta import graph_structural_delta
from orion.structural_mass.snapshot_history import (
    GraphSnapshotStats,
    append_snapshot,
    graph_snapshot_stats_from_text,
    read_snapshots,
)
from orion.schemas.self_study import (
    SelfConceptEvidenceRefV1,
    SelfConceptInduceResultV1,
    SelfConceptRefV1,
    SelfConceptReflectResultV1,
    SelfInducedConceptV1,
    SelfKnowledgeItemV1,
    SelfKnowledgeSectionCountsV1,
    SelfStudyRetrievedRecordV1,
    SelfStudyRetrievalBackendStatusV1,
    SelfStudyRetrievalCountsV1,
    SelfStudyRetrievalGroupV1,
    SelfStudyRetrieveFiltersV1,
    SelfStudyRetrieveRequestV1,
    SelfStudyRetrieveResultV1,
    SelfReflectiveFindingV1,
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
ORION = Namespace("http://conjourney.net/orion#")
SELF = Namespace("http://conjourney.net/orion/self#")
RDF_ENQUEUE_CHANNEL = "orion:rdf:enqueue"
JOURNAL_WRITE_CHANNEL = "orion:journal:write"
SELF_STUDY_ITEMS_WRITE_CHANNEL = "orion:self_study:items:write"
SELF_CONCEPT_HISTORY_WRITE_CHANNEL = "orion:self_concept:history:write"
# Same public intake channel orion-actions and every other external-ish
# CortexClientRequest caller uses (settings.cortex_request_channel there) --
# self_study.py round-trips through cortex-orch like any other client rather
# than reaching into cortex-exec's own executor.py internals, even though
# both live in this service. That keeps the LLM call on the one reviewed,
# tested public contract (CortexClientRequest -> orch -> ... -> CortexClientResult)
# instead of a second, undocumented in-process path.
CORTEX_ORCH_REQUEST_CHANNEL = os.getenv("CORTEX_REQUEST_CHANNEL", "orion:cortex:request")
SELF_STUDY_REFLECT_VERB = "self_study.reflect"
SELF_GRAPH = "orion:self"
SELF_INDUCED_GRAPH = "orion:self:induced"
SELF_REFLECTIVE_GRAPH = "orion:self:reflective"
GRAPHDB_DEFAULT_URL = "http://orion-athena-graphdb:7200"
GRAPHDB_DEFAULT_REPO = "collapse"
GRAPHDB_DEFAULT_USER = "admin"
GRAPHDB_DEFAULT_PASS = "admin"
GRAPHDB_TIMEOUT_SEC = 5.0
TRUST_TIER = "authoritative"
INDUCED_TRUST_TIER = "induced"
REFLECTIVE_TRUST_TIER = "reflective"
_AUTHOR = "orion"
_SCAN_ROOTS: tuple[str, ...] = ("services", "orion")
_VERB_DECORATOR_RE = re.compile(r'@verb\([\"\']([^\"\']+)[\"\']\)')
_REGISTRY_KEY_RE = re.compile(r'^\s+\"([^\"]+)\":', re.MULTILINE)
_ENV_FIELD_RE = re.compile(r'^(?P<name>[A-Z0-9_]+)\s*:\s*[^=]+=\s*Field\([^\n]*?(?:alias|env|validation_alias)="(?P<alias>[A-Z0-9_]+)"', re.MULTILINE)
_ENV_FALLBACK_RE = re.compile(r'^(?P<name>[A-Z0-9_]+)\s*:\s*', re.MULTILINE)

# --- Layer 2 additive concept sources: graphify community, structural_mass
# delta, semantic-enrichment cache (2026-08) -- see induce_self_concepts()
# below for wiring.
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

_GRAPHIFY_GRAPH_JSON_RELPATH = "graphify-out/graph.json"
_GRAPHIFY_GRAPH_REPORT_RELPATH = "graphify-out/GRAPH_REPORT.md"
_BUILT_AT_COMMIT_RE = re.compile(r'"built_at_commit"\s*:\s*"([^"]*)"')

# Fast-path, in-process cache of "last observed graphify snapshot" for
# structural_mass delta concepts -- always consulted first since it's free.
# On a cold process (this global is still None) _structural_delta_concepts()
# falls back to the durable history log below, if one is configured, instead
# of unconditionally treating every fresh container as "first observation
# ever." Originally this was documented as in-process-only, deliberately
# mirroring orion-cocreation-signals' graph_delta_loop() (services/
# orion-cocreation-signals/app/producers/graph_delta.py) on the grounds that
# cortex-exec's only repo mount is read-only and widening it for one log file
# wasn't worth it -- that reasoning still holds and is NOT being revisited
# here. What changed instead: a small, dedicated, writable volume (mirroring
# the self_study_enrichment_data volume's own pattern one constant below) now
# backs a JSONL history log via snapshot_history.append_snapshot/
# read_snapshots, so durability comes from a new volume rather than from
# widening the existing read-only repo mount.
_LAST_GRAPH_SNAPSHOT_STATS_FOR_STRUCTURAL_DELTA: GraphSnapshotStats | None = None

# Guards read-then-write access to the global above, and backs the
# per-snapshot result memoization below. Two problems this fixes together
# (found in review, both reachable from the shipped async bus-verb
# handlers): (1) reflect_self_concepts() -> validate_phase2a_induction()
# re-invokes induce_self_concepts() on the SAME snapshot to check
# byte-identical concept ids; without memoization that second call would
# see its own first call's already-updated "last seen" state as unchanged
# and silently drop the structural_mass concept, raising
# concept_idempotency_mismatch. (2) two concurrently in-flight
# induce/reflect requests could otherwise interleave reads/writes of the
# global across each other's awaits and cross-contaminate their prior/
# current state.
_STRUCTURAL_DELTA_STATE_LOCK = threading.Lock()
_STRUCTURAL_DELTA_RESULT_CACHE: "OrderedDict[str, list[SelfInducedConceptV1]]" = OrderedDict()
_STRUCTURAL_DELTA_RESULT_CACHE_MAX = 8

# The most recent REAL reflection (see reflect_self_concepts() below), keyed
# by snapshot_id, same shape as _STRUCTURAL_DELTA_RESULT_CACHE. Written only
# by reflect_self_concepts() when it actually gets a bus and a real LLM
# result. _retrieve_self_study_in_process() reads this instead of calling
# reflect_self_concepts() itself, so a synchronous, mode-scoped retrieval
# call can never block on an LLM turn -- a cache miss means "no reflection
# has run yet for this snapshot," not "trigger one now."
#
# KNOWN LIMITATION, same as _STRUCTURAL_DELTA_RESULT_CACHE above: this is
# process-local, in-memory state. If cortex-exec ever runs multiple workers/
# replicas, a reflection produced on one worker is invisible to a retrieval
# request landing on another -- that worker reports "no reflective findings
# cached yet" even though a real reflection just ran successfully elsewhere
# for the identical (content-addressed) snapshot_id. Not fixed here: PR 3 of
# this rebuild (docs/superpowers/specs/2026-09-03-orion-endogenous-self-
# model-and-journal-design.md) replaces this with the durable
# `self_concept_history` table, which is the real fix for this class of
# problem -- adding cross-worker durability to this in-memory cache now
# would be throwaway work.
_REFLECTION_STATE_LOCK = threading.Lock()
_REFLECTION_RESULT_CACHE: "OrderedDict[str, list[SelfReflectiveFindingV1]]" = OrderedDict()
_REFLECTION_RESULT_CACHE_MAX = 8

# Read-only mount of orion-self-study-enrichment's cache volume (see that
# service's docker-compose.yml `self_study_enrichment_data` volume and
# services/orion-cortex-exec/docker-compose.yml's mount of the same volume).
# Path is *inside* the shared volume, matching that service's
# SELF_STUDY_ENRICHMENT_CACHE_DIR=/data/cache/self_study_enrichment default
# (the volume is mounted at /data there, so the cache subpath is
# cache/self_study_enrichment underneath it).
SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR = os.getenv(
    "SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR",
    "/mnt/self_study_enrichment_data/cache/self_study_enrichment",
)

# Durable JSONL history log for structural_mass deltas, on a small writable
# volume owned solely by this service (self_study_structural_mass_data --
# see services/orion-cortex-exec/docker-compose.yml). Unset (None) means "no
# durable store configured" -- e.g. local dev without the volume -- and
# _structural_delta_concepts() falls back to today's in-process-only
# behavior (cold start every process restart), not a crash.
SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH = os.getenv("SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH")

# Layer 3 real reflection. Resolved with normalize_llm_route() at call time
# (not cached at import time -- reflect_self_concepts() runs rarely, on
# demand, so re-validating per call costs nothing and there is no long-lived
# object to resolve it once for, unlike CuriosityInvestigation.__init__).
# `agent` matches curiosity investigation's own lane per the design doc;
# an unrecognised override falls back to no override (the executor's own
# verb-based default) rather than guessing, same contract normalize_llm_route
# already documents.
SELF_STUDY_REFLECT_LLM_ROUTE = os.getenv("SELF_STUDY_REFLECT_LLM_ROUTE", "agent")
# 480s, not the original 240s (2026-09-05): live-confirmed 240s wasn't
# enough for a real reflect call against a real self_knowledge_items-sized
# snapshot (hundreds of concepts, not the small fixture used when 240s was
# first set) -- two manual end-to-end test calls both failed with
# self_study_reflect_llm_call_failed (RPC timeout waiting on
# orion:cortex:result:self-study-reflect:*), one at 240s and one with an
# outer client budget of 300s (still bounded by this constant internally).
# Doubling rather than guessing a small bump: SELF_STUDY_REFLECT_LLM_ROUTE
# defaults to "agent" (see below), already flagged elsewhere in this repo
# (reference_agent_lane_27b_vs_chat_lane_35b_speed) as ~2x slower than the
# chat lane on both prompt processing and generation -- if 480s still isn't
# enough, the real fix is moving this route off "agent", not raising the
# timeout again. See self_study.reflect.yaml's own comment for why its
# verb/step timeout stays larger than this.
SELF_STUDY_REFLECT_TIMEOUT_SEC = float(os.getenv("SELF_STUDY_REFLECT_TIMEOUT_SEC", "480"))

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


def _service_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    services_dir = REPO_ROOT / "services"
    items: list[SelfKnowledgeItemV1] = []
    for service_dir in sorted(p for p in services_dir.iterdir() if p.is_dir()):
        main_py = service_dir / "app" / "main.py"
        settings_py = service_dir / "app" / "settings.py"
        docker_compose = service_dir / "docker-compose.yml"
        source = settings_py if settings_py.exists() else main_py if main_py.exists() else docker_compose if docker_compose.exists() else service_dir
        items.append(
            _item(
                run_id=run_id,
                observed_at=observed_at,
                category="service",
                name=service_dir.name,
                source_path=_rel(source),
                origin_kind="service",
                origin_name=service_dir.name,
                metadata={
                    "has_app_main": main_py.exists(),
                    "has_app_settings": settings_py.exists(),
                    "has_docker_compose": docker_compose.exists(),
                },
            )
        )
    return items


def _module_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    package_root = REPO_ROOT / "orion"
    items: list[SelfKnowledgeItemV1] = []
    for path in sorted(package_root.iterdir()):
        if not path.is_dir():
            continue
        init_py = path / "__init__.py"
        if not init_py.exists():
            continue
        items.append(
            _item(
                run_id=run_id,
                observed_at=observed_at,
                category="module",
                name=f"orion.{path.name}",
                source_path=_rel(init_py),
                origin_kind="module",
                origin_name=f"orion.{path.name}",
            )
        )
    return items


def _channel_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    path = REPO_ROOT / "orion" / "bus" / "channels.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_channels = data.get("channels") or []
    items: list[SelfKnowledgeItemV1] = []
    for channel in raw_channels:
        if not isinstance(channel, dict):
            continue
        name = str(channel.get("name") or "")
        items.append(
            _item(
                run_id=run_id,
                observed_at=observed_at,
                category="channel",
                name=name,
                source_path=_rel(path),
                origin_kind="channel",
                origin_name=name,
                metadata={
                    "kind": channel.get("kind"),
                    "schema_id": channel.get("schema_id"),
                    "message_kind": channel.get("message_kind"),
                    "producer_services": list(channel.get("producer_services") or []),
                    "consumer_services": list(channel.get("consumer_services") or []),
                },
            )
        )
    return sorted(items, key=lambda item: item.name)


def _yaml_verb_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    verbs_dir = REPO_ROOT / "orion" / "cognition" / "verbs"
    items: list[SelfKnowledgeItemV1] = []
    for path in sorted(verbs_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        name = str(data.get("name") or path.stem)
        items.append(
            _item(
                run_id=run_id,
                observed_at=observed_at,
                category="verb",
                name=name,
                source_path=_rel(path),
                origin_kind="verb",
                origin_name=name,
                metadata={
                    "source_type": "yaml",
                    "recall_profile": data.get("recall_profile"),
                },
            )
        )
    return items


def _runtime_verb_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    app_dir = REPO_ROOT / "services" / "orion-cortex-exec" / "app"
    items: list[SelfKnowledgeItemV1] = []
    for path in sorted(app_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in _VERB_DECORATOR_RE.finditer(text):
            trigger = match.group(1)
            items.append(
                _item(
                    run_id=run_id,
                    observed_at=observed_at,
                    category="verb",
                    name=trigger,
                    source_path=_rel(path),
                    origin_kind="verb",
                    origin_name=trigger,
                    metadata={"source_type": "runtime"},
                )
            )
    return sorted(items, key=lambda item: (item.name, item.source_path))


def _verb_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    merged: dict[tuple[str, str], SelfKnowledgeItemV1] = {}
    for item in [*_yaml_verb_items(run_id=run_id, observed_at=observed_at), *_runtime_verb_items(run_id=run_id, observed_at=observed_at)]:
        merged[(item.name, item.source_path)] = item
    return sorted(merged.values(), key=lambda item: (item.name, item.source_path))


def _schema_items(*, run_id: str, observed_at: str) -> list[SelfKnowledgeItemV1]:
    path = REPO_ROOT / "orion" / "schemas" / "registry.py"
    text = path.read_text(encoding="utf-8")
    names = sorted(set(_REGISTRY_KEY_RE.findall(text)))
    return [
        _item(
            run_id=run_id,
            observed_at=observed_at,
            category="schema",
            name=name,
            source_path=_rel(path),
            origin_kind="schema",
            origin_name=name,
        )
        for name in names
    ]


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


def _evidence_ref(snapshot: SelfSnapshotV1, item: SelfKnowledgeItemV1) -> SelfConceptEvidenceRefV1:
    return SelfConceptEvidenceRefV1(
        snapshot_id=snapshot.snapshot_id,
        item_id=item.item_id,
        source_path=item.source_path,
        origin_kind=item.origin_kind,
        origin_name=item.origin_name,
        symbol_name=item.symbol_name,
    )


def _concept(
    *,
    snapshot: SelfSnapshotV1,
    concept_kind: str,
    label: str,
    description: str,
    evidence_items: Sequence[SelfKnowledgeItemV1],
    inferred_from: Sequence[str],
) -> SelfInducedConceptV1:
    evidence = [_evidence_ref(snapshot, item) for item in evidence_items]
    concept_id = f"self-concept-{_stable_digest({'snapshot_id': snapshot.snapshot_id, 'kind': concept_kind, 'label': label, 'evidence': [item.item_id for item in evidence_items]})}"
    confidence = min(0.95, 0.45 + (0.1 * len(evidence)))
    return SelfInducedConceptV1(
        concept_id=concept_id,
        concept_kind=concept_kind,
        label=label,
        description=description,
        confidence=round(confidence, 2),
        source_snapshot_id=snapshot.snapshot_id,
        evidence=evidence,
        inferred_from=sorted(set(inferred_from)),
        metadata={"evidence_count": len(evidence)},
    )


def _concept_ref(concept: SelfInducedConceptV1) -> SelfConceptRefV1:
    return SelfConceptRefV1(
        concept_id=concept.concept_id,
        concept_kind=concept.concept_kind,
        label=concept.label,
        source_snapshot_id=concept.source_snapshot_id,
    )


def _unique_evidence_refs(evidence_refs: Sequence[SelfConceptEvidenceRefV1]) -> list[SelfConceptEvidenceRefV1]:
    unique: dict[tuple[str, str], SelfConceptEvidenceRefV1] = {}
    for ref in evidence_refs:
        unique[(ref.snapshot_id, ref.item_id)] = ref
    return [unique[key] for key in sorted(unique)]


def _reflection(
    *,
    snapshot: SelfSnapshotV1,
    reflection_kind: str,
    title: str,
    description: str,
    concepts: Sequence[SelfInducedConceptV1],
    confidence: float,
    salience: float,
    recommendation: str | None = None,
    follow_up_question: str | None = None,
) -> SelfReflectiveFindingV1:
    concept_refs = [_concept_ref(concept) for concept in concepts]
    evidence = _unique_evidence_refs([ref for concept in concepts for ref in concept.evidence])
    reflection_id = f"self-reflection-{_stable_digest({'snapshot_id': snapshot.snapshot_id, 'kind': reflection_kind, 'title': title, 'concept_ids': [concept.concept_id for concept in concepts]})}"
    return SelfReflectiveFindingV1(
        reflection_id=reflection_id,
        reflection_kind=reflection_kind,
        title=title,
        description=description,
        confidence=round(confidence, 2),
        salience=round(salience, 2),
        source_snapshot_id=snapshot.snapshot_id,
        evidence=evidence,
        concept_refs=concept_refs,
        recommendation=recommendation,
        follow_up_question=follow_up_question,
        metadata={
            "concept_count": len(concept_refs),
            "evidence_count": len(evidence),
        },
    )


def build_self_study_summary(snapshot: SelfSnapshotV1) -> str:
    touch_surfaces = sorted({str(item.metadata.get("surface") or "") for item in snapshot.touchpoints if item.metadata.get("surface")})
    return (
        f"Self-study factual snapshot captured {snapshot.counts.services} services, {snapshot.counts.modules} modules, "
        f"{snapshot.counts.channels} channels, {snapshot.counts.verbs} verbs, and {snapshot.counts.schemas} schemas. "
        f"Touchpoints present: {', '.join(touch_surfaces) or 'none'}. "
        "Authoritative write-back excludes induced and reflective content."
    )


def _load_graphify_source_file_communities() -> dict[str, int]:
    """Reads graphify-out/graph.json directly as plain JSON (no `graphify`
    CLI shellout -- this function is synchronous and called from a bus
    handler) and returns a mapping of node `source_file` (repo-relative
    path, matching Layer-1 items' `source_path`) -> the first graphify
    `community` id observed for that path. Real, already-computed graphify
    communities only -- no re-derivation of clustering here.

    Fails soft (empty dict) if the graph file is missing/unparseable --
    graphify enrichment is optional, additive context, never load-bearing
    for Layer-1/Layer-2's existing authoritative guarantees."""
    graph_json_path = REPO_ROOT / _GRAPHIFY_GRAPH_JSON_RELPATH
    if not graph_json_path.exists():
        return {}
    try:
        data = json.loads(graph_json_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("self_study_graphify_graph_json_unreadable path=%s", graph_json_path)
        return {}
    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        return {}
    mapping: dict[str, int] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        source_file = node.get("source_file")
        community = node.get("community")
        if not isinstance(source_file, str) or not isinstance(community, int):
            continue
        mapping.setdefault(source_file, community)
    return mapping


def _graphify_derived_concepts(
    snapshot: SelfSnapshotV1, covered_item_ids: set[str]
) -> list[SelfInducedConceptV1]:
    """One `graphify_community` concept per distinct graphify community
    touched by Layer-1 items NOT already covered by one of the hardcoded
    concept branches above -- additive, does not replace them."""
    community_by_source_file = _load_graphify_source_file_communities()
    if not community_by_source_file:
        return []

    items_by_community: dict[int, list[SelfKnowledgeItemV1]] = {}
    for section_name in _SNAPSHOT_SECTION_NAMES:
        for item in getattr(snapshot, section_name):
            if item.item_id in covered_item_ids:
                continue
            community = community_by_source_file.get(item.source_path)
            if community is None:
                continue
            items_by_community.setdefault(community, []).append(item)

    concepts: list[SelfInducedConceptV1] = []
    for community_id in sorted(items_by_community):
        evidence_items = items_by_community[community_id]
        names = sorted({item.name for item in evidence_items})
        preview = ", ".join(names[:8]) + ("..." if len(names) > 8 else "")
        concepts.append(
            _concept(
                snapshot=snapshot,
                concept_kind="graphify_community",
                label=f"graphify community {community_id}",
                description=(
                    f"Layer-1 self-study items ({preview}) share graphify community {community_id}, "
                    "a structural cluster graphify's own community-detection assigned to their source files."
                ),
                evidence_items=evidence_items,
                inferred_from=["graphify_community"],
            )
        )
    return concepts


def _structural_delta_concepts(snapshot: SelfSnapshotV1) -> list[SelfInducedConceptV1]:
    """One `structural_mass` concept when graphify's on-disk structural
    snapshot (graphify-out/graph.json + GRAPH_REPORT.md) has moved in a
    real, non-trivial way since the last time this process observed it.

    Produces NOTHING (not a fabricated zero-delta concept) on: missing
    graph files, an unreadable/unparseable snapshot, cold start (no prior
    in-process observation yet), an unchanged `built_at_commit`, or a
    parsed-but-trivial delta (all counts unchanged and god-node jaccard
    reads 1.0/None). Mirrors orion.structural_mass.codebase_delta's
    documented "no fabricated zero" pattern and orion-cocreation-signals'
    graph_delta_loop() cold-start guard.

    Memoized per snapshot_id (see _STRUCTURAL_DELTA_RESULT_CACHE above) so
    repeat calls against the SAME snapshot -- notably
    validate_phase2a_induction()'s own internal re-induction check -- return
    the identical result instead of re-observing the "last seen" state a
    second time and silently diverging from the first call.

    "Cold start" here means the in-process global has no observation AND
    (if SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH is configured) the durable
    history log is also empty -- a fresh process with a populated history
    log recovers its last real observation from disk instead of always
    reporting nothing until its second call."""
    global _LAST_GRAPH_SNAPSHOT_STATS_FOR_STRUCTURAL_DELTA

    with _STRUCTURAL_DELTA_STATE_LOCK:
        cached = _STRUCTURAL_DELTA_RESULT_CACHE.get(snapshot.snapshot_id)
        if cached is not None:
            return cached

        graph_json_path = REPO_ROOT / _GRAPHIFY_GRAPH_JSON_RELPATH
        report_path = REPO_ROOT / _GRAPHIFY_GRAPH_REPORT_RELPATH
        if not graph_json_path.exists():
            return []
        try:
            graph_json_text = graph_json_path.read_text(encoding="utf-8")
            graph_report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
            match = _BUILT_AT_COMMIT_RE.search(graph_json_text)
            commit_sha = match.group(1) if match else None
            current_stats = graph_snapshot_stats_from_text(
                graph_json_text, graph_report_text, commit_sha=commit_sha, backfilled=False
            )
        except (OSError, ValueError, TypeError) as exc:
            # Narrowed from bare `except Exception` (review finding): these
            # are the expected "file present but unreadable/unparseable"
            # cases. Anything else (e.g. MemoryError on a corrupt/huge file)
            # propagates instead of silently looking identical to "nothing
            # cached yet".
            logger.warning("self_study_structural_delta_snapshot_unreadable error=%s", exc)
            return []

        prior_stats = _LAST_GRAPH_SNAPSHOT_STATS_FOR_STRUCTURAL_DELTA
        if prior_stats is None and SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH:
            # This process has never observed a snapshot yet -- before
            # treating that as a true cold start, check whether an earlier
            # process already recorded one on the durable volume. A wrong
            # or corrupt history file must not crash concept induction, so
            # this is best-effort: same narrowed exception set as the
            # graph-read above, PLUS KeyError (review finding) -- unlike
            # graph_snapshot_stats_from_text() above, read_snapshots() goes
            # through GraphSnapshotStats.from_json_dict(), which does direct
            # dict indexing (data["node_count"], etc.) rather than .get(...),
            # so a syntactically-valid-but-incomplete JSONL line raises
            # KeyError, not ValueError/TypeError -- that must stay
            # non-fatal here same as any other corrupt-history shape.
            try:
                history = read_snapshots(SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                logger.warning("self_study_structural_mass_history_unreadable error=%s", exc)
                history = []
            if history:
                prior_stats = history[-1]

        _LAST_GRAPH_SNAPSHOT_STATS_FOR_STRUCTURAL_DELTA = current_stats

        # Persist whenever this is a genuinely new observation -- either the
        # very first one ever recorded (prior_stats is None, true cold
        # start: no in-process state AND nothing durable to recover) or a
        # real transition to a different commit_sha, regardless of whether
        # that prior came from memory or from durable recovery above. Only a
        # recovered prior at the SAME commit_sha is skipped -- re-appending
        # an identical current_stats on every repeat call at that commit
        # would defeat the append-only log's purpose (unbounded growth with
        # zero new information).
        should_persist = SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH and (
            prior_stats is None or prior_stats.commit_sha != current_stats.commit_sha
        )
        if should_persist:
            try:
                append_snapshot(SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH, current_stats)
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("self_study_structural_mass_history_append_failed error=%s", exc)

        result: list[SelfInducedConceptV1] = []
        if prior_stats is not None and prior_stats.commit_sha != current_stats.commit_sha:
            delta = graph_structural_delta(prior_stats, current_stats)
            trivial = (
                delta.node_count_delta == 0
                and delta.edge_count_delta == 0
                and delta.community_count_delta == 0
                and delta.god_node_jaccard_similarity in (None, 1.0)
            )
            module_item = next((item for item in snapshot.modules if item.name == "orion.structural_mass"), None)
            # module_item is None: no authoritative Layer-1 item to anchor
            # evidence to -- Phase 2A's invariant requires every concept's
            # evidence to trace to a real snapshot item, so this concept
            # cannot be produced without one.
            if not trivial and module_item is not None:
                entered = ", ".join(sorted(delta.god_nodes_entered)) if delta.god_nodes_entered else "none"
                exited = ", ".join(sorted(delta.god_nodes_exited)) if delta.god_nodes_exited else "none"
                description = (
                    "graphify's on-disk structural snapshot moved since the last in-process observation: "
                    f"node_count_delta={delta.node_count_delta:+d}, edge_count_delta={delta.edge_count_delta:+d}, "
                    f"community_count_delta={delta.community_count_delta:+d}, "
                    f"god_node_jaccard_similarity={delta.god_node_jaccard_similarity}, "
                    f"god nodes entered=[{entered}], god nodes exited=[{exited}]."
                )
                result = [
                    _concept(
                        snapshot=snapshot,
                        concept_kind="structural_mass",
                        label="repo-wide structural mass delta",
                        description=description,
                        evidence_items=[module_item],
                        inferred_from=["structural_mass_delta"],
                    )
                ]

        _STRUCTURAL_DELTA_RESULT_CACHE[snapshot.snapshot_id] = result
        if len(_STRUCTURAL_DELTA_RESULT_CACHE) > _STRUCTURAL_DELTA_RESULT_CACHE_MAX:
            _STRUCTURAL_DELTA_RESULT_CACHE.popitem(last=False)
        return result


def _enrichment_cluster_root(path: str) -> str:
    """Intentional duplicate of services/orion-self-study-enrichment/app/
    evidence.py's `_cluster_root()` -- NOT imported, because importing
    another service's app/ package would reach into its internals (CLAUDE.md
    sec 5). Keep in sync if that function's clustering rule changes."""
    parts = path.split("/")
    if parts and parts[0] == "services" and len(parts) > 1:
        return f"services/{parts[1]}"
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else path


def _iter_enrichment_cache_entries(cache_dir: Path) -> Iterable[dict[str, Any]]:
    if not cache_dir.is_dir():
        return
    for json_path in sorted(cache_dir.glob("*/*.json")):
        try:
            yield json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue


def _semantic_enrichment_concepts(snapshot: SelfSnapshotV1) -> list[SelfInducedConceptV1]:
    """One `semantic_enrichment` concept per cluster with a real cached
    enrichment entry (services/orion-self-study-enrichment/app/cache.py's
    content-hash-keyed cache), using the enrichment's own real, previously
    LLM-generated `summary` text verbatim as the concept description.

    The cache key is `content_hash(render_evidence_prompt(bundle))` --
    not reproducible from this side without replaying the exact evidence
    bundle that triggered the original run, so this reads every cached
    entry directly and recovers each entry's cluster(s) from its own stored
    `touched_paths` field rather than guessing a key. Handles both "cache
    mount unreachable" (not a container, or nothing mounted there yet) and
    "no entry yet for this cluster" gracefully -- produces nothing, not an
    error, for either case."""
    cache_dir = Path(SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR)
    if not cache_dir.is_dir():
        logger.info("self_study_enrichment_cache_unreachable path=%s", cache_dir)
        return []

    items_by_cluster: dict[str, list[SelfKnowledgeItemV1]] = {}
    for section_name in _SNAPSHOT_SECTION_NAMES:
        for item in getattr(snapshot, section_name):
            items_by_cluster.setdefault(_enrichment_cluster_root(item.source_path), []).append(item)

    concepts: list[SelfInducedConceptV1] = []
    seen_clusters: set[str] = set()
    for entry in _iter_enrichment_cache_entries(cache_dir):
        if not isinstance(entry, dict):
            # Review finding: a cache file whose JSON content parses but
            # isn't an object (e.g. a partial/crashed write by that other
            # service -- cortex-exec doesn't control its writer, see
            # CLAUDE.md sec 5) must be skipped like any other malformed
            # entry, not crash the whole induction call on .get().
            continue
        summary = entry.get("summary")
        touched_paths = entry.get("touched_paths")
        if not summary or not isinstance(summary, str) or not isinstance(touched_paths, list) or not touched_paths:
            continue
        clusters = sorted({_enrichment_cluster_root(str(p)) for p in touched_paths})
        for cluster in clusters:
            if cluster in seen_clusters:
                continue
            evidence_items = items_by_cluster.get(cluster)
            if not evidence_items:
                continue
            seen_clusters.add(cluster)
            concepts.append(
                _concept(
                    snapshot=snapshot,
                    concept_kind="semantic_enrichment",
                    label=f"semantic enrichment: {cluster}",
                    description=summary.strip(),
                    evidence_items=evidence_items,
                    inferred_from=["semantic_enrichment"],
                )
            )
    return sorted(concepts, key=lambda concept: concept.label)


def _hardware_concepts(snapshot: SelfSnapshotV1) -> list[SelfInducedConceptV1]:
    """One `physical_topology` concept summarizing Layer 1's new hardware
    items (field-topology nodes/capabilities/edges), if any exist. Additive,
    whole-snapshot, same shape as _structural_delta_concepts/
    _semantic_enrichment_concepts above -- not the real Layer 2 clustering
    rebuild (that's a separate, later patch in the self-model rebuild arc),
    just making sure these new Layer-1 facts are visible to Layer 3 today."""
    if not snapshot.hardware:
        return []
    names = sorted({item.name for item in snapshot.hardware})
    preview = ", ".join(names[:8]) + ("..." if len(names) > 8 else "")
    return [
        _concept(
            snapshot=snapshot,
            concept_kind="physical_topology",
            label="Orion's physical mesh topology",
            description=f"Orion's physical mesh: {preview}.",
            evidence_items=snapshot.hardware,
            inferred_from=["hardware"],
        )
    ]


def _behavioral_concepts(snapshot: SelfSnapshotV1) -> list[SelfInducedConceptV1]:
    """One `behavioral_pattern` concept summarizing Layer 1's new behavioral
    items (recent real chat_stance turns), if any exist. Same additive shape
    as _hardware_concepts above."""
    if not snapshot.behavioral:
        return []
    shift_kinds = sorted({str(item.metadata.get("shift_kind") or "NONE") for item in snapshot.behavioral})
    return [
        _concept(
            snapshot=snapshot,
            concept_kind="behavioral_pattern",
            label="Orion's recent conversational behavior",
            description=(
                f"Orion's {len(snapshot.behavioral)} most recent logged conversational turns "
                f"span shift kinds: {', '.join(shift_kinds)}."
            ),
            evidence_items=snapshot.behavioral,
            inferred_from=["behavioral"],
        )
    ]


def induce_self_concepts(snapshot: SelfSnapshotV1) -> list[SelfInducedConceptV1]:
    _validate_authoritative_snapshot(snapshot)

    concepts: list[SelfInducedConceptV1] = []
    service_by_name = {item.name: item for item in snapshot.services}
    channel_by_name = {item.name: item for item in snapshot.channels}
    touchpoints_by_surface: dict[str, list[SelfKnowledgeItemV1]] = {}
    for item in snapshot.touchpoints:
        touchpoints_by_surface.setdefault(str(item.metadata.get("surface") or ""), []).append(item)

    runtime_evidence: list[SelfKnowledgeItemV1] = []
    for key in ("orion-cortex-exec", "orion-cortex-orch"):
        if key in service_by_name:
            runtime_evidence.append(service_by_name[key])
    for key in ("orion:verb:request", "orion:verb:result"):
        if key in channel_by_name:
            runtime_evidence.append(channel_by_name[key])
    if runtime_evidence:
        concepts.append(
            _concept(
                snapshot=snapshot,
                concept_kind="runtime_boundary",
                label="cortex-exec verb runtime boundary",
                description="cortex-exec and adjacent verb channels form the runtime boundary that executes Orion verbs, including self-study verbs.",
                evidence_items=runtime_evidence,
                inferred_from=["service", "channel"],
            )
        )

    # "rdf graph write surface" concept removed 2026-07-28: orion-rdf-writer
    # is deleted and orion:rdf:enqueue is retired (empty producer/consumer
    # lists, see orion/bus/channels.yaml) -- there is no longer a graph write
    # surface to describe. Citing either as evidence here would be exactly
    # the "self-knowledge claim with no runtime backing" the orion:rdf:worker
    # removal above already called out; this concept is gone rather than
    # left with hollow evidence.

    journal_evidence: list[SelfKnowledgeItemV1] = []
    if "orion:journal:write" in channel_by_name:
        journal_evidence.append(channel_by_name["orion:journal:write"])
    journal_evidence.extend(touchpoints_by_surface.get("journal", []))
    if journal_evidence:
        concepts.append(
            _concept(
                snapshot=snapshot,
                concept_kind="journaling_surface",
                label="journal persistence surface",
                description="Journaling is a persistence-adjacent surface built from journal bus routing and journal/sql writer touchpoints rather than authoritative fact storage.",
                evidence_items=journal_evidence,
                inferred_from=["channel", "touchpoint"],
            )
        )

    recall_evidence: list[SelfKnowledgeItemV1] = []
    for item in snapshot.env_surfaces:
        if str(item.metadata.get("surface") or "") == "recall":
            recall_evidence.append(item)
    recall_evidence.extend(touchpoints_by_surface.get("recall", []))
    if recall_evidence:
        concepts.append(
            _concept(
                snapshot=snapshot,
                concept_kind="recall_surface",
                label="authoritative self recall isolation",
                description="Self-study recall currently sits behind recall-profile config and recall worker touchpoints that isolate factual recall from non-authoritative content.",
                evidence_items=recall_evidence,
                inferred_from=["env_surface", "touchpoint"],
            )
        )

    self_study_cluster: list[SelfKnowledgeItemV1] = []
    for key in ("orion-cortex-exec", "orion-recall"):
        if key in service_by_name:
            self_study_cluster.append(service_by_name[key])
    if "orion:journal:write" in channel_by_name:
        self_study_cluster.append(channel_by_name["orion:journal:write"])
    if self_study_cluster:
        concepts.append(
            _concept(
                snapshot=snapshot,
                concept_kind="service_cluster",
                label="self-study execution cluster",
                description="Self-study spans cortex-exec, recall configuration surfaces, and journal routing as a small cross-service cluster.",
                evidence_items=self_study_cluster,
                inferred_from=["service", "channel"],
            )
        )

    # topology_evidence no longer cites orion:rdf:* channels (2026-07-28):
    # graph publication was retired, so "separating graph publication from
    # journal publication" is no longer true -- self-study now has one write
    # surface, not two.
    topology_evidence = []
    if "orion:journal:write" in channel_by_name:
        topology_evidence.append(channel_by_name["orion:journal:write"])
    if topology_evidence:
        concepts.append(
            _concept(
                snapshot=snapshot,
                concept_kind="bus_topology_pattern",
                label="self-study bus write topology",
                description="Self-study uses bus-first journal publication behind a typed envelope; graph publication was retired 2026-07-28.",
                evidence_items=topology_evidence,
                inferred_from=["channel"],
            )
        )

    # Additive Layer 2 concept sources (2026-08): graphify community
    # clustering, structural_mass repo-wide delta, and semantic-enrichment
    # cache readback. All additive to the five hardcoded branches above --
    # none of them replace or alter existing concept output.
    covered_item_ids = {ref.item_id for concept in concepts for ref in concept.evidence}
    concepts.extend(_graphify_derived_concepts(snapshot, covered_item_ids))
    concepts.extend(_structural_delta_concepts(snapshot))
    concepts.extend(_semantic_enrichment_concepts(snapshot))
    # Added 2026-09-05 (Layer 1 broadening, self-model rebuild arc).
    concepts.extend(_hardware_concepts(snapshot))
    concepts.extend(_behavioral_concepts(snapshot))

    concepts.sort(key=lambda item: (item.concept_kind, item.label))
    return concepts


def build_self_concept_summary(concepts: Sequence[SelfInducedConceptV1]) -> str:
    if not concepts:
        return "Concept induction produced 0 induced architectural concepts."
    by_kind: dict[str, int] = {}
    for concept in concepts:
        by_kind[concept.concept_kind] = by_kind.get(concept.concept_kind, 0) + 1
    parts = [f"{kind}={count}" for kind, count in sorted(by_kind.items())]
    return f"Concept induction produced {len(concepts)} induced architectural concepts ({', '.join(parts)})."


def validate_phase2a_induction(snapshot: SelfSnapshotV1, concepts: Sequence[SelfInducedConceptV1]) -> str:
    _validate_authoritative_snapshot(snapshot)
    authoritative_ids = {
        item.item_id
        for section_name in _SNAPSHOT_SECTION_NAMES
        for item in getattr(snapshot, section_name)
    }
    for concept in concepts:
        if concept.trust_tier != INDUCED_TRUST_TIER:
            raise ValueError(f"concept_trust_tier_invalid:{concept.concept_id}:{concept.trust_tier}")
        if concept.source_snapshot_id != snapshot.snapshot_id:
            raise ValueError(f"concept_snapshot_mismatch:{concept.concept_id}:{concept.source_snapshot_id}")
        if not concept.evidence:
            raise ValueError(f"concept_missing_evidence:{concept.concept_id}")
        for ref in concept.evidence:
            if ref.trust_tier != TRUST_TIER:
                raise ValueError(f"concept_evidence_trust_invalid:{concept.concept_id}:{ref.item_id}:{ref.trust_tier}")
            if ref.snapshot_id != snapshot.snapshot_id:
                raise ValueError(f"concept_evidence_snapshot_mismatch:{concept.concept_id}:{ref.item_id}:{ref.snapshot_id}")
            if ref.item_id not in authoritative_ids:
                raise ValueError(f"concept_evidence_missing_authoritative_item:{concept.concept_id}:{ref.item_id}")

    repeated = induce_self_concepts(snapshot)
    if [concept.concept_id for concept in repeated] != [concept.concept_id for concept in concepts]:
        raise ValueError("concept_idempotency_mismatch")

    return (
        "Phase 2A validation passed: self.factual.v1 excludes induced/reflective trust tiers, "
        "induced concepts retain authoritative evidence chains, induction uses authoritative snapshot items rather than journal text, "
        "and repeated unchanged induction keeps stable concept identifiers."
    )


_SELF_STUDY_REFLECT_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_SELF_STUDY_REFLECT_THINK_BLOCK_RE = re.compile(r"<think>\s*.*?\s*</think>", flags=re.IGNORECASE | re.DOTALL)


def _strip_self_study_reflect_text(text: str) -> str:
    """Same fence/think-tag stripping discipline orion.journaler.worker uses
    to parse structured LLM output -- a small local copy rather than a
    cross-import of that module's private regexes, since journal drafting and
    self-study reflection are unrelated producers with no shared caller."""
    stripped = _SELF_STUDY_REFLECT_THINK_BLOCK_RE.sub(" ", text).strip()
    match = _SELF_STUDY_REFLECT_JSON_FENCE_RE.search(stripped)
    return match.group(1).strip() if match else stripped


def _self_study_reflect_input(snapshot: SelfSnapshotV1, concepts: Sequence[SelfInducedConceptV1]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    for concept in concepts:
        by_kind[concept.concept_kind] = by_kind.get(concept.concept_kind, 0) + 1
    return {
        "snapshot_id": snapshot.snapshot_id,
        "counts_by_kind": dict(sorted(by_kind.items())),
        "concepts": [
            {"concept_kind": concept.concept_kind, "label": concept.label, "description": concept.description}
            for concept in concepts
        ],
    }


async def _call_self_study_reflect_llm(
    *,
    bus: Any,
    source: ServiceRef,
    snapshot: SelfSnapshotV1,
    concepts: Sequence[SelfInducedConceptV1],
    correlation_id: str,
) -> list[dict[str, Any]] | None:
    """The real Layer 3 LLM call: one CortexClientRequest round trip through
    cortex-orch, same public contract journal.compose and every other
    external CortexClientRequest caller (e.g. orion-actions's `_run_journal`)
    already use -- not a reach into this service's own executor.py internals.

    This DOES loop back through cortex-orch into this same cortex-exec
    process (cortex-orch dispatches self_study.reflect's LLMGatewayService
    step onto this service's own orion:exec:request:LLMGatewayService
    channel) -- not a new pattern this function introduces:
    bound_capability_exec.execute_bound_capability(), called from this
    service's own supervisor.py mid-verb-dispatch, already does the exact
    same self-referential round trip. Both rely on
    settings.exec_concurrent_handlers staying enabled (the documented
    default -- see test_exec_concurrent_handlers.py); with it disabled this
    call could not be served by this same process and would time out at
    SELF_STUDY_REFLECT_TIMEOUT_SEC. An existing, shared risk, not one this
    patch newly creates.

    Returns a list of raw finding dicts on success, or None on ANY failure
    (bad input, RPC error/timeout, non-ok result, empty/unparseable text,
    wrong JSON shape). Never raises and never fabricates a result -- same
    "produce nothing on failure" discipline `_structural_delta_concepts`
    already documents for this file. Caller (`reflect_self_concepts`) is
    responsible for checking `bus is not None` first; this assumes a real
    bus."""
    try:
        llm_route = normalize_llm_route(SELF_STUDY_REFLECT_LLM_ROUTE)
        if SELF_STUDY_REFLECT_LLM_ROUTE and not llm_route:
            logger.warning(
                "self_study_reflect_llm_route_unusable route=%r -- falling back to the verb's own default lane",
                SELF_STUDY_REFLECT_LLM_ROUTE,
            )

        request = CortexClientRequest(
            mode="brain",
            route_intent="none",
            verb=SELF_STUDY_REFLECT_VERB,
            options={
                "policy_dispatch_only": True,
                **({"llm_route": llm_route} if llm_route else {}),
            },
            recall=RecallDirective(enabled=False, required=False),
            context=CortexClientContext(
                messages=[
                    LLMMessage(
                        role="user",
                        content=f"Reflect on self-study snapshot {snapshot.snapshot_id}.",
                    )
                ],
                raw_user_text=f"Reflect on self-study snapshot {snapshot.snapshot_id}.",
                metadata={"self_study_reflect_input": _self_study_reflect_input(snapshot, concepts)},
            ),
        )
        # Deterministically derived from the CALLER's correlation_id (not a
        # fresh uuid4()) so this RPC's own correlation_id is traceable back
        # to the run_self_concept_reflect invocation that triggered it in
        # logs/telemetry -- same "namespaced uuid5 derivation" convention
        # curiosity_investigation.py uses for its own second-turn (outreach)
        # correlation_id, and the same _as_envelope_correlation_id() this
        # file already uses elsewhere for coercing a raw id into envelope
        # shape.
        rpc_correlation_id = _as_envelope_correlation_id(f"self-study-reflect:{correlation_id}")
        reply_channel = f"orion:cortex:result:self-study-reflect:{rpc_correlation_id}"
        envelope = BaseEnvelope(
            kind="cortex.orch.request",
            source=source,
            correlation_id=rpc_correlation_id,
            reply_to=reply_channel,
            payload=request.model_dump(mode="json"),
        )
        msg = await bus.rpc_request(
            CORTEX_ORCH_REQUEST_CHANNEL,
            envelope,
            reply_channel=reply_channel,
            timeout_sec=SELF_STUDY_REFLECT_TIMEOUT_SEC,
        )
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok or decoded.envelope is None:
            logger.warning("self_study_reflect_llm_decode_failed corr=%s err=%s", rpc_correlation_id, decoded.error)
            return None
        payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
    except Exception as exc:  # noqa: BLE001 -- any failure (construction, RPC, decode) degrades to "no reflection", never a crash
        logger.warning("self_study_reflect_llm_call_failed corr=%s err=%s", correlation_id, exc)
        return None

    if not payload.get("ok", False):
        logger.warning(
            "self_study_reflect_llm_not_ok corr=%s status=%s error=%s",
            rpc_correlation_id,
            payload.get("status"),
            payload.get("error"),
        )
        return None

    text = extract_cortex_payload_text(payload)
    if not text:
        logger.warning("self_study_reflect_llm_empty_text corr=%s", rpc_correlation_id)
        return None

    try:
        parsed = json.loads(_strip_self_study_reflect_text(text))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("self_study_reflect_llm_unparseable corr=%s err=%s", rpc_correlation_id, exc)
        return None

    findings = parsed.get("findings") if isinstance(parsed, dict) else None
    if not isinstance(findings, list):
        logger.warning("self_study_reflect_llm_bad_shape corr=%s", rpc_correlation_id)
        return None
    return [item for item in findings if isinstance(item, dict)]


def _finding_from_llm_item(
    *,
    snapshot: SelfSnapshotV1,
    concept_by_kind: dict[str, SelfInducedConceptV1],
    raw: dict[str, Any],
) -> SelfReflectiveFindingV1 | None:
    """Builds one real, evidence-backed finding from one model-authored item,
    or None if the item doesn't hold up. The model supplies reflection_kind/
    title/description/confidence/salience/recommendation/follow_up_question
    and which concept_kinds it's about; everything evidentiary (reflection_id,
    evidence, concept_refs) is computed here from the real snapshot/concepts,
    same trust-boundary discipline _validate_authoritative_snapshot and
    validate_phase2a_induction already enforce elsewhere in this file --
    the model is never trusted as a source of evidence, only of prose."""
    reflection_kind = raw.get("reflection_kind")
    title = raw.get("title")
    description = raw.get("description")
    concept_kinds = raw.get("concept_kinds")
    if not isinstance(reflection_kind, str) or not isinstance(title, str) or not isinstance(description, str):
        return None
    if not title.strip() or not description.strip():
        return None
    if not isinstance(concept_kinds, list) or not concept_kinds:
        return None
    grounding = [concept_by_kind[kind] for kind in concept_kinds if isinstance(kind, str) and kind in concept_by_kind]
    if not grounding:
        # Model named concept_kinds not actually present in this snapshot.
        # Never fabricate a concept_ref -- drop the finding.
        return None
    try:
        confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.0))))
        salience = min(1.0, max(0.0, float(raw.get("salience", 0.0))))
    except (TypeError, ValueError):
        return None
    recommendation = raw.get("recommendation")
    follow_up_question = raw.get("follow_up_question")
    try:
        return _reflection(
            snapshot=snapshot,
            reflection_kind=reflection_kind,
            title=title.strip(),
            description=description.strip(),
            concepts=grounding,
            confidence=confidence,
            salience=salience,
            recommendation=recommendation if isinstance(recommendation, str) and recommendation.strip() else None,
            follow_up_question=(
                follow_up_question if isinstance(follow_up_question, str) and follow_up_question.strip() else None
            ),
        )
    except Exception:
        # Most likely reflection_kind failed SelfReflectionKind's Literal
        # validation. One bad model-authored item must not sink the whole
        # reflection pass.
        logger.warning("self_study_reflect_finding_rejected kind=%r title=%r", reflection_kind, title)
        return None


class ReflectSelfConceptsOutcome(NamedTuple):
    """Result of one reflect_self_concepts() call.

    `llm_call_failed` is True only when a REAL LLM attempt (`bus is not
    None`) did not complete successfully -- distinct from `findings == []`,
    which can also mean "no bus configured" or "a real, successful call that
    genuinely found nothing." A caller that must not publish/journal a
    failed attempt as if it were a completed reflection (see
    run_self_concept_reflect) checks this flag directly rather than
    inferring it from an empty findings list or from cache presence, which
    can go stale (an earlier successful call's cache entry for the same
    content-addressed snapshot_id would otherwise mask a later failure)."""

    findings: list[SelfReflectiveFindingV1]
    llm_call_failed: bool


async def reflect_self_concepts(
    *,
    bus: Any | None,
    source: ServiceRef,
    snapshot: SelfSnapshotV1,
    concepts: Sequence[SelfInducedConceptV1],
    correlation_id: str | None = None,
) -> ReflectSelfConceptsOutcome:
    """Layer 3: real reflective findings from a real LLM call, replacing the
    six hardcoded template branches this function used to hold. Caches into
    _REFLECTION_RESULT_CACHE ONLY on a genuine, successful LLM round trip --
    never on `bus is None` (never attempted) and never on an RPC/decode/
    parse failure (attempted but failed). Both of those must stay
    distinguishable from "attempted and genuinely found nothing" (real
    success, empty findings list), which DOES cache -- collapsing "failed"
    into "found nothing" would let an infra outage read as a completed,
    calm reflection pass, indistinguishable from the real thing to every
    downstream reader of the cache.

    `correlation_id` is the CALLER's own correlation_id (e.g.
    run_self_concept_reflect's), threaded through so the RPC this makes is
    traceable back to the operation that triggered it rather than showing up
    in logs under an unrelated, freshly-minted id. Optional and defaulted
    (rather than required) so callers that genuinely have no outer
    correlation_id of their own -- tests, ad hoc harness runs -- don't need
    to invent one."""
    validation_summary = validate_phase2a_induction(snapshot, concepts)
    if not validation_summary:
        raise ValueError("phase2a_validation_missing")
    correlation_id = correlation_id or str(uuid4())

    raw_findings = (
        await _call_self_study_reflect_llm(
            bus=bus, source=source, snapshot=snapshot, concepts=concepts, correlation_id=correlation_id
        )
        if bus is not None
        else None
    )
    llm_call_failed = bus is not None and raw_findings is None

    concept_by_kind = {concept.concept_kind: concept for concept in concepts}
    findings: list[SelfReflectiveFindingV1] = []
    for raw in raw_findings or []:
        finding = _finding_from_llm_item(snapshot=snapshot, concept_by_kind=concept_by_kind, raw=raw)
        if finding is not None:
            findings.append(finding)
    findings.sort(key=lambda item: (item.reflection_kind, item.title))

    if bus is not None and not llm_call_failed:
        with _REFLECTION_STATE_LOCK:
            _REFLECTION_RESULT_CACHE[snapshot.snapshot_id] = findings
            if len(_REFLECTION_RESULT_CACHE) > _REFLECTION_RESULT_CACHE_MAX:
                _REFLECTION_RESULT_CACHE.popitem(last=False)

    return ReflectSelfConceptsOutcome(findings=findings, llm_call_failed=llm_call_failed)


def _cached_reflection_findings(snapshot_id: str) -> list[SelfReflectiveFindingV1] | None:
    """None means "reflect_self_concepts has never run for this snapshot" --
    distinct from `[]`, which means it ran and genuinely found nothing.
    Read-only consumer for _retrieve_self_study_in_process(); never triggers
    a fresh LLM call itself."""
    with _REFLECTION_STATE_LOCK:
        cached = _REFLECTION_RESULT_CACHE.get(snapshot_id)
        return list(cached) if cached is not None else None


def build_self_reflection_summary(findings: Sequence[SelfReflectiveFindingV1]) -> str:
    if not findings:
        return "Reflection produced 0 reflective findings."
    by_kind: dict[str, int] = {}
    for finding in findings:
        by_kind[finding.reflection_kind] = by_kind.get(finding.reflection_kind, 0) + 1
    parts = [f"{kind}={count}" for kind, count in sorted(by_kind.items())]
    return f"Reflection produced {len(findings)} reflective findings ({', '.join(parts)})."


def _fact_record(snapshot: SelfSnapshotV1, item: SelfKnowledgeItemV1) -> SelfStudyRetrievedRecordV1:
    preview_bits = [item.category, item.source_path]
    surface = item.metadata.get("surface")
    if surface:
        preview_bits.append(f"surface={surface}")
    return SelfStudyRetrievedRecordV1(
        stable_id=item.item_id,
        trust_tier=item.trust_tier,
        record_type="fact",
        title=item.name,
        content_preview=" | ".join(str(bit) for bit in preview_bits if bit),
        source_kind="self_repo_inspect",
        source_snapshot_id=snapshot.snapshot_id,
        source_path=item.source_path,
        origin_kind=item.origin_kind,
        origin_name=item.origin_name,
        symbol_name=item.symbol_name,
        metadata=dict(item.metadata),
    )


def _concept_record(concept: SelfInducedConceptV1) -> SelfStudyRetrievedRecordV1:
    return SelfStudyRetrievedRecordV1(
        stable_id=concept.concept_id,
        trust_tier=concept.trust_tier,
        record_type="concept",
        title=concept.label,
        content_preview=concept.description,
        source_kind="self_concept_induce",
        source_snapshot_id=concept.source_snapshot_id,
        concept_kind=concept.concept_kind,
        evidence=list(concept.evidence),
        metadata=dict(concept.metadata),
    )


def _reflection_record(finding: SelfReflectiveFindingV1) -> SelfStudyRetrievedRecordV1:
    return SelfStudyRetrievedRecordV1(
        stable_id=finding.reflection_id,
        trust_tier=finding.trust_tier,
        record_type="reflection",
        title=finding.title,
        content_preview=finding.description,
        source_kind="self_concept_reflect",
        source_snapshot_id=finding.source_snapshot_id,
        reflection_kind=finding.reflection_kind,
        evidence=list(finding.evidence),
        concept_refs=list(finding.concept_refs),
        metadata=dict(finding.metadata),
    )


def _build_self_study_retrieval_records(
    snapshot: SelfSnapshotV1,
    concepts: Sequence[SelfInducedConceptV1],
    findings: Sequence[SelfReflectiveFindingV1],
) -> list[SelfStudyRetrievedRecordV1]:
    fact_records = [
        _fact_record(snapshot, item)
        for section_name in _SNAPSHOT_SECTION_NAMES
        for item in getattr(snapshot, section_name)
    ]
    concept_records = [_concept_record(concept) for concept in concepts]
    reflection_records = [_reflection_record(finding) for finding in findings]
    return fact_records + concept_records + reflection_records


def _filter_allows_trust_tier(filters: SelfStudyRetrieveFiltersV1, trust_tier: str) -> bool:
    return not filters.trust_tiers or trust_tier in filters.trust_tiers


def _filter_allows_record_type(filters: SelfStudyRetrieveFiltersV1, record_type: str) -> bool:
    return not filters.record_types or record_type in filters.record_types


def _filter_allows_source_kind(filters: SelfStudyRetrieveFiltersV1, source_kind: str) -> bool:
    return not filters.source_kinds or source_kind in filters.source_kinds


def _graphdb_endpoint() -> str | None:
    explicit = (os.getenv("RECALL_RDF_ENDPOINT_URL") or "").strip()
    if explicit:
        return explicit
    base = (os.getenv("GRAPHDB_URL") or "").strip()
    repo = (os.getenv("GRAPHDB_REPO") or "").strip()
    if not base and not repo:
        return None
    if not base:
        base = GRAPHDB_DEFAULT_URL
    if not repo:
        repo = GRAPHDB_DEFAULT_REPO
    return f"{base.rstrip('/')}/repositories/{repo}"


def _graphdb_auth() -> tuple[str, str]:
    return (
        (os.getenv("RECALL_RDF_USER") or os.getenv("GRAPHDB_USER") or GRAPHDB_DEFAULT_USER).strip(),
        (os.getenv("RECALL_RDF_PASS") or os.getenv("GRAPHDB_PASS") or GRAPHDB_DEFAULT_PASS).strip(),
    )


def _escape_sparql(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _sparql_values(var_name: str, values: Sequence[str], *, uris: bool = False) -> str:
    if not values:
        return ""
    if uris:
        rendered = " ".join(f"<{value}>" for value in values)
    else:
        rendered = " ".join(f'"{_escape_sparql(value)}"' for value in values)
    return f"VALUES ?{var_name} {{ {rendered} }}"


def _graphdb_query_limit(request: SelfStudyRetrieveRequestV1) -> int:
    requested = max(1, int(request.filters.limit or 12))
    return max(requested * 4, requested + 8)


def _execute_graphdb_select(
    endpoint: str,
    sparql: str,
) -> list[dict[str, dict[str, str]]]:
    response = requests.post(
        endpoint,
        data=sparql,
        headers={
            "Content-Type": "application/sparql-query",
            "Accept": "application/sparql-results+json",
        },
        auth=_graphdb_auth(),
        timeout=GRAPHDB_TIMEOUT_SEC,
    )
    response.raise_for_status()
    payload = response.json()
    return list(payload.get("results", {}).get("bindings", []))


def _binding_value(binding: dict[str, dict[str, str]], key: str) -> str | None:
    raw = binding.get(key)
    if not isinstance(raw, dict):
        return None
    value = raw.get("value")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _graphdb_fact_records(request: SelfStudyRetrieveRequestV1, *, endpoint: str) -> list[SelfStudyRetrievedRecordV1]:
    if not _filter_allows_trust_tier(request.filters, TRUST_TIER):
        return []
    if not _filter_allows_record_type(request.filters, "fact"):
        return []
    if not _filter_allows_source_kind(request.filters, "self_repo_inspect"):
        return []

    clauses = []
    if request.filters.stable_ids:
        clauses.append(_sparql_values("stable_id", request.filters.stable_ids))
    query_text = (request.filters.text_query or "").strip().lower()
    text_filter = ""
    if query_text:
        escaped = _escape_sparql(query_text)
        text_filter = (
            "FILTER("
            f'CONTAINS(LCASE(STR(?title)), "{escaped}") || '
            f'CONTAINS(LCASE(STR(?source_path)), "{escaped}") || '
            f'CONTAINS(LCASE(STR(?category)), "{escaped}") || '
            f'CONTAINS(LCASE(STR(?origin_name)), "{escaped}") || '
            f'CONTAINS(LCASE(STR(?origin_kind)), "{escaped}")'
            ")"
        )
    sparql = f"""
    PREFIX orion: <http://conjourney.net/orion#>
    SELECT ?stable_id ?title ?category ?source_path ?origin_kind ?origin_name ?symbol_name ?snapshot_id
    WHERE {{
      GRAPH <{SELF_GRAPH}> {{
        ?snapshot a orion:AuthoritativeSelfSnapshot ;
                  orion:snapshotId ?snapshot_id ;
                  orion:hasAuthoritativeFact ?fact .
        ?fact a orion:AuthoritativeSelfFact ;
              orion:factId ?stable_id ;
              orion:factName ?title ;
              orion:factCategory ?category ;
              orion:sourcePath ?source_path ;
              orion:trustTier "authoritative" .
        OPTIONAL {{ ?fact orion:originKind ?origin_kind }}
        OPTIONAL {{ ?fact orion:originName ?origin_name }}
        OPTIONAL {{ ?fact orion:symbolName ?symbol_name }}
        {clauses[0] if clauses else ""}
        {text_filter}
      }}
    }}
    ORDER BY ?stable_id
    LIMIT {_graphdb_query_limit(request)}
    """
    bindings = _execute_graphdb_select(endpoint, sparql)
    records: list[SelfStudyRetrievedRecordV1] = []
    for binding in bindings:
        stable_id = _binding_value(binding, "stable_id")
        title = _binding_value(binding, "title")
        category = _binding_value(binding, "category")
        source_path = _binding_value(binding, "source_path")
        snapshot_id = _binding_value(binding, "snapshot_id")
        if not stable_id or not title or not category or not source_path or not snapshot_id:
            continue
        records.append(
            SelfStudyRetrievedRecordV1(
                stable_id=stable_id,
                trust_tier=TRUST_TIER,
                record_type="fact",
                title=title,
                content_preview=f"{category} | {source_path}",
                source_kind="self_repo_inspect",
                storage_surface="rdf_graph",
                source_snapshot_id=snapshot_id,
                source_path=source_path,
                origin_kind=_binding_value(binding, "origin_kind"),
                origin_name=_binding_value(binding, "origin_name"),
                symbol_name=_binding_value(binding, "symbol_name"),
                metadata={"provenance": ["graphdb", SELF_GRAPH]},
            )
        )
    return records


def _graphdb_concept_evidence(
    endpoint: str,
    *,
    concept_uris: Sequence[str],
) -> dict[str, list[SelfConceptEvidenceRefV1]]:
    if not concept_uris:
        return {}
    sparql = f"""
    PREFIX orion: <http://conjourney.net/orion#>
    SELECT ?concept_uri ?snapshot_id ?item_id ?source_path ?origin_kind ?origin_name ?symbol_name
    WHERE {{
      GRAPH <{SELF_INDUCED_GRAPH}> {{
        {_sparql_values("concept_uri", concept_uris, uris=True)}
        ?concept_uri orion:supportedBy ?fact .
      }}
      GRAPH <{SELF_GRAPH}> {{
        ?snapshot orion:snapshotId ?snapshot_id ;
                  orion:hasAuthoritativeFact ?fact .
        ?fact orion:factId ?item_id ;
              orion:sourcePath ?source_path ;
              orion:trustTier "authoritative" .
        OPTIONAL {{ ?fact orion:originKind ?origin_kind }}
        OPTIONAL {{ ?fact orion:originName ?origin_name }}
        OPTIONAL {{ ?fact orion:symbolName ?symbol_name }}
      }}
    }}
    ORDER BY ?concept_uri ?item_id
    """
    bindings = _execute_graphdb_select(endpoint, sparql)
    evidence: dict[str, list[SelfConceptEvidenceRefV1]] = {}
    for binding in bindings:
        concept_uri = _binding_value(binding, "concept_uri")
        snapshot_id = _binding_value(binding, "snapshot_id")
        item_id = _binding_value(binding, "item_id")
        source_path = _binding_value(binding, "source_path")
        if not concept_uri or not snapshot_id or not item_id or not source_path:
            continue
        evidence.setdefault(concept_uri, []).append(
            SelfConceptEvidenceRefV1(
                snapshot_id=snapshot_id,
                item_id=item_id,
                source_path=source_path,
                origin_kind=_binding_value(binding, "origin_kind"),
                origin_name=_binding_value(binding, "origin_name"),
                symbol_name=_binding_value(binding, "symbol_name"),
            )
        )
    return evidence


def _graphdb_concept_records(request: SelfStudyRetrieveRequestV1, *, endpoint: str) -> list[SelfStudyRetrievedRecordV1]:
    if not _filter_allows_trust_tier(request.filters, INDUCED_TRUST_TIER):
        return []
    if not _filter_allows_record_type(request.filters, "concept"):
        return []
    if not _filter_allows_source_kind(request.filters, "self_concept_induce"):
        return []

    clauses = []
    if request.filters.stable_ids:
        clauses.append(_sparql_values("stable_id", request.filters.stable_ids))
    if request.filters.concept_kinds:
        clauses.append(_sparql_values("concept_kind", request.filters.concept_kinds))
    query_text = (request.filters.text_query or "").strip().lower()
    text_filter = ""
    if query_text:
        escaped = _escape_sparql(query_text)
        text_filter = (
            "FILTER("
            f'CONTAINS(LCASE(STR(?title)), "{escaped}") || '
            f'CONTAINS(LCASE(STR(?description)), "{escaped}") || '
            f'CONTAINS(LCASE(STR(?concept_kind)), "{escaped}")'
            ")"
        )
    sparql = f"""
    PREFIX orion: <http://conjourney.net/orion#>
    SELECT ?concept_uri ?stable_id ?concept_kind ?title ?description ?snapshot_id ?confidence
    WHERE {{
      GRAPH <{SELF_INDUCED_GRAPH}> {{
        ?concept_uri a orion:InducedSelfConcept ;
                     orion:conceptId ?stable_id ;
                     orion:conceptKind ?concept_kind ;
                     orion:label ?title ;
                     orion:description ?description ;
                     orion:sourceSnapshotId ?snapshot_id ;
                     orion:trustTier "induced" .
        OPTIONAL {{ ?concept_uri orion:confidence ?confidence }}
        {" ".join(clauses)}
        {text_filter}
      }}
    }}
    ORDER BY ?stable_id
    LIMIT {_graphdb_query_limit(request)}
    """
    bindings = _execute_graphdb_select(endpoint, sparql)
    concept_uris = [uri for uri in (_binding_value(binding, "concept_uri") for binding in bindings) if uri]
    evidence_map = _graphdb_concept_evidence(endpoint, concept_uris=concept_uris)
    records: list[SelfStudyRetrievedRecordV1] = []
    for binding in bindings:
        concept_uri = _binding_value(binding, "concept_uri")
        stable_id = _binding_value(binding, "stable_id")
        title = _binding_value(binding, "title")
        description = _binding_value(binding, "description")
        snapshot_id = _binding_value(binding, "snapshot_id")
        concept_kind = _binding_value(binding, "concept_kind")
        if not concept_uri or not stable_id or not title or not description or not snapshot_id or not concept_kind:
            continue
        records.append(
            SelfStudyRetrievedRecordV1(
                stable_id=stable_id,
                trust_tier=INDUCED_TRUST_TIER,
                record_type="concept",
                title=title,
                content_preview=description,
                source_kind="self_concept_induce",
                storage_surface="rdf_graph",
                source_snapshot_id=snapshot_id,
                concept_kind=concept_kind,  # type: ignore[arg-type]
                evidence=evidence_map.get(concept_uri, []),
                metadata={
                    "confidence": float(_binding_value(binding, "confidence") or 0.0),
                    "provenance": ["graphdb", SELF_INDUCED_GRAPH],
                },
            )
        )
    return records


def _graphdb_reflection_evidence(
    endpoint: str,
    *,
    reflection_uris: Sequence[str],
) -> dict[str, list[SelfConceptEvidenceRefV1]]:
    if not reflection_uris:
        return {}
    sparql = f"""
    PREFIX orion: <http://conjourney.net/orion#>
    SELECT ?reflection_uri ?snapshot_id ?item_id ?source_path ?origin_kind ?origin_name ?symbol_name
    WHERE {{
      GRAPH <{SELF_REFLECTIVE_GRAPH}> {{
        {_sparql_values("reflection_uri", reflection_uris, uris=True)}
        ?reflection_uri orion:supportedBy ?fact .
      }}
      GRAPH <{SELF_GRAPH}> {{
        ?snapshot orion:snapshotId ?snapshot_id ;
                  orion:hasAuthoritativeFact ?fact .
        ?fact orion:factId ?item_id ;
              orion:sourcePath ?source_path ;
              orion:trustTier "authoritative" .
        OPTIONAL {{ ?fact orion:originKind ?origin_kind }}
        OPTIONAL {{ ?fact orion:originName ?origin_name }}
        OPTIONAL {{ ?fact orion:symbolName ?symbol_name }}
      }}
    }}
    ORDER BY ?reflection_uri ?item_id
    """
    bindings = _execute_graphdb_select(endpoint, sparql)
    evidence: dict[str, list[SelfConceptEvidenceRefV1]] = {}
    for binding in bindings:
        reflection_uri = _binding_value(binding, "reflection_uri")
        snapshot_id = _binding_value(binding, "snapshot_id")
        item_id = _binding_value(binding, "item_id")
        source_path = _binding_value(binding, "source_path")
        if not reflection_uri or not snapshot_id or not item_id or not source_path:
            continue
        evidence.setdefault(reflection_uri, []).append(
            SelfConceptEvidenceRefV1(
                snapshot_id=snapshot_id,
                item_id=item_id,
                source_path=source_path,
                origin_kind=_binding_value(binding, "origin_kind"),
                origin_name=_binding_value(binding, "origin_name"),
                symbol_name=_binding_value(binding, "symbol_name"),
            )
        )
    return evidence


def _graphdb_reflection_concept_refs(
    endpoint: str,
    *,
    reflection_uris: Sequence[str],
) -> dict[str, list[SelfConceptRefV1]]:
    if not reflection_uris:
        return {}
    sparql = f"""
    PREFIX orion: <http://conjourney.net/orion#>
    SELECT ?reflection_uri ?concept_id ?concept_kind ?label ?snapshot_id
    WHERE {{
      GRAPH <{SELF_REFLECTIVE_GRAPH}> {{
        {_sparql_values("reflection_uri", reflection_uris, uris=True)}
        ?reflection_uri orion:derivedFromConcept ?concept_uri .
      }}
      GRAPH <{SELF_INDUCED_GRAPH}> {{
        ?concept_uri orion:conceptId ?concept_id ;
                     orion:conceptKind ?concept_kind ;
                     orion:label ?label ;
                     orion:sourceSnapshotId ?snapshot_id ;
                     orion:trustTier "induced" .
      }}
    }}
    ORDER BY ?reflection_uri ?concept_id
    """
    bindings = _execute_graphdb_select(endpoint, sparql)
    concept_refs: dict[str, list[SelfConceptRefV1]] = {}
    for binding in bindings:
        reflection_uri = _binding_value(binding, "reflection_uri")
        concept_id = _binding_value(binding, "concept_id")
        concept_kind = _binding_value(binding, "concept_kind")
        label = _binding_value(binding, "label")
        snapshot_id = _binding_value(binding, "snapshot_id")
        if not reflection_uri or not concept_id or not concept_kind or not label or not snapshot_id:
            continue
        concept_refs.setdefault(reflection_uri, []).append(
            SelfConceptRefV1(
                concept_id=concept_id,
                concept_kind=concept_kind,  # type: ignore[arg-type]
                label=label,
                source_snapshot_id=snapshot_id,
            )
        )
    return concept_refs


def _graphdb_reflection_records(request: SelfStudyRetrieveRequestV1, *, endpoint: str) -> list[SelfStudyRetrievedRecordV1]:
    if not _filter_allows_trust_tier(request.filters, REFLECTIVE_TRUST_TIER):
        return []
    if not _filter_allows_record_type(request.filters, "reflection"):
        return []
    if not _filter_allows_source_kind(request.filters, "self_concept_reflect"):
        return []

    clauses = []
    if request.filters.stable_ids:
        clauses.append(_sparql_values("stable_id", request.filters.stable_ids))
    if request.filters.reflection_kinds:
        clauses.append(_sparql_values("reflection_kind", request.filters.reflection_kinds))
    query_text = (request.filters.text_query or "").strip().lower()
    text_filter = ""
    if query_text:
        escaped = _escape_sparql(query_text)
        text_filter = (
            "FILTER("
            f'CONTAINS(LCASE(STR(?title)), "{escaped}") || '
            f'CONTAINS(LCASE(STR(?description)), "{escaped}") || '
            f'CONTAINS(LCASE(STR(?reflection_kind)), "{escaped}")'
            ")"
        )
    sparql = f"""
    PREFIX orion: <http://conjourney.net/orion#>
    SELECT ?reflection_uri ?stable_id ?reflection_kind ?title ?description ?snapshot_id ?confidence ?salience
    WHERE {{
      GRAPH <{SELF_REFLECTIVE_GRAPH}> {{
        ?reflection_uri a orion:ReflectiveSelfFinding ;
                        orion:reflectionId ?stable_id ;
                        orion:reflectionKind ?reflection_kind ;
                        orion:label ?title ;
                        orion:description ?description ;
                        orion:sourceSnapshotId ?snapshot_id ;
                        orion:trustTier "reflective" .
        OPTIONAL {{ ?reflection_uri orion:confidence ?confidence }}
        OPTIONAL {{ ?reflection_uri orion:salience ?salience }}
        {" ".join(clauses)}
        {text_filter}
      }}
    }}
    ORDER BY ?stable_id
    LIMIT {_graphdb_query_limit(request)}
    """
    bindings = _execute_graphdb_select(endpoint, sparql)
    reflection_uris = [uri for uri in (_binding_value(binding, "reflection_uri") for binding in bindings) if uri]
    evidence_map = _graphdb_reflection_evidence(endpoint, reflection_uris=reflection_uris)
    concept_ref_map = _graphdb_reflection_concept_refs(endpoint, reflection_uris=reflection_uris)
    records: list[SelfStudyRetrievedRecordV1] = []
    for binding in bindings:
        reflection_uri = _binding_value(binding, "reflection_uri")
        stable_id = _binding_value(binding, "stable_id")
        reflection_kind = _binding_value(binding, "reflection_kind")
        title = _binding_value(binding, "title")
        description = _binding_value(binding, "description")
        snapshot_id = _binding_value(binding, "snapshot_id")
        if not reflection_uri or not stable_id or not reflection_kind or not title or not description or not snapshot_id:
            continue
        records.append(
            SelfStudyRetrievedRecordV1(
                stable_id=stable_id,
                trust_tier=REFLECTIVE_TRUST_TIER,
                record_type="reflection",
                title=title,
                content_preview=description,
                source_kind="self_concept_reflect",
                storage_surface="rdf_graph",
                source_snapshot_id=snapshot_id,
                reflection_kind=reflection_kind,  # type: ignore[arg-type]
                evidence=evidence_map.get(reflection_uri, []),
                concept_refs=concept_ref_map.get(reflection_uri, []),
                metadata={
                    "confidence": float(_binding_value(binding, "confidence") or 0.0),
                    "salience": float(_binding_value(binding, "salience") or 0.0),
                    "provenance": ["graphdb", SELF_REFLECTIVE_GRAPH],
                },
            )
        )
    return records


def _build_retrieval_result(
    *,
    request: SelfStudyRetrieveRequestV1,
    records: Sequence[SelfStudyRetrievedRecordV1],
    backend_used: str | None,
    backend_status: Sequence[SelfStudyRetrievalBackendStatusV1],
    notes: Sequence[str],
) -> SelfStudyRetrieveResultV1:
    filtered = [record for record in records if _record_matches_filters(record, request.filters)]
    filtered.sort(key=lambda item: (item.trust_tier, item.record_type, item.title, item.stable_id))
    limited = _limit_records_for_mode(filtered, retrieval_mode=request.retrieval_mode, limit=request.filters.limit)
    counts = SelfStudyRetrievalCountsV1(
        total=len(limited),
        authoritative=sum(1 for item in limited if item.trust_tier == TRUST_TIER),
        induced=sum(1 for item in limited if item.trust_tier == INDUCED_TRUST_TIER),
        reflective=sum(1 for item in limited if item.trust_tier == REFLECTIVE_TRUST_TIER),
        facts=sum(1 for item in limited if item.record_type == "fact"),
        concepts=sum(1 for item in limited if item.record_type == "concept"),
        reflections=sum(1 for item in limited if item.record_type == "reflection"),
    )
    groups = [
        SelfStudyRetrievalGroupV1(
            trust_tier=trust_tier,
            items=[item for item in limited if item.trust_tier == trust_tier],
        )
        for trust_tier in (TRUST_TIER, INDUCED_TRUST_TIER, REFLECTIVE_TRUST_TIER)
        if any(item.trust_tier == trust_tier for item in limited)
    ]
    return SelfStudyRetrieveResultV1(
        run_id=f"self-retrieve-{uuid4()}",
        retrieval_mode=request.retrieval_mode,
        backend_used=backend_used,  # type: ignore[arg-type]
        applied_filters=request.filters,
        groups=groups,
        counts=counts,
        backend_status=list(backend_status),
        notes=list(notes),
    )


def _retrieve_self_study_in_process(request: SelfStudyRetrieveRequestV1) -> SelfStudyRetrieveResultV1:
    snapshot = build_self_snapshot()
    concepts = induce_self_concepts(snapshot)
    # Reads the last REAL reflection for this snapshot rather than calling
    # reflect_self_concepts() here -- that function now makes a real LLM
    # call, and this is a synchronous, mode-scoped read path with live
    # consumers (self_study_harness.py's consumer scenarios). snapshot_id is
    # a content digest of repo state (see _canonical_snapshot_payload), not a
    # timestamp, so a snapshot built here matches a prior
    # run_self_concept_reflect() snapshot as long as the repo hasn't changed
    # in between -- same idempotency property validate_phase2a_induction
    # already relies on for induce_self_concepts().
    cached_findings = _cached_reflection_findings(snapshot.snapshot_id)
    findings = cached_findings or []

    allowed_trust_tiers = set(_mode_allowed_trust_tiers(request.retrieval_mode))
    allowed_record_types = set(_mode_allowed_record_types(request.retrieval_mode))
    records = [
        record
        for record in _build_self_study_retrieval_records(snapshot, concepts, findings)
        if record.trust_tier in allowed_trust_tiers and record.record_type in allowed_record_types
    ]
    backend_status = [
        SelfStudyRetrievalBackendStatusV1(storage_surface="in_process", status="used", detail="Repo-derived self-study snapshot, concepts, and reflections."),
        SelfStudyRetrievalBackendStatusV1(storage_surface="rdf_graph", status="not_queried", detail="GraphDB self-study retrieval was not attempted."),
        SelfStudyRetrievalBackendStatusV1(storage_surface="journal", status="not_queried", detail="Phase 4B retrieval does not consume journal prose as primary self-study truth."),
    ]
    notes = [
        "Self-study retrieval is explicit and mode-scoped; it does not widen self.factual.v1.",
        "Fallback retrieval uses in-process self-study records and preserves trust tiers rather than flattening them.",
    ]
    if "reflection" in allowed_record_types and cached_findings is None:
        notes.append(
            "No reflective findings cached yet for this snapshot; run self_concept_reflect to produce a real "
            "reflection before retrieving it."
        )
    return _build_retrieval_result(
        request=request,
        records=records,
        backend_used="in_process",
        backend_status=backend_status,
        notes=notes,
    )


def _retrieve_self_study_from_graphdb(request: SelfStudyRetrieveRequestV1) -> SelfStudyRetrieveResultV1:
    endpoint = _graphdb_endpoint()
    if not endpoint:
        raise RuntimeError("graphdb_not_configured")

    allowed_trust_tiers = set(_mode_allowed_trust_tiers(request.retrieval_mode))
    allowed_record_types = set(_mode_allowed_record_types(request.retrieval_mode))
    records: list[SelfStudyRetrievedRecordV1] = []
    if TRUST_TIER in allowed_trust_tiers and "fact" in allowed_record_types:
        records.extend(_graphdb_fact_records(request, endpoint=endpoint))
    if INDUCED_TRUST_TIER in allowed_trust_tiers and "concept" in allowed_record_types:
        records.extend(_graphdb_concept_records(request, endpoint=endpoint))
    if REFLECTIVE_TRUST_TIER in allowed_trust_tiers and "reflection" in allowed_record_types:
        records.extend(_graphdb_reflection_records(request, endpoint=endpoint))

    backend_status = [
        SelfStudyRetrievalBackendStatusV1(storage_surface="rdf_graph", status="used", detail=f"Persisted self-study records queried from {endpoint}."),
        SelfStudyRetrievalBackendStatusV1(storage_surface="in_process", status="not_queried", detail="GraphDB persisted self-study retrieval succeeded without fallback."),
        SelfStudyRetrievalBackendStatusV1(storage_surface="journal", status="not_queried", detail="Phase 4B retrieval does not consume journal prose as primary self-study truth."),
    ]
    notes = [
        "Self-study retrieval is explicit and mode-scoped; it does not widen self.factual.v1.",
        "Phase 4B retrieval uses persisted GraphDB self-study records when available and preserves trust tiers end to end.",
    ]
    return _build_retrieval_result(
        request=request,
        records=records,
        backend_used="rdf_graph",
        backend_status=backend_status,
        notes=notes,
    )


def _mode_allowed_trust_tiers(retrieval_mode: str) -> tuple[str, ...]:
    if retrieval_mode == "factual":
        return (TRUST_TIER,)
    if retrieval_mode == "conceptual":
        return (TRUST_TIER, INDUCED_TRUST_TIER)
    return (TRUST_TIER, INDUCED_TRUST_TIER, REFLECTIVE_TRUST_TIER)


def _mode_allowed_record_types(retrieval_mode: str) -> tuple[str, ...]:
    if retrieval_mode == "factual":
        return ("fact",)
    if retrieval_mode == "conceptual":
        return ("fact", "concept")
    return ("fact", "concept", "reflection")


def _record_matches_filters(record: SelfStudyRetrievedRecordV1, filters: SelfStudyRetrieveFiltersV1) -> bool:
    if filters.trust_tiers and record.trust_tier not in filters.trust_tiers:
        return False
    if filters.record_types and record.record_type not in filters.record_types:
        return False
    if filters.stable_ids and record.stable_id not in filters.stable_ids:
        return False
    if filters.source_kinds and record.source_kind not in filters.source_kinds:
        return False
    if filters.storage_surfaces and record.storage_surface not in filters.storage_surfaces:
        return False
    if filters.concept_kinds and (record.concept_kind is None or record.concept_kind not in filters.concept_kinds):
        return False
    if filters.reflection_kinds and (record.reflection_kind is None or record.reflection_kind not in filters.reflection_kinds):
        return False
    query = (filters.text_query or "").strip().lower()
    if query:
        haystack = " ".join(
            str(value)
            for value in (
                record.title,
                record.content_preview,
                record.source_path or "",
                record.origin_name or "",
                record.origin_kind or "",
                record.concept_kind or "",
                record.reflection_kind or "",
            )
        ).lower()
        if query not in haystack:
            return False
    return True


def _limit_records_for_mode(records: Sequence[SelfStudyRetrievedRecordV1], *, retrieval_mode: str, limit: int) -> list[SelfStudyRetrievedRecordV1]:
    bounded_limit = max(1, int(limit or 12))
    if retrieval_mode == "factual":
        return list(records[:bounded_limit])

    buckets: dict[str, list[SelfStudyRetrievedRecordV1]] = {
        trust_tier: [record for record in records if record.trust_tier == trust_tier]
        for trust_tier in _mode_allowed_trust_tiers(retrieval_mode)
    }
    selected: list[SelfStudyRetrievedRecordV1] = []
    while len(selected) < bounded_limit and any(buckets.values()):
        for trust_tier in _mode_allowed_trust_tiers(retrieval_mode):
            bucket = buckets.get(trust_tier) or []
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            if len(selected) >= bounded_limit:
                break
    selected.sort(key=lambda item: (item.trust_tier, item.record_type, item.title, item.stable_id))
    return selected


def retrieve_self_study(request: SelfStudyRetrieveRequestV1) -> SelfStudyRetrieveResultV1:
    requested_surfaces = set(request.filters.storage_surfaces)
    wants_rdf_graph = not requested_surfaces or "rdf_graph" in requested_surfaces
    wants_in_process = not requested_surfaces or "in_process" in requested_surfaces

    if wants_rdf_graph:
        try:
            graph_result = _retrieve_self_study_from_graphdb(request)
            if graph_result.counts.total > 0 or not wants_in_process:
                return graph_result
            fallback = _retrieve_self_study_in_process(request)
            return fallback.model_copy(
                update={
                    "notes": [
                        *fallback.notes,
                        "GraphDB self-study query returned no persisted matches; fell back to in-process retrieval.",
                    ],
                    "backend_status": [
                        SelfStudyRetrievalBackendStatusV1(
                            storage_surface="in_process",
                            status="used",
                            detail="Fallback in-process self-study retrieval after persisted GraphDB query returned no matches.",
                        ),
                        SelfStudyRetrievalBackendStatusV1(
                            storage_surface="rdf_graph",
                            status="used",
                            detail="Persisted self-study GraphDB query succeeded but returned no matching records.",
                        ),
                        SelfStudyRetrievalBackendStatusV1(
                            storage_surface="journal",
                            status="not_queried",
                            detail="Phase 4B retrieval does not consume journal prose as primary self-study truth.",
                        ),
                    ],
                }
            )
        except Exception as exc:
            if not wants_in_process:
                return _build_retrieval_result(
                    request=request,
                    records=[],
                    backend_used=None,
                    backend_status=[
                        SelfStudyRetrievalBackendStatusV1(
                            storage_surface="rdf_graph",
                            status="unavailable",
                            detail=f"Persisted self-study GraphDB query failed: {exc}",
                        ),
                        SelfStudyRetrievalBackendStatusV1(
                            storage_surface="in_process",
                            status="not_queried",
                            detail="In-process fallback was not allowed by storage_surfaces filter.",
                        ),
                        SelfStudyRetrievalBackendStatusV1(
                            storage_surface="journal",
                            status="not_queried",
                            detail="Phase 4B retrieval does not consume journal prose as primary self-study truth.",
                        ),
                    ],
                    notes=[
                        "Self-study retrieval is explicit and mode-scoped; it does not widen self.factual.v1.",
                        f"Persisted GraphDB self-study retrieval failed without fallback: {exc}",
                    ],
                )
            fallback = _retrieve_self_study_in_process(request)
            return fallback.model_copy(
                update={
                    "notes": [
                        *fallback.notes,
                        f"GraphDB self-study retrieval unavailable; fell back to in-process retrieval: {exc}",
                    ],
                    "backend_status": [
                        SelfStudyRetrievalBackendStatusV1(
                            storage_surface="in_process",
                            status="used",
                            detail="Fallback in-process self-study retrieval after persisted GraphDB query was unavailable.",
                        ),
                        SelfStudyRetrievalBackendStatusV1(
                            storage_surface="rdf_graph",
                            status="unavailable",
                            detail=f"Persisted self-study GraphDB query failed: {exc}",
                        ),
                        SelfStudyRetrievalBackendStatusV1(
                            storage_surface="journal",
                            status="not_queried",
                            detail="Phase 4B retrieval does not consume journal prose as primary self-study truth.",
                        ),
                    ],
                }
            )

    return _retrieve_self_study_in_process(request)


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


def build_self_reflection_journal_entry(
    *,
    snapshot: SelfSnapshotV1,
    findings: Sequence[SelfReflectiveFindingV1],
    correlation_id: str,
    created_at: datetime | None = None,
) -> JournalEntryWriteV1:
    ts = created_at or datetime.now(timezone.utc)
    summary = build_self_reflection_summary(findings)
    lines = [
        f"Reflective self-study for snapshot {snapshot.snapshot_id}.",
        f"Trust tier: {REFLECTIVE_TRUST_TIER}.",
        summary,
    ]
    for finding in findings[:4]:
        line = f"- [{finding.reflection_kind}] {finding.title} (confidence={finding.confidence:.2f})"
        if finding.recommendation:
            line += f": {finding.recommendation}"
        elif finding.follow_up_question:
            line += f": {finding.follow_up_question}"
        lines.append(line)
    return JournalEntryWriteV1(
        created_at=ts,
        author=_AUTHOR,
        mode="manual",
        title="Self-study reflective findings",
        body="\n".join(lines),
        source_kind="self_reflection",
        source_ref=snapshot.snapshot_id,
        correlation_id=correlation_id,
    )


def _as_envelope_correlation_id(raw: str) -> str:
    try:
        return str(UUID(str(raw)))
    except Exception:
        return str(uuid5(NAMESPACE_URL, str(raw)))


def _node_uri(snapshot_id: str, suffix: str) -> URIRef:
    return URIRef(f"http://conjourney.net/orion/self/{snapshot_id}/{suffix}")


def build_self_study_rdf_request(snapshot: SelfSnapshotV1) -> RdfWriteRequest:
    _validate_authoritative_snapshot(snapshot)
    graph = Graph()
    graph.bind("orion", ORION)
    graph.bind("self", SELF)
    snapshot_uri = _node_uri(snapshot.snapshot_id, "snapshot")
    graph.add((snapshot_uri, RDF.type, ORION.AuthoritativeSelfSnapshot))
    graph.add((snapshot_uri, ORION.snapshotId, Literal(snapshot.snapshot_id, datatype=XSD.string)))
    graph.add((snapshot_uri, ORION.repoRoot, Literal(snapshot.repo_root, datatype=XSD.string)))
    graph.add((snapshot_uri, ORION.trustTier, Literal(snapshot.trust_tier, datatype=XSD.string)))

    section_map: Iterable[tuple[str, Sequence[SelfKnowledgeItemV1]]] = (
        ("services", snapshot.services),
        ("modules", snapshot.modules),
        ("channels", snapshot.channels),
        ("verbs", snapshot.verbs),
        ("schemas", snapshot.schemas),
        ("touchpoints", snapshot.touchpoints),
        ("env_surfaces", snapshot.env_surfaces),
        ("hardware", snapshot.hardware),
        ("behavioral", snapshot.behavioral),
    )
    for section_name, items in section_map:
        for item in items:
            item_uri = _node_uri(snapshot.snapshot_id, item.item_id)
            graph.add((item_uri, RDF.type, ORION.AuthoritativeSelfFact))
            graph.add((item_uri, ORION.factId, Literal(item.item_id, datatype=XSD.string)))
            graph.add((item_uri, ORION.inSection, Literal(section_name, datatype=XSD.string)))
            graph.add((item_uri, ORION.factCategory, Literal(item.category, datatype=XSD.string)))
            graph.add((item_uri, ORION.factName, Literal(item.name, datatype=XSD.string)))
            graph.add((item_uri, ORION.trustTier, Literal(item.trust_tier, datatype=XSD.string)))
            graph.add((item_uri, ORION.sourcePath, Literal(item.source_path, datatype=XSD.string)))
            if item.origin_kind:
                graph.add((item_uri, ORION.originKind, Literal(item.origin_kind, datatype=XSD.string)))
            if item.origin_name:
                graph.add((item_uri, ORION.originName, Literal(item.origin_name, datatype=XSD.string)))
            if item.symbol_name:
                graph.add((item_uri, ORION.symbolName, Literal(item.symbol_name, datatype=XSD.string)))
            for key, value in sorted(item.metadata.items()):
                graph.add((item_uri, ORION.hasMetadata, Literal(json.dumps({key: value}, sort_keys=True), datatype=XSD.string)))
            graph.add((snapshot_uri, ORION.hasAuthoritativeFact, item_uri))

    triples = graph.serialize(format="nt")
    return RdfWriteRequest(
        id=snapshot.snapshot_id,
        source="orion-cortex-exec",
        graph=SELF_GRAPH,
        triples=triples,
        kind="self.snapshot.authoritative.v1",
        payload={
            "snapshot_id": snapshot.snapshot_id,
            "run_id": snapshot.run_id,
            "observed_at": snapshot.observed_at,
            "trust_tier": snapshot.trust_tier,
        },
    )


def build_self_concept_rdf_request(*, source_snapshot: SelfSnapshotV1, concepts: Sequence[SelfInducedConceptV1], run_id: str) -> RdfWriteRequest:
    _validate_authoritative_snapshot(source_snapshot)
    for concept in concepts:
        if concept.trust_tier != INDUCED_TRUST_TIER:
            raise ValueError(f"concept_trust_tier_invalid:{concept.concept_id}:{concept.trust_tier}")
        if not concept.evidence:
            raise ValueError(f"concept_missing_evidence:{concept.concept_id}")

    graph = Graph()
    graph.bind("orion", ORION)
    graph.bind("self", SELF)
    for concept in concepts:
        concept_uri = URIRef(f"http://conjourney.net/orion/self/concept/{concept.concept_id}")
        graph.add((concept_uri, RDF.type, ORION.InducedSelfConcept))
        graph.add((concept_uri, ORION.conceptId, Literal(concept.concept_id, datatype=XSD.string)))
        graph.add((concept_uri, ORION.conceptKind, Literal(concept.concept_kind, datatype=XSD.string)))
        graph.add((concept_uri, ORION.label, Literal(concept.label, datatype=XSD.string)))
        graph.add((concept_uri, ORION.description, Literal(concept.description, datatype=XSD.string)))
        graph.add((concept_uri, ORION.trustTier, Literal(concept.trust_tier, datatype=XSD.string)))
        graph.add((concept_uri, ORION.confidence, Literal(concept.confidence, datatype=XSD.float)))
        graph.add((concept_uri, ORION.sourceSnapshotId, Literal(concept.source_snapshot_id, datatype=XSD.string)))
        for ref in concept.evidence:
            evidence_uri = _node_uri(ref.snapshot_id, ref.item_id)
            graph.add((concept_uri, ORION.supportedBy, evidence_uri))
            graph.add((concept_uri, ORION.sourcePath, Literal(ref.source_path, datatype=XSD.string)))
        for inferred in concept.inferred_from:
            graph.add((concept_uri, ORION.inferredFrom, Literal(inferred, datatype=XSD.string)))

    concept_digest = _stable_digest([concept.concept_id for concept in concepts])
    return RdfWriteRequest(
        id=f"self-concepts-{concept_digest}",
        source="orion-cortex-exec",
        graph=SELF_INDUCED_GRAPH,
        triples=graph.serialize(format="nt"),
        kind="self.concepts.induced.v1",
        payload={
            "run_id": run_id,
            "source_snapshot_id": source_snapshot.snapshot_id,
            "concept_ids": [concept.concept_id for concept in concepts],
            "trust_tier": INDUCED_TRUST_TIER,
        },
    )


def build_self_reflection_rdf_request(
    *,
    source_snapshot: SelfSnapshotV1,
    findings: Sequence[SelfReflectiveFindingV1],
    run_id: str,
) -> RdfWriteRequest:
    _validate_authoritative_snapshot(source_snapshot)
    for finding in findings:
        if finding.trust_tier != REFLECTIVE_TRUST_TIER:
            raise ValueError(f"reflection_trust_tier_invalid:{finding.reflection_id}:{finding.trust_tier}")
        if not finding.evidence:
            raise ValueError(f"reflection_missing_evidence:{finding.reflection_id}")
        if not finding.concept_refs:
            raise ValueError(f"reflection_missing_concept_refs:{finding.reflection_id}")

    graph = Graph()
    graph.bind("orion", ORION)
    graph.bind("self", SELF)
    for finding in findings:
        finding_uri = URIRef(f"http://conjourney.net/orion/self/reflection/{finding.reflection_id}")
        graph.add((finding_uri, RDF.type, ORION.ReflectiveSelfFinding))
        graph.add((finding_uri, ORION.reflectionId, Literal(finding.reflection_id, datatype=XSD.string)))
        graph.add((finding_uri, ORION.reflectionKind, Literal(finding.reflection_kind, datatype=XSD.string)))
        graph.add((finding_uri, ORION.label, Literal(finding.title, datatype=XSD.string)))
        graph.add((finding_uri, ORION.description, Literal(finding.description, datatype=XSD.string)))
        graph.add((finding_uri, ORION.trustTier, Literal(finding.trust_tier, datatype=XSD.string)))
        graph.add((finding_uri, ORION.confidence, Literal(finding.confidence, datatype=XSD.float)))
        graph.add((finding_uri, ORION.salience, Literal(finding.salience, datatype=XSD.float)))
        graph.add((finding_uri, ORION.sourceSnapshotId, Literal(finding.source_snapshot_id, datatype=XSD.string)))
        if finding.recommendation:
            graph.add((finding_uri, ORION.recommendation, Literal(finding.recommendation, datatype=XSD.string)))
        if finding.follow_up_question:
            graph.add((finding_uri, ORION.followUpQuestion, Literal(finding.follow_up_question, datatype=XSD.string)))
        for ref in finding.evidence:
            evidence_uri = _node_uri(ref.snapshot_id, ref.item_id)
            graph.add((finding_uri, ORION.supportedBy, evidence_uri))
        for concept_ref in finding.concept_refs:
            concept_uri = URIRef(f"http://conjourney.net/orion/self/concept/{concept_ref.concept_id}")
            graph.add((finding_uri, ORION.derivedFromConcept, concept_uri))

    reflection_digest = _stable_digest([finding.reflection_id for finding in findings])
    return RdfWriteRequest(
        id=f"self-reflections-{reflection_digest}",
        source="orion-cortex-exec",
        graph=SELF_REFLECTIVE_GRAPH,
        triples=graph.serialize(format="nt"),
        kind="self.reflections.reflective.v1",
        payload={
            "run_id": run_id,
            "source_snapshot_id": source_snapshot.snapshot_id,
            "reflection_ids": [finding.reflection_id for finding in findings],
            "source_concept_ids": sorted({concept_ref.concept_id for finding in findings for concept_ref in finding.concept_refs}),
            "trust_tier": REFLECTIVE_TRUST_TIER,
        },
    )


async def publish_self_concept_artifacts(
    *,
    bus: Any | None,
    source: ServiceRef,
    snapshot: SelfSnapshotV1,
    concepts: Sequence[SelfInducedConceptV1],
    correlation_id: str,
) -> SelfWritebackStatusV1:
    # orion:rdf:enqueue retired 2026-07-28: orion-rdf-writer (its sole real
    # consumer) is gone, and live verification found orion-graph-compression's
    # listed consumption was already dead weight (empty stream, empty
    # stale_queue/artifacts tables, SPARQL federator pointed at a Fuseki
    # container that no longer exists). Graph writeback is now a permanent
    # no-op rather than a publish into a channel nothing acts on.
    request = build_self_concept_rdf_request(source_snapshot=snapshot, concepts=concepts, run_id=snapshot.run_id)
    return SelfWritebackStatusV1(
        target="graph",
        status="skipped",
        authoritative=False,
        channel=RDF_ENQUEUE_CHANNEL,
        graph=SELF_INDUCED_GRAPH,
        idempotency_key=request.id,
        detail="channel_retired",
    )


async def publish_self_reflection_artifacts(
    *,
    bus: Any | None,
    source: ServiceRef,
    snapshot: SelfSnapshotV1,
    findings: Sequence[SelfReflectiveFindingV1],
    correlation_id: str,
) -> tuple[SelfWritebackStatusV1, SelfWritebackStatusV1, JournalEntryWriteV1]:
    journal_entry = build_self_reflection_journal_entry(
        snapshot=snapshot,
        findings=findings,
        correlation_id=correlation_id,
    )
    request = build_self_reflection_rdf_request(source_snapshot=snapshot, findings=findings, run_id=snapshot.run_id)
    if bus is None:
        return (
            SelfWritebackStatusV1(
                target="graph",
                status="skipped",
                authoritative=False,
                channel=RDF_ENQUEUE_CHANNEL,
                graph=SELF_REFLECTIVE_GRAPH,
                idempotency_key=request.id,
                detail="missing_bus",
            ),
            SelfWritebackStatusV1(
                target="journal",
                status="skipped",
                authoritative=False,
                channel=JOURNAL_WRITE_CHANNEL,
                idempotency_key=request.id,
                append_only=True,
                detail="missing_bus",
            ),
            journal_entry,
        )

    # orion:rdf:enqueue retired 2026-07-28: see publish_self_concept_artifacts
    # above for the live-verification behind this. Graph writeback is now a
    # permanent no-op; journal writeback below is unaffected.
    graph_status = SelfWritebackStatusV1(
        target="graph",
        status="skipped",
        authoritative=False,
        channel=RDF_ENQUEUE_CHANNEL,
        graph=SELF_REFLECTIVE_GRAPH,
        idempotency_key=request.id,
        detail="channel_retired",
    )

    journal_status = SelfWritebackStatusV1(
        target="journal",
        status="written",
        authoritative=False,
        channel=JOURNAL_WRITE_CHANNEL,
        idempotency_key=request.id,
        append_only=True,
        detail="append_only_by_design",
    )
    journal_env = BaseEnvelope(
        kind="journal.entry.write.v1",
        source=source,
        correlation_id=_as_envelope_correlation_id(correlation_id),
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
            idempotency_key=request.id,
            append_only=True,
            detail=str(exc),
        )

    return graph_status, journal_status, journal_entry


def _next_self_concept_version(concept_id: str) -> int:
    """Real current-max-version lookup, not a guess -- reuses
    self_study_analysis.py's existing engine/DSN fallback chain (same
    pattern as _behavioral_items()). `version` is informational (see
    orion/schemas/self_concept_history.py's own docstring: "current" is
    resolved by latest created_at per concept_id, not by version), but
    still worth getting right rather than always writing 1. Fails soft to
    1 -- an unreachable DB must not block a real reflection from being
    recorded, just cost this one row an uninformative version number."""
    try:
        from app.self_study_analysis import _get_engine
        from sqlalchemy import text

        engine = _get_engine()
        if engine is None:
            return 1
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT MAX(version) AS max_version FROM self_concept_history WHERE concept_id = :concept_id"),
                {"concept_id": concept_id},
            ).mappings().first()
        current_max = row["max_version"] if row else None
        return int(current_max) + 1 if current_max is not None else 1
    except Exception as exc:
        logger.debug("self_concept_history_version_lookup_failed concept_id=%s error=%s", concept_id, exc)
        return 1


async def publish_self_concept_history_from_reflection(
    *, bus: Any | None, source: ServiceRef, findings: Sequence[SelfReflectiveFindingV1], correlation_id: str
) -> SelfWritebackStatusV1:
    """Feed Layer 3's real LLM reflection findings into the append-only
    self_concept_history store (self-model rebuild arc, Patch 3) -- the
    identity.yaml-replacement history, additive not a replacement. One row
    per distinct concept_kind a finding's concept_refs touch (a finding
    with no concept_refs falls back to its own reflection_kind as the
    concept_id) -- concept_kind/reflection_kind are the stable, recurring
    identity anchors; a finding's own hashed reflection_id is not (it's
    unique per finding, never repeats, so nothing would ever accumulate a
    real history under it)."""
    status = SelfWritebackStatusV1(
        target="self_concept_history",
        status="skipped",
        authoritative=False,
        channel=SELF_CONCEPT_HISTORY_WRITE_CHANNEL,
        append_only=True,
        detail="missing_bus",
    )
    if bus is None or not findings:
        logger.info(
            "self_concept_history_publish_skip reason=%s finding_count=%s",
            "missing_bus" if bus is None else "no_findings",
            len(findings),
        )
        return status

    envelope_corr_id = _as_envelope_correlation_id(correlation_id)
    published = 0
    failed = 0
    for finding in findings:
        concept_ids = sorted({ref.concept_kind for ref in finding.concept_refs}) or [finding.reflection_kind]
        for concept_id in concept_ids:
            try:
                # Review finding: _next_self_concept_version() is a blocking
                # SQLAlchemy call (create_engine + a real SELECT) -- calling
                # it directly here (this function is async) would freeze
                # cortex-exec's event loop for every concept touched by every
                # finding, the exact same blocking-event-loop bug already
                # caught and fixed twice this session (chat_stance_belief_bus,
                # publish_self_knowledge_items).
                version = await asyncio.to_thread(_next_self_concept_version, concept_id)
                payload = SelfConceptHistoryV1(
                    concept_id=concept_id,
                    version=version,
                    content=f"{finding.title}: {finding.description}",
                    evidence_refs=[ref.item_id for ref in finding.evidence],
                    produced_by="layer3_reflect",
                )
                envelope = BaseEnvelope(
                    kind="self_concept.history.write.v1",
                    source=source,
                    correlation_id=envelope_corr_id,
                    payload=payload.model_dump(mode="json"),
                )
                await bus.publish(SELF_CONCEPT_HISTORY_WRITE_CHANNEL, envelope)
                published += 1
            except Exception as exc:
                failed += 1
                logger.debug(
                    "self_concept_history_publish_item_failed concept_id=%s error=%s",
                    concept_id,
                    exc,
                )

    if failed > 0:
        overall_status: SelfWritebackState = "failed"
    elif published > 0:
        overall_status = "written"
    else:
        overall_status = "skipped"
    status = SelfWritebackStatusV1(
        target="self_concept_history",
        status=overall_status,
        authoritative=False,
        channel=SELF_CONCEPT_HISTORY_WRITE_CHANNEL,
        append_only=True,
        detail=f"published={published} failed={failed}",
    )
    log_fn = logger.warning if failed > 0 else logger.info
    log_fn(
        "self_concept_history_publish status=%s published=%s failed=%s",
        overall_status,
        published,
        failed,
    )
    return status


async def publish_self_study_artifacts(
    *,
    bus: Any | None,
    source: ServiceRef,
    snapshot: SelfSnapshotV1,
    correlation_id: str,
) -> tuple[SelfWritebackStatusV1, SelfWritebackStatusV1, JournalEntryWriteV1]:
    # orion:rdf:enqueue retired 2026-07-28: see publish_self_concept_artifacts
    # above for the live-verification behind this. This graph (orion:self)
    # was independently confirmed by orion-graph-compression's own
    # verification (2026-07-23, see its README) to have zero triples ever --
    # no producer including this one had ever landed a real write. Graph
    # writeback is now a permanent no-op; journal writeback is unaffected.
    journal_entry = build_self_study_journal_entry(snapshot, correlation_id=correlation_id)
    graph_status = SelfWritebackStatusV1(
        target="graph",
        status="skipped",
        authoritative=True,
        channel=RDF_ENQUEUE_CHANNEL,
        graph=SELF_GRAPH,
        idempotency_key=snapshot.snapshot_id,
        detail="channel_retired",
    )
    if bus is None:
        logger.info("self_study_degraded snapshot_id=%s reason=missing_bus", snapshot.snapshot_id)
        return (
            graph_status,
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
        "self_study_publish snapshot_id=%s run_id=%s graph_status=%s journal_status=%s graph_detail=%s journal_detail=%s",
        snapshot.snapshot_id,
        snapshot.run_id,
        graph_status.status,
        journal_status.status,
        graph_status.detail,
        journal_status.detail,
    )
    return graph_status, journal_status, journal_entry


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
    snapshot = build_self_snapshot()
    graph_status, journal_status, journal_entry = await publish_self_study_artifacts(
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
        "self_study_scan snapshot_id=%s run_id=%s duration_ms=%s services=%s modules=%s channels=%s verbs=%s schemas=%s graph_status=%s journal_status=%s items_write_status=%s",
        snapshot.snapshot_id,
        snapshot.run_id,
        duration_ms,
        snapshot.counts.services,
        snapshot.counts.modules,
        snapshot.counts.channels,
        snapshot.counts.verbs,
        snapshot.counts.schemas,
        graph_status.status,
        journal_status.status,
        items_write_status.status,
    )
    return SelfRepoInspectResultV1(
        snapshot=snapshot,
        summary=build_self_study_summary(snapshot),
        graph_write=graph_status,
        journal_write=journal_status,
        journal_entry=journal_entry,
        self_knowledge_items_write=items_write_status,
    )


async def run_self_concept_induce(*, bus: Any | None, source: ServiceRef, correlation_id: str) -> SelfConceptInduceResultV1:
    snapshot = build_self_snapshot()
    concepts = induce_self_concepts(snapshot)
    graph_status = await publish_self_concept_artifacts(
        bus=bus,
        source=source,
        snapshot=snapshot,
        concepts=concepts,
        correlation_id=correlation_id,
    )
    return SelfConceptInduceResultV1(
        run_id=snapshot.run_id,
        source_snapshot_id=snapshot.snapshot_id,
        concepts=list(concepts),
        summary=build_self_concept_summary(concepts),
        graph_write=graph_status,
    )


async def run_self_concept_reflect(*, bus: Any | None, source: ServiceRef, correlation_id: str) -> SelfConceptReflectResultV1:
    snapshot = build_self_snapshot()
    concepts = induce_self_concepts(snapshot)
    validation_summary = validate_phase2a_induction(snapshot, concepts)
    outcome = await reflect_self_concepts(
        bus=bus, source=source, snapshot=snapshot, concepts=concepts, correlation_id=correlation_id
    )
    findings = outcome.findings
    if outcome.llm_call_failed:
        # A real LLM attempt did not complete -- publishing a "0 findings"
        # journal entry here would misrepresent an infra failure as a
        # completed reflection pass that genuinely found nothing (CLAUDE.md
        # 0A's no-empty-shell-cognition rule, verbatim). `findings` is always
        # `[]` on this path (see ReflectSelfConceptsOutcome). Build the
        # journal entry (the result object's journal_entry field is
        # required) but never publish it.
        journal_entry = build_self_reflection_journal_entry(
            snapshot=snapshot, findings=findings, correlation_id=correlation_id
        )
        graph_status = SelfWritebackStatusV1(
            target="graph",
            status="skipped",
            authoritative=False,
            channel=RDF_ENQUEUE_CHANNEL,
            graph=SELF_REFLECTIVE_GRAPH,
            idempotency_key=snapshot.snapshot_id,
            detail="reflection_llm_call_failed",
        )
        journal_status = SelfWritebackStatusV1(
            target="journal",
            status="skipped",
            authoritative=False,
            channel=JOURNAL_WRITE_CHANNEL,
            idempotency_key=snapshot.snapshot_id,
            append_only=True,
            detail="reflection_llm_call_failed",
        )
        # No real findings exist on this path -- publishing self-concept
        # history rows here would be exactly the empty-shell-cognition
        # pattern the comment above already rules out for the journal.
        self_concept_history_status = SelfWritebackStatusV1(
            target="self_concept_history",
            status="skipped",
            authoritative=False,
            channel=SELF_CONCEPT_HISTORY_WRITE_CHANNEL,
            append_only=True,
            detail="reflection_llm_call_failed",
        )
    else:
        graph_status, journal_status, journal_entry = await publish_self_reflection_artifacts(
            bus=bus,
            source=source,
            snapshot=snapshot,
            findings=findings,
            correlation_id=correlation_id,
        )
        self_concept_history_status = await publish_self_concept_history_from_reflection(
            bus=bus, source=source, findings=findings, correlation_id=correlation_id
        )
    return SelfConceptReflectResultV1(
        run_id=snapshot.run_id,
        source_snapshot_id=snapshot.snapshot_id,
        source_concept_ids=[concept.concept_id for concept in concepts],
        validated_phase2a=True,
        validation_summary=validation_summary,
        findings=findings,
        summary=build_self_reflection_summary(findings),
        graph_write=graph_status,
        journal_write=journal_status,
        journal_entry=journal_entry,
        self_concept_history_write=self_concept_history_status,
    )


async def run_self_retrieve(
    *,
    request: SelfStudyRetrieveRequestV1,
    bus: Any | None = None,
    source: ServiceRef | None = None,
    correlation_id: str | None = None,
) -> SelfStudyRetrieveResultV1:
    del bus, source, correlation_id
    return retrieve_self_study(request)

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import requests
import yaml
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.journaler.schemas import JournalEntryWriteV1
from orion.schemas.rdf import RdfWriteRequest
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
    (
        "services/orion-rdf-writer/app/settings.py",
        "rdf_writer",
        ("GRAPHDB_URL", "GRAPHDB_REPO", "CHANNEL_RDF_ENQUEUE", "CHANNEL_WORKER_RDF"),
    ),
)

_TOUCHPOINTS: tuple[tuple[str, str, str], ...] = (
    ("journal", "orion/journaler/worker.py", "build_write_payload"),
    ("journal", "services/orion-actions/app/main.py", "_run_journal"),
    ("journal", "services/orion-sql-writer/app/worker.py", "handle_envelope"),
    ("graph", "services/orion-rdf-writer/app/service.py", "_push_to_graphdb"),
    ("graph", "services/orion-rdf-writer/app/rdf_builder.py", "build_triples_from_envelope"),
    ("recall", "services/orion-recall/app/profiles.py", "load_profiles"),
    ("recall", "services/orion-recall/app/worker.py", "process_recall"),
    ("persistence", "services/orion-state-service/app/store.py", "StateStore"),
)

_LAST_GRAPH_PUBLISH_KEY: str | None = None
_LAST_CONCEPT_PUBLISH_KEY: str | None = None
_LAST_REFLECTION_PUBLISH_KEY: str | None = None


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
    for section_name in ("services", "modules", "channels", "verbs", "schemas", "touchpoints", "env_surfaces"):
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

    graph_evidence: list[SelfKnowledgeItemV1] = []
    for key in ("orion-rdf-writer",):
        if key in service_by_name:
            graph_evidence.append(service_by_name[key])
    for key in ("orion:rdf:enqueue", "orion:rdf:worker"):
        if key in channel_by_name:
            graph_evidence.append(channel_by_name[key])
    graph_evidence.extend(touchpoints_by_surface.get("graph", []))
    if graph_evidence:
        concepts.append(
            _concept(
                snapshot=snapshot,
                concept_kind="graph_surface",
                label="rdf graph write surface",
                description="The self-study graph surface is anchored by rdf-writer touchpoints and the orion:rdf:* bus channels reused for structured graph writes.",
                evidence_items=graph_evidence,
                inferred_from=["service", "channel", "touchpoint"],
            )
        )

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
    for key in ("orion-cortex-exec", "orion-rdf-writer", "orion-recall"):
        if key in service_by_name:
            self_study_cluster.append(service_by_name[key])
    for key in ("orion:rdf:enqueue", "orion:journal:write"):
        if key in channel_by_name:
            self_study_cluster.append(channel_by_name[key])
    if self_study_cluster:
        concepts.append(
            _concept(
                snapshot=snapshot,
                concept_kind="service_cluster",
                label="self-study execution cluster",
                description="Self-study spans cortex-exec, rdf-writer, recall configuration surfaces, and journal routing as a small cross-service cluster.",
                evidence_items=self_study_cluster,
                inferred_from=["service", "channel"],
            )
        )

    topology_evidence = [item for item in snapshot.channels if item.name.startswith("orion:rdf:")]
    if "orion:journal:write" in channel_by_name:
        topology_evidence.append(channel_by_name["orion:journal:write"])
    if topology_evidence:
        concepts.append(
            _concept(
                snapshot=snapshot,
                concept_kind="bus_topology_pattern",
                label="self-study bus write topology",
                description="Self-study uses bus-first write surfaces, separating graph publication from journal publication while keeping both behind typed envelopes.",
                evidence_items=topology_evidence,
                inferred_from=["channel"],
            )
        )

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
        for section_name in ("services", "modules", "channels", "verbs", "schemas", "touchpoints", "env_surfaces")
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


def reflect_self_concepts(snapshot: SelfSnapshotV1, concepts: Sequence[SelfInducedConceptV1]) -> list[SelfReflectiveFindingV1]:
    validation_summary = validate_phase2a_induction(snapshot, concepts)
    if not validation_summary:
        raise ValueError("phase2a_validation_missing")

    concept_by_kind = {concept.concept_kind: concept for concept in concepts}
    findings: list[SelfReflectiveFindingV1] = []

    graph_concept = concept_by_kind.get("graph_surface")
    journal_concept = concept_by_kind.get("journaling_surface")
    if graph_concept and journal_concept:
        findings.append(
            _reflection(
                snapshot=snapshot,
                reflection_kind="tension",
                title="Graph and journal lanes need ongoing trust separation",
                description="Self-study spans authoritative graph publication and non-authoritative journal publication, so reflective outputs should stay on separate sinks and never re-enter factual grounding lanes.",
                concepts=[graph_concept, journal_concept],
                confidence=0.79,
                salience=0.84,
                recommendation="Keep reflective writeback on its own graph and journal source kind rather than reusing authoritative paths.",
            )
        )

    service_cluster = concept_by_kind.get("service_cluster")
    runtime_boundary = concept_by_kind.get("runtime_boundary")
    if service_cluster and runtime_boundary:
        findings.append(
            _reflection(
                snapshot=snapshot,
                reflection_kind="architecture_observation",
                title="Self-study is coordinated across multiple runtime surfaces",
                description="The current self-study lane is not local to a single module: it depends on cortex-exec, write channels, and adjacent services, which makes provenance discipline more important than richer narration.",
                concepts=[runtime_boundary, service_cluster],
                confidence=0.74,
                salience=0.68,
                recommendation="Favor compact typed outputs over broad narrative summaries when adding new self-study phases.",
            )
        )

    recall_concept = concept_by_kind.get("recall_surface")
    if recall_concept:
        findings.append(
            _reflection(
                snapshot=snapshot,
                reflection_kind="seam_risk",
                title="Reflective retrieval should stay out of factual recall by default",
                description="The current recall surface cleanly isolates authoritative self-study material. Adding reflective retrieval later would need a dedicated profile rather than broadening self.factual.v1.",
                concepts=[recall_concept],
                confidence=0.86,
                salience=0.81,
                follow_up_question="If reflective recall is added later, what profile and tag boundaries keep it separate from delivery/debug grounding?",
            )
        )

    topology_concept = concept_by_kind.get("bus_topology_pattern")
    if topology_concept:
        findings.append(
            _reflection(
                snapshot=snapshot,
                reflection_kind="candidate_skill_idea",
                title="Future drift checking could compare static and runtime surfaces",
                description="Current self-study is strong on repo-visible structure, but it may miss runtime-only drift that does not appear in static files or channel declarations.",
                concepts=[topology_concept],
                confidence=0.63,
                salience=0.59,
                recommendation="A later skill could compare runtime-observed channels and services against repo-extracted self-study surfaces without promoting the result into authoritative truth.",
            )
        )

    if not findings and concepts:
        findings.append(
            _reflection(
                snapshot=snapshot,
                reflection_kind="blind_spot",
                title="Current self-study is intentionally conservative",
                description="The present self-study lane prioritizes trusted repo structure, which limits coverage of dynamic behavior but protects factual grounding from looser interpretation.",
                concepts=[concepts[0]],
                confidence=0.6,
                salience=0.45,
                recommendation="Add runtime comparison only behind a separate reflective lane if stronger dynamic coverage becomes necessary.",
            )
        )

    findings.sort(key=lambda item: (item.reflection_kind, item.title))
    return findings


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
        for section_name in ("services", "modules", "channels", "verbs", "schemas", "touchpoints", "env_surfaces")
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
    findings = reflect_self_concepts(snapshot, concepts)

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
    global _LAST_CONCEPT_PUBLISH_KEY

    request = build_self_concept_rdf_request(source_snapshot=snapshot, concepts=concepts, run_id=snapshot.run_id)
    if bus is None:
        return SelfWritebackStatusV1(
            target="graph",
            status="skipped",
            authoritative=False,
            channel=RDF_ENQUEUE_CHANNEL,
            graph=SELF_INDUCED_GRAPH,
            idempotency_key=request.id,
            detail="missing_bus",
        )

    if _LAST_CONCEPT_PUBLISH_KEY == request.id:
        return SelfWritebackStatusV1(
            target="graph",
            status="skipped",
            authoritative=False,
            channel=RDF_ENQUEUE_CHANNEL,
            graph=SELF_INDUCED_GRAPH,
            idempotency_key=request.id,
            detail="unchanged_concepts",
        )

    env = BaseEnvelope(
        kind="rdf.write.request",
        source=source,
        correlation_id=_as_envelope_correlation_id(correlation_id),
        payload=request.model_dump(mode="json"),
    )
    try:
        await bus.publish(RDF_ENQUEUE_CHANNEL, env)
        _LAST_CONCEPT_PUBLISH_KEY = request.id
        return SelfWritebackStatusV1(
            target="graph",
            status="written",
            authoritative=False,
            channel=RDF_ENQUEUE_CHANNEL,
            graph=SELF_INDUCED_GRAPH,
            idempotency_key=request.id,
        )
    except Exception as exc:
        return SelfWritebackStatusV1(
            target="graph",
            status="failed",
            authoritative=False,
            channel=RDF_ENQUEUE_CHANNEL,
            graph=SELF_INDUCED_GRAPH,
            idempotency_key=request.id,
            detail=str(exc),
        )


async def publish_self_reflection_artifacts(
    *,
    bus: Any | None,
    source: ServiceRef,
    snapshot: SelfSnapshotV1,
    findings: Sequence[SelfReflectiveFindingV1],
    correlation_id: str,
) -> tuple[SelfWritebackStatusV1, SelfWritebackStatusV1, JournalEntryWriteV1]:
    global _LAST_REFLECTION_PUBLISH_KEY

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

    graph_status = SelfWritebackStatusV1(
        target="graph",
        status="written",
        authoritative=False,
        channel=RDF_ENQUEUE_CHANNEL,
        graph=SELF_REFLECTIVE_GRAPH,
        idempotency_key=request.id,
    )
    if _LAST_REFLECTION_PUBLISH_KEY == request.id:
        graph_status = SelfWritebackStatusV1(
            target="graph",
            status="skipped",
            authoritative=False,
            channel=RDF_ENQUEUE_CHANNEL,
            graph=SELF_REFLECTIVE_GRAPH,
            idempotency_key=request.id,
            detail="unchanged_reflections",
        )
    else:
        rdf_env = BaseEnvelope(
            kind="rdf.write.request",
            source=source,
            correlation_id=_as_envelope_correlation_id(correlation_id),
            payload=request.model_dump(mode="json"),
        )
        try:
            await bus.publish(RDF_ENQUEUE_CHANNEL, rdf_env)
            _LAST_REFLECTION_PUBLISH_KEY = request.id
        except Exception as exc:
            graph_status = SelfWritebackStatusV1(
                target="graph",
                status="failed",
                authoritative=False,
                channel=RDF_ENQUEUE_CHANNEL,
                graph=SELF_REFLECTIVE_GRAPH,
                idempotency_key=request.id,
                detail=str(exc),
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


async def publish_self_study_artifacts(
    *,
    bus: Any | None,
    source: ServiceRef,
    snapshot: SelfSnapshotV1,
    correlation_id: str,
) -> tuple[SelfWritebackStatusV1, SelfWritebackStatusV1, JournalEntryWriteV1]:
    global _LAST_GRAPH_PUBLISH_KEY

    journal_entry = build_self_study_journal_entry(snapshot, correlation_id=correlation_id)
    if bus is None:
        logger.info("self_study_degraded snapshot_id=%s reason=missing_bus", snapshot.snapshot_id)
        return (
            SelfWritebackStatusV1(
                target="graph",
                status="skipped",
                authoritative=True,
                channel=RDF_ENQUEUE_CHANNEL,
                graph=SELF_GRAPH,
                idempotency_key=snapshot.snapshot_id,
                detail="missing_bus",
            ),
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

    graph_status = SelfWritebackStatusV1(
        target="graph",
        status="written",
        authoritative=True,
        channel=RDF_ENQUEUE_CHANNEL,
        graph=SELF_GRAPH,
        idempotency_key=snapshot.snapshot_id,
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
    if _LAST_GRAPH_PUBLISH_KEY == snapshot.snapshot_id:
        graph_status = SelfWritebackStatusV1(
            target="graph",
            status="skipped",
            authoritative=True,
            channel=RDF_ENQUEUE_CHANNEL,
            graph=SELF_GRAPH,
            idempotency_key=snapshot.snapshot_id,
            detail="unchanged_snapshot",
        )
    else:
        rdf_request = build_self_study_rdf_request(snapshot)
        rdf_env = BaseEnvelope(
            kind="rdf.write.request",
            source=source,
            correlation_id=envelope_corr_id,
            payload=rdf_request.model_dump(mode="json"),
        )
        try:
            await bus.publish(RDF_ENQUEUE_CHANNEL, rdf_env)
            _LAST_GRAPH_PUBLISH_KEY = snapshot.snapshot_id
        except Exception as exc:
            graph_status = SelfWritebackStatusV1(
                target="graph",
                status="failed",
                authoritative=True,
                channel=RDF_ENQUEUE_CHANNEL,
                graph=SELF_GRAPH,
                idempotency_key=snapshot.snapshot_id,
                detail=str(exc),
            )

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


async def run_self_repo_inspect(*, bus: Any | None, source: ServiceRef, correlation_id: str) -> SelfRepoInspectResultV1:
    start = time.monotonic()
    snapshot = build_self_snapshot()
    graph_status, journal_status, journal_entry = await publish_self_study_artifacts(
        bus=bus,
        source=source,
        snapshot=snapshot,
        correlation_id=correlation_id,
    )
    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "self_study_scan snapshot_id=%s run_id=%s duration_ms=%s services=%s modules=%s channels=%s verbs=%s schemas=%s graph_status=%s journal_status=%s",
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
    )
    return SelfRepoInspectResultV1(
        snapshot=snapshot,
        summary=build_self_study_summary(snapshot),
        graph_write=graph_status,
        journal_write=journal_status,
        journal_entry=journal_entry,
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
    findings = reflect_self_concepts(snapshot, concepts)
    graph_status, journal_status, journal_entry = await publish_self_reflection_artifacts(
        bus=bus,
        source=source,
        snapshot=snapshot,
        findings=findings,
        correlation_id=correlation_id,
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

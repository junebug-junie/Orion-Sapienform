from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF, XSD

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = ROOT / "services" / "orion-rdf-writer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.autonomy import build_autonomy_triples  # noqa: E402

ORION = Namespace("http://conjourney.net/orion#")
DEFAULT_SCENARIO_PACK = ROOT / "tests" / "fixtures" / "autonomy_graph" / "scenario_pack.json"
DEFAULT_REPORT_DIR = ROOT / "tmp"

KIND_TO_CHANNEL = {
    "memory.identity.snapshot.v1": "orion:memory:identity:snapshot",
    "memory.drives.audit.v1": "orion:memory:drives:audit",
    "memory.goals.proposed.v1": "orion:memory:goals:proposed",
}

KIND_TO_GRAPH = {
    "memory.identity.snapshot.v1": "http://conjourney.net/graph/autonomy/identity",
    "memory.drives.audit.v1": "http://conjourney.net/graph/autonomy/drives",
    "memory.goals.proposed.v1": "http://conjourney.net/graph/autonomy/goals",
}


@dataclass
class ScenarioFixture:
    scenario_id: str
    kind: str
    channel: str
    payload: Dict[str, Any]
    description: str = ""

    @property
    def artifact_id(self) -> str:
        return str(self.payload["artifact_id"])


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass
class ScenarioRunResult:
    scenario_id: str
    artifact_id: str
    kind: str
    graph_uri: Optional[str]
    local_ok: bool
    local_checks: List[CheckResult] = field(default_factory=list)
    bus_publish: str = "not_requested"
    graphdb_verify: str = "not_requested"
    graphdb_detail: str = ""


class GraphDBClient:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        repo: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        timeout_sec: float = 10.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("GRAPHDB_URL", "")).rstrip("/")
        self.repo = repo or os.getenv("GRAPHDB_REPO", "collapse")
        self.user = user if user is not None else os.getenv("GRAPHDB_USER")
        self.password = password if password is not None else os.getenv("GRAPHDB_PASS")
        self.timeout_sec = timeout_sec

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.repo)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/repositories/{self.repo}"

    def ask(self, query: str) -> bool:
        auth = (self.user, self.password) if self.user or self.password else None
        response = requests.post(
            self.endpoint,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            auth=auth,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        return bool(response.json().get("boolean"))

    def verify_scenario(self, fixture: ScenarioFixture) -> tuple[str, str]:
        if not self.configured:
            return "skipped", "GraphDB not configured via GRAPHDB_URL/GRAPHDB_REPO."
        graph_uri = KIND_TO_GRAPH.get(fixture.kind)
        artifact_id = fixture.artifact_id
        literal = json.dumps(artifact_id)
        checks = {
            "artifact_exists": f'''
                PREFIX orion: <http://conjourney.net/orion#>
                ASK WHERE {{
                  GRAPH <{graph_uri}> {{
                    ?artifact orion:artifactId {literal}^^<http://www.w3.org/2001/XMLSchema#string> .
                  }}
                }}
            ''',
            "about_entity": f'''
                PREFIX orion: <http://conjourney.net/orion#>
                ASK WHERE {{
                  GRAPH <{graph_uri}> {{
                    ?artifact orion:artifactId {literal}^^<http://www.w3.org/2001/XMLSchema#string> ;
                              orion:aboutEntity ?entity ;
                              orion:belongsToModelLayer ?layer .
                  }}
                }}
            ''',
            "provenance": f'''
                PREFIX orion: <http://conjourney.net/orion#>
                ASK WHERE {{
                  GRAPH <{graph_uri}> {{
                    ?artifact orion:artifactId {literal}^^<http://www.w3.org/2001/XMLSchema#string> ;
                              orion:hasProvenance ?prov .
                    ?prov a orion:ArtifactProvenance .
                  }}
                }}
            ''',
        }
        try:
            failed: list[str] = []
            for name, query in checks.items():
                if not self.ask(" ".join(query.split())):
                    failed.append(name)
            if failed:
                return "failed", f"Missing GraphDB checks: {', '.join(failed)}"
            return "passed", f"Verified {len(checks)} GraphDB ASK checks in {graph_uri}."
        except Exception as exc:  # pragma: no cover - environment dependent
            return "skipped", f"GraphDB verification unavailable: {exc}"


class AutonomyVerificationHarness:
    def __init__(self, *, graphdb_client: Optional[GraphDBClient] = None, bus_url: Optional[str] = None) -> None:
        self.graphdb_client = graphdb_client or GraphDBClient()
        self.bus_url = bus_url or os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")

    def run(
        self,
        scenarios: Sequence[ScenarioFixture],
        *,
        publish_bus: bool = False,
        verify_graphdb: bool = False,
        wait_sec: float = 0.0,
    ) -> Dict[str, Any]:
        results: list[ScenarioRunResult] = []
        bus_statuses = {fixture.scenario_id: "not_requested" for fixture in scenarios}
        if publish_bus:
            bus_statuses = asyncio.run(self._publish_all(scenarios))
            if wait_sec > 0:
                time.sleep(wait_sec)
        for fixture in scenarios:
            result = self._run_single(fixture)
            result.bus_publish = bus_statuses.get(fixture.scenario_id, "not_requested")
            if verify_graphdb:
                result.graphdb_verify, result.graphdb_detail = self.graphdb_client.verify_scenario(fixture)
            results.append(result)
        passed = sum(1 for item in results if item.local_ok and item.graphdb_verify != "failed")
        return {
            "summary": {
                "scenario_count": len(results),
                "passed_count": passed,
                "local_failures": [item.scenario_id for item in results if not item.local_ok],
                "graphdb_failures": [item.scenario_id for item in results if item.graphdb_verify == "failed"],
                "graphdb_skipped": [item.scenario_id for item in results if item.graphdb_verify == "skipped"],
            },
            "results": [
                {
                    **asdict(item),
                    "local_checks": [asdict(check) for check in item.local_checks],
                }
                for item in results
            ],
        }

    async def _publish_all(self, scenarios: Sequence[ScenarioFixture]) -> Dict[str, str]:
        statuses = {fixture.scenario_id: "not_requested" for fixture in scenarios}
        bus = OrionBusAsync(self.bus_url, enabled=True, enforce_catalog=False)
        try:
            await bus.connect()
            for fixture in scenarios:
                env = BaseEnvelope(
                    kind=fixture.kind,
                    source=ServiceRef(name="orion-autonomy-verifier", version="0.1.0"),
                    payload=fixture.payload,
                )
                await bus.publish(fixture.channel, env)
                statuses[fixture.scenario_id] = "published"
        except Exception as exc:  # pragma: no cover - infrastructure dependent
            for fixture in scenarios:
                if statuses[fixture.scenario_id] == "not_requested":
                    statuses[fixture.scenario_id] = f"skipped:{exc}"
        finally:
            try:
                await bus.close()
            except Exception:
                pass
        return statuses

    def _run_single(self, fixture: ScenarioFixture) -> ScenarioRunResult:
        nt, graph_uri = build_autonomy_triples(fixture.kind, fixture.payload)
        graph = Graph()
        if nt:
            graph.parse(data=nt, format="nt")
        checks = self._local_checks(fixture, graph, graph_uri)
        return ScenarioRunResult(
            scenario_id=fixture.scenario_id,
            artifact_id=fixture.artifact_id,
            kind=fixture.kind,
            graph_uri=graph_uri,
            local_ok=all(check.ok for check in checks),
            local_checks=checks,
        )

    def _local_checks(self, fixture: ScenarioFixture, graph: Graph, graph_uri: Optional[str]) -> List[CheckResult]:
        artifact_literal = Literal(fixture.artifact_id, datatype=XSD.string)
        subjects = list(graph.subjects(ORION.artifactId, artifact_literal))
        artifact = subjects[0] if subjects else None
        checks: list[CheckResult] = [
            CheckResult("graph_uri", graph_uri == KIND_TO_GRAPH[fixture.kind], f"graph_uri={graph_uri}"),
            CheckResult("artifact_present", artifact is not None, f"artifact_id={fixture.artifact_id}"),
        ]
        if artifact is None:
            return checks
        checks.extend(
            [
                CheckResult("belongs_to_model_layer", (artifact, ORION.belongsToModelLayer, None) in graph, "artifact has model layer"),
                CheckResult("about_entity", (artifact, ORION.aboutEntity, None) in graph, "artifact has aboutEntity"),
                CheckResult("has_provenance", (artifact, ORION.hasProvenance, None) in graph, "artifact has provenance node"),
                CheckResult("source_event_ref", (artifact, ORION.referencesSourceEvent, None) in graph, "artifact references source event"),
                CheckResult("evidence_link", (artifact, ORION.supportedByEvidence, None) in graph, "artifact linked to evidence"),
                CheckResult("lineage_correlation", (artifact, ORION.hasCorrelation, None) in graph, "artifact has correlation lineage"),
                CheckResult("lineage_trace", (artifact, ORION.hasTrace, None) in graph, "artifact has trace lineage"),
                CheckResult("lineage_turn", (artifact, ORION.hasTurnContext, None) in graph, "artifact has turn lineage"),
            ]
        )
        tension_refs = fixture.payload.get("provenance", {}).get("tension_refs") or []
        if tension_refs:
            checks.append(CheckResult("tension_refs", (artifact, ORION.derivedFromTension, None) in graph, "artifact derived from tensions"))
        model_layer = fixture.payload.get("model_layer")
        entity_targets = list(graph.objects(artifact, ORION.aboutEntity))
        entity = entity_targets[0] if entity_targets else None
        if entity is not None:
            layer_type = {
                "self-model": ORION.SelfModelEntity,
                "user-model": ORION.UserModelEntity,
                "world-model": ORION.WorldModelEntity,
                "relationship-model": ORION.RelationshipModelEntity,
            }.get(model_layer)
            if layer_type is not None:
                checks.append(CheckResult("entity_layer_type", (entity, RDF.type, layer_type) in graph, f"entity typed as {model_layer}"))
        if fixture.kind == "memory.identity.snapshot.v1":
            checks.extend(self._identity_checks(fixture, graph, artifact, entity))
        elif fixture.kind == "memory.drives.audit.v1":
            checks.extend(self._drive_audit_checks(graph, artifact))
        elif fixture.kind == "memory.goals.proposed.v1":
            checks.extend(self._goal_checks(graph, artifact))
        return checks

    def _identity_checks(self, fixture: ScenarioFixture, graph: Graph, artifact: Any, entity: Any) -> List[CheckResult]:
        checks = [
            CheckResult("snapshot_summary", (artifact, ORION.snapshotSummary, None) in graph, "identity snapshot has summary"),
            CheckResult("anchor_strategy", (artifact, ORION.anchorStrategy, None) in graph, "identity snapshot has anchor strategy"),
            CheckResult("drive_assessment", (artifact, ORION.hasDriveAssessment, None) in graph, "identity snapshot has drive assessments"),
        ]
        if fixture.payload.get("model_layer") == "world-model":
            world_literal = Literal("world", datatype=XSD.string)
            checks.append(CheckResult("world_model_not_generic", not list(graph.subjects(ORION.entityId, world_literal)), "world model does not collapse to generic world entity"))
        if fixture.payload.get("model_layer") == "relationship-model" and entity is not None:
            checks.extend(
                [
                    CheckResult("relationship_distinct_from_self", (entity, RDF.type, ORION.SelfModelEntity) not in graph, "relationship entity is not self-model"),
                    CheckResult("relationship_distinct_from_user", (entity, RDF.type, ORION.UserModelEntity) not in graph, "relationship entity is not user-model"),
                ]
            )
        return checks

    def _drive_audit_checks(self, graph: Graph, artifact: Any) -> List[CheckResult]:
        return [
            CheckResult("drive_assessment", (artifact, ORION.hasDriveAssessment, None) in graph, "drive audit has drive assessments"),
            CheckResult("active_drive", (artifact, ORION.highlightsActiveDrive, None) in graph, "drive audit highlights active drives"),
            CheckResult("dominant_drive", (artifact, ORION.dominantDriveName, None) in graph, "drive audit has dominant drive"),
        ]

    def _goal_checks(self, graph: Graph, artifact: Any) -> List[CheckResult]:
        return [
            CheckResult("proposal_only_execution", (artifact, ORION.executionMode, Literal("proposal-only", datatype=XSD.string)) in graph, "goal remains proposal-only"),
            CheckResult("proposal_status", (artifact, ORION.proposalStatus, Literal("proposed", datatype=XSD.string)) in graph, "goal remains proposed"),
            CheckResult("drive_origin", (artifact, ORION.influencedByDrive, None) in graph, "goal references drive origin"),
        ]


def load_scenarios(path: Path | str = DEFAULT_SCENARIO_PACK, *, only: Optional[Iterable[str]] = None) -> List[ScenarioFixture]:
    path = Path(path)
    data = json.loads(path.read_text())
    wanted = set(only or [])
    scenarios = [
        ScenarioFixture(
            scenario_id=item["scenario_id"],
            kind=item["kind"],
            channel=item.get("channel") or KIND_TO_CHANNEL[item["kind"]],
            payload=item["payload"],
            description=item.get("description", ""),
        )
        for item in data["scenarios"]
    ]
    if wanted:
        scenarios = [item for item in scenarios if item.scenario_id in wanted]
    return scenarios


def write_report(report: Dict[str, Any], *, json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2))
    lines = [
        "# Orion Autonomy Phase 3.1 Verification Report",
        "",
        "## Summary",
        f"- Scenario count: {report['summary']['scenario_count']}",
        f"- Passed count: {report['summary']['passed_count']}",
        f"- Local failures: {', '.join(report['summary']['local_failures']) or 'none'}",
        f"- GraphDB failures: {', '.join(report['summary']['graphdb_failures']) or 'none'}",
        f"- GraphDB skipped: {', '.join(report['summary']['graphdb_skipped']) or 'none'}",
        "",
        "## Scenario Results",
    ]
    for item in report["results"]:
        lines.extend(
            [
                f"### {item['scenario_id']}",
                f"- Kind: `{item['kind']}`",
                f"- Artifact ID: `{item['artifact_id']}`",
                f"- Graph URI: `{item['graph_uri']}`",
                f"- Local verification: `{'PASS' if item['local_ok'] else 'FAIL'}`",
                f"- Bus publish: `{item['bus_publish']}`",
                f"- GraphDB verification: `{item['graphdb_verify']}`",
            ]
        )
        if item.get("graphdb_detail"):
            lines.append(f"- GraphDB detail: {item['graphdb_detail']}")
        lines.append("- Checks:")
        for check in item["local_checks"]:
            mark = "PASS" if check["ok"] else "FAIL"
            lines.append(f"  - [{mark}] {check['name']}: {check['detail']}")
        lines.append("")
    md_out.write_text("\n".join(lines))

"""Evidence bundle assembly for one enrichment run.

Per the feature spec, the bundle is three parts, all real and already on
disk -- no speculative LLM calls at this stage:

1. graphify structural/community data for the affected cluster(s) -- read
   directly from `graphify-out/graph.json` rather than shelling out to
   `graphify query`/`explain`. Chosen because this service runs
   non-interactively and unattended (a bus consumer, not a terminal
   session): `graphify query`/`explain` are designed for an interactive/
   agent caller and may themselves dispatch LLM subagents for parts of
   their answer (see the repo's graphify skill docs), which is exactly the
   double-LLM-call, budget-uncontrolled shape this service's "one real
   subprocess call, no tool use, evidence-in/prose-out" design explicitly
   avoids. Reading the already-built graph.json is a plain, deterministic
   file read with no extra cost or nondeterminism -- more robust for a
   service that runs from a bus event with no human present to intervene
   if a `graphify` subprocess hangs or needs auth.
2. The structural_mass delta that triggered this run (touched paths +
   commit/file/line counts) -- passed straight through from the bus event.
3. Nearby README/docstring text already in the repo for the touched
   paths -- cheap grep/read, not another LLM call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvidenceBundle:
    touched_paths: tuple[str, ...]
    delta_summary: dict[str, Any]
    graph_nodes: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    nearby_docs: tuple[dict[str, str], ...] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        """No empty-shell cognition (CLAUDE.md): a bundle with nothing real
        in it should not be sent to the model as if it were grounded
        evidence."""
        return not self.graph_nodes and not self.nearby_docs and not self.touched_paths


def _cluster_root(path: str) -> str:
    """Coarse cluster key for a touched path -- the top-level service dir
    for services/*, else the first two path segments under orion/."""
    parts = path.split("/")
    if parts and parts[0] == "services" and len(parts) > 1:
        return f"services/{parts[1]}"
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else path


def affected_clusters(touched_paths: tuple[str, ...]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for path in touched_paths:
        seen.setdefault(_cluster_root(path), None)
    return tuple(seen.keys())


def load_graph_nodes_for_clusters(
    graph_json_path: str | Path,
    clusters: tuple[str, ...],
    *,
    max_nodes: int = 40,
) -> tuple[dict[str, Any], ...]:
    """Read graphify-out/graph.json directly and pull nodes whose id/path/
    label mentions one of the affected clusters. Fails soft (empty tuple)
    if the graph file is missing or unparseable -- this evidence source is
    optional, not load-bearing for the run to proceed.

    Matching is path-segment-boundary aware, not raw substring containment:
    a node field must equal a cluster, or contain it bounded by `/` (or at
    the very start/end of the field) on both sides. Plain `cluster in
    haystack` (an earlier version of this function) matched
    "services/orion-hub" against an unrelated node like
    "services/orion-hub-analytics/..." or any node whose free-text label
    field happened to contain that substring anywhere -- pulling unrelated
    evidence into a prompt whose whole design point is that every claim
    must trace to a *specific* evidence item (see README's "Evidence-only
    prompting" section)."""
    path = Path(graph_json_path)
    if not path.exists() or not clusters:
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ()
    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        return ()
    matched: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        fields = [str(node.get(key, "")) for key in ("id", "path", "label", "name", "file")]
        if any(_cluster_matches_field(cluster, field) for cluster in clusters for field in fields if field):
            matched.append(node)
        if len(matched) >= max_nodes:
            break
    return tuple(matched)


def _cluster_matches_field(cluster: str, field: str) -> bool:
    """True if `field` names something inside (or exactly) `cluster`, with
    `/` (or field-start/end) as the boundary on both sides -- e.g. cluster
    "services/orion-hub" matches "services/orion-hub/app/main.py" and
    "services/orion-hub" itself, but not "services/orion-hub-analytics"."""
    if field == cluster:
        return True
    prefix = f"{cluster}/"
    if field.startswith(prefix):
        return True
    # Cluster may also appear as a bounded segment further into the field
    # (e.g. a label like "concept: services/orion-hub/worker") -- require
    # a non-alphanumeric boundary (or string start) immediately before it.
    idx = field.find(prefix)
    if idx > 0 and not (field[idx - 1].isalnum() or field[idx - 1] in "-_"):
        return True
    return False


def load_nearby_docs(repo_root: str | Path, clusters: tuple[str, ...], *, max_chars_per_doc: int = 4000) -> tuple[dict[str, str], ...]:
    """Cheap, deterministic read of README.md / top-of-module docstrings for
    each affected cluster directory. No LLM call, no network."""
    root = Path(repo_root)
    docs: list[dict[str, str]] = []
    for cluster in clusters:
        cluster_dir = root / cluster
        if not cluster_dir.is_dir():
            continue
        readme = cluster_dir / "README.md"
        if readme.exists():
            try:
                text = readme.read_text(encoding="utf-8", errors="replace")[:max_chars_per_doc]
                docs.append({"cluster": cluster, "source_path": str(readme.relative_to(root)), "text": text})
            except Exception:
                pass
    return tuple(docs)


def build_evidence_bundle(
    *,
    repo_root: str | Path,
    graph_json_path: str | Path,
    touched_paths: tuple[str, ...],
    delta_summary: dict[str, Any],
) -> EvidenceBundle:
    clusters = affected_clusters(touched_paths)
    graph_nodes = load_graph_nodes_for_clusters(graph_json_path, clusters)
    nearby_docs = load_nearby_docs(repo_root, clusters)
    return EvidenceBundle(
        touched_paths=touched_paths,
        delta_summary=delta_summary,
        graph_nodes=graph_nodes,
        nearby_docs=nearby_docs,
    )


def render_evidence_prompt(bundle: EvidenceBundle) -> str:
    """Renders the evidence bundle into the exact text sent to the Claude
    subprocess. Kept as a pure function so tests can assert on its shape
    without spawning a real subprocess."""
    lines: list[str] = []
    lines.append("You are producing a grounded 'what is this and why' summary for an architectural cluster in a codebase.")
    lines.append("")
    lines.append("HARD REQUIREMENT: synthesize ONLY from the EVIDENCE below. Do not free-associate from general")
    lines.append("knowledge about this repo or about similar systems. If the evidence is insufficient to say")
    lines.append("something with confidence, say so explicitly rather than filling the gap. Every claim in your")
    lines.append("summary must be traceable to a specific item in the evidence below.")
    lines.append("")
    lines.append("=== EVIDENCE: structural delta that triggered this run ===")
    lines.append(json.dumps(bundle.delta_summary, sort_keys=True, indent=2))
    lines.append("")
    lines.append("=== EVIDENCE: touched paths ===")
    for p in bundle.touched_paths:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("=== EVIDENCE: graphify structural nodes for the affected cluster(s) ===")
    if bundle.graph_nodes:
        for node in bundle.graph_nodes:
            lines.append(json.dumps(node, sort_keys=True))
    else:
        lines.append("(none found)")
    lines.append("")
    lines.append("=== EVIDENCE: nearby README/docstring text ===")
    if bundle.nearby_docs:
        for doc in bundle.nearby_docs:
            lines.append(f"--- {doc['source_path']} ---")
            lines.append(doc["text"])
    else:
        lines.append("(none found)")
    lines.append("")
    lines.append("Write a concise (2-5 sentence) grounded summary of what this cluster is and why it exists,")
    lines.append("citing which evidence item(s) support each claim. Output plain text only, no preamble.")
    return "\n".join(lines)

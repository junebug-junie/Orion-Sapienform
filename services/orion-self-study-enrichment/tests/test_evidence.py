from __future__ import annotations

import json
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.evidence import (  # noqa: E402
    EvidenceBundle,
    affected_clusters,
    build_evidence_bundle,
    load_graph_nodes_for_clusters,
    load_nearby_docs,
    render_evidence_prompt,
)


def test_affected_clusters_dedupes_service_dirs():
    paths = (
        "services/orion-foo/app/main.py",
        "services/orion-foo/README.md",
        "orion/bus/channels.yaml",
    )
    assert affected_clusters(paths) == ("services/orion-foo", "orion/bus")


def test_load_graph_nodes_for_clusters_matches_and_caps(tmp_path):
    graph = {
        "nodes": [
            {"id": "services/orion-foo/app/main.py", "label": "main"},
            {"id": "services/orion-bar/app/main.py", "label": "main"},
        ]
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph))
    nodes = load_graph_nodes_for_clusters(graph_path, ("services/orion-foo",))
    assert len(nodes) == 1
    assert nodes[0]["id"] == "services/orion-foo/app/main.py"


def test_load_graph_nodes_for_clusters_missing_file_fails_soft(tmp_path):
    assert load_graph_nodes_for_clusters(tmp_path / "nope.json", ("x",)) == ()


def test_load_graph_nodes_for_clusters_does_not_match_similarly_prefixed_cluster(tmp_path):
    # Regression: "services/orion-hub" must NOT match
    # "services/orion-hub-analytics/..." -- plain substring containment did.
    graph = {
        "nodes": [
            {"id": "services/orion-hub/app/main.py", "label": "main"},
            {"id": "services/orion-hub-analytics/app/main.py", "label": "main"},
        ]
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph))
    nodes = load_graph_nodes_for_clusters(graph_path, ("services/orion-hub",))
    assert len(nodes) == 1
    assert nodes[0]["id"] == "services/orion-hub/app/main.py"


def test_load_nearby_docs_reads_readme(tmp_path):
    cluster_dir = tmp_path / "services" / "orion-foo"
    cluster_dir.mkdir(parents=True)
    (cluster_dir / "README.md").write_text("This is orion-foo.")
    docs = load_nearby_docs(tmp_path, ("services/orion-foo",))
    assert len(docs) == 1
    assert docs[0]["text"] == "This is orion-foo."


def test_evidence_bundle_is_empty_when_nothing_real(tmp_path):
    bundle = EvidenceBundle(touched_paths=(), delta_summary={})
    assert bundle.is_empty()


def test_build_evidence_bundle_not_empty_with_real_touched_paths(tmp_path):
    (tmp_path / "services" / "orion-foo").mkdir(parents=True)
    bundle = build_evidence_bundle(
        repo_root=tmp_path,
        graph_json_path=tmp_path / "missing.json",
        touched_paths=("services/orion-foo/app/main.py",),
        delta_summary={"commit_count": 1},
    )
    assert not bundle.is_empty()


def test_render_evidence_prompt_requires_evidence_grounding():
    bundle = EvidenceBundle(
        touched_paths=("services/orion-foo/app/main.py",),
        delta_summary={"commit_count": 1},
        graph_nodes=({"id": "services/orion-foo"},),
        nearby_docs=({"cluster": "services/orion-foo", "source_path": "services/orion-foo/README.md", "text": "foo"},),
    )
    prompt = render_evidence_prompt(bundle)
    assert "ONLY from the EVIDENCE" in prompt
    assert "services/orion-foo/app/main.py" in prompt
    assert "foo" in prompt

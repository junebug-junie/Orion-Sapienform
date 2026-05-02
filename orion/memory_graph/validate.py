from __future__ import annotations

from pathlib import Path
from typing import List

import pyshacl
from rdflib import Graph


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _shapes_path() -> Path:
    return _repo_root() / "ontology" / "memory" / "shapes-orionmem-v2026-05.ttl"


def validate_graph(data_graph: Graph) -> List[str]:
    """SHACL validation using packaged shapes; returns human-readable violation lines."""
    shapes_path = _shapes_path()
    if not shapes_path.is_file():
        return [f"shapes_file_missing:{shapes_path}"]

    shapes_graph = Graph()
    shapes_graph.parse(shapes_path.as_posix(), format="turtle")

    conforms, _, results_text = pyshacl.validate(
        data_graph,
        shacl_graph=shapes_graph,
        inference="none",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=False,
    )
    if conforms:
        return []
    lines = [ln.strip() for ln in str(results_text).splitlines() if ln.strip()]
    return lines or ["shacl_validation_failed"]

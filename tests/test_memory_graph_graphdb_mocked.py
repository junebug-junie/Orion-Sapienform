from __future__ import annotations

from unittest.mock import MagicMock

from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.graphdb import insert_batch
from orion.memory_graph.json_to_rdf import draft_to_graph


def test_insert_batch_posts_named_graph_param() -> None:
    draft = SuggestDraftV1.model_validate(
        {
            "ontology_version": "orionmem-2026-05",
            "utterance_ids": ["t1"],
            "entities": [],
            "situations": [],
            "edges": [],
            "dispositions": [],
        }
    )
    g = draft_to_graph(draft)
    sess = MagicMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    sess.post.return_value = resp
    insert_batch(
        g,
        named_graph="https://example.test/graph/ng1",
        graphdb_url="http://gdb.test",
        repo="collapse",
        session=sess,
    )
    assert sess.post.called
    url = sess.post.call_args[0][0]
    assert "named-graph-uri=" in url
    assert "https%3A%2F%2Fexample.test%2Fgraph%2Fng1" in url or "example.test" in url

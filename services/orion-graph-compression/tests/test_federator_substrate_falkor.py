from app.federators.substrate_falkor import FalkorSubstrateFederator


class _FakeClient:
    def __init__(self, rows=None, *, raises=False):
        self._rows = rows or []
        self._raises = raises
        self.calls = []

    def graph_query(self, cypher, params=None):
        self.calls.append((cypher, params))
        if self._raises:
            raise RuntimeError("falkor unreachable")
        return self._rows


def test_fetch_returns_triples_from_rows():
    client = _FakeClient(
        rows=[
            {"s": "concept-1", "p": "CONTRADICTS", "o": "concept-2"},
            {"s": "concept-2", "p": "SUPPORTS", "o": "concept-3"},
        ]
    )
    f = FalkorSubstrateFederator(client=client)
    triples = f.fetch()
    assert ("concept-1", "CONTRADICTS", "concept-2") in triples
    assert ("concept-2", "SUPPORTS", "concept-3") in triples
    assert len(triples) == 2


def test_fetch_query_matches_substrate_node_label_generically():
    """Must not hardcode a specific predicate/label set -- relationship type
    is recovered via type(r), matching any predicate upsert_edge() writes."""
    client = _FakeClient(rows=[])
    FalkorSubstrateFederator(client=client).fetch()
    cypher = client.calls[0][0]
    assert "SubstrateNode" in cypher
    assert "type(r)" in cypher


def test_fetch_degrades_to_empty_list_on_client_error():
    client = _FakeClient(raises=True)
    triples = FalkorSubstrateFederator(client=client).fetch()
    assert triples == []


def test_fetch_degrades_to_empty_list_when_no_client_configured(monkeypatch):
    monkeypatch.delenv("FALKORDB_URI", raising=False)
    triples = FalkorSubstrateFederator().fetch()
    assert triples == []


def test_fetch_skips_rows_with_missing_fields():
    client = _FakeClient(
        rows=[
            {"s": "concept-1", "p": None, "o": "concept-2"},
            {"s": "concept-3", "p": "REFINES", "o": "concept-4"},
        ]
    )
    triples = FalkorSubstrateFederator(client=client).fetch()
    assert triples == [("concept-3", "REFINES", "concept-4")]

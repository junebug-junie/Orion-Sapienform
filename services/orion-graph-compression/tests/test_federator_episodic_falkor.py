from app.federators.episodic_falkor import FalkorEpisodicFederator, _to_iri


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


def test_fetch_returns_triples_across_node_labels():
    client = _FakeClient(
        rows=[
            {"s": "session-1", "p": "HAS_TURN", "o": "turn-1"},
            {"s": "turn-1", "p": "MENTIONS_ENTITY", "o": "Juniper"},
            {"s": "turn-1", "p": "HAS_TAG", "o": "gpu"},
        ]
    )
    triples = FalkorEpisodicFederator(client=client).fetch()
    assert (_to_iri("session-1"), "HAS_TURN", _to_iri("turn-1")) in triples
    assert (_to_iri("turn-1"), "MENTIONS_ENTITY", _to_iri("Juniper")) in triples
    assert (_to_iri("turn-1"), "HAS_TAG", _to_iri("gpu")) in triples
    assert len(triples) == 3


def test_fetch_wraps_free_text_entity_names_as_well_formed_iris():
    """Real orion_recall Entity/Tag .name values are raw free chat text (live
    examples: "solar system", "the 'sentience striving program'") -- spaces
    and apostrophes are illegal inside a SPARQL IRIREF, and writer.py's
    _build_sparql_update interpolates node identity strings straight into
    one with zero escaping. Every returned node must be a well-formed IRI
    regardless of how messy the underlying Falkor property value is."""
    client = _FakeClient(
        rows=[{"s": "turn-1", "p": "MENTIONS_ENTITY", "o": "the 'sentience striving program'"}]
    )
    triples = FalkorEpisodicFederator(client=client).fetch()
    s, p, o = triples[0]
    assert o.startswith("http://conjourney.net/orion/recall/falkor/")
    assert " " not in o and "'" not in o and "<" not in o and ">" not in o


def test_fetch_query_covers_all_three_relationship_types():
    client = _FakeClient(rows=[])
    FalkorEpisodicFederator(client=client).fetch()
    cypher = client.calls[0][0]
    assert "HAS_TURN" in cypher
    assert "HAS_TAG" in cypher
    assert "MENTIONS_ENTITY" in cypher
    assert "coalesce" in cypher


def test_fetch_query_coalesce_includes_collapse_id():
    """CollapseEvent (services/orion-meta-tags's collapse-triage writer,
    2026-07-22) has no turn_id/session_id/name property -- only collapse_id.
    Without it in the coalesce, a CollapseEvent-sourced edge resolves to a
    NULL node id and gets silently dropped by the `if s and p and o` check
    below, once RECALL_FALKOR_COLLAPSE_TRIAGE_ENABLED flips on."""
    client = _FakeClient(rows=[])
    FalkorEpisodicFederator(client=client).fetch()
    cypher = client.calls[0][0]
    assert "collapse_id" in cypher


def test_fetch_does_not_drop_collapse_event_sourced_rows():
    client = _FakeClient(
        rows=[{"s": "collapse_abc123", "p": "MENTIONS_ENTITY", "o": "gnostic"}]
    )
    triples = FalkorEpisodicFederator(client=client).fetch()
    assert triples == [(_to_iri("collapse_abc123"), "MENTIONS_ENTITY", _to_iri("gnostic"))]


def test_fetch_degrades_to_empty_list_on_client_error():
    client = _FakeClient(raises=True)
    triples = FalkorEpisodicFederator(client=client).fetch()
    assert triples == []


def test_fetch_degrades_to_empty_list_when_no_client_configured(monkeypatch):
    monkeypatch.delenv("FALKORDB_URI", raising=False)
    triples = FalkorEpisodicFederator().fetch()
    assert triples == []


def test_fetch_skips_rows_with_missing_fields():
    client = _FakeClient(
        rows=[
            {"s": "turn-1", "p": "MENTIONS_ENTITY", "o": None},
            {"s": "turn-2", "p": "HAS_TAG", "o": "circe"},
        ]
    )
    triples = FalkorEpisodicFederator(client=client).fetch()
    assert triples == [(_to_iri("turn-2"), "HAS_TAG", _to_iri("circe"))]

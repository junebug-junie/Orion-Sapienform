"""Who writes or reads a Chroma collection / FalkorDB graph, from the code that names it."""
from __future__ import annotations

import ast
import textwrap

import pytest

from orion.body_atlas.__main__ import summarize
from orion.body_atlas.build import build_atlas
from orion.body_atlas.sources import BodyInputs, StoreIn
from orion.body_atlas.store_code import code_store_refs, scan_module, verb_relation
from orion.body_atlas.tests.test_build import CATALOG, NOW


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


def _scan(src, anchors):
    return scan_module(ast.parse(textwrap.dedent(src)), anchors)


def _refs(root, names):
    out = code_store_refs(root, names)
    return {s: {(r.service, r.relation) for r in refs} for s, refs in out.refs.items()}, out


# --- direction ------------------------------------------------------------------------------------

def test_chroma_writer_and_reader_from_the_call():
    writer = """
        NAME = "orion_social_room"
        col = client.get_or_create_collection(NAME)
        col.upsert(ids=["a"], documents=["x"])
    """
    reader = """
        def search(client):
            return client.get_collection("orion_social_room").query(query_texts=["x"])
    """
    anchors = {"orion_social_room": "orion_social_room"}
    assert _scan(writer, anchors) == {"orion_social_room": "writes"}
    assert _scan(reader, anchors) == {"orion_social_room": "reads"}


def test_a_handle_on_self_is_followed_across_methods():
    src = """
        COLLECTION = "orion_chat"
        class Store:
            def __init__(self, client):
                self._col = client.get_or_create_collection(name=COLLECTION)
            def put(self, doc):
                self._col.add(ids=[doc.id], documents=[doc.text])
    """
    assert _scan(src, {"orion_chat": "orion_chat"}) == {"orion_chat": "writes"}


def test_cypher_text_decides_a_falkor_query_and_read_only_reads():
    write = """
        g = db.select_graph("orion_kg")
        g.query("MERGE (n:Concept {id: $id}) SET n.seen = 1", {"id": 1})
    """
    read = """
        g = db.select_graph("orion_kg")
        rows = g.query("MATCH (n) RETURN count(n)")
    """
    ro = """
        client = RedisGraphQueryClient(uri=u, graph_name="orion_kg", read_only=True)
    """
    a = {"orion_kg": "orion_kg"}
    assert _scan(write, a) == {"orion_kg": "writes"}
    assert _scan(read, a) == {"orion_kg": "reads"}
    assert _scan(ro, a) == {"orion_kg": "reads"}


def test_a_call_whose_verb_says_neither_is_uses():
    src = """
        cfg = SearchConfig(collection="orion_atlas_concepts", min_similarity=0.4)
    """
    assert _scan(src, {"orion_atlas_concepts": "orion_atlas_concepts"}) == {"orion_atlas_concepts": "uses"}
    # a publish helper whose name carries the write verb is a write
    pub = """
        await _publish_semantic_upsert(env=env, collection_name="orion_social_room")
    """
    assert _scan(pub, {"orion_social_room": "orion_social_room"}) == {"orion_social_room": "writes"}


def test_verbs_split_snake_and_camel_case_and_ambiguous_names_say_nothing():
    assert verb_relation("_publish_semantic_upsert") == "writes"
    assert verb_relation("graphQuery") == "reads"
    assert verb_relation("get_or_create_collection") is None  # both: a handle, not an act
    assert verb_relation("SearchConfig") is None  # a constructor, not an act
    assert verb_relation("Settings") is None
    assert verb_relation("build_store") is None  # "store" is a noun here, not an act


# --- naming: exact, never a substring --------------------------------------------------------------

def test_no_false_match_on_substrings_docstrings_or_fstrings():
    src = '''
        """This module mirrors orion_kg nightly."""  # a sentence that mentions it
        OTHER = "orion_kg_v2"
        PREFIX = f"orion_{kind}"
        LABEL = "orion_kg backup"
        # GRAPH = "orion_kg"
    '''
    assert _scan(src, {"orion_kg": "orion_kg"}) == {}


# --- the repo scan: services, env keys, shared modules ---------------------------------------------

def test_service_module_names_a_store_and_tests_evals_are_not_the_body(tmp_path):
    _write(tmp_path, "services/orion-vector-host/app/main.py", """
        SOCIAL = "orion_social_room"
        async def h(env):
            await publish_upsert(collection_name=SOCIAL)
    """)
    _write(tmp_path, "services/orion-hub/tests/test_x.py", 'X = "orion_social_room"\n')
    _write(tmp_path, "services/orion-hub/evals/run_x.py", 'X = "orion_social_room"\n')
    _write(tmp_path, "orion/body_atlas/sources.py", 'GAP = "orion_social_room"\n')  # the observer itself
    refs, _ = _refs(tmp_path, {"orion_social_room"})
    assert refs == {"orion_social_room": {("orion-vector-host", "writes")}}


def test_env_key_indirection_through_env_example_and_compose_default(tmp_path):
    _write(tmp_path, "services/orion-bus-mirror/.env_example", "FALKORDB_BUS_GRAPH=orion_bus_synapse\n")
    _write(tmp_path, "services/orion-bus-mirror/app/main.py", """
        import os
        g = db.select_graph(os.getenv("FALKORDB_BUS_GRAPH"))
        g.query("CREATE (n:Ping)")
    """)
    _write(tmp_path, "services/orion-meta-tags/docker-compose.yml", """
        services:
          meta:
            environment:
              - RECALL_GRAPH=${RECALL_GRAPH:-orion_recall}
    """)
    _write(tmp_path, "services/orion-meta-tags/app/main.py", """
        client = RedisGraphQueryClient(uri=u, graph_name=settings.RECALL_GRAPH, read_only=True)
    """)
    # a service whose env does not set the key gets nothing from reading it
    _write(tmp_path, "services/orion-other/app/main.py", """
        import os
        g = db.select_graph(os.getenv("FALKORDB_BUS_GRAPH"))
    """)
    refs, _ = _refs(tmp_path, {"orion_bus_synapse", "orion_recall"})
    assert refs == {"orion_bus_synapse": {("orion-bus-mirror", "writes")},
                    "orion_recall": {("orion-meta-tags", "reads")}}


def test_shared_module_goes_to_the_services_that_import_it(tmp_path):
    _write(tmp_path, "orion/world_pulse_read/concept_search.py", """
        ATLAS_CONCEPT_COLLECTION = "orion_atlas_concepts"
    """)
    _write(tmp_path, "orion/world_pulse_read/__init__.py", "")
    # imports the constant and searches with it
    _write(tmp_path, "services/orion-hub/scripts/main.py", """
        from orion.world_pulse_read.concept_search import ATLAS_CONCEPT_COLLECTION
        hits = chroma.search(collection=ATLAS_CONCEPT_COLLECTION)
    """)
    # imports the module by its parent package path
    _write(tmp_path, "services/orion-reader/app/main.py", """
        from orion.world_pulse_read import concept_search
    """)
    # a shared module that names a store but no service imports
    _write(tmp_path, "orion/lonely/graphs.py", 'G = "orion_express"\n')
    refs, out = _refs(tmp_path, {"orion_atlas_concepts", "orion_express"})
    assert refs == {"orion_atlas_concepts": {("orion-hub", "reads"), ("orion-reader", "uses")}}
    assert out.unattributed == {"orion_express": ["orion/lonely/graphs.py"]}
    via = {r.service: r.via for r in out.refs["orion_atlas_concepts"]}
    assert via["orion-reader"] == "orion/world_pulse_read/concept_search.py:ATLAS_CONCEPT_COLLECTION (imported by orion-reader)"


def test_an_importer_gets_the_stores_of_the_symbols_it_uses_not_the_whole_module(tmp_path):
    # live case: orion/substrate/falkor_store.py builds the main graph and two side graphs;
    # importing the main builder is not writing the side graphs
    _write(tmp_path, "orion/substrate/falkor_store.py", """
        def build_store(name="orion_substrate"):
            return db.select_graph(name)
        def build_aitown_store():
            return build_store(name="orion_substrate_aitown")
    """)
    _write(tmp_path, "services/orion-recall/app/main.py", """
        from orion.substrate.falkor_store import build_store
    """)
    _write(tmp_path, "services/orion-hub/app/routes.py", """
        from orion.substrate import falkor_store as fs
        store = fs.build_aitown_store()
    """)
    refs, out = _refs(tmp_path, {"orion_substrate", "orion_substrate_aitown"})
    assert refs == {"orion_substrate": {("orion-recall", "uses")},
                    "orion_substrate_aitown": {("orion-hub", "uses")}}
    assert out.unattributed == {}


def test_env_key_inside_a_shared_module_resolves_per_importing_service(tmp_path):
    _write(tmp_path, "orion/curiosity/write_stamp.py", """
        import os
        def stamp(db):
            db.select_graph(os.getenv("CURIOSITY_GRAPH")).query("MERGE (s:Stamp)")
    """)
    _write(tmp_path, "services/orion-hub/.env_example", "CURIOSITY_GRAPH=orion_worldview\n")
    _write(tmp_path, "services/orion-hub/app/main.py", "from orion.curiosity.write_stamp import stamp\n")
    _write(tmp_path, "services/orion-peer/.env_example", "CURIOSITY_GRAPH=orion_peer_graph\n")
    _write(tmp_path, "services/orion-peer/app/main.py", "from orion.curiosity import write_stamp\n")
    refs, _ = _refs(tmp_path, {"orion_worldview", "orion_peer_graph"})
    assert refs == {"orion_worldview": {("orion-hub", "writes")},
                    "orion_peer_graph": {("orion-peer", "writes")}}


def test_a_file_that_does_not_parse_is_reported_not_fatal(tmp_path):
    _write(tmp_path, "services/orion-x/app/broken.py", 'X = "orion_kg"\ndef (:\n')
    refs, out = _refs(tmp_path, {"orion_kg"})
    assert refs == {} and out.skipped == ["services/orion-x/app/broken.py"]


# --- the body: stores join their writers' organs; orphans are named --------------------------------

def test_stores_named_in_code_leave_singletons_and_orphans_are_listed():
    stores = [
        StoreIn(name="orion_social_room", store="chroma", rows=40,
                links=[("orion-vision-host", "writes", "declared_code", "services/orion-vision-host/app/main.py")]),
        StoreIn(name="orion_atlas_concepts", store="chroma", rows=9,
                links=[("orion-cortex-exec", "reads", "declared_code", "services/orion-cortex-exec/app/a.py"),
                       ("orion-cortex-exec", "uses", "declared_env", "services/orion-cortex-exec/.env_example")]),
        StoreIn(name="orion_kg", store="falkordb", rows=0, named_in_unimported=["orion/lonely/graphs.py"]),
    ]
    atlas = build_atlas(CATALOG, None, now=NOW, body=BodyInputs(stores=stores))
    organ_of = {m: o for o in atlas.organs for m in o.members}
    assert "part:orion-vision-host" in organ_of["part:chroma:orion_social_room"].members
    assert "part:orion-cortex-exec" in organ_of["part:chroma:orion_atlas_concepts"].members
    assert organ_of["part:falkor:orion_kg"].members == ["part:falkor:orion_kg"]
    flows = {(f.source, f.target): f for f in atlas.flows}
    one = flows[("part:orion-cortex-exec", "part:chroma:orion_atlas_concepts")]
    # one edge per service: its strongest relation, every evidence kept, the whole unit
    assert one.relation == "reads" and one.evidence == ["declared_code", "declared_env"]
    assert one.weight == pytest.approx(1.0) and one.channels == []
    parts = {p.name: p for p in atlas.parts}
    assert parts["chroma:orion_social_room"].facts["written_by"] == "orion-vision-host"
    assert "orion/lonely/graphs.py" in parts["falkor:orion_kg"].facts["evidence"]
    text = summarize(atlas, out=None)
    assert "UNATTRIBUTED 1 store(s)" in text and "falkor:orion_kg (empty)" in text


def test_a_registry_naming_every_store_keeps_a_small_share():
    """The rarity rule: a module that names one store outweighs a registry that names ten."""
    registry = "orion/memory/vector_collection_policy.py (imported by orion-equilibrium-service)"
    stores = [StoreIn(name=f"c{i}", store="chroma", rows=1,
                      links=[("orion-equilibrium-service", "reads", "declared_code", registry)])
              for i in range(10)]
    stores[0].links.append(("orion-vision-host", "uses", "declared_code", "services/orion-vision-host/app/x.py"))
    atlas = build_atlas(CATALOG, None, now=NOW, body=BodyInputs(stores=stores))
    w = {f.source: f.weight for f in atlas.flows if f.target == "part:chroma:c0"}
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["part:orion-vision-host"] > 2 * w["part:orion-equilibrium-service"]
    # equal sources: the plain one-unit split
    even = [StoreIn(name="s", store="chroma", rows=1,
                    links=[(svc, "uses", "declared_env", "") for svc in ("orion-cortex-exec", "orion-vision-host")])]
    atlas = build_atlas(CATALOG, None, now=NOW, body=BodyInputs(stores=even))
    assert [f.weight for f in atlas.flows if f.target == "part:chroma:s"] == pytest.approx([0.5, 0.5])

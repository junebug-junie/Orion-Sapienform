import time
import requests
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD
from app.settings import settings
from app.utils import logger

# Namespace for Orion Collapse terms
CM = Namespace("http://orion.ai/collapse#")

# Derived endpoints
REST_REPOS = f"{settings.GRAPHDB_URL}/rest/repositories"
STATEMENTS_URL = f"{settings.GRAPHDB_URL}/repositories/{settings.GRAPHDB_REPO}/statements"


def wait_for_graphdb(max_wait_sec: int = 120, interval: int = 3):
    """
    Poll GraphDB until it responds or timeout expires.
    """
    start = time.time()
    while time.time() - start < max_wait_sec:
        try:
            r = requests.get(REST_REPOS, timeout=5)
            if r.status_code == 200:
                logger.info("✅ GraphDB reachable")
                return
        except requests.RequestException:
            pass
        logger.info("⏳ Waiting for GraphDB…")
        time.sleep(interval)
    raise RuntimeError("❌ GraphDB did not become ready in time")


def ensure_repo_exists():
    """
    Ensure the target repository exists; create it if missing.
    """
    try:
        r = requests.get(REST_REPOS, timeout=10)
        r.raise_for_status()
        repos = r.json()
        if any(x.get("id") == settings.GRAPHDB_REPO for x in repos):
            logger.info("✅ Repository '%s' present", settings.GRAPHDB_REPO)
            return
    except Exception as e:
        logger.warning("⚠️ Failed to list repositories: %s", e)

    cfg_ttl = f"""
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rep:     <http://www.openrdf.org/config/repository#> .
@prefix sr:      <http://www.openrdf.org/config/repository/sail#> .
@prefix sail:    <http://www.openrdf.org/config/sail#> .
@prefix graphdb: <http://www.ontotext.com/trree/graphdb#> .

[] a rep:Repository ;
   rep:repositoryID "{settings.GRAPHDB_REPO}" ;
   rdfs:label "Orion GDB Client Repository" ;
   rep:repositoryImpl [
     rep:repositoryType "graphdb:SailRepository" ;
     sr:sailImpl [
       sail:sailType "graphdb:SailStore" ;
       graphdb:ruleset "rdfsplus-optimized"
     ]
   ] .
""".strip()

    r = requests.post(
        REST_REPOS,
        headers={"Content-Type": "text/turtle"},
        data=cfg_ttl.encode("utf-8"),
        timeout=15,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"❌ Repo creation failed: {r.status_code} {r.text[:300]}")
    logger.info("✅ Created repository '%s'", settings.GRAPHDB_REPO)


def process_raw_collapse(entry_id: str, payload: dict) -> int:
    """
    Converts a raw collapse event into the base RDF triples for that event.
    """
    g = Graph()
    g.bind("cm", CM)
    subject = URIRef(f"{CM}{entry_id}")
    g.add((subject, RDF.type, CM.Collapse))
    g.add((subject, CM.id, Literal(entry_id, datatype=XSD.string)))

    for key, val in payload.items():
        if val is None or key in ["id", "service_name", "text"]:
            continue
        if isinstance(val, list):
            val = ", ".join(map(str, val))
        g.add((subject, URIRef(str(CM) + key), Literal(val, datatype=XSD.string)))
    push_graph(g)
    return len(g)


def process_enrichment(entry_id: str, payload: dict) -> int:
    """
    Adds enrichment data (tags, entities) as new triples to an existing node.
    """
    collapse_id = payload.get("collapse_id", entry_id)
    g = Graph()
    g.bind("cm", CM)
    subject = URIRef(f"{CM}{collapse_id}")

    for tag in payload.get("tags", []):
        g.add((subject, CM.hasTag, Literal(tag, datatype=XSD.string)))
    for entity in payload.get("entities", []):
        if entity.get("value") and entity.get("type"):
            g.add((subject, CM.hasEntity, Literal(f"{entity['value']} ({entity['type']})", datatype=XSD.string)))
    if g:
        push_graph(g)
    return len(g)


def push_graph(g: Graph):
    """
    Push the RDF graph into GraphDB.
    """
    if not g:
        return
    ttl = g.serialize(format="turtle")
    r = requests.post(
        STATEMENTS_URL,
        headers={"Content-Type": "text/turtle"},
        data=ttl.encode("utf-8"),
        timeout=15,
    )
    r.raise_for_status()


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
   rep:repositoryID "{settings.graphdb_repo}" ;
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


def build_graph(entry_id: str, payload: dict) -> Graph:
    """
    Convert a JSON message into RDF triples.
    """
    g = Graph()
    g.bind("cm", CM)

    s = URIRef(f"{CM}{entry_id}")
    g.add((s, RDF.type, CM.Collapse))
    g.add((s, CM.id, Literal(entry_id, datatype=XSD.string)))

    for key, val in payload.items():
        if val is None:
            continue
        if isinstance(val, list):
            val = ", ".join(map(str, val))
        g.add((s, URIRef(str(CM) + key), Literal(val, datatype=XSD.string)))

    return g


def push_graph(g: Graph):
    """
    Push the RDF graph into GraphDB.
    """
    ttl = g.serialize(format="turtle")
    r = requests.post(
        STATEMENTS_URL,
        headers={"Content-Type": "text/turtle"},
        data=ttl,
        timeout=15,
    )
    if r.status_code not in (200, 204):
        raise RuntimeError(f"❌ RDF push failed: {r.status_code} {r.text[:200]}")

import os
import sys
import json
import time
import redis
import requests

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

# ---- Environment / Config
REDIS_URL     = os.getenv("REDIS_URL", "redis://orion-redis:6379/0")
CHANNEL       = os.getenv("BUS_CHANNEL", "collapse:new")
GRAPHDB_BASE  = os.getenv("GRAPHDB_URL", "http://graphdb:7200")
REPO          = os.getenv("GRAPHDB_REPO", "collapse")

REST_REPOS    = f"{GRAPHDB_BASE}/rest/repositories"
STATEMENTS_URL= f"{GRAPHDB_BASE}/repositories/{REPO}/statements"

CM = Namespace("http://orion.ai/collapse#")

# ---- Logging helper
def log(*args):
    print(*args, flush=True)

# ---- GraphDB readiness + repo management
def wait_for_graphdb(max_wait_sec=120, interval=3):
    start = time.time()
    while time.time() - start < max_wait_sec:
        try:
            r = requests.get(REST_REPOS, timeout=5)
            if r.status_code == 200:
                log("✅ GraphDB reachable")
                return
        except requests.exceptions.RequestException:
            pass
        log("⏳ Waiting for GraphDB…")
        time.sleep(interval)
    raise RuntimeError("❌ GraphDB did not become ready in time")

def ensure_repo_exists():
    try:
        r = requests.get(REST_REPOS, timeout=10)
        r.raise_for_status()
        repos = r.json()
        if any(x.get("id") == REPO for x in repos):
            log(f"✅ Repository '{REPO}' present")
            return
    except Exception as e:
        log(f"⚠️ Failed to list repositories: {e}")

    # GraphDB 11 config (use graphdb: vocab, SailStore)
    cfg_ttl = f"""
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rep:     <http://www.openrdf.org/config/repository#> .
@prefix sr:      <http://www.openrdf.org/config/repository/sail#> .
@prefix sail:    <http://www.openrdf.org/config/sail#> .
@prefix graphdb: <http://www.ontotext.com/trree/graphdb#> .

[] a rep:Repository ;
   rep:repositoryID "{REPO}" ;
   rdfs:label "Collapse Mirror Repository" ;
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
        timeout=15
    )
    if r.status_code >= 400:
        raise RuntimeError(f"❌ Repo create failed: {r.status_code} {r.text[:500]}")
    log(f"✅ Created repository '{REPO}'")

# ---- RDF builder and push
def build_graph(entry_id: str, payload: dict) -> Graph:
    """
    Map collapse payload to RDF. We:
    - type the subject as cm:Collapse
    - add all scalar fields as cm:<key> "value"
    - collapse lists (e.g., observer_state) to comma-joined strings
    """
    g = Graph()
    g.bind("cm", CM)

    s = URIRef(f"{CM}{entry_id}")
    g.add((s, RDF.type, CM.Collapse))
    g.add((s, CM.id, Literal(entry_id, datatype=XSD.string)))

    for k, v in payload.items():
        if v is None:
            continue
        # flatten lists to a readable string
        if isinstance(v, list):
            v = ", ".join(map(str, v))
        # store everything as xsd:string for simplicity
        g.add((s, URIRef(str(CM) + k), Literal(v, datatype=XSD.string)))

    return g

def push_graph(g: Graph):
    ttl = g.serialize(format="turtle")
    r = requests.post(
        STATEMENTS_URL,
        headers={"Content-Type": "text/turtle"},
        data=ttl,
        timeout=15
    )
    if r.status_code not in (200, 204):
        raise RuntimeError(f"❌ Push failed: {r.status_code} {r.text[:500]}")

# ---- Redis subscriber loop
def main():
    log(f"ENV → REDIS_URL={REDIS_URL}  GRAPHDB={GRAPHDB_BASE}  REPO={REPO}  CHANNEL={CHANNEL}")
    wait_for_graphdb()
    ensure_repo_exists()

    r = redis.Redis.from_url(REDIS_URL)
    ps = r.pubsub()
    ps.subscribe(CHANNEL)
    log(f"📡 Listening on Redis channel '{CHANNEL}'")

    for msg in ps.listen():
        if msg.get("type") != "message":
            continue
        try:
            raw = msg["data"]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw)

            entry_id = data.pop("id", None)
            if not entry_id:
                log("⚠️ Skipping message without 'id'"); continue

            # Build and push RDF
            g = build_graph(entry_id, data)
            push_graph(g)
            log(f"✅ Ingested {entry_id} → GraphDB ({len(g)} triples)")
        except Exception as e:
            log(f"❌ Error ingesting message: {e}",)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"💥 Fatal: {e}")
        sys.exit(1)

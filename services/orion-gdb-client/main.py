import os
import json
import redis
import requests
import time

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
GRAPHDB_BASE = os.getenv("GRAPHDB_URL", "http://graphdb:7200")
REPO = os.getenv("GRAPHDB_REPO", "collapse")
STATEMENTS_URL = f"{GRAPHDB_BASE}/repositories/{REPO}/statements"
REPOS_URL = f"{GRAPHDB_BASE}/rest/repositories"

r = redis.Redis.from_url(REDIS_URL)
CM = Namespace("http://orion.ai/collapse#")

def wait_for_graphdb(timeout=60, interval=3):
    """Poll GraphDB REST API until it's ready or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(REPOS_URL, timeout=5)
            if resp.status_code == 200:
                print("✅ GraphDB is ready")
                return True
        except requests.exceptions.RequestException:
            pass
        print("⏳ Waiting for GraphDB to start...")
        time.sleep(interval)
    raise RuntimeError("❌ GraphDB did not become ready in time")

def ensure_repo_exists():
    resp = requests.get(REPOS_URL)
    if resp.status_code == 200 and REPO in resp.text:
        print(f"ℹ️ Repository '{REPO}' already exists")
        return

    storage_path = f"/opt/graphdb/home/data/repositories/{REPO}"

    ttl = f"""
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix rep: <http://www.openrdf.org/config/repository#> .
    @prefix sr: <http://www.openrdf.org/config/repository/sail#> .
    @prefix sail: <http://www.openrdf.org/config/sail#> .
    @prefix owlim: <http://www.ontotext.com/trree/owlim#> .

    [] a rep:Repository ;
       rep:repositoryID "{REPO}" ;
       rdfs:label "Collapse Mirror Repository" ;
       rep:repositoryImpl [
          rep:repositoryType "graphdb:SailRepository" ;
          sr:sailImpl [
             sail:sailType "graphdb:Sail" ;
             owlim:ruleset "rdfsplus-optimized" ;
             owlim:storage-folder "{storage_path}" ;
          ]
       ] .
    """

    files = {"config": ("repo.ttl", ttl, "application/x-turtle")}
    resp = requests.post(REPOS_URL, files=files)
    print("↩️ Repo creation response:", resp.status_code, resp.text[:500])
    resp.raise_for_status()
    print(f"✅ Created repository '{REPO}'")


def insert_triples(triples):
    """Insert a batch of RDF triples into GraphDB."""
    g = Graph()
    for s, p, o in triples:
        g.add((URIRef(s), URIRef(p), Literal(o)))
    data = g.serialize(format="nt")
    resp = requests.post(STATEMENTS_URL, data=data, headers={"Content-Type": "application/n-triples"})
    resp.raise_for_status()


def handle_event(msg):
    try:
        payload = msg["data"]
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        data = json.loads(payload)

        collapse_id = data.get("id")
        triples = []
        if collapse_id and data.get("trigger"):
            triples.append((f"{CM}{collapse_id}", f"{CM}trigger", data["trigger"]))
        if collapse_id and data.get("reflection"):
            triples.append((f"{CM}{collapse_id}", f"{CM}reflection", data["reflection"]))

        if triples:
            insert_triples(triples)
            print(f"✅ Stored collapse {collapse_id} into GraphDB")
        else:
            print(f"⚠️ No valid triples for event: {data}")

    except Exception as e:
        print(f"⚠️ Error handling event: {e}")


if __name__ == "__main__":
    wait_for_graphdb(timeout=120)
    ensure_repo_exists()
    print(f"🎉 Repository '{REPO}' is ready, waiting for Redis events…")

    pubsub = r.pubsub()
    pubsub.subscribe("collapse:new")
    print("📡 Listening for collapse:new events...")

    for message in pubsub.listen():
        if message["type"] == "message":
            handle_event(message)
1

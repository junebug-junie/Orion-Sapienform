# memory/tier3_ontology/blaze_writer.py
import os
import requests
from rdflib import Graph

GRAPHDB_URL = os.getenv("GRAPHDB_URL", "http://localhost:7200")
GRAPHDB_REPO = os.getenv("GRAPHDB_REPO", "conjourney")
GRAPHDB_CONTEXT = os.getenv("GRAPHDB_CONTEXT", None)


def post_to_blazegraph(graph: Graph, context: str = None):
    """
    Push an RDFLib Graph to Blazegraph.
    """
    headers = {"Content-Type": "text/turtle"}
    params = {}
    if context or GRAPHDB_CONTEXT:
        params["context"] = f"<{context or GRAPHDB_CONTEXT}>"

    try:
        ttl_data = graph.serialize(format="turtle")
        url = f"{GRAPHDB_URL}/repositories/{GRAPHDB_REPO}/statements"
        resp = requests.post(url, headers=headers, data=ttl_data, params=params)
        resp.raise_for_status()
        print(f"[BlazeWriter] Pushed {len(graph)} triples to {GRAPHDB_REPO}.")
    except Exception as e:
        print(f"[BlazeWriter][Error] {e}")


def load_ttl_and_post(path):
    g = Graph()
    g.parse(path, format="turtle")
    post_to_blazegraph(g)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python blaze_writer.py /path/to/file.ttl")
    else:
        load_ttl_and_post(sys.argv[1])


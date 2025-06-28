# memory/tier3_ontology/blaze_writer.py
import requests
from rdflib import Graph

def post_graph_to_blazegraph(graph: Graph, endpoint_url: str, graph_uri: str = None):
    """
    Post an RDFLib Graph to a Blazegraph SPARQL endpoint.
    """
    headers = {"Content-Type": "application/x-turtle"}
    data = graph.serialize(format="turtle")
    params = {}

    if graph_uri:
        params["context"] = f"<{graph_uri}>"

    response = requests.post(
        url=endpoint_url,
        headers=headers,
        data=data,
        params=params,
        timeout=10
    )

    if response.status_code not in (200, 204):
        raise RuntimeError(f"[BlazeWriter] Failed to post RDF: {response.status_code} {response.text}")

    print(f"[BlazeWriter] Graph successfully posted to Blazegraph at {endpoint_url}")


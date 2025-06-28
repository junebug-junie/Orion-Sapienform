# memory/tier3_ontology/graphdb_sync.py
import requests
from datetime import datetime

def sync_from_blaze_to_graphdb(blaze_url: str, graphdb_url: str, graph_uri: str = None):
    """
    Pull RDF data from Blazegraph and push it to GraphDB.
    Assumes both endpoints accept SPARQL update/insert.
    """
    construct_query = """
    CONSTRUCT {
        ?s ?p ?o .
    } WHERE {
        ?s ?p ?o .
    }
    """

    blaze_params = {"query": construct_query}
    headers = {"Accept": "text/turtle", "Content-Type": "application/sparql-query"}

    # Pull from Blazegraph
    blaze_response = requests.post(blaze_url, headers=headers, data=construct_query)
    if blaze_response.status_code != 200:
        raise RuntimeError(f"[GraphSync] Blazegraph query failed: {blaze_response.status_code} {blaze_response.text}")

    ttl_data = blaze_response.text

    # Push to GraphDB
    graphdb_headers = {"Content-Type": "text/turtle"}
    graphdb_params = {"context": f"<{graph_uri}>"} if graph_uri else {}

    graphdb_response = requests.post(
        url=graphdb_url,
        headers=graphdb_headers,
        data=ttl_data,
        params=graphdb_params,
        timeout=15
    )

    if graphdb_response.status_code not in (200, 204):
        raise RuntimeError(f"[GraphSync] GraphDB update failed: {graphdb_response.status_code} {graphdb_response.text}")

    print(f"[GraphSync] Successfully synced from Blazegraph to GraphDB at {datetime.utcnow().isoformat()}")


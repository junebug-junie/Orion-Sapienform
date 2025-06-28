# memory/tier3_ontology/graphdb_import_schema.py
import requests
from pathlib import Path

def import_schema_to_graphdb(schema_path: Path, graphdb_url: str, context_uri: str = None):
    """
    Import the GraphDB schema TTL file into a target repository.
    """
    with open(schema_path, "r") as f:
        ttl_data = f.read()

    headers = {"Content-Type": "text/turtle"}
    params = {"context": f"<{context_uri}>"} if context_uri else {}

    response = requests.post(
        url=graphdb_url,
        headers=headers,
        data=ttl_data,
        params=params,
        timeout=10
    )

    if response.status_code not in (200, 204):
        raise RuntimeError(f"[GraphDB Import] Failed: {response.status_code} {response.text}")

    print(f"[GraphDB Import] Schema imported from {schema_path} to {graphdb_url}")

if __name__ == "__main__":
    SCHEMA_FILE = Path("schema/graphdb_schema.ttl")
    GRAPHDB_REPO_URL = "http://localhost:7200/repositories/emergence/statements"
    import_schema_to_graphdb(SCHEMA_FILE, GRAPHDB_REPO_URL)


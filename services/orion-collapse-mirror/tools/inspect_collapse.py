import os
import sqlite3
from chromadb import PersistentClient
import requests

DB_PATH = os.getenv("DB_PATH", "/mnt/storage/collapse-mirrors/collapse.db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "/mnt/storage/collapse-mirrors/chroma")
GRAPHDB_URL = os.getenv("GRAPHDB_URL", "http://localhost:7200")
GRAPHDB_REPO = os.getenv("GRAPHDB_REPO", "collapse")

def inspect(collapse_id: str):
    print(f"\n=== Inspecting {collapse_id} ===")

    # SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, observer, timestamp, summary FROM collapse_mirror WHERE id = ?", 
        (collapse_id,)
    )
    row = cursor.fetchone()
    print("\nSQLite:")
    print(row if row else "❌ Not found")
    conn.close()

    # Chroma
    client = PersistentClient(path=CHROMA_PATH)
    coll = client.get_or_create_collection("collapse_mirror")
    res = coll.get(ids=[collapse_id])
    print("\nChroma:")
    print(res if res["ids"] else "❌ Not found")

    # GraphDB
    query = f"""
    PREFIX cm: <http://orion.ai/collapse#>
    SELECT ?p ?o WHERE {{
      cm:{collapse_id} ?p ?o .
    }}
    """
    resp = requests.post(
        f"{GRAPHDB_URL}/repositories/{GRAPHDB_REPO}",
        data={"query": query},
        headers={"Accept": "application/sparql-results+json"}
    )
    if resp.status_code == 200:
        results = resp.json()["results"]["bindings"]
        print("\nGraphDB:")
        if results:
            for b in results:
                print(f"{b['p']['value']} -> {b['o']['value']}")
        else:
            print("❌ Not found")
    else:
        print("\nGraphDB:")
        print(f"❌ Query failed {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inspect_collapse.py <collapse_id>")
    else:
        inspect(sys.argv[1])

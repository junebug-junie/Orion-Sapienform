import requests
from rdflib import Namespace
from app.settings import settings

CM = Namespace("http://orion.ai/collapse#")

def enrich_from_graphdb_ids(fragments):
    for f in fragments:
        try:
            q = f"""
            PREFIX cm: <{CM}>
            SELECT ?tag ?entity WHERE {{
                ?s cm:id '{f.id}' .
                OPTIONAL {{ ?s cm:hasTag ?tag. }}
                OPTIONAL {{ ?s cm:hasEntity ?entity. }}
            }}
            """
            endpoint = settings.rdf_sparql_endpoint
            if not endpoint:
                continue
            r = requests.post(
                endpoint,
                data=q,
                headers={
                    "Content-Type": "application/sparql-query",
                    "Accept": "application/sparql-results+json",
                },
                auth=settings.rdf_sparql_auth,
                timeout=5,
            )
            if r.status_code == 200 and "application/json" in r.headers.get("Content-Type", ""):
                res = r.json().get("results", {}).get("bindings", [])
                for b in res:
                    if "tag" in b:    f.tags.append(b["tag"]["value"])
                    if "entity" in b: f.tags.append(b["entity"]["value"])
        except Exception:
            continue
    return fragments

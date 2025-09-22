# schema/ttl_audit_listener.py

import os
import requests
from datetime import datetime
from rdflib import Graph, Namespace
from emergence.core.redis_bus import RedisBus
from schema.ttl_auditor import (
    load_latest_ttls,
    diff_graphs,
    validate_graph_schema,
    report_broken_threads,
)

CJ = Namespace("http://conjourney.net/schema#")

# ————————————————
# Configuration (via env or defaults)
# ————————————————
TTL_LOG_DIR      = os.getenv("TTL_LOG_DIR",      os.path.expanduser("~/conjourney/logs/ttl"))
AUDIT_LOG_DIR    = os.getenv("AUDIT_LOG_DIR",    os.path.expanduser("~/conjourney/logs/audit"))
GRAPHDB_URL      = os.getenv("GRAPHDB_URL",      "http://localhost:7200")
GRAPHDB_REPO     = os.getenv("GRAPHDB_REPO",     "conjourney")
GRAPHDB_CONTEXT  = os.getenv("GRAPHDB_AUTO_PUSH_CONTEXT", None)
AUTO_PUSH_AUDIT  = os.getenv("AUTO_PUSH_AUDIT",  "false").lower() in ("1","true","yes")

def run_audit():
    # ensure output dir
    os.makedirs(AUDIT_LOG_DIR, exist_ok=True)

    # load the two most recent TTLs
    graphs = load_latest_ttls(n=2, ttl_dir=TTL_LOG_DIR)
    if len(graphs) < 2:
        print("[Audit] Not enough TTL files to diff.")
        return

    base, compare = graphs[1], graphs[0]
    diff          = diff_graphs(base, compare)
    schema_issues = validate_graph_schema(compare)
    broken        = report_broken_threads(compare)

    # merge audit graphs
    audit = Graph()
    audit.bind("cj", CJ)
    for g in (diff, schema_issues, broken):
        for t in g:
            audit.add(t)

    # write to disk
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(AUDIT_LOG_DIR, f"audit_report_{ts}.ttl")
    audit.serialize(destination=out, format="turtle")
    print(f"[Audit] Written report to {out}")

    # optionally push to GraphDB
    if AUTO_PUSH_AUDIT:
        headers = {"Content-Type": "text/turtle"}
        params  = {}
        if GRAPHDB_CONTEXT:
            params["context"] = f"<{GRAPHDB_CONTEXT}>"
        resp = requests.post(
            f"{GRAPHDB_URL}/repositories/{GRAPHDB_REPO}/statements",
            headers=headers,
            data=audit.serialize(format="turtle"),
            params=params
        )
        resp.raise_for_status()
        print(f"[Audit] Pushed to GraphDB repo '{GRAPHDB_REPO}' (status {resp.status_code})")

def start_listener():
    bus = RedisBus()
    bus.subscribe("memory:ttl:updated", lambda msg: run_audit())
    print("[Audit Listener] Subscribed to 'memory:ttl:updated'")
    bus.listen_forever()

if __name__ == "__main__":
    start_listener()


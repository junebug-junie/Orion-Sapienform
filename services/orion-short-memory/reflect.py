import os
import uuid
import psycopg2
import requests
from datetime import datetime, timedelta

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rdflib import Graph, Namespace, Literal, RDF, XSD, URIRef

# ================== CONFIG ==================
ORION_BRAIN_URL = os.getenv("ORION_BRAIN_URL", "http://orion-brain-service:8088")
CHROMA_DIR = os.getenv("CHROMA_DIR", "/data/chroma")
PG_HOST = os.getenv("PG_HOST", "chrysalis")
PG_DB   = os.getenv("PG_DB", "conjourney")
PG_USER = os.getenv("PG_USER", "juniper")
PG_PASS = os.getenv("PG_PASS", "yourpassword")
GRAPHDB_URL = os.getenv("GRAPHDB_URL", "http://graphdb:7200/repositories/orion")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ================== LLM Summarizer ==================
llm = ChatOllama(
    base_url=ORION_BRAIN_URL,
    model=os.getenv("LLM_MODEL", "mistral:instruct")
)

# ================== Warm Memory ==================
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# ================== RDF NS ==================
ORION = Namespace("http://conjourney.net/orion#")

def summarize_text(text: str) -> str:
    prompt = f"Summarize the following conversation turn for long-term memory:\\n\\n{text}"
    response = llm.invoke(prompt)
    return response.content.strip()

def push_rdf_summary(summary: str, ts: datetime):
    g = Graph()
    summary_id = str(uuid.uuid4())
    summary_uri = URIRef(f"http://conjourney.net/orion#summary_{summary_id}")

    g.add((summary_uri, RDF.type, ORION.ReflectionSummary))
    g.add((summary_uri, ORION.summaryText, Literal(summary)))
    g.add((summary_uri, ORION.timestamp, Literal(ts.isoformat(), datatype=XSD.dateTime)))

    ntriples = g.serialize(format="nt")
    update = f"INSERT DATA {{\\n{ntriples}\\n}}"
    r = requests.post(
        GRAPHDB_URL + "/statements",
        data=update.encode("utf-8"),
        headers={"Content-Type": "application/sparql-update"}
    )
    if r.status_code != 200:
        print(f"[Reflection RDF] Failed: {r.status_code} {r.text}")

def reflect():
    since = datetime.utcnow() - timedelta(hours=24)
    conn = psycopg2.connect(host=PG_HOST, dbname=PG_DB, user=PG_USER, password=PG_PASS)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, user_msg, bot_msg, ts FROM conversation_memory WHERE ts >= %s;",
        (since,)
    )
    rows = cur.fetchall()
    for turn_id, user_msg, bot_msg, ts in rows:
        combined = f"USER: {user_msg}\\nORION: {bot_msg}"
        summary = summarize_text(combined)
        # 1) Store into Chroma
        try:
            vectorstore.add_texts([summary])
        except Exception as e:
            print(f"[Reflection] Chroma add_texts error: {e}")
        # 2) Push RDF summary
        try:
            push_rdf_summary(summary, ts)
        except Exception as e:
            print(f"[Reflection] RDF push error: {e}")
        print(f"[Reflection] Stored summary for turn {turn_id}: {summary[:100]}...")
    cur.close()
    conn.close()

if __name__ == "__main__":
    reflect()

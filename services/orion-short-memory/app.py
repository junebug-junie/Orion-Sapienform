import os
import uuid
from datetime import datetime

import psycopg2
import requests
from fastapi import FastAPI, Body

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_community.chat_models import ChatOllama
from langchain_community.memory import VectorStoreRetrieverMemory
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

# ================== LLM ROUTER ==================
llm = ChatOllama(
    base_url=ORION_BRAIN_URL,
    model=os.getenv("LLM_MODEL", "mistral:instruct")
)

# ================== EMBEDDINGS / WARM STORE ==================
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
warm_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
warm_memory = VectorStoreRetrieverMemory(retriever=warm_retriever)

# ================== HOT / SUMMARY ==================
hot_memory = ConversationBufferMemory(return_messages=True)
summary_memory = ConversationSummaryMemory(llm=llm)

# ================== RDF NAMESPACE ==================
ORION = Namespace("http://conjourney.net/orion#")

def log_to_cold_memory(user_msg: str, bot_msg: str):
    """Persist turn to Postgres and GraphDB (RDF)."""
    # --- Postgres insert ---
    try:
        conn = psycopg2.connect(host=PG_HOST, dbname=PG_DB, user=PG_USER, password=PG_PASS)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS conversation_memory (id SERIAL PRIMARY KEY, ts TIMESTAMPTZ DEFAULT NOW(), user_msg TEXT NOT NULL, bot_msg TEXT NOT NULL);"
        )
        cur.execute(
            "INSERT INTO conversation_memory (user_msg, bot_msg) VALUES (%s, %s);",
            (user_msg, bot_msg)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[ColdMemory] DB insert failed: {e}")

    # --- RDF triples (GraphDB) ---
    try:
        g = Graph()
        turn_id = str(uuid.uuid4())
        turn_uri = URIRef(f"http://conjourney.net/orion#turn_{turn_id}")

        g.add((turn_uri, RDF.type, ORION.ConversationTurn))
        g.add((turn_uri, ORION.hasUserMessage, Literal(user_msg)))
        g.add((turn_uri, ORION.hasBotMessage, Literal(bot_msg)))
        g.add((turn_uri, ORION.timestamp, Literal(datetime.utcnow().isoformat(), datatype=XSD.dateTime)))

        ntriples = g.serialize(format="nt")
        update = f"INSERT DATA {{\n{ntriples}\n}}"
        r = requests.post(
            GRAPHDB_URL + "/statements",
            data=update.encode("utf-8"),
            headers={"Content-Type": "application/sparql-update"}
        )
        if r.status_code != 200:
            print(f"[ColdMemory RDF] Failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"[ColdMemory RDF] Error: {e}")

# ================== API ==================
app = FastAPI(title="Orion LangChain Memory Service")

@app.post("/query")
async def query(payload: dict = Body(...)):
    user_msg = payload.get("message", "Hello from Orion")

    # 1) Hot buffer
    hot_vars = hot_memory.load_memory_variables({})
    hot_context = hot_vars.get("history", "")

    # 2) Warm semantic recall
    warm_vars = warm_memory.load_memory_variables({"prompt": user_msg})
    warm_context = warm_vars.get("history", "")

    # 3) Compose context
    context = f"Hot:\\n{hot_context}\\n\\nWarm:\\n{warm_context}\\n\\nUser:\\n{user_msg}"

    # 4) Call Brain (LLM)
    response = llm.invoke(context)
    bot_msg = response.content

    # 5) Update memories
    hot_memory.save_context({"input": user_msg}, {"output": bot_msg})
    # Persist to warm store: add raw message + bot reply for recall
    try:
        vectorstore.add_texts([f"USER: {user_msg}", f"ORION: {bot_msg}"])
    except Exception as e:
        print(f"[WarmMemory] add_texts error: {e}")

    # 6) Cold storage
    log_to_cold_memory(user_msg, bot_msg)

    return {"response": bot_msg}

# to run:
# docker exec -it orion-janus-vector-writer python /app/app/collection_browse.py

import chromadb

# Connect to your Chroma service
client = chromadb.HttpClient(host="orion-janus-vector-db", port=8000)

# Get your collection
coll = client.get_collection("orion_main_store")

# --- Basic stats ---
print("📦 Collection name:", coll.name)
print("🧮 Total items:", coll.count())

# --- Peek at documents ---
docs = coll.get(include=["documents", "metadatas"], limit=5)
for i, doc in enumerate(zip(docs["ids"], docs["documents"], docs["metadatas"])):
    _id, text, meta = doc
    print(f"\n[{i}] ID: {_id}\nText: {text[:120]}...\nMetadata: {meta}")

# to run:
# docker exec -it orion-janus-vector-writer python /app/app/query_similarity.py

import chromadb

# Connect to your Chroma service
client = chromadb.HttpClient(host="orion-janus-vector-db", port=8000)

# Get your collection
coll = client.get_collection("orion_main_store")

query = "looking back at me"
results = coll.query(query_texts=[query], n_results=3)

for doc, meta, score in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"🔍 Score: {score:.4f}\nText: {doc[:120]}...\nMetadata: {meta}\n")

import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

collection = None
embedder = None

if os.getenv("CHRONICLE_MODE", "local") != "cloud":
    chroma_client = PersistentClient(path="/mnt/storage/collapse-mirrors/chroma")
    collection = chroma_client.get_or_create_collection("collapse_mirror")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        _ = embedder.encode("warmup")
        print("✅ Embedder warm-up success")
    except Exception as e:
        print("⚠️ Embedder warm-up failed:", e)

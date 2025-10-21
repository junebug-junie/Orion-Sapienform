import random, traceback
import numpy as np
from chromadb import HttpClient
from sklearn.metrics.pairwise import cosine_similarity
from app.settings import settings

DREAM_TEMPERATURE = 0.35
def enrich_from_chroma(fragments, seed=None, temperature=DREAM_TEMPERATURE):
    """
    Hybrid enrichment using Chroma vectors:
      • Recall stored embeddings by ID
      • Perform associative drift queries when missing or randomly selected
      • Compute salience via cosine similarity
    """
    try:
        client = HttpClient(
            host=settings.VECTOR_DB_HOST,
            port=settings.VECTOR_DB_PORT
        )
        coll = client.get_or_create_collection(settings.VECTOR_DB_COLLECTION)

        for f in fragments:
            # --------------------------------------------------
            # Normalize tags to ensure list type
            # --------------------------------------------------
            if isinstance(f.tags, str):
                f.tags = [f.tags]
            elif f.tags is None:
                f.tags = []

            drift = random.random() < temperature

            # --------------------------------------------------
            # Direct recall of stored embeddings
            # --------------------------------------------------
            try:
                doc = coll.get(ids=[f.id])
                if doc and doc.get("embeddings"):
                    f.embedding = np.array(doc["embeddings"][0])
                    f.tags.append("recalled")

                    # Skip association unless drift is triggered
                    if not drift:
                        continue
            except Exception:
                pass

            # --------------------------------------------------
            # Associative drift — semantic search for nearby memories
            # --------------------------------------------------
            base = seed or f.text or "dream fragment"

            if drift:
                base += " " + random.choice([
                    "holographic memory",
                    "entangled meaning",
                    "the warmth of a signal",
                    "echoes of thought",
                    "folded timelines",
                ])

            try:
                res = coll.query(query_texts=[base], n_results=3)
                if res and res.get("documents"):
                    assoc_docs = res["documents"][0]
                    f.tags.extend([f"assoc:{d[:40]}" for d in assoc_docs])
                    f.tags.append("associated")
            except Exception as e:
                print(f"⚠️ association failed for {f.id}: {e}")

        # --------------------------------------------------
        # Compute embedding cohesion for salience weighting
        # --------------------------------------------------
        embeddings = [
            f.embedding for f in fragments
            if getattr(f, "embedding", None) is not None
        ]

        if embeddings:
            sims = cosine_similarity(embeddings)
            valid_frags = [
                x for x in fragments
                if getattr(x, "embedding", None) is not None
            ]
            for i, f in enumerate(valid_frags):
                f.salience = float(np.mean(sims[i]))

    except Exception:
        print(f"⚠️ Chroma hybrid enrich failed: {traceback.format_exc()}")

    return fragments

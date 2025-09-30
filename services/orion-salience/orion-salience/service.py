from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gzip, os, json

app = FastAPI(title="Orion Salience Service")

# Load cheap models
nlp = spacy.load("en_core_web_sm")
sentiment = SentimentIntensityAnalyzer()

class EventIn(BaseModel):
    id: str
    text: str
    type: Optional[str] = None

# ---------- Tier 1: Online ----------
def compute_online(event: EventIn) -> Dict:
    doc = nlp(event.text)

    entities = [ent.text for ent in doc.ents]
    entity_density = min(1.0, len(entities) / 5)

    score = sentiment.polarity_scores(event.text)
    emotion = abs(score["pos"] - score["neg"])

    ritual_boost = 0.2 if event.type == "ritual" else 0.0

    return {
        "emotion": emotion,
        "entity_density": entity_density,
        "ritual_boost": ritual_boost,
    }

# ---------- Tier 2: Offline ----------
def compute_offline(event: EventIn) -> Dict:
    raw_len = len(event.text.encode("utf-8"))
    comp_len = len(gzip.compress(event.text.encode("utf-8")))
    compression = 1 - (comp_len / raw_len)

    # Placeholders for future graph/chroma metrics
    graph_degree = 0.0
    embedding_density = 0.0
    prediction_error = 0.0

    return {
        "compression": compression,
        "graph_degree": graph_degree,
        "embedding_density": embedding_density,
        "prediction_error": prediction_error,
    }

# ---------- API Routes ----------
@app.post("/salience/online")
def salience_online(event: EventIn):
    return {"id": event.id, "salience_components": compute_online(event)}

@app.post("/salience/offline")
def salience_offline(event: EventIn):
    return {"id": event.id, "salience_components": compute_offline(event)}

@app.post("/salience/full")
def salience_full(event: EventIn):
    online = compute_online(event)
    offline = compute_offline(event)
    return {"id": event.id, "salience_components": {**online, **offline}}

@app.get("/health")
def health():
    return {"status": "ok"}

STATE_FILE = "/app/last_cron.json"

@app.get("/health/cron")
def health_cron():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"status": "no dream cycle recorded yet"}

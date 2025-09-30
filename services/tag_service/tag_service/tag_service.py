from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import spacy
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Tag Service")

# Load NLP + embedding model once
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

class EventIn(BaseModel):
    id: str
    text: str

class TaggedEvent(BaseModel):
    id: str
    tags: List[str]
    entities: List[Dict[str, str]]
    salience: float
    embedding: List[float]

@app.post("/tag", response_model=TaggedEvent)
def tag_event(event: EventIn):
    doc = nlp(event.text)

    # Extract entities
    entities = [{"type": ent.label_, "value": ent.text} for ent in doc.ents]

    # Tags = nouns + verbs
    tags = list(set([t.lemma_ for t in doc if t.pos_ in ("NOUN", "VERB") and not t.is_stop]))

    # Embedding
    embedding = embedder.encode([event.text])[0].tolist()

    # Salience (naive: length + entity count)
    salience = min(1.0, (len(tags) + len(entities)) / 10)

    return TaggedEvent(
        id=event.id,
        tags=tags,
        entities=entities,
        salience=salience,
        embedding=embedding,
    )

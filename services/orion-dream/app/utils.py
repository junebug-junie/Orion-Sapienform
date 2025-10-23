# ==================================================
# utils.py
# ==================================================

import json
import uuid
import numpy as np
from datetime import datetime
from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import List, Optional

# --- Text + Biometrics Helpers ---

def clean_text(val: str | list | dict | None) -> str:
    """Normalizes text fragments (handles JSON, lists, or weird spacing)."""
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        val = ", ".join(map(str, val))
    elif isinstance(val, dict):
        val = json.dumps(val, ensure_ascii=False)
    # collapse spaces like "p i p e l i n e"
    val = " ".join(val.split())
    return val.strip()


@dataclass
class Fragment:
    id: str
    kind: str
    text: str
    tags: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    salience: float = 0.0
    ts: float = 0.0

def compute_salience(f: Fragment) -> float:
    v = abs(f.valence or 0)
    a = f.arousal or 0
    novelty = np.random.random() * 0.3  # Placeholder until semantic comparison available
    recurrence = np.random.random() * 0.2
    return 0.5*v + 0.3*a + 0.2*novelty + 0.1*recurrence

def now_ts() -> float:
    return datetime.utcnow().timestamp()

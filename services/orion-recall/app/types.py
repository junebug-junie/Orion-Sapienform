# app/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Fragment:
    id: str
    kind: str                   # "chat", "collapse", "enrichment", "assoc", "biometrics", "rdf", ...
    source: str = "sql"         # "sql", "vector", "rdf", ...
    text: str = ""
    ts: float = 0.0             # epoch seconds
    tags: List[str] = field(default_factory=list)

    salience: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0

    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhiContext:
    valence: float = 0.0
    energy: float = 0.0
    coherence: float = 0.0
    novelty: float = 0.0


@dataclass
class RecallQuery:
    query_text: str = ""
    max_items: int = 16
    time_window_days: int = 30
    mode: str = "hybrid"
    tags: List[str] = field(default_factory=list)
    phi: Optional[PhiContext] = None

    trace_id: Optional[str] = None
    source: str = "unknown"
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    packs: List[str] = field(default_factory=list)


@dataclass
class RecallResult:
    fragments: List[Fragment]
    debug: Dict[str, Any] = field(default_factory=dict)

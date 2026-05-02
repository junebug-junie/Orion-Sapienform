from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IntentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal[
        "biographical",
        "episodic_recent",
        "episodic_historical",
        "factual_project",
        "associative",
        "reflective",
        "unknown",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


_BIO_PATTERNS = re.compile(
    r"\b(who am i|where do i live|my birthday|how old am i|my name|about me|my hometown|my family)\b",
    re.I,
)
_EPISODIC_RECENT = re.compile(r"\b(just now|earlier today|this morning|last night|yesterday|recently)\b", re.I)
_EPISODIC_OLD = re.compile(r"\b(last year|years ago|back when|in \d{4}|childhood|used to)\b", re.I)
_PROJECT = re.compile(r"\b(project|repo|codebase|ticket|jira|pr\b|pull request)\b", re.I)
_REFLECT = re.compile(r"\b(feel|felt|reflect|meaning|why do i|journal)\b", re.I)


def classify_intent_v1(query_text: str) -> IntentClassification:
    text = (query_text or "").strip()
    if not text:
        return IntentClassification(intent="unknown", confidence=0.0, rationale="empty")

    if _BIO_PATTERNS.search(text):
        return IntentClassification(intent="biographical", confidence=0.82, rationale="biographical cue")
    if _REFLECT.search(text):
        return IntentClassification(intent="reflective", confidence=0.65, rationale="reflective cue")
    if _EPISODIC_RECENT.search(text):
        return IntentClassification(intent="episodic_recent", confidence=0.6, rationale="recent episodic cue")
    if _EPISODIC_OLD.search(text):
        return IntentClassification(intent="episodic_historical", confidence=0.6, rationale="historical episodic cue")
    if _PROJECT.search(text):
        return IntentClassification(intent="factual_project", confidence=0.55, rationale="project cue")
    if len(text.split()) <= 4:
        return IntentClassification(intent="associative", confidence=0.35, rationale="short query")

    return IntentClassification(intent="unknown", confidence=0.4, rationale="no strong cue")


def resolve_profile_for_intent(intent: str, *, fallback_profile: str) -> str:
    if intent == "biographical":
        return "biographical.v1"
    if intent == "reflective":
        return "reflect.v1"
    if intent in {"episodic_recent", "episodic_historical"}:
        return "chat.general.v1"
    if intent == "factual_project":
        return "self.factual.v1"
    return fallback_profile


def intent_telemetry_payload(*, query_text: str, intent: str, profile: str, override: bool) -> dict:
    h = hashlib.sha256((query_text or "").encode("utf-8")).hexdigest()[:16]
    return {
        "kind": "recall.intent.v1",
        "query_hash16": h,
        "intent": intent,
        "selected_profile": profile,
        "profile_explicit": override,
    }

"""One naming function for every place the self-monitor organ speaks. Pure, no imports beyond re.

Juniper (2026-11-03): "use plain names ALONG with internal ids". Every Orion-facing text about a
faculty (the stance/reverie cue, the E2 self-question, the E3 self-finding prompt, the Temporal Self
faculty arcs read back in the Day letter and recent arcs) names it as ``named(faculty)``: the plain
words first, then the internal id in brackets, e.g. "walkway-grade journal runs
(act.actions.journal.make.walkway_grade)". The plain words say what it is; the id lets Orion (and
Juniper) find the exact record, and is the token the E3 grounding guard checks.

``plain_name`` maps a faculty key to a plural noun phrase that reads after "My " (a small table plus
rules per key family; anything unrecognised becomes its humanised words plus "bets").
"""

from __future__ import annotations

import re

# Exact keys whose generic rule reads badly.
_EXACT: dict[str, str] = {
    "turn.self.steps": "guesses about how many steps my turns take",
    "turn.self.reply_length": "guesses about how long my replies run",
    "turn.self.asks_question": "guesses that I'll ask a question",
    "turn.self.fetches_outside": "guesses that I'll look something up outside",
    "vision.percept_expectation": "guesses about what I'll see next",
    "act.hub.world_pulse_read.look": "world-pulse reads",
    "act.topic_foundry.run.maintain": "topic-foundry maintenance runs",
    "act.aitown.session.play": "AI Town sessions",
}
# Act verbs -> the plural noun for one try of that act.
_VERB_NOUNS: dict[str, str] = {
    "make": "runs", "run": "runs", "predict": "predictions", "infer": "inferences", "look": "looks",
    "say": "messages", "ask": "questions", "play": "sessions", "sense": "checks",
    "maintain": "maintenance runs", "act": "acts",
}
# Doers that name the hosting service, not the faculty: dropped from the plain name unless nothing
# else is left ("act.hub.say" -> "hub messages", never a bare "messages").
_HOST_WORDS = frozenset({"actions", "hub", "durable", "thought"})
# Whole-segment and suffix tokens -> words; unit suffixes (_c, _w, ...) are dropped.
_TOKEN_WORDS: dict[str, str] = {"temp_c": "temperature", "temp": "temperature", "rh": "humidity"}
_UNIT_SUFFIX = re.compile(r"_(c|f|w|kw|kwh|pct|ms|s|sec)$")


def _words(segment: str) -> str:
    seg = segment.split(":")[-1]
    if seg in _TOKEN_WORDS:
        return _TOKEN_WORDS[seg]
    for token, word in _TOKEN_WORDS.items():
        if seg.endswith("_" + token):
            seg = seg[: -len(token) - 1] + " " + word
            break
    seg = _UNIT_SUFFIX.sub("", seg)
    return re.sub(r"[_\s-]+", " ", seg).strip()


def _hyphen(segment: str) -> str:
    return _words(segment).replace(" ", "-")


def _generic(parts: list[str]) -> str:
    words = " ".join(w for w in (_words(p) for p in parts if p != "act") if w)
    return f"{words or 'unnamed'} bets"


def plain_name(faculty: str) -> str:
    """Faculty key -> a plain plural noun phrase that reads after "My ". Deterministic."""
    if faculty in _EXACT:
        return _EXACT[faculty]
    parts = [p for p in faculty.split(".") if p]
    head, rest = (parts[0], parts[1:]) if parts else ("", [])
    if head == "act" and rest:
        if rest[:2] == ["actions", "journal"] and len(rest) >= 4:
            return f"{_hyphen(rest[3])} journal runs"
        verb_at = next((i for i in range(len(rest) - 1, -1, -1) if rest[i] in _VERB_NOUNS), None)
        if verb_at is not None and verb_at >= len(rest) - 2:
            doer = [w for w in rest[:verb_at] if w not in _HOST_WORDS] or rest[:verb_at]
            name = rest[verb_at + 1:]
            words = " ".join(w for w in (_words(x) for x in doer + name) if w)
            if not words:
                return _generic(parts)  # "act.make" -> "make bets", never a bare "runs"
            noun = _VERB_NOUNS[rest[verb_at]]
            if doer == ["prime"] and not name:
                words = "Prime"
            if words.endswith(" " + noun.rstrip("s")) or words == noun.rstrip("s"):
                return f"{words}s"  # "session sessions" -> "sessions"
            return f"{words} {noun}"
    if head == "forecast" and rest:
        words = " ".join(w for w in (_words(x) for x in rest) if w)
        if words:
            return f"{words} forecast ranges"
    if head == "turn":
        words = " ".join(w for w in (_words(x) for x in rest if x != "self") if w)
        if words:
            return f"guesses about my turns' {words}"
    if head == "body" and len(rest) >= 2:
        domain, skill = _words(rest[0]), _words(rest[-1])
        if domain and skill:
            return f"{domain} {skill} moves"
    return _generic(parts)


def named(faculty: str) -> str:
    """Plain name first, internal id after it in brackets. Deterministic."""
    return f"{plain_name(faculty)} ({faculty})"


FACULTY_SUBJECT_PREFIX = "faculty:"


def faculty_arc_words(subject_ref: str, subject_label: str) -> str:
    """A Temporal Self faculty arc in words: "<plain name> (<id>) <status>".

    The arc's stored label stays "<faculty> <status>" (``orion.temporal_self.arcs._fold_faculty``;
    ``faculty_status_of`` reads the status back from it), so only the rendered text changes. An arc
    whose subject is not ``faculty:<id>`` renders its label unchanged."""
    label = subject_label.strip()
    if not subject_ref.startswith(FACULTY_SUBJECT_PREFIX):
        return label or subject_ref
    faculty = subject_ref[len(FACULTY_SUBJECT_PREFIX):]
    if not faculty:
        return label or subject_ref
    status = label.rsplit(" ", 1)[-1] if " " in label else ""
    return f"{named(faculty)} {status}".rstrip()


__all__ = ["FACULTY_SUBJECT_PREFIX", "faculty_arc_words", "named", "plain_name"]

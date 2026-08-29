"""The declared shape of a segment's ``meaning``/``sentiment``/``aspects``, plus
the coercion that keeps the database honest about it.

Why this module exists (confirmed live 2026-08-29):

``SegmentRecord`` declares ``meaning: Optional[Dict[str, Any]]`` and
``sentiment: Optional[Dict[str, Any]]``, but nothing ever enforced that.
``_llm_prompt`` asked for "JSON with keys: title, aspects, aspect_scores,
sentiment, meaning, evidence_spans" without saying what shape ``sentiment``
and ``meaning`` are, so the model answered with prose strings, and
``_finalize_enrichment``'s ``setdefault`` is a no-op for a key that is
present-but-wrong-typed. Those strings went straight into ``jsonb``.

Measured damage at the time this module was written:

- 552 of 554 enriched ``topic_foundry_segments`` rows had ``jsonb_typeof =
  'string'`` for BOTH ``meaning`` and ``sentiment``; 2 had objects.
- ``GET /segments`` returned **HTTP 500** for any run containing one of those
  rows -- ``pydantic_core.ValidationError: 2 validation errors for
  SegmentRecord ... dict_type``. That is not a cosmetic API failure: it is
  the only source of ``segment_speakers`` for the participation edges added
  in PR #1932, and ``concept_atlas_routes`` degrades a segments-fetch failure
  to an empty map. So a live ingest reported ``available: true`` and wrote 19
  concepts while silently producing ``participation_edges: 0`` -- the shipped
  feature was dead on the live path with every test green.
- ``kg_edges._edges_from_segment`` swallowed the resulting ``JSONDecodeError``
  into ``meaning = {}``, which is why ``topic_foundry_edges`` has 0 rows for
  all time.

What this module DOES fix: everything that reads these columns through
``SegmentRecord`` (every HTTP route) and everything written from
``_finalize_enrichment`` onward.

What it deliberately does NOT fix, so nobody reads more into it than is
there -- these run in SQL, against raw ``jsonb``, before pydantic exists:

- ``repository.fetch_segments``'s ``friction``/``valence`` sort keys are
  ``COALESCE((sentiment->>'friction')::float, 0)``, which is NULL (not an
  error) for a jsonb string. All legacy prose rows still sort as a constant
  0, and so does a NEW row whose enrichment was prose, since the coerced
  ``{"summary": ...}`` object has no ``friction`` key either.
- ``repository.segment_facets``'s ``meaning ? 'intent'`` facet is false for
  those same rows.

Closing those needs a backfill of the legacy rows, not a read-side coercion.

The prompt text below is GENERATED from the same constants the coercion
validates against, so the KEY NAMES and the numeric RANGES given to the model
cannot drift from what the coercion accepts. It does not make the model
obey -- ``coerce_*`` is what handles disobedience -- and unknown keys are
deliberately preserved rather than rejected, so "use exactly these keys" is
an instruction, not an invariant.

``MEANING_EDGE_PREDICATES`` is the single source of the four list keys, so
``kg_edges`` derives its predicates from the same table the prompt is built
from. Renaming a key there moves the prompt, the coercion and the edge
builder together; before this they were four hardcoded ``meaning.get(...)``
calls that a rename would have silently orphaned -- the exact failure class
this module exists to fix.

Pure module: no DB, no network, no LLM, no settings. See
``tests/test_enrichment_contract.py``.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Tuple

# ``meaning``'s list-valued keys -> the (predicate, confidence) each one
# becomes in ``kg_edges._edges_from_segment``. That function guards with
# ``isinstance(values, list)`` and silently returns [] otherwise, so a
# non-list here is a silent edge drop, not an error.
MEANING_EDGE_PREDICATES: Dict[str, Tuple[str, float]] = {
    "entities": ("mentions", 0.6),
    "questions": ("asks_about", 0.5),
    "claims": ("claims_about", 0.7),
    "next_steps": ("next_step", 0.8),
}

MEANING_LIST_KEYS: Tuple[str, ...] = tuple(MEANING_EDGE_PREDICATES)

# ``meaning``'s scalar text keys, matching ``_heuristic_enrich``'s output --
# the only shape in this codebase that was ever actually correct.
MEANING_TEXT_KEYS: Tuple[str, ...] = ("intent", "outcome")

# ``sentiment``'s numeric keys and their real ranges. These are NOT uniform,
# and getting that wrong would have been a fresh bug: ``_heuristic_enrich``
# emits arousal/uncertainty/friction strictly in 0..1, and the Topic Studio
# UI (services/orion-hub/static/js/app.js) buckets friction as 0-0.3 /
# 0.3-0.7 / 0.7-1.0 -- a negative friction would land in the *low*-friction
# bucket and read as calm. Only valence and stance are signed.
SENTIMENT_RANGES: Dict[str, Tuple[float, float]] = {
    "valence": (-1.0, 1.0),
    "arousal": (0.0, 1.0),
    "stance": (-1.0, 1.0),
    "uncertainty": (0.0, 1.0),
    "friction": (0.0, 1.0),
}

SENTIMENT_SCALAR_KEYS: Tuple[str, ...] = tuple(SENTIMENT_RANGES)

# Marker written onto a coerced-from-prose object. A consumer can tell "the
# enricher produced unstructured text" apart from "the enricher produced a
# real object", instead of both arriving as an indistinguishable dict.
UNSTRUCTURED_KEY = "unstructured"
SUMMARY_KEY = "summary"

# Keys checked, in order, when a list item arrives as an object rather than a
# string ({"name": "orion", "type": "person"} is a very common LLM answer
# shape for `entities`).
_ITEM_NAME_KEYS = ("name", "text", "label", "title", "value", "entity")


def _as_text(item: Any) -> str:
    """Render one list item as text without destroying it.

    ``str()`` on a dict yields a Python repr -- single quotes, ``True``,
    ``None`` -- which is not valid JSON and cannot be parsed back, and
    ``_finalize_enrichment`` persists it to ``jsonb`` permanently. That is
    worse than the bug this module fixes: it does not drop the data, it
    corrupts it irreversibly. So an object is read for its name-ish key if it
    has one, and otherwise re-encoded as real JSON.
    """
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in _ITEM_NAME_KEYS:
            candidate = item.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    if item is None:
        return ""
    try:
        return json.dumps(item, ensure_ascii=False, default=str).strip()
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(item).strip()


def _as_object(value: Any) -> Optional[Dict[str, Any]]:
    """Return ``value`` as a dict, or ``None`` if there is nothing there.

    A JSON string that parses to an object is that object -- the enricher
    double-encoding its answer is a formatting mistake, not a different
    meaning. Prose, or a structure that is not an object, is preserved under
    ``summary`` and flagged ``unstructured``: what the model wrote is real
    output and dropping it would be lying about what happened, but it is not
    a structured object and must not masquerade as one.

    A bare scalar (``0``, ``4.2``, ``True``) is NOT prose and gets no
    fabricated summary -- it is returned as ``None``, since inventing
    ``{"summary": "0"}`` would also trip the "enricher returned prose"
    warning for something that is not prose.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return None
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return dict(parsed)
        return {SUMMARY_KEY: text, UNSTRUCTURED_KEY: True}
    summary = _as_text(value)
    if not summary:
        return None
    return {SUMMARY_KEY: summary, UNSTRUCTURED_KEY: True}


def coerce_meaning(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize a ``meaning`` value to the declared object shape.

    Unknown keys are preserved -- the declared type is ``Dict[str, Any]`` and
    silently dropping whatever else the enricher found would be a worse
    failure than the one this fixes.
    """
    obj = _as_object(value)
    if obj is None:
        return None
    for key in MEANING_LIST_KEYS:
        if key not in obj:
            continue
        items = obj[key]
        if isinstance(items, list):
            obj[key] = [text for text in (_as_text(item) for item in items) if text]
        elif isinstance(items, str) and items.strip():
            # A bare string is read as a one-element list, the literal
            # reading. Deliberately NOT split on commas: "orion, juniper" may
            # be one entity or two and this module does not get to guess.
            obj[key] = [items.strip()]
        else:
            obj[key] = []
    for key in MEANING_TEXT_KEYS:
        if key in obj and obj[key] is not None and not isinstance(obj[key], str):
            obj[key] = _as_text(obj[key])
    return obj


def coerce_sentiment(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize a ``sentiment`` value to the declared object shape.

    A scalar key that is not a real, in-range number is DROPPED rather than
    defaulted or clamped. A fabricated 0.0 valence is indistinguishable from
    a measured neutral one, and a clamped -0.8 friction silently becomes a
    confident 0.0 "calm" reading in the Topic Studio UI. ``fetch_segments``'s
    ``COALESCE(..., 0)`` already renders absence as 0 for sorting without
    also persisting a number nobody produced.
    """
    obj = _as_object(value)
    if obj is None:
        return None
    for key, (low, high) in SENTIMENT_RANGES.items():
        if key not in obj:
            continue
        raw = obj[key]
        # bool is an int subclass: float(True) is 1.0, and a boolean is not a
        # measured scalar.
        if isinstance(raw, bool):
            obj.pop(key)
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            obj.pop(key)
            continue
        # json.dumps emits a bare NaN/Infinity token, which Postgres jsonb
        # rejects outright -- so an un-guarded NaN would fail the whole
        # segment's write, where the old prose string persisted harmlessly.
        if not math.isfinite(number) or not (low <= number <= high):
            obj.pop(key)
            continue
        obj[key] = number
    return obj


def coerce_aspects(value: Any) -> Optional[list]:
    """Normalize ``aspects`` to ``Optional[List[str]]``.

    Same latent defect as ``meaning``/``sentiment``: ``_finalize_enrichment``
    used ``setdefault("aspects", [])``, which is a no-op for a present-but-
    wrong-typed key, and ``SegmentRecord.aspects: Optional[List[str]]`` would
    500 the same endpoint the same way. Live check 2026-08-29 found
    ``jsonb_typeof(aspects) = 'array'`` on all 701 enriched rows, so this is
    latent rather than live -- fixed here because it is one line and the same
    bug.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [text for text in (_as_text(item) for item in value) if text]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _range_spec(key: str) -> str:
    low, high = SENTIMENT_RANGES[key]
    return f"<float {low} to {high}>"


def describe_enrichment_shape() -> str:
    """The shape block appended to the enrichment prompt.

    Built from the constants above so the key names and numeric ranges the
    model is told about cannot drift from what the coercion accepts. Adding a
    key to ``MEANING_EDGE_PREDICATES`` tells the model about it in the same
    commit, by construction.
    """
    meaning_fields = [f'  "{key}": "<short string>"' for key in MEANING_TEXT_KEYS]
    meaning_fields += [f'  "{key}": ["<string>", ...]' for key in MEANING_LIST_KEYS]
    sentiment_fields = [f'  "{key}": {_range_spec(key)}' for key in SENTIMENT_SCALAR_KEYS]
    return (
        "\n"
        "`meaning` and `sentiment` MUST be JSON objects, not strings. Do not "
        "return a JSON-encoded string for either. Use these keys:\n"
        "\n"
        '"meaning": {\n' + ",\n".join(meaning_fields) + "\n}\n"
        "\n"
        '"sentiment": {\n' + ",\n".join(sentiment_fields) + "\n}\n"
        "\n"
        "Every `sentiment` value must be a plain number inside the range shown; "
        "a value outside it is discarded.\n"
        "`aspects` is a flat list of strings.\n"
        "Each item of `entities`, `questions`, `claims` and `next_steps` is a "
        "plain string, not an object. `entities` is the list of concrete named "
        "things the segment is about (people, services, files, projects). Use [] "
        "when there are none -- never a prose sentence.\n"
    )

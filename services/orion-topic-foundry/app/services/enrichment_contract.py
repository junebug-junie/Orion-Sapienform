"""The declared shape of a segment's ``meaning`` and ``sentiment``, plus the
coercion that keeps the database honest about it.

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
- ``repository.fetch_segments``'s ``friction``/``valence`` sort keys are
  ``(sentiment->>'friction')::float``, which is NULL for a JSON string, so
  sorting by either silently collapsed to a constant 0.

The prompt text below is GENERATED from the same constants the coercion
validates against, so the instruction given to the model and the shape the
database accepts cannot drift apart. That is the actual seam here; the
coercion is what makes the 552 already-written rows readable without paying
to re-enrich them.

Pure module: no DB, no network, no LLM, no settings. See
``tests/test_enrichment_contract.py``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

# ``meaning``'s list-valued keys. These are exactly the keys
# ``kg_edges._edges_from_segment`` reads to build ``mentions`` /
# ``asks_about`` / ``claims_about`` / ``next_step`` edges -- it guards with
# ``isinstance(values, list)`` and silently returns [] otherwise, so a
# non-list here is a silent edge drop, not an error.
MEANING_LIST_KEYS = ("questions", "claims", "next_steps", "entities")

# ``meaning``'s scalar text keys, matching ``_heuristic_enrich``'s output --
# the only shape in this codebase that was ever actually correct.
MEANING_TEXT_KEYS = ("intent", "outcome")

# ``sentiment``'s numeric keys. ``repository.fetch_segments`` sorts on
# ``friction`` and ``valence`` via ``->>``, so those two in particular have a
# live consumer that needs real numbers.
SENTIMENT_SCALAR_KEYS = ("valence", "arousal", "stance", "uncertainty", "friction")

# Marker written onto a coerced-from-prose object. A consumer can tell "the
# enricher produced unstructured text" apart from "the enricher produced a
# real object", instead of both arriving as an indistinguishable dict.
UNSTRUCTURED_KEY = "unstructured"
SUMMARY_KEY = "summary"


def _as_object(value: Any) -> Optional[Dict[str, Any]]:
    """Return ``value`` as a dict, or ``None`` if there is nothing there.

    A JSON string that parses to an object is that object -- the enricher
    double-encoding its answer is a formatting mistake, not a different
    meaning. Anything else non-empty is preserved verbatim under
    ``summary`` and flagged ``unstructured``: prose the model wrote is real
    output and dropping it would be lying about what happened, but it is not
    a structured object and must not masquerade as one.
    """
    if value is None:
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
    return {SUMMARY_KEY: str(value), UNSTRUCTURED_KEY: True}


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
            obj[key] = [str(item).strip() for item in items if str(item).strip()]
        elif isinstance(items, str) and items.strip():
            # A bare string is read as a one-element list, the literal
            # reading. Deliberately NOT split on commas: "orion, juniper" may
            # be one entity or two and this module does not get to guess.
            obj[key] = [items.strip()]
        else:
            obj[key] = []
    for key in MEANING_TEXT_KEYS:
        if key in obj and obj[key] is not None and not isinstance(obj[key], str):
            obj[key] = str(obj[key])
    return obj


def coerce_sentiment(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize a ``sentiment`` value to the declared object shape.

    A non-numeric scalar key is dropped rather than defaulted to 0.0: a
    fabricated 0.0 valence is indistinguishable from a measured neutral one,
    and ``fetch_segments``'s ``COALESCE(..., 0)`` already renders absence as
    0 for sorting without also persisting a number nobody produced.
    """
    obj = _as_object(value)
    if obj is None:
        return None
    for key in SENTIMENT_SCALAR_KEYS:
        if key not in obj:
            continue
        try:
            obj[key] = float(obj[key])
        except (TypeError, ValueError):
            obj.pop(key)
    return obj


def _shape_line(name: str, spec: str) -> str:
    return f'  "{name}": {spec}'


def describe_enrichment_shape() -> str:
    """The shape block appended to the enrichment prompt.

    Built from the constants above so the instruction and the validator
    cannot drift. If a key is added to ``MEANING_LIST_KEYS``, the model is
    told about it in the same commit, by construction.
    """
    meaning_fields = [_shape_line(key, '"<short string>"') for key in MEANING_TEXT_KEYS]
    meaning_fields += [_shape_line(key, "[\"<string>\", ...]") for key in MEANING_LIST_KEYS]
    sentiment_fields = [_shape_line(key, "<float -1.0..1.0>") for key in SENTIMENT_SCALAR_KEYS]
    return (
        "\n"
        "`meaning` and `sentiment` MUST be JSON objects, not strings. Do not "
        "return a JSON-encoded string for either. Use exactly these keys:\n"
        "\n"
        '"meaning": {\n' + ",\n".join(meaning_fields) + "\n}\n"
        "\n"
        '"sentiment": {\n' + ",\n".join(sentiment_fields) + "\n}\n"
        "\n"
        "`entities` is the list of concrete named things the segment is about "
        "(people, services, files, projects). Use [] when there are none -- "
        "never a prose sentence.\n"
    )

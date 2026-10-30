"""References: the ids Orion may cite to another part of itself, and the ONE way to open each.

Rule (Orion Prime step 1, 2026-10-30): an id may cross from a producer to a consumer only as a
structured reference ``{kind, id}`` whose kind is in this closed set. Each kind has exactly one
resolver below. The producer (Prime, services/orion-mind/app/prime_loop.py) resolves every
reference with it BEFORE sending; the consumer (a curiosity run, through the dig kit
``python -P -m orion.dig <kind> <id>``) opens it with the same function. So a reference a producer
could send is one the consumer can open, by construction.

Why: Prime's live question on 2026-10-30 said "arc 8d87adeca2ce06a5", and a curiosity run had no
way to open an arc.

Kinds (both read from orion-durable-runs, the service that already holds both stores):

* ``arc`` -- one Temporal Self arc: GET /temporal-self/arcs/{arc_id} (table temporal_self_arc,
  kept for the chronicle's arc retention; an arc past it no longer resolves).
* ``act`` -- one act ledger row: GET /acts/{act_id} (table grammar_acts).

Adding a kind: add it to ``RefKind``, give it a ``RefKindSpec`` in ``KINDS``, a dig command in
orion/dig/__main__.py, and a sample in orion/dig/tests/test_refs_contract.py ``SAMPLES``. The test there fails
if any of the four is missing.

Free text: ``id_tokens_not_in`` finds id-looking words in text (see ``ID_TOKEN_RE``) so a producer
can refuse text that cites an id outside its references.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, get_args

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

RefKind = Literal["arc", "act"]
REF_KINDS: tuple[str, ...] = get_args(RefKind)
MAX_REFS = 4

# The durable-runs base URL. Prime passes its own PRIME_DURABLE_RUNS_URL; the dig kit (harness
# governor) reuses ORION_PHYSIOLOGY_REPORT_URL, which is the same service (orion/physiology/report_text.py).
DURABLE_RUNS_URL_ENV = "ORION_PHYSIOLOGY_REPORT_URL"
DEFAULT_DURABLE_RUNS_URL = "http://orion-athena-durable-runs:8121"
HTTP_TIMEOUT_SEC = 5.0

# An id-looking word: a hex run of 12+ characters that mixes digits and letters a-f (arc, thought,
# concern and run ids), an act id ``act_<hex>``, or a UUID (focus ids). Twelve, because the shortest
# id Orion shows is a 12-hex run id; mixed, so a long plain number or a word is never an id.
# An underscore does not hide an id ("run_34d7345e36e7", "prime_act_..."): only a letter or digit
# next to the run does.
ID_TOKEN_RE = re.compile(
    r"(?<![0-9A-Za-z])("
    r"act_[0-9a-f]{8,}"
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{12,}"
    r")(?![0-9A-Za-z])",
    re.IGNORECASE,
)


class RefV1(BaseModel):
    """One structured reference. The id must match its kind's shape (no other text reaches a URL)."""

    model_config = ConfigDict(extra="forbid")
    kind: RefKind
    id: str

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str, info) -> str:
        kind = info.data.get("kind")
        spec = KINDS.get(kind) if kind else None
        value = str(value).strip()
        if spec is not None and not spec.id_re.fullmatch(value):
            raise ValueError(f"{kind} id must look like {spec.id_example}, got {value[:80]!r}")
        return value


@dataclass(frozen=True)
class RefKindSpec:
    kind: str
    id_re: re.Pattern
    id_example: str
    path: Callable[[str], str]              # durable-runs path for one id
    render: Callable[[dict[str, Any]], list[str]]
    what: str                               # one line, plain words


@dataclass
class Resolution:
    ref: RefV1
    ok: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    lines: list[str] = field(default_factory=list)


def _s(value: Any) -> str:
    return " ".join(str(value).split()) if value not in (None, "") else "-"


def _render_arc(a: dict[str, Any]) -> list[str]:
    label = a.get("subject_label") or a.get("subject_ref") or "?"
    lines = [
        f"arc {a.get('arc_id')}: {a.get('kind')} {label!r}, {a.get('status')}"
        + (f" ({a.get('closed_reason')})" if a.get("closed_reason") else ""),
        f"day {a.get('day_id')}; began {_s(a.get('began_at'))}; ended {_s(a.get('ended_at'))}; "
        f"last seen {_s(a.get('last_seen_at'))}",
        f"attention returned {a.get('attention_returns', 0)}x; dwell {a.get('cumulative_dwell_sec', 0)} s; "
        f"segments {len(a.get('segments') or [])}",
        f"subject_ref {_s(a.get('subject_ref'))}",
    ]
    if a.get("carried_from_arc_id"):
        lines.append(f"carried from arc {a['carried_from_arc_id']} (previous day)")
    for key in ("related_refs", "evidence_refs"):
        refs = a.get(key) or []
        if refs:
            lines.append(f"{key} ({len(refs)}): " + ", ".join(str(r) for r in refs[:6])
                         + (" ..." if len(refs) > 6 else ""))
    return lines


def _render_act(a: dict[str, Any]) -> list[str]:
    outcome = ("still open" if a.get("closed_at") is None
               else f"closed {a.get('closed_at')}: outcome {a.get('outcome_value')} ({a.get('outcome_source')})")
    lines = [
        f"act {a.get('act_id')}: {a.get('actor')} {a.get('verb')} ({a.get('level')}) on {_s(a.get('object_ref'))}",
        f"opened {_s(a.get('opened_at'))}; expected {a.get('signal_id')} by {_s(a.get('horizon_at'))}; {outcome}",
    ]
    if a.get("outcome_note"):
        lines.append(f"close note: {_s(a['outcome_note'])[:400]}")
    for key in ("goal_ref", "parent_act_id", "test_before"):
        if a.get(key):
            lines.append(f"{key}: {a[key]}")
    if a.get("expect_basis"):
        lines.append(f"what it expected: {_s(a['expect_basis'])[:400]}")
    return lines


KINDS: dict[str, RefKindSpec] = {
    "arc": RefKindSpec(
        kind="arc", id_re=re.compile(r"[0-9a-f]{16}"), id_example="16 hex characters",
        path=lambda i: f"/temporal-self/arcs/{i}", render=_render_arc,
        what="one Temporal Self arc (a stretch of attention on one subject)",
    ),
    "act": RefKindSpec(
        kind="act", id_re=re.compile(r"act_[0-9a-f]{24}"), id_example="act_ + 24 hex characters",
        path=lambda i: f"/acts/{i}", render=_render_act,
        what="one act on the act ledger (what Orion did, what it expected, how it closed)",
    ),
}


def durable_runs_url(base: Optional[str] = None) -> str:
    return (base or os.environ.get(DURABLE_RUNS_URL_ENV) or DEFAULT_DURABLE_RUNS_URL).rstrip("/")


def dig_command(ref: RefV1) -> str:
    """The exact command a curiosity run uses to open this reference."""
    return f"python -P -m orion.dig {ref.kind} {ref.id}"


def _get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SEC) as resp:  # noqa: S310 -- internal service URL
        return json.loads(resp.read())


def resolve(ref: RefV1, *, base_url: Optional[str] = None,
            get_json: Callable[[str], Any] = _get_json) -> Resolution:
    """Open one reference. Never raises: ``ok`` False carries the reason in plain words."""
    spec = KINDS[ref.kind]
    url = durable_runs_url(base_url) + spec.path(ref.id)
    try:
        data = get_json(url)
    except urllib.error.HTTPError as exc:
        why = "not found" if exc.code == 404 else f"HTTP {exc.code}"
        return Resolution(ref, False, error=f"{ref.kind} {ref.id}: {why}")
    except Exception as exc:  # noqa: BLE001 -- unreachable is unresolved, never "fine"
        return Resolution(ref, False, error=f"{ref.kind} {ref.id}: could not read ({type(exc).__name__})")
    if not isinstance(data, dict) or not data:
        return Resolution(ref, False, error=f"{ref.kind} {ref.id}: empty answer")
    try:
        lines = spec.render(data)
    except Exception as exc:  # noqa: BLE001
        return Resolution(ref, False, data=data, error=f"{ref.kind} {ref.id}: unreadable ({type(exc).__name__})")
    return Resolution(ref, True, data=data, lines=lines)


def error_words(exc: ValidationError) -> str:
    """The first validation problem in plain words (pydantic's own message, no URL)."""
    errs = exc.errors()
    return str(errs[0].get("msg", "invalid"))[:160] if errs else "invalid"


def parse_refs(raw: Any) -> tuple[list[RefV1], list[str]]:
    """(valid refs, plain-words problems). A list of dicts in; anything malformed is a problem."""
    if raw in (None, []):
        return [], []
    if not isinstance(raw, list):
        return [], ["refs must be a list of {kind, id}"]
    refs, problems = [], []
    for item in raw:
        try:
            ref = RefV1.model_validate(item)
        except ValidationError as exc:
            problems.append(f"ref {json.dumps(item, default=str)[:120]} is not usable "
                            f"(kinds: {', '.join(REF_KINDS)}): {error_words(exc)}")
            continue
        if ref not in refs:
            refs.append(ref)
    if len(refs) > MAX_REFS:
        problems.append(f"at most {MAX_REFS} refs, got {len(refs)}")
    return refs, problems


def id_tokens_not_in(text: str, refs: list[RefV1]) -> list[str]:
    """Id-looking words in ``text`` that are not one of ``refs``' ids (case-insensitive)."""
    allowed = {r.id.lower() for r in refs}
    out: list[str] = []
    for m in ID_TOKEN_RE.finditer(text or ""):
        tok = m.group(1).lower()
        if tok not in allowed and tok not in out:
            out.append(tok)
    return out


__all__ = [
    "ID_TOKEN_RE",
    "KINDS",
    "MAX_REFS",
    "REF_KINDS",
    "RefKind",
    "RefKindSpec",
    "RefV1",
    "Resolution",
    "dig_command",
    "durable_runs_url",
    "error_words",
    "id_tokens_not_in",
    "parse_refs",
    "resolve",
]

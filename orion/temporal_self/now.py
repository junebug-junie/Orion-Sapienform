"""Orion's now: the few most salient experiences of other organs since a reader last looked. Pure.

Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, Part A (A2-A5).

* An **experience** is one ``temporal_self_event`` row. Its **organ** comes from the organ registry
  (``config/integration/organs.yaml``); its **salience** is the share of the same source kind's events
  in the 30 days before it that were strictly smaller, by the kind's declared magnitude
  (``MAGNITUDE_RULES``; the rule itself is ``orion.physiology.event_wants.source_salience``). Under 20
  past events, or a kind with no magnitude: salience is None, never 0. Use never feeds salience.
* ``build_now`` picks at most 12 items, at most 2 per organ, never the reader's own organ, highest
  salience first (None last), newest first on ties.
* ``record_use`` checks a reader's cites against what it was shown (the rule of
  ``orion/temporal_self/citations.py``: only a shown id counts; anything else is removed and counted)
  and returns the rows to store. At most 3 count per run.
* ``use_rate`` is the integration metric: the share of experiences a different organ cited, validly,
  within 24 h; and the organs whose experiences another organ used ("closed organs").

No I/O (the package rule, tests/test_sources.py): ``orion.integration.organs.load_organs`` reads the
registry file and orion-durable-runs reads rows and stores what this returns.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, get_args

from orion.curiosity.value import ARM_SELF_INQUIRY
from orion.physiology.event_wants import MIN_SOURCE_HISTORY, SOURCE_HISTORY, source_salience
from orion.schemas.temporal_self import (
    NOW_LABEL_MAX,
    NOW_MAX_CITES,
    NOW_MAX_ITEMS,
    NOW_MAX_PER_ORGAN,
    NOW_RELATED_REFS_CAP,
    ExperienceUseRateV1,
    ExperienceUseResultV1,
    ExperienceUseRowV1,
    OrganUseV1,
    TemporalNowItemV1,
    TemporalNowV1,
    TemporalSourceKind,
)

NOW_MAX_PINNED = 3  # a need's content rows a reader may pin into its now (build_now ``pinned``)
USE_WINDOW = timedelta(hours=24)  # a use counts for the metric when cited within 24 h of the event
EXPERIENCE_PREFIX = "experience:"

__all__ = [
    "MAGNITUDE_RULES",
    "NOW_MAX_PINNED",
    "MIN_SOURCE_HISTORY",
    "SOURCE_HISTORY",
    "OrganRegistry",
    "build_now",
    "magnitude",
    "now_label",
    "parse_organs",
    "record_use",
    "salience_of",
    "stored_payload_surprise",
    "use_rate",
]


# --------------------------------------------------------------------------- the registry


@dataclass(frozen=True)
class OrganRegistry:
    """``config/integration/organs.yaml``, parsed. ``organ_of`` maps a source kind to its one organ."""

    organs: tuple[str, ...]
    organ_of: Mapping[str, str]
    magnitude_rule: Mapping[str, str]  # source kind -> MAGNITUDE_RULES key
    raw: Mapping[str, Any] = field(default_factory=dict)

    def kinds_of(self, organ: str) -> list[str]:
        return sorted(k for k, o in self.organ_of.items() if o == organ)


def parse_organs(doc: Mapping[str, Any]) -> OrganRegistry:
    """Validate and index the registry. Raises ValueError on a kind claimed twice, an unknown kind, or
    a magnitude rule that does not exist or names a kind of another organ."""
    organs = (doc or {}).get("organs")
    if not isinstance(organs, Mapping) or not organs:
        raise ValueError("organs.yaml: top-level 'organs' map is missing or empty")
    known_kinds = set(get_args(TemporalSourceKind))
    organ_of: dict[str, str] = {}
    rule_of: dict[str, str] = {}
    for name, spec in organs.items():
        spec = spec or {}
        kinds = spec.get("temporal_sources") or []
        if not isinstance(kinds, list):
            raise ValueError(f"organs.yaml: {name}.temporal_sources must be a list")
        for k in kinds:
            if k not in known_kinds:
                raise ValueError(f"organs.yaml: {name} claims {k!r}, not a TemporalSourceKind")
            if k in organ_of:
                raise ValueError(f"organs.yaml: {k!r} is claimed by both {organ_of[k]} and {name}")
            organ_of[k] = str(name)
        for k, rule in (spec.get("salience_magnitude") or {}).items():
            if k not in kinds:
                raise ValueError(f"organs.yaml: {name}.salience_magnitude names {k!r}, not one of its temporal_sources")
            if rule not in MAGNITUDE_RULES:
                raise ValueError(f"organs.yaml: {name}.salience_magnitude.{k} = {rule!r} is not a rule in now.MAGNITUDE_RULES")
            rule_of[k] = str(rule)
    return OrganRegistry(organs=tuple(str(o) for o in organs), organ_of=organ_of, magnitude_rule=rule_of,
                         raw=dict(organs))


# --------------------------------------------------------------------------- magnitude and salience
# A rule reads one STORED row shape: ``payload`` is temporal_self_event.payload_json (the event's
# payload plus ``ended_at``/``privacy_class``), ``verdict`` and ``surprise_nats`` are its columns. The
# chronicle scores a new event with the same shape (temporal_self_store.stored_payload), so history
# read back from the table and a new event are measured by one function.


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _len(v: Any) -> Optional[float]:
    return float(len(v)) if isinstance(v, list) else None


def _visit_seconds(row: Mapping[str, Any]) -> Optional[float]:
    payload = row.get("payload") or {}
    end, start = payload.get("ended_at"), row.get("occurred_at")
    if not end or start is None:
        return None
    try:
        end_dt = end if isinstance(end, datetime) else datetime.fromisoformat(str(end))
        return max(0.0, (end_dt - start).total_seconds())
    except (TypeError, ValueError):
        return None


def _sum_present(*vals: Any) -> Optional[float]:
    nums = [n for n in (_num(v) for v in vals) if n is not None]
    return float(sum(nums)) if nums else None


_SEVERITY = {"degraded": 1.0, "critical": 2.0}

MAGNITUDE_RULES: dict[str, Callable[[Mapping[str, Any]], Optional[float]]] = {
    # dream: hypotheses the cycle formed (dream_cycle.hypothesis_count)
    "dream_hypothesis_count": lambda r: _num((r.get("payload") or {}).get("hypothesis_count")),
    # curiosity: priors the run moved plus priors it formed (curiosity_run_outcomes)
    "curiosity_moved_formed": lambda r: _sum_present((r.get("payload") or {}).get("n_moved"),
                                                     (r.get("payload") or {}).get("n_formed")),
    # camera: how long one office visit lasted (vision_office_session started_at .. ended_at)
    "visit_seconds": _visit_seconds,
    # play: how long one Retro session lasted, in minutes (retro_play_sessions.duration_sec: a person's
    # sitting, a co-play handoff or a solo leisure session). A count of time, so rest is a true zero.
    "sitting_minutes": lambda r: (lambda sec: sec / 60.0 if sec is not None else None)(
        _num((r.get("payload") or {}).get("duration_sec"))),
    # body: how long a background GPU lease waited (gpu_pool_events.waited_ms)
    "waited_ms": lambda r: _num((r.get("payload") or {}).get("waited_ms")),
    # metacognition: degraded 1, critical 2 (orion_metacog.severity; only those two are emitted)
    "metacog_severity": lambda r: _SEVERITY.get(str(r.get("verdict") or "")),
    # metacognition: broadcast ticks one node held dominance (field_dominance_run.tick_count)
    "dominance_ticks": lambda r: _num((r.get("payload") or {}).get("tick_count")),
    # a real prediction's surprisal: the act ledger's close (grammar_acts / substrate_action_outcomes)
    # or an exceedance forecast's -ln P(observed). The column, else the payload copy.
    "surprise_nats": lambda r: _num(r.get("surprise_nats")) if _num(r.get("surprise_nats")) is not None
    else _num((r.get("payload") or {}).get("surprise_nats")),
    # prediction: a finished randomised experiment's |estimate| / SE of its primary outcome
    # (sources.experiment_result). A clear effect is more notable than a null; z does not drift with age.
    "effect_z": lambda r: (lambda z: abs(z) if z is not None else None)(_num((r.get("payload") or {}).get("effect_z"))),
}


def magnitude(rule: Optional[str], row: Mapping[str, Any]) -> Optional[float]:
    """The declared magnitude of one stored-shape row, or None (no rule, or the field is missing)."""
    if not rule:
        return None
    fn = MAGNITUDE_RULES.get(rule)
    return fn(row) if fn is not None else None


def stored_payload_surprise(payload: Mapping[str, Any]) -> Optional[float]:
    """``surprise_nats`` for the column: only where the source carried a real prediction's surprisal."""
    return _num((payload or {}).get("surprise_nats"))


def salience_of(mag: Optional[float], at: datetime, history: Iterable[tuple[datetime, float]]) -> Optional[float]:
    """``source_salience`` against the same kind's magnitudes in the 30 days strictly before ``at``."""
    if mag is None:
        return None
    lo = at - SOURCE_HISTORY
    return source_salience(mag, [m for t, m in history if lo <= t < at])


# --------------------------------------------------------------------------- labels

_TEMPLATES: dict[str, Callable[[Mapping[str, Any]], str]] = {
    # Never the experiment arm (#3458): the three-arm curiosity test (#3183) and the dream/control hypothesis
    # test are blinded, and the now is read by Prime, every chat turn's Prime block and the roundup. The arm
    # stays in the stored payload for the analysis scripts only. self_inquiry is a line, not an offer arm.
    "curiosity_run": lambda p: (f"{'self-inquiry run' if p.get('arm') == ARM_SELF_INQUIRY else 'curiosity run'}: "
                                f"tested {p.get('n_tested')}, moved {p.get('n_moved')}, formed {p.get('n_formed')}"),
    "dream_cycle": lambda p: (f"dream cycle: {p.get('hypothesis_count')} hypotheses from "
                              f"{p.get('replay_count')} replayed items"),
    "dream_hypothesis": lambda p: "dream hypothesis",
    "gpu_wait": lambda p: f"{p.get('holder') or 'a background lease'} waited {_num(p.get('waited_ms')) or 0:.0f} ms for a GPU",
    "action_outcome": lambda p: f"act outcome {p.get('target_id')} ({p.get('dispatch_kind')})",
    "vision_percept": lambda p: "saw " + ", ".join(str(x) for x in (p.get("entities") or [])[:6]),
    "metacog_observation": lambda p: f"metacog trigger {p.get('trigger_kind')}",
    "presence_transition": lambda p: f"room presence {p.get('from_state')} -> {p.get('to_state')}",
    "memory_episode": lambda p: f"memory episode ({p.get('purpose') or 'no purpose'})",
    "consolidation_window_close": lambda p: f"memory window closed ({p.get('close_reason')})",
    "chat_turn": lambda p: f"chat turn by {p.get('speaker') or 'unknown'}",
    "attention_row": lambda p: f"attention: {p.get('process')} {p.get('reason') or ''}".rstrip(),
    "visual_deferral": lambda p: f"painting deferred ({p.get('reason') or p.get('thermal_state') or 'no reason'})",
    "field_dominance_run": lambda p: f"held attention for {p.get('tick_count')} ticks",
}


def now_label(source_kind: str, label: Optional[str], verdict: Optional[str], payload: Mapping[str, Any]) -> str:
    """A code-written label of at most 200 characters: the row's own label when it has one (prefixed by
    its kind's word), else a template over the row's own fields. Never model prose added here."""
    own = " ".join(str(label or "").split())
    if own:
        text = f"{source_kind.replace('_', ' ')}: {own}"
    else:
        tmpl = _TEMPLATES.get(source_kind)
        try:
            text = tmpl(payload or {}) if tmpl else source_kind.replace("_", " ")
        except Exception:  # noqa: BLE001 - a malformed payload falls back to the kind's word
            text = source_kind.replace("_", " ")
    if verdict and str(verdict) not in text:
        text = f"{text} [{verdict}]"
    text = " ".join(text.split())
    return text if len(text) <= NOW_LABEL_MAX else text[: NOW_LABEL_MAX - 1] + "…"


# --------------------------------------------------------------------------- the now


def _sort_key(c: Mapping[str, Any]) -> tuple:
    sal = c.get("salience")
    return (0 if sal is not None else 1, -(sal or 0.0), -c["occurred_at"].timestamp(), str(c["event_id"]))


def build_now(
    candidates: Sequence[Mapping[str, Any]],
    reader_organ: str,
    *,
    since: datetime,
    as_of: datetime,
    shown_id: str,
    registry: OrganRegistry,
    reader_run_ref: Optional[str] = None,
    max_items: int = NOW_MAX_ITEMS,
    max_per_organ: int = NOW_MAX_PER_ORGAN,
    candidates_total: Optional[int] = None,
    pinned: Sequence[Mapping[str, Any]] = (),
) -> TemporalNowV1:
    """Pick the now from candidate rows (stored shape: event_id, source_kind, occurred_at, label, verdict,
    payload, related_refs, salience, surprise_nats). ``candidates_total``: how many rows the caller's
    query matched before its own per-kind cut (defaults to ``len(candidates)``).

    ``pinned`` (Prime hunger, integration Part B): rows the reader is about to show anyway as a need's content
    (at most ``NOW_MAX_PINNED``, other organs only, same row shape), so a cite of one is valid against this shown
    row. They come first, outside the per-organ cap, and take slots from the total cap."""
    if reader_organ not in registry.organs:
        raise ValueError(f"unknown organ {reader_organ!r}; registered: {', '.join(registry.organs)}")
    max_items = max(0, min(int(max_items), NOW_MAX_ITEMS))
    max_per_organ = max(1, int(max_per_organ))
    unregistered = sorted({str(c["source_kind"]) for c in candidates if c["source_kind"] not in registry.organ_of})
    pins: list[Mapping[str, Any]] = []
    for c in pinned:
        if (c["source_kind"] in registry.organ_of and registry.organ_of[c["source_kind"]] != reader_organ
                and str(c["event_id"]) not in {str(p["event_id"]) for p in pins} and len(pins) < NOW_MAX_PINNED):
            pins.append(c)
    pin_ids = {str(p["event_id"]) for p in pins}
    max_items = max(0, max_items - len(pins))
    pool = [c for c in candidates
            if c["source_kind"] in registry.organ_of and registry.organ_of[c["source_kind"]] != reader_organ
            and str(c["event_id"]) not in pin_ids]
    # The organ cap first (over every candidate), then the total cap over what is left: both counts
    # are then independent of where the cut falls.
    per_organ: dict[str, int] = {}
    within: list[Mapping[str, Any]] = []
    for c in sorted(pool, key=_sort_key):
        organ = registry.organ_of[c["source_kind"]]
        if per_organ.get(organ, 0) < max_per_organ:
            per_organ[organ] = per_organ.get(organ, 0) + 1
            within.append(c)
    picked = within[:max_items]
    dropped_organ, dropped_total = len(pool) - len(within), len(within) - len(picked)
    items = [
        TemporalNowItemV1(
            event_id=str(c["event_id"]), organ=registry.organ_of[c["source_kind"]], source_kind=c["source_kind"],
            label=now_label(c["source_kind"], c.get("label"), c.get("verdict"), c.get("payload") or {}),
            occurred_at=c["occurred_at"], salience=c.get("salience"), surprise_nats=_num(c.get("surprise_nats")),
            related_refs=[str(r) for r in (c.get("related_refs") or [])][:NOW_RELATED_REFS_CAP],
        )
        for c in [*pins, *picked]
    ]
    total = len(pool) if candidates_total is None else int(candidates_total)
    return TemporalNowV1(
        shown_id=shown_id, reader_organ=reader_organ, reader_run_ref=reader_run_ref, since=since, as_of=as_of,
        items=items, candidates=total, dropped_by_organ_cap=dropped_organ, dropped_by_total_cap=dropped_total,
        unregistered_kinds=unregistered,
    )


# --------------------------------------------------------------------------- cites


def normalize_cite(cite: Any) -> str:
    s = str(cite or "").strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    return s[len(EXPERIENCE_PREFIX):] if s.startswith(EXPERIENCE_PREFIX) else s


def use_id(shown_id: str, event_id: str) -> str:
    return hashlib.sha256(f"{shown_id}|{event_id}".encode()).hexdigest()[:24]


def record_use(
    *,
    shown_id: str,
    reader_organ: str,
    shown_event_ids: Iterable[str],
    cites: Iterable[Any],
    citing_ref: str,
    cited_at: datetime,
    max_cites: int = NOW_MAX_CITES,
    already_used: Iterable[str] = (),
) -> ExperienceUseResultV1:
    """Check cites against the shown row. A cite not in the shown list is dropped and counted
    (``rejected_not_shown``); valid cites past ``max_cites`` are dropped and counted
    (``rejected_over_cap``). Duplicates count once. The citing organ is the one that was shown.

    The cap is per shown row, not per call: ``already_used`` (event ids already stored for this shown
    row) count against it, and re-sending one of them stores nothing new."""
    shown = set(shown_event_ids)
    stored = set(already_used)
    max_cites = max(0, max_cites - len(stored))
    seen: set[str] = set(stored)
    uses: list[ExperienceUseRowV1] = []
    not_shown: list[str] = []
    over_cap: list[str] = []
    for raw in cites:
        eid = normalize_cite(raw)
        if not eid or eid in seen:
            continue
        seen.add(eid)
        if eid not in shown:
            not_shown.append(eid)
        elif len(uses) >= max_cites:
            over_cap.append(eid)
        else:
            uses.append(ExperienceUseRowV1(use_id=use_id(shown_id, eid), event_id=eid, citing_organ=reader_organ,
                                           citing_ref=citing_ref, shown_id=shown_id, cited_at=cited_at))
    return ExperienceUseResultV1(shown_id=shown_id, citing_organ=reader_organ, uses=uses,
                                 rejected_not_shown=not_shown, rejected_over_cap=over_cap)


# --------------------------------------------------------------------------- the integration metric


def use_rate(
    kind_counts: Mapping[str, int],
    uses: Iterable[tuple[str, str, str]],
    *,
    window_start: datetime,
    window_end: datetime,
    registry: OrganRegistry,
    cites_made: Optional[Mapping[str, int]] = None,
) -> ExperienceUseRateV1:
    """``kind_counts``: experiences recorded in the window, per source kind. ``uses``: (event_id,
    source_kind, citing_organ) for valid cites of those experiences within 24 h (the caller's query
    applies the 24 h rule). A use by the event's own organ never counts. ``cites_made``: valid cites
    each organ made in the window (for the per-organ line; not part of the rate)."""
    per_organ_exp: dict[str, int] = {o: 0 for o in registry.organs}
    unregistered: dict[str, int] = {}
    for kind, n in kind_counts.items():
        organ = registry.organ_of.get(kind)
        if organ is None:
            unregistered[kind] = unregistered.get(kind, 0) + int(n)
        else:
            per_organ_exp[organ] += int(n)
    used: dict[str, set[str]] = {o: set() for o in registry.organs}
    for event_id, kind, citing in uses:
        organ = registry.organ_of.get(kind)
        if organ is None or citing == organ:
            continue
        used[organ].add(event_id)
    organs = []
    for o in registry.organs:
        n, u = per_organ_exp[o], len(used[o])
        organs.append(OrganUseV1(organ=o, experiences=n, used_by_other=u,
                                 cites_made=int((cites_made or {}).get(o, 0)),
                                 rate=(u / n) if n else None))
    total = sum(per_organ_exp.values())
    total_used = sum(len(v) for v in used.values())
    return ExperienceUseRateV1(
        window_start=window_start, window_end=window_end, experiences=total, used_by_other=total_used,
        rate=(total_used / total) if total else None,
        closed_organs=[o for o in registry.organs if used[o]], registered_organs=len(registry.organs),
        organs=organs, unregistered_kinds=unregistered,
    )

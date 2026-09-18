"""Self-question pool: seed YAML, family draw, pinned floor.

Pure module — no Hub, no Postgres. Task 3 persists ask records.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Sequence

import yaml

Family = Literal["lived", "anatomy"]
MintedBy = Literal["juniper", "orion"]
QuestionStatus = Literal["open", "answered", "parked"]

_DEFAULT_SEED_PATH = Path(__file__).with_name("self_question_seed.yaml")
_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class SelfQuestion:
    question_id: str
    text: str
    family: Family
    pinned: bool
    minted_by: MintedBy
    status: QuestionStatus
    ask_count: int
    last_asked_at: Optional[datetime]


def load_seed_questions(path: Path | None = None) -> list[SelfQuestion]:
    """Load Juniper's pinned starter pack from YAML."""
    seed_path = path or _DEFAULT_SEED_PATH
    raw = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
    rows = raw.get("questions") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"seed file {seed_path} must contain a questions list")

    out: list[SelfQuestion] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            SelfQuestion(
                question_id=str(row["question_id"]),
                text=str(row["text"]),
                family=row["family"],
                pinned=bool(row.get("pinned", False)),
                minted_by=row.get("minted_by", "juniper"),
                status="open",
                ask_count=0,
                last_asked_at=None,
            )
        )
    return out


def with_ask_recorded(q: SelfQuestion, *, now: datetime) -> SelfQuestion:
    """Return a copy with ask_count incremented and last_asked_at set."""
    return replace(q, ask_count=q.ask_count + 1, last_asked_at=now)


def _eligible(pool: Sequence[SelfQuestion]) -> list[SelfQuestion]:
    return [q for q in pool if q.status != "parked"]


def _age_days(last_asked_at: Optional[datetime], now: datetime) -> Optional[float]:
    if last_asked_at is None:
        return None
    ref = last_asked_at
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    anchor = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return (anchor - ref).total_seconds() / 86400.0


def _needs_pinned_floor(q: SelfQuestion, *, pinned_floor_days: float, now: datetime) -> bool:
    if not (q.pinned and q.family == "lived"):
        return False
    age = _age_days(q.last_asked_at, now)
    return age is None or age >= pinned_floor_days


def _pick_sort_key(q: SelfQuestion) -> tuple:
    """Never-asked first, then oldest, then lowest ask_count."""
    return (
        q.last_asked_at is not None,
        q.last_asked_at or _EPOCH,
        q.ask_count,
    )


def _want_lived(recent_families: Sequence[str], lived_weight: float, rng: random.Random) -> bool:
    window = list(recent_families)[-12:]
    if not window:
        return rng.random() < lived_weight
    lived_share = sum(1 for f in window if f == "lived") / len(window)
    if lived_share < lived_weight:
        return True
    if lived_share > lived_weight + 0.05:
        return False
    return rng.random() < lived_weight


def pick_question(
    *,
    pool: Sequence[SelfQuestion],
    recent_families: Sequence[str],
    lived_weight: float = 0.75,
    pinned_floor_days: float = 7.0,
    now: datetime | None = None,
    rng: random.Random | None = None,
) -> SelfQuestion:
    """Choose the next self-inquiry question from the pool."""
    anchor = now or datetime.now(timezone.utc)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    roll = rng or random.Random()

    candidates = _eligible(pool)
    if not candidates:
        raise ValueError("pool has no eligible (non-parked) questions")

    floor = [q for q in candidates if _needs_pinned_floor(q, pinned_floor_days=pinned_floor_days, now=anchor)]
    if floor:
        return sorted(floor, key=_pick_sort_key)[0]

    family: Family = "lived" if _want_lived(recent_families, lived_weight, roll) else "anatomy"
    in_family = [q for q in candidates if q.family == family]
    if not in_family:
        other: Family = "anatomy" if family == "lived" else "lived"
        in_family = [q for q in candidates if q.family == other]
    if not in_family:
        return sorted(candidates, key=_pick_sort_key)[0]

    return sorted(in_family, key=_pick_sort_key)[0]

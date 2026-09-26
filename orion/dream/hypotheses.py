"""Dream hypotheses: how a dream reaches waking curiosity, and how it is scored.

Three seams, one module, so the contract cannot drift between them:

  offer      Hub's curiosity kickoff takes up to N never-offered, unexpired
             hypotheses (both arms, shuffled, arm hidden) and stamps them
             offered. Each hypothesis is shown to Orion exactly once.
  prompt     the kickoff section. Invitational: Orion decides whether any is
             worth holding. The dream never writes a :Prior -- a belief is
             Orion's to form (see orion/curiosity/peer_briefs.py: system
             writers never write belief labels).
  scorecard  pure function joining offered hypotheses to the priors Orion
             formed from them (`formed_from` starts with FORMED_FROM_PREFIX)
             and counting outcomes per arm. The control arm is what makes
             this falsifiable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from orion.schemas.dream_cycle import FORMED_FROM_PREFIX

logger = logging.getLogger("orion.dream.hypotheses")

# Hypothesis ids are `dh-<hex>`; anything after (comma, "; crystallization:x",
# prose) is Orion's own annotation and must not break the join.
_HID_RE = re.compile(r"[A-Za-z0-9_-]+")

# Worldview prior statuses (orion/curiosity/worldview.py). Duplicated as
# literals, not imported, to keep this module importable without Hub's graph
# client; tests/test_dream_hypotheses pins them to the worldview constants.
_SUPPORTED = "supported"
_REVISED = "revised"
_REFUTED = "refuted"
_RETIRED = "retired_unresolvable"
_OPEN = "open"

TAKE_FOR_OFFER_SQL = """
UPDATE dream_hypothesis
   SET offered_at = now(), offered_run_id = $1
 WHERE hypothesis_id IN (
        SELECT hypothesis_id FROM dream_hypothesis
         WHERE offered_at IS NULL AND expires_at > now()
         ORDER BY random()
         LIMIT $2
         FOR UPDATE SKIP LOCKED)
RETURNING hypothesis_id, claim, why
"""


RELEASE_FOR_RUN_SQL = """
UPDATE dream_hypothesis
   SET offered_at = NULL, offered_run_id = NULL
 WHERE offered_run_id = $1
"""


@dataclass(frozen=True)
class OfferedHypothesis:
    hypothesis_id: str
    claim: str
    why: str = ""


async def take_hypotheses_for_offer(pool: Any, *, run_id: str, limit: int) -> tuple[OfferedHypothesis, ...]:
    """Claim up to `limit` hypotheses for one curiosity run. () on any failure.

    Silence over a false section: a missing table (migration not applied) or a
    down pool must not break the kickoff, so every failure returns empty.
    """
    if pool is None or limit <= 0:
        return ()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(TAKE_FOR_OFFER_SQL, run_id, int(limit))
    except Exception as exc:  # noqa: BLE001 -- see docstring
        logger.warning("dream_hypotheses_offer_failed run=%s err=%s", run_id, exc)
        return ()
    out = []
    for r in rows:
        hid = str(r["hypothesis_id"] or "").strip()
        claim = str(r["claim"] or "").strip()
        if hid and claim:
            out.append(OfferedHypothesis(hid, claim, str(r["why"] or "").strip()))
    # RETURNING order follows the physical update order, which can correlate
    # with insert order (dream arm is inserted first). Sort on id so position
    # carries no arm signal.
    out.sort(key=lambda h: h.hypothesis_id)
    return tuple(out)


async def release_hypotheses_for_run(pool: Any, *, run_id: str) -> None:
    """Un-claim a run's hypotheses when the run was cancelled before Orion saw it.

    Keeps `offered` meaning "shown to Orion" -- otherwise a Hub restart mid-turn
    inflates the scorecard's denominator for both arms. Never raises.
    """
    if pool is None or not run_id:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(RELEASE_FOR_RUN_SQL, run_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("dream_hypotheses_release_failed run=%s err=%s", run_id, exc)


def format_dream_section(hypotheses: Sequence[OfferedHypothesis]) -> list[str]:
    """Kickoff lines. Empty list when there is nothing to offer."""
    if not hypotheses:
        return []
    lines = [
        "WHILE YOU SLEPT (optional). Your last dream cycle replayed things that "
        "stayed unresolved and paired memories that do not usually meet. These "
        "are the links it proposed. Some may be noise. None is a belief yet -- "
        "each is shown to you once.",
        "",
    ]
    for h in hypotheses:
        lines.append(f"  hypothesis {h.hypothesis_id}: {h.claim}")
        if h.why:
            lines.append(f"    why the dream linked them: {h.why}")
    lines += [
        "",
        "If one is worth holding, form a Prior from it the usual way and set "
        f'`formed_from = "{FORMED_FROM_PREFIX}<hypothesis id>"` so it stays '
        "traceable to the dream. Ignoring all of them is a fine answer.",
        "",
    ]
    return lines


# --- scorecard ---------------------------------------------------------------


@dataclass
class ArmScore:
    offered: int = 0
    adopted: int = 0
    tested: int = 0
    supported: int = 0
    revised: int = 0
    refuted: int = 0
    retired: int = 0

    @property
    def adoption_rate(self) -> Optional[float]:
        return self.adopted / self.offered if self.offered else None

    @property
    def support_rate(self) -> Optional[float]:
        """Supported-or-revised over tested. None until something was tested."""
        return (self.supported + self.revised) / self.tested if self.tested else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "offered": self.offered,
            "adopted": self.adopted,
            "tested": self.tested,
            "supported": self.supported,
            "revised": self.revised,
            "refuted": self.refuted,
            "retired": self.retired,
            "adoption_rate": self.adoption_rate,
            "support_rate": self.support_rate,
        }


@dataclass
class Scorecard:
    arms: dict[str, ArmScore] = field(default_factory=lambda: {"dream": ArmScore(), "control": ArmScore()})
    unmatched_priors: int = 0

    def verdict(self, *, min_offered: int = 20) -> str:
        """Plain-language read. Refuses to call it before there is enough data."""
        d, c = self.arms["dream"], self.arms["control"]
        if d.offered < min_offered or c.offered < max(1, min_offered // 4):
            return (
                f"too early: dream offered {d.offered}, control offered {c.offered} "
                f"(need {min_offered} / {max(1, min_offered // 4)})"
            )
        da, ca = d.adoption_rate or 0.0, c.adoption_rate or 0.0
        if da <= ca:
            return f"recombination not beating random: adoption dream {da:.2f} <= control {ca:.2f}"
        return f"dream ahead of random: adoption dream {da:.2f} > control {ca:.2f}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "arms": {k: v.as_dict() for k, v in self.arms.items()},
            "unmatched_priors": self.unmatched_priors,
            "verdict": self.verdict(),
        }


def hypothesis_id_from_formed_from(formed_from: Any) -> Optional[str]:
    text = str(formed_from or "").strip()
    if not text.startswith(FORMED_FROM_PREFIX):
        return None
    m = _HID_RE.match(text[len(FORMED_FROM_PREFIX):].strip())
    return m.group(0) if m else None


def score_hypotheses(
    offered_rows: Iterable[dict[str, Any]],
    prior_rows: Iterable[dict[str, Any]],
) -> Scorecard:
    """Join offered hypotheses to the priors formed from them.

    offered_rows: {hypothesis_id, arm} for hypotheses with offered_at set.
    prior_rows:   {prior_id, formed_from, status, times_tested} from Orion's graph.

    A hypothesis is adopted if at least one prior names it. Several priors
    naming one hypothesis (a forked node) count once, using the most-tested one.
    """
    card = Scorecard()
    arm_of: dict[str, str] = {}
    for row in offered_rows:
        hid = str(row.get("hypothesis_id") or "").strip()
        arm = str(row.get("arm") or "").strip()
        if not hid or arm not in card.arms:
            continue
        arm_of[hid] = arm
        card.arms[arm].offered += 1

    best: dict[str, dict[str, Any]] = {}
    unmatched: set[str] = set()
    for row in prior_rows:
        hid = hypothesis_id_from_formed_from(row.get("formed_from"))
        if hid is None:
            continue
        if hid not in arm_of:
            unmatched.add(hid)
            continue
        tested = _as_int(row.get("times_tested"))
        cur = best.get(hid)
        if cur is None or tested > _as_int(cur.get("times_tested")):
            best[hid] = row

    card.unmatched_priors = len(unmatched)
    for hid, row in best.items():
        score = card.arms[arm_of[hid]]
        score.adopted += 1
        status = str(row.get("status") or _OPEN).strip()
        if _as_int(row.get("times_tested")) > 0 or status in (_SUPPORTED, _REVISED, _REFUTED):
            score.tested += 1
        if status == _SUPPORTED:
            score.supported += 1
        elif status == _REVISED:
            score.revised += 1
        elif status == _REFUTED:
            score.refuted += 1
        elif status == _RETIRED:
            score.retired += 1
    return card


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0

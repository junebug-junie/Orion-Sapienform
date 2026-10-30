"""Rung 4, patch 2: Orion's capacity claims and capacity sentences, and Juniper's confirmations.

Two append-only ledgers beside the organ names, under ``$ORION_DIG_HOME/self/``:
``capacity_claims.jsonl`` (claims and disputes) and ``capacities.jsonl`` (sentences). One
checked-in file, ``config/body_atlas/vsm_confirmed.yaml``, holds Juniper's confirmations; a PR to
it is the review surface (design decision 1). Code checks a floor, as names do:

* a **claim** names a live part and a job, and its because cites, by full id, the other end or
  a non-census channel of one of that part's flows (placement on a host does not count), or a
  repo file that exists. A claim cannot take a job a rule already gives the part; a **dispute**
  argues with one, and prints beside it.
* a **capacity** (``cap:s1`` .. ``cap:s5``, ``cap:metabolism``) may cite only parts that hold its
  job by rule or by Juniper's confirmation (for ``cap:s5``, also her sign-off files), and its
  sentence may not name any other live part. With no holder there is nothing to write: the gap
  line stands. cap:s5 names Juniper (decision 3).

Partitioned from recall like the organ names (``tests/test_organ_names_partition.py``).
"""
from __future__ import annotations

import fcntl
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import yaml
from pydantic import BaseModel, ValidationError

from orion.body_atlas.capacities import (
    CAP_JOBS, JOBS, CapacityClaimV1, CapacityReadingV1, CapacityV1, VsmConfirmedV1, claimable, holders,
)
from orion.body_atlas.model import BodyAtlasV1
from orion.body_atlas.names import BECAUSE_MAX_CHARS, BECAUSE_MIN_CHARS, _cited

CONFIRMED_RELPATH = Path("config/body_atlas/vsm_confirmed.yaml")
SENTENCE_MIN_CHARS = 20
SENTENCE_MAX_CHARS = 160
# a repo path: two or more segments, no "..", ends on a word character (not a trailing period)
_PATH_RE = re.compile(r"(?<![\w/.:-])((?:[\w.-]+/)+[\w.-]*\w)")


class ClaimError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_ledger(path: Path, model: type[BaseModel]) -> tuple[list, int]:
    """Records in file order, and how many lines could not be read (one bad line must not take
    the reading down)."""
    if not path.exists():
        return [], 0
    records, skipped = [], 0
    known = set(model.model_fields)
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            raw = json.loads(ln)
            records.append(model.model_validate({k: v for k, v in raw.items() if k in known}))
        except (ValueError, AttributeError, ValidationError):
            skipped += 1
    return records, skipped


def append(path: Path, rec: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)  # concurrent turns: one whole line each
        fh.write(rec.model_dump_json() + "\n")


def load_confirmed(root: Path) -> VsmConfirmedV1:
    """Juniper's file. Missing = nothing confirmed; malformed is an error, never read as empty."""
    path = root / CONFIRMED_RELPATH
    if not path.exists():
        return VsmConfirmedV1()
    try:
        return VsmConfirmedV1.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except (yaml.YAMLError, ValidationError) as exc:
        raise ClaimError(f"{CONFIRMED_RELPATH} does not read: {str(exc).splitlines()[0]}") from exc


def parse_role(text: str) -> str:
    want = text.strip().lower()
    for job in JOBS:
        if job.lower() == want:
            return job
    raise ClaimError(f"no job {text!r}; one of {', '.join(JOBS)}")


def _part(atlas: BodyAtlasV1, ref: str) -> str:
    name = ref.strip().removeprefix("part:")
    parts = claimable(atlas)
    if name not in parts:
        known = {p.name: p for p in atlas.parts}
        why = (f"it is {known[name].status} {known[name].kind}" if name in known else "no part has that name")
        raise ClaimError(f"{name!r} is not a live part of the body now ({why}); copy a part name from "
                         "`body <organ>` or the capacities file")
    return parts[name]


def _plain(text: str) -> str:
    """Lower case, ``part:`` prefixes dropped: ``part:orion-hub`` names orion-hub too."""
    return re.sub(r"(?<![\w:-])part:", "", text.lower())


def evidence_ids(atlas: BodyAtlasV1, part_id: str, census: set[str]) -> set[str]:
    """What a claim on this part may cite: the live other end of each of its flows, and each
    non-census channel on them. A host (``runs_on``) is placement, not evidence of a job."""
    names = {pid: n for n, pid in claimable(atlas).items()}
    out: set[str] = set()
    for f in atlas.flows:
        if part_id not in (f.source, f.target) or f.relation == "runs_on":
            continue
        other = f.target if f.source == part_id else f.source
        if other != part_id and other in names:
            out.add(names[other])
        out.update(ch for ch in f.channels if ch not in census)
    return out


def repo_files(text: str, root: Path) -> list[str]:
    out = []
    for m in _PATH_RE.finditer(text):
        rel = m.group(1)
        if ".." in PurePosixPath(rel).parts:
            continue
        # a real file inside the checkout: no dot dirs (.venv, .git), no symlink out of the repo
        if any(seg.startswith(".") for seg in PurePosixPath(rel).parts):
            continue
        path = root / rel
        if path.is_file() and path.resolve().is_relative_to(root.resolve()):
            out.append(rel)
    return list(dict.fromkeys(out))


def _because(text: str) -> str:
    because = " ".join(text.split())
    if not BECAUSE_MIN_CHARS <= len(because) <= BECAUSE_MAX_CHARS:
        raise ClaimError(f"the because is {BECAUSE_MIN_CHARS}-{BECAUSE_MAX_CHARS} chars, got {len(because)}")
    return because


def check_claim(reading: CapacityReadingV1, atlas: BodyAtlasV1, ref: str, role: str, because: str, *,
                kind: str, root: Path, dig: str) -> CapacityClaimV1:
    part_id = _part(atlas, ref)
    job = parse_role(role)
    because = _because(because)
    ruled = [r for r in reading.roles if r.part_id == part_id and r.role == job]
    if kind == "claim" and any(r.strength == "rule" for r in ruled):
        raise ClaimError(f"{part_id.removeprefix('part:')} already holds {job} by {ruled[0].rule}; nothing to "
                         "claim (`capacity dispute` if you think the rule is wrong)")
    if kind == "dispute" and not ruled:
        raise ClaimError(f"no rule gives {part_id.removeprefix('part:')} {job}; there is nothing to dispute "
                         "(`capacity claim` to propose a role)")
    census = {c.channel for c in reading.census if c.census}
    low = _plain(because)
    cites = sorted(e for e in evidence_ids(atlas, part_id, census) if _cited(e, low))
    cites += repo_files(because, root)
    if not cites:
        sample = sorted(evidence_ids(atlas, part_id, census))[:3]
        raise ClaimError("the because must cite, by full id, a part or channel on one of this part's flows"
                         + (f" (e.g. {', '.join(sample)})" if sample else "") + ", or a repo file that exists")
    return CapacityClaimV1(part_id=part_id, role=job, kind=kind, because=because, cites=cites, at=_now(), dig=dig)


def check_capacity(reading: CapacityReadingV1, atlas: BodyAtlasV1, written: list[CapacityV1], cap_id: str,
                   sentence: str, because: str, *, dig: str) -> CapacityV1:
    cap_id = cap_id.strip().lower()
    if cap_id not in CAP_JOBS:
        raise ClaimError(f"no capacity {cap_id!r}; one of {', '.join(CAP_JOBS)}")
    sentence = " ".join(sentence.split())
    if not SENTENCE_MIN_CHARS <= len(sentence) <= SENTENCE_MAX_CHARS:
        raise ClaimError(f"a capacity is one sentence of {SENTENCE_MIN_CHARS}-{SENTENCE_MAX_CHARS} chars, "
                         f"got {len(sentence)}")
    because = _because(because)
    hs = holders(reading, cap_id)
    jobs = "/".join(CAP_JOBS[cap_id])
    if not hs:
        raise ClaimError(f"no part holds {jobs} by rule or by Juniper's confirmation, so there is nothing to "
                         "cite; the gap line stands (`capacity claim` a part, then Juniper confirms)")
    held = [h.removeprefix("part:") for h in hs]
    s_low = _plain(sentence)
    others = sorted(n for n in claimable(atlas) if n not in held and _cited(n, s_low))
    if others:
        raise ClaimError(f"the sentence names {', '.join(others)}, which hold no {jobs} by rule or confirmation; "
                         f"a capacity may name only {', '.join(held[:6])}")
    text = f"{s_low} {_plain(because)}"
    cites = [h for h in held if _cited(h, text)]
    if not cites:
        raise ClaimError(f"cite at least one holder of {jobs} by full id: {', '.join(held[:6])}")
    s5 = reading.s5_shared_with
    if cap_id == "cap:s5" and s5 is not None and not re.search(rf"\b{re.escape(s5.who.lower())}\b", s_low):
        raise ClaimError(f"cap:s5 says it is shared with {s5.who} (Juniper's decision 3); name {s5.who} in the sentence")
    prior = [w for w in written if w.cap_id == cap_id]
    return CapacityV1(cap_id=cap_id, sentence=sentence, because=because, cites=cites, at=_now(), dig=dig,
                      replaces=prior[-1].at if prior else None)


def claim_rule() -> str:
    """The floor ``check_claim`` and ``check_capacity`` enforce, from the same constants."""
    return (f"Claim: a live part, a job ({', '.join(JOBS)}), and a because ({BECAUSE_MIN_CHARS}-"
            f"{BECAUSE_MAX_CHARS} chars) citing by full id a part or channel on one of its flows, or a repo "
            "file that exists. Not a role until Juniper confirms it in config/body_atlas/vsm_confirmed.yaml. Capacity "
            f"(cap:s1..s5, cap:metabolism): one sentence ({SENTENCE_MIN_CHARS}-{SENTENCE_MAX_CHARS} chars) "
            "naming only parts that hold the job by rule or confirmation.")

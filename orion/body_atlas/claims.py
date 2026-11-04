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
  job by rule, by Juniper's confirmation or by checked evidence (for ``cap:s5``, also her sign-off files), and its
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
    CAP_JOBS, JOBS, CapacityClaimV1, CapacityReadingV1, CapacityV1, EvidenceCheckV1, VsmConfirmedV1, _Index,
    active_claims, claimable, holders, rule_broadcast_keys, rule_census_reader,
)
from orion.body_atlas.model import BodyAtlasV1
from orion.body_atlas.names import BECAUSE_MAX_CHARS, BECAUSE_MIN_CHARS, _cited

CONFIRMED_RELPATH = Path("config/body_atlas/vsm_confirmed.yaml")
SENTENCE_MIN_CHARS = 20
SENTENCE_MAX_CHARS = 160
# a repo path: two or more segments, no "..", ends on a word character (not a trailing period)
_PATH_RE = re.compile(r"(?<![\w/.:-])((?:[\w.-]+/)+[\w.-]*\w)")


_PLACEHOLDER_RE = re.compile(r"<\w[^<>]*>")


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


def flow_dirs(atlas: BodyAtlasV1, part_id: str, census: set[str]) -> dict[str, set[str]]:
    """What a claim on this part may cite, each with the part's own direction on it: the live other
    end of each of its flows that carries more than the census, and each non-census channel on them. "out": the part sends, calls,
    writes (or is the store a part reads); "in": the reverse; "?": a ``uses`` flow (direction
    unknown). A host (``runs_on``) is placement, not evidence of a job."""
    names = {pid: n for n, pid in claimable(atlas).items()}
    out: dict[str, set[str]] = {}
    for f in atlas.flows:
        if part_id not in (f.source, f.target) or f.relation == "runs_on":
            continue
        if f.channels and set(f.channels) <= census:  # a health heartbeat links everyone to everyone
            continue
        other = f.target if f.source == part_id else f.source
        if f.relation == "uses":
            d = "?"
        elif f.relation == "reads":  # source reads target: the data flows target -> source
            d = "in" if f.source == part_id else "out"
        else:
            d = "out" if f.source == part_id else "in"
        if other != part_id and other in names:
            out.setdefault(names[other], set()).add(d)
        for ch in f.channels:
            if ch not in census:
                out.setdefault(ch, set()).add(d)
    return out


def evidence_ids(atlas: BodyAtlasV1, part_id: str, census: set[str]) -> set[str]:
    return set(flow_dirs(atlas, part_id, census))


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
    if _PLACEHOLDER_RE.search(because):  # a printed `capacity claim` line run as is (2026-11-04 review)
        raise ClaimError("the because still holds a <...> placeholder; write your own reason in its place")
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


# a file reference: a repo path whose last segment has an extension (``orion/x/y.py``), so
# "and/or" or "watts/dollars" in prose is not read as a path
_EXT_RE = re.compile(r"\.\w{1,8}$")


def _id_prefixes(atlas: BodyAtlasV1) -> set[str]:
    """The id prefixes the atlas itself uses (``pg``, ``redis``, ``orion`` ...): read from its parts
    and channels, never a hand list."""
    ids = [p.name for p in atlas.parts] + [ch for f in atlas.flows for ch in f.channels]
    return {i.split(":", 1)[0].lower() for i in ids if ":" in i and i.split(":", 1)[0]}


def because_refs(atlas: BodyAtlasV1, part_id: str, because: str) -> tuple[list[str], list[str], list[str]]:
    """What a because points at, by id matching only: (atlas ids it names, id-shaped tokens that
    are in no part or channel of the atlas, repo-path tokens with an extension). A part name counts
    only when it is id-shaped (has ``-`` or ``:``), so a part called ``debug`` is not read out of prose.
    The claimed part's own name is not a reference."""
    low = _plain(because)
    self_name = part_id.removeprefix("part:").lower()
    names = {p.name.lower() for p in atlas.parts if "-" in p.name or ":" in p.name}
    channels = {ch.lower() for f in atlas.flows for ch in f.channels}
    known = (names | channels) - {self_name}
    found = sorted(i for i in known if _cited(i, low))
    unknown: list[str] = []
    prefixes = _id_prefixes(atlas)
    if prefixes:
        alt = "|".join(sorted(map(re.escape, prefixes), key=len, reverse=True))
        for m in re.finditer(rf"(?<![\w:/.-])((?:{alt}):[\w:.*/-]*[\w*])", low):
            tok = m.group(1)
            if tok != self_name and tok not in known and not any(_cited(k, tok) for k in found):
                unknown.append(tok)
    files = [m.group(1) for m in _PATH_RE.finditer(because) if _EXT_RE.search(m.group(1))]
    return found, list(dict.fromkeys(unknown)), list(dict.fromkeys(files))


def check_evidence(atlas: BodyAtlasV1, claim: CapacityClaimV1, root: Path, *, idx: _Index | None = None,
                   audit: set[str] | None = None, broadcast: set[str] | None = None) -> EvidenceCheckV1:
    """Re-check one claim against today's atlas (rung 4, 2026-11-04 decision note): every reference
    its because names (and every id it cited when written) must be on a live, non-census flow of the
    claimed part, or a repo file that exists; at least one must be atlas wiring; and the wiring must
    fit the job's shape (``EVIDENCE_SHAPES``). S5 never passes: it is Juniper's to confirm."""
    idx = idx or _Index(atlas)
    census = idx.census
    dirs = flow_dirs(atlas, claim.part_id, census)
    by_low = {k.lower(): k for k in dirs}
    parts = {p.name.lower(): p for p in atlas.parts}
    found, unknown, files = because_refs(atlas, claim.part_id, claim.because)
    cited_files = [c for c in claim.cites if "/" in c and _EXT_RE.search(c)]
    refs = list(dict.fromkeys(found + unknown + [c.lower() for c in claim.cites if c not in cited_files]))
    channels = {ch.lower() for f in atlas.flows for ch in f.channels}
    held: list[str] = []
    failed: list[str] = []
    wiring: dict[str, set[str]] = {}
    for ref in refs:
        if ref == claim.part_id.removeprefix("part:").lower():
            continue
        if ref in by_low:
            wiring[by_low[ref]] = dirs[by_low[ref]]
            held.append(f"{by_low[ref]} ({'/'.join(sorted(dirs[by_low[ref]]))})")
        elif ref in census:
            failed.append(f"{ref}: the census, which every part sends")
        elif ref in parts and parts[ref].status != "live":
            rows = parts[ref].facts.get("rows_est")
            failed.append(f"{ref}: {parts[ref].status}, not live" + (f" ({rows} rows)" if rows is not None else ""))
        elif ref not in parts and ref not in channels:
            failed.append(f"{ref}: in no part or channel of my atlas")
        else:
            failed.append(f"{ref}: not on a live flow of {claim.part_id.removeprefix('part:')}")
    for rel in dict.fromkeys(files + cited_files):
        if repo_files(rel, root) == [rel]:
            held.append(f"{rel} (file)")
        else:
            failed.append(f"{rel}: no such file in the repo")
    shape_miss = None
    if claim.role == "S5":
        shape_miss = "S5 is shared with Juniper; code cannot check it into a role"
    elif not wiring:
        shape_miss = "it cites no wiring of the part (files alone are not checked as a role)"
    else:
        ins = any("in" in d for d in wiring.values())
        outs = any("out" in d for d in wiring.values())
        if claim.role == "S2" and not (ins and outs):
            shape_miss = "S2 needs a cited flow in and one out"
        elif claim.role in ("S3", "S4") and not outs:
            shape_miss = f"{claim.role} needs a cited flow out to another part"
        elif claim.role == "S3*":
            audit = audit if audit is not None else {r.part_id for r in rule_census_reader(idx)}
            if claim.part_id not in audit:
                shape_miss = "S3* needs R-census-reader to find it listening to the whole body"
            elif not outs:
                shape_miss = "S3* needs a cited flow out (it reports back)"
        elif claim.role == "algedonic":
            broadcast = broadcast if broadcast is not None else {b.part_id for b in rule_broadcast_keys(atlas, idx)}
            hit = [w for w in wiring if f"part:{w}" in broadcast and "out" in wiring[w]]
            if not hit:
                shape_miss = "algedonic needs a cited Redis key it writes that R-broadcast-key finds"
    if shape_miss:
        failed.append(f"shape: {shape_miss}")
    return EvidenceCheckV1(part_id=claim.part_id, role=claim.role, at=claim.at, holds=not failed,
                           held=held, failed=failed)


def check_claims(atlas: BodyAtlasV1, claims: list[CapacityClaimV1], root: Path) -> list[EvidenceCheckV1]:
    """Each standing claim (latest per part and job, not withdrawn), re-checked against today's
    atlas. Deterministic; stores nothing: run on every read, so a claim whose wiring is gone drops back."""
    standing = active_claims(claims)
    if not standing:
        return []
    idx = _Index(atlas)
    audit = {r.part_id for r in rule_census_reader(idx)}
    broadcast = {b.part_id for b in rule_broadcast_keys(atlas, idx)}
    return [check_evidence(atlas, c, root, idx=idx, audit=audit, broadcast=broadcast) for c in standing.values()]


def withdraw_claim(atlas: BodyAtlasV1, claims: list[CapacityClaimV1], ref: str, role: str, *,
                   dig: str) -> CapacityClaimV1:
    """Take back a standing claim (any status; a part that is no longer live included)."""
    job = parse_role(role)
    name = ref.strip().removeprefix("part:")
    part_id = {p.name: p.part_id for p in atlas.parts}.get(name, f"part:{name}")
    if (part_id, job) not in active_claims(claims):
        raise ClaimError(f"I have no standing claim of {job} for {name} to withdraw")
    return CapacityClaimV1(part_id=part_id, role=job, kind="withdraw", because="withdrawn", cites=[], at=_now(), dig=dig)


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
        raise ClaimError(f"no part holds {jobs} by rule, by Juniper's confirmation or by checked evidence, so "
                         "there is nothing to cite; the gap line stands (`capacity claim` a part with wiring that holds)")
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
            "file that exists. On every read, code re-checks the cited wiring against the atlas: it counts "
            "as a role (EVIDENCE-CHECKED, not Juniper's) while it holds and fits the job; S5 waits for Juniper, "
            "and her rejection in config/body_atlas/vsm_confirmed.yaml wins. Capacity "
            f"(cap:s1..s5, cap:metabolism): one sentence ({SENTENCE_MIN_CHARS}-{SENTENCE_MAX_CHARS} chars) "
            "naming only parts that hold the job by rule, confirmation or checked evidence.")

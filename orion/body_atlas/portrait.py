"""Rung 5: the self-portrait (design ``docs/superpowers/specs/2026-11-06-self-portrait-design.md``).

One page rebuilt from live sources on every read: each machine (what runs there, watts, organs
placed there, jobs held there), each organ by Orion's name, each capacity sentence, and the gap
lines as "I cannot yet". Beside it, one append-only ledger of Orion's own words,
``$ORION_DIG_HOME/self/portrait.jsonl``:

* a **machine line** (subject ``host:<name>``): what that machine does and what it affects. Every id
  it names must be in the machine's **reach**: a live part running there, or a live part or
  non-census channel a running part sends to, calls or writes. Organ ids must be organs placed
  there; Orion's organ names count as a cite when that organ is placed there (a name placed
  elsewhere is not an error: no word lists). At least one cite runs there or is an organ there.
* the **whole** line (subject ``whole``): the "I am ..." paragraph, written from the capacities
  alone. Every ``cap:`` id it names must read ``written`` now, at least one is named, and every part
  id it names must hold a job.

Code runs the same check on write and on every read: a write that would not hold writes nothing;
a written line that stops holding prints STALE with the id that failed. Id matching proves a thing
is wired to the machine, not that the sentence is true (the rung-4 limit).

Partitioned from recall like the other self ledgers (``tests/test_organ_names_partition.py``).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.body_atlas.capacities import CAP_JOBS, JOBS, CapacityReadingV1, CapacityViewV1, claimable, holders
from orion.body_atlas.claims import (
    _PLACEHOLDER_RE, ClaimError, _because, _now, _plain, because_refs, flow_dirs, repo_files,
)
from orion.body_atlas.model import BodyAtlasV1

WHOLE = "whole"
MACHINE_MIN_CHARS = 20
MACHINE_MAX_CHARS = 200
WHOLE_MIN_CHARS = 40
WHOLE_MAX_CHARS = 600
SHOWN_CHARS = 85  # a quoted line in the printout; the full text is in the file
_ORGAN_ID_RE = re.compile(r"(?<![0-9a-z])(?:organ:)?([0-9a-f]{10})(?![0-9a-z])")
_CAP_ID_RE = re.compile(r"(?<![\w:])(cap:[a-z0-9_]+)(?![\w:])")


class PortraitError(ClaimError):
    pass


class PortraitLineV1(BaseModel):
    """One line of Orion's portrait ledger. The latest line per ``subject`` reads; old ones stay."""

    model_config = ConfigDict(extra="forbid")

    subject: str  # "host:<name>" or "whole"
    sentence: str
    because: str
    cites: list[str]  # what held when written: part / channel names, organ ids or names, cap ids, files
    at: str
    dig: str
    replaces: str | None = None  # ``at`` of the line it supersedes


class PortraitCheckV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    holds: bool
    held: list[str] = Field(default_factory=list)  # "<ref> (<why it counts>)"
    failed: list[str] = Field(default_factory=list)  # "<ref>: <why not>"


class ReachV1(BaseModel):
    """What one machine runs and affects on this read (the ids a machine line may name)."""

    model_config = ConfigDict(extra="forbid")

    running: list[str] = Field(default_factory=list)  # live part names with a runs_on flow to it
    feeds: list[str] = Field(default_factory=list)  # live part / non-census channel a running part sends to
    organs: dict[str, int] = Field(default_factory=dict)  # organ id -> members placed there


class MachineViewV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: str
    role: str
    hardware: str
    last_seen_at: str | None = None
    reach: ReachV1
    organs: list[str] = Field(default_factory=list)  # "<name or id> <n>", most members first
    jobs: dict[str, list[str]] = Field(default_factory=dict)  # job -> holders running there
    watts: dict[str, float] = Field(default_factory=dict)  # now, rest, usd_month (body economy)
    line: Literal["holds", "stale", "not written", "cannot be checked"]
    sentence: str | None = None
    at: str | None = None
    failed: list[str] = Field(default_factory=list)


class OrganViewV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organ_id: str
    name: str | None = None
    size: int
    placement: dict[str, int] = Field(default_factory=dict)
    gives_to: list[str] = Field(default_factory=list)  # organs it sends most to, strongest first
    jobs: dict[str, list[str]] = Field(default_factory=dict)


class WholeViewV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line: Literal["holds", "stale", "not written"]
    sentence: str | None = None
    at: str | None = None
    failed: list[str] = Field(default_factory=list)


class PortraitV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["portrait.v1"] = "portrait.v1"
    atlas_as_of: str
    whole: WholeViewV1
    cannot_yet: list[str]  # code's gap lines, "<job>: <text>"
    machines: list[MachineViewV1]
    organs: list[OrganViewV1]
    capacities: list[CapacityViewV1]
    watts_read: str  # "read" or "UNREAD: <why>"
    notes: list[str] = Field(default_factory=list)


# --- reach --------------------------------------------------------------------------------------

def host_part(atlas: BodyAtlasV1, name: str):
    want = name.strip().lower().removeprefix("part:").removeprefix("host:")
    return next((p for p in atlas.parts if p.kind == "host" and p.name.removeprefix("host:").lower() == want), None)


def machine_reach(atlas: BodyAtlasV1, host_name: str, census: set[str]) -> ReachV1:
    hid = f"part:host:{host_name}"
    live = {pid: n for n, pid in claimable(atlas).items()}
    busy = {p.part_id: p.publish_count for p in atlas.parts}
    ids = {f.source for f in atlas.flows if f.relation == "runs_on" and f.target == hid and f.source in live}
    ids_sorted = sorted(ids, key=lambda pid: (-busy.get(pid, 0), live[pid]))  # busiest first (the hint)
    running = [live[pid] for pid in ids_sorted]
    feeds: set[str] = set()
    for pid in ids_sorted:
        for ref, dirs in flow_dirs(atlas, pid, census).items():
            if dirs & {"out", "?"}:  # it sends, calls or writes there ("?": a uses flow, direction unknown)
                feeds.add(ref)
    organs = {o.organ_id: o.placement[host_name] for o in atlas.organs if o.placement.get(host_name)}
    return ReachV1(running=running, feeds=sorted(feeds - set(running)), organs=organs)


# --- the check (write and read) -------------------------------------------------------------------

def _organ_label(atlas: BodyAtlasV1, organ_id: str) -> str:
    o = next((o for o in atlas.organs if o.organ_id == organ_id), None)
    return o.name if o is not None and o.name else organ_id.removeprefix("organ:")


NAME_FAMILY_MIN = 3  # a first word shared by this many part names is a naming family (``orion-``, ``vision-``)


def _unknown_names(atlas: BodyAtlasV1, text: str) -> list[str]:
    """Hyphenated words that look like a part name of mine but name no part: their first word is shared
    by at least NAME_FAMILY_MIN part names (read from the atlas, never a hand list). ``because_refs``
    sees only ids with a ``prefix:``, so "orion-llm-worker-a" (no such part) would pass unseen.
    Smoke 2026-11-06: a circe line naming it was refused only for its missing anchor."""
    names = {p.name.lower() for p in atlas.parts}
    first: dict[str, int] = {}
    for n in names:
        if "-" in n and ":" not in n:
            first[n.split("-", 1)[0]] = first.get(n.split("-", 1)[0], 0) + 1
    families = {f for f, k in first.items() if k >= NAME_FAMILY_MIN}
    out = []
    for m in re.finditer(r"(?<![\w:/.-])([a-z][a-z0-9]*(?:-[a-z0-9]+)+)(?![\w:/-])", _plain(text)):
        tok = m.group(1)
        if tok.split("-", 1)[0] in families and tok not in names:
            out.append(tok)
    return list(dict.fromkeys(out))


def _named(name: str, low: str) -> bool:
    """An organ name in prose: whole words, so a colon or comma after it still counts ("cortex: ...")."""
    return re.search(rf"(?<![\w-]){re.escape(name.lower())}(?![\w-])", low) is not None


def check_machine(atlas: BodyAtlasV1, host_name: str, text: str, census: set[str], root: Path) -> PortraitCheckV1:
    subject = f"host:{host_name}"
    reach = machine_reach(atlas, host_name, census)
    held: list[str] = []
    failed: list[str] = []
    anchored = False
    if not reach.running:
        host = host_part(atlas, host_name)
        status = host.status if host is not None else "not in my atlas"
        failed.append(f"{host_name}: nothing live of mine runs there now ({status}), so no line about it can be checked")
        return PortraitCheckV1(subject=subject, holds=False, failed=failed)
    running = {n.lower(): n for n in reach.running}
    feeds = {n.lower(): n for n in reach.feeds}
    parts = {p.name.lower(): p for p in atlas.parts}
    census_low = {c.lower() for c in census}
    found, unknown, files = because_refs(atlas, f"part:{subject}", text)
    for ref in found:
        if ref in running:
            held.append(f"{running[ref]} (runs on {host_name})")
            anchored = True
        elif ref in feeds:
            held.append(f"{feeds[ref]} (fed from {host_name})")
        elif ref in census_low:
            failed.append(f"{ref}: the census, which every part sends")
        elif ref in parts and parts[ref].kind == "host":
            failed.append(f"{ref}: another machine; name what runs on {host_name} or what it feeds")
        elif ref in parts and parts[ref].status != "live":
            failed.append(f"{ref}: {parts[ref].status}, not live")
        else:
            failed.append(f"{ref}: neither runs on {host_name} nor is fed by a part that does")
    failed += [f"{tok}: in no part or channel of my atlas" for tok in [*unknown, *_unknown_names(atlas, text)]]
    for rel in files:
        if repo_files(rel, root) == [rel]:
            held.append(f"{rel} (file)")
        else:
            failed.append(f"{rel}: no such file in the repo")
    low = _plain(text)
    organ_ids = {o.organ_id.removeprefix("organ:") for o in atlas.organs}
    for m in _ORGAN_ID_RE.finditer(low):
        oid = f"organ:{m.group(1)}"
        if oid in reach.organs:
            held.append(f"{oid} (organ on {host_name})")
            anchored = True
        elif m.group(1) in organ_ids:
            failed.append(f"{oid}: an organ with no member running on {host_name}")
        else:
            failed.append(f"{oid}: no such organ in today's body (organ ids change; names carry)")
    for oid in reach.organs:  # a name counts only as a cite, never as an error (no word lists)
        name = _organ_label(atlas, oid)
        if name != oid.removeprefix("organ:") and _named(name, low):
            held.append(f'"{name}" (organ on {host_name})')
            anchored = True
    if not anchored and not failed:
        failed.append(f"anchor: cite at least one part that runs on {host_name} (e.g. {', '.join(reach.running[:3])}) "
                      "or an organ placed there")
    return PortraitCheckV1(subject=subject, holds=not failed, held=list(dict.fromkeys(held)), failed=failed)


def check_whole(atlas: BodyAtlasV1, reading: CapacityReadingV1, text: str, root: Path) -> PortraitCheckV1:
    held: list[str] = []
    failed: list[str] = []
    status = {c.cap_id: c.status for c in reading.capacities}
    low = _plain(text)
    caps = list(dict.fromkeys(_CAP_ID_RE.findall(low)))
    for cap in caps:
        if cap not in CAP_JOBS:
            failed.append(f"{cap}: no such capacity (one of {', '.join(CAP_JOBS)})")
        elif status.get(cap) == "written":
            held.append(f"{cap} (written)")
        else:
            failed.append(f"{cap}: {status.get(cap, 'not read')} now, not a written capacity")
    if not caps:
        failed.append("cite: name at least one written capacity by id (cap:s1 ...); rung 5 is written from them")
    job_holders = {h.removeprefix("part:").lower() for cap in CAP_JOBS for h in holders(reading, cap)}
    found, unknown, files = because_refs(atlas, "part:", text)
    for ref in found:
        if ref in job_holders:
            held.append(f"{ref} (holds a job)")
        else:
            failed.append(f"{ref}: holds no job by rule, confirmation or checked claim")
    failed += [f"{tok}: in no part or channel of my atlas" for tok in [*unknown, *_unknown_names(atlas, text)]]
    for rel in files:
        if rel.lower() in job_holders and repo_files(rel, root) == [rel]:
            held.append(f"{rel} (Juniper's sign-off file)")
        elif repo_files(rel, root) == [rel]:
            failed.append(f"{rel}: a file, but no job cites it")
        else:
            failed.append(f"{rel}: no such file in the repo")
    return PortraitCheckV1(subject=WHOLE, holds=not failed, held=list(dict.fromkeys(held)), failed=failed)


def check_line(atlas: BodyAtlasV1, reading: CapacityReadingV1, line: PortraitLineV1, root: Path) -> PortraitCheckV1:
    text = f"{line.sentence} {line.because}"
    if line.subject == WHOLE:
        return check_whole(atlas, reading, text, root)
    census = {c.channel for c in reading.census if c.census}
    return check_machine(atlas, line.subject.removeprefix("host:"), text, census, root)


def _ref(held: str) -> str:
    return held.rsplit(" (", 1)[0].strip('"')


def check_write(atlas: BodyAtlasV1, reading: CapacityReadingV1, lines: list[PortraitLineV1], subject: str,
                sentence: str, because: str, *, root: Path, dig: str) -> PortraitLineV1:
    """The write runs the read's own check: a line that would not hold now writes nothing."""
    raw = subject.strip()
    sentence = " ".join(sentence.split())
    if raw.lower() == WHOLE:
        subj, lo, hi = WHOLE, WHOLE_MIN_CHARS, WHOLE_MAX_CHARS
    else:
        host = host_part(atlas, raw)
        if host is None:
            names = ", ".join(sorted(p.name.removeprefix("host:") for p in atlas.parts if p.kind == "host"))
            raise PortraitError(f"no machine {raw!r} in my atlas; one of {names}, or `whole` for the I-am paragraph")
        subj, lo, hi = host.name, MACHINE_MIN_CHARS, MACHINE_MAX_CHARS
    if _PLACEHOLDER_RE.search(sentence):
        raise PortraitError("the sentence still holds a <...> placeholder; write your own words in its place")
    if not lo <= len(sentence) <= hi:
        raise PortraitError(f"a {'paragraph' if subj == WHOLE else 'machine line'} is {lo}-{hi} chars, got {len(sentence)}")
    try:
        because = _because(because)
    except ClaimError as exc:
        raise PortraitError(str(exc)) from exc
    rec = PortraitLineV1(subject=subj, sentence=sentence, because=because, cites=[], at=_now(), dig=dig)
    check = check_line(atlas, reading, rec, root)
    if not check.holds:
        raise PortraitError("not written: it would not hold on today's atlas: " + "; ".join(check.failed[:4])
                            + (f" (+{len(check.failed) - 4} more)" if len(check.failed) > 4 else ""))
    prior = latest(lines).get(subj)
    return rec.model_copy(update={"cites": [_ref(h) for h in check.held],
                                  "replaces": prior.at if prior else None})


def latest(lines: list[PortraitLineV1]) -> dict[str, PortraitLineV1]:
    """subject -> the line that reads: the latest by ``at``, then file order."""
    out: dict[str, PortraitLineV1] = {}
    for _, ln in sorted(enumerate(lines), key=lambda il: (il[1].at, il[0])):
        out[ln.subject] = ln
    return out


# --- the read -------------------------------------------------------------------------------------

def read_portrait(atlas: BodyAtlasV1, reading: CapacityReadingV1, lines: list[PortraitLineV1], *, root: Path,
                  watts: dict[str, dict] | str) -> PortraitV1:
    """``reading``: ``capacities.attach_ledgers`` output on this same atlas. ``watts``: host ->
    body-economy hosts row, or the reason the report could not be read."""
    census = {c.channel for c in reading.census if c.census}
    mine = latest(lines)
    job_of: dict[str, list[str]] = {}
    for r in reading.roles + reading.confirmed + reading.claimed:
        if r.strength != "candidate":
            job_of.setdefault(r.part_id.removeprefix("part:"), []).append(r.role)
    machines: list[MachineViewV1] = []
    hosts = sorted((p for p in atlas.parts if p.kind == "host"), key=lambda p: (p.status != "live", p.name))
    for h in hosts:
        name = h.name.removeprefix("host:")
        reach = machine_reach(atlas, name, census)
        jobs: dict[str, list[str]] = {}
        for part in reach.running:
            for job in job_of.get(part, []):
                jobs.setdefault(job, []).append(part)
        jobs = {j: sorted(set(jobs[j])) for j in JOBS if j in jobs}  # S1 .. metabolism order
        organs = [f"{_organ_label(atlas, oid)} {n}" for oid, n in sorted(reach.organs.items(), key=lambda kv: -kv[1])]
        row = watts.get(name) if isinstance(watts, dict) else None
        w = {k: float(row[src]) for k, src in (("now", "chassis_w_now"), ("rest", "chassis_w_rest"),
                                               ("usd_month", "usd_per_month_at_this_rate"))
             if row and row.get(src) is not None}
        ln = mine.get(h.name)
        view = dict(name=name, status=h.status, role=h.facts.get("role", "?"), hardware=h.facts.get("hardware", "not recorded"),
                    last_seen_at=h.last_seen_at, reach=reach, organs=organs, jobs=jobs, watts=w)
        if ln is not None:
            chk = check_line(atlas, reading, ln, root)
            machines.append(MachineViewV1(**view, line="holds" if chk.holds else "stale", sentence=ln.sentence,
                                          at=ln.at, failed=chk.failed))
        else:
            machines.append(MachineViewV1(**view, line="not written" if reach.running else "cannot be checked"))
    whole_ln = mine.get(WHOLE)
    if whole_ln is None:
        whole = WholeViewV1(line="not written")
    else:
        chk = check_line(atlas, reading, whole_ln, root)
        whole = WholeViewV1(line="holds" if chk.holds else "stale", sentence=whole_ln.sentence, at=whole_ln.at,
                            failed=chk.failed)
    mixes = {m.organ_id: m.roles for m in reading.organ_mixes}
    organs = []
    for o in sorted(atlas.organs, key=lambda o: -len(o.members)):
        if len(o.members) < 2:
            continue
        out = sorted((f for f in atlas.organ_flows if f.source == o.organ_id and f.target != o.organ_id),
                     key=lambda f: -f.weight)
        organs.append(OrganViewV1(organ_id=o.organ_id, name=o.name, size=len(o.members), placement=dict(o.placement),
                                  gives_to=[_organ_label(atlas, f.target) for f in out[:3]],
                                  jobs={k: [p.removeprefix("part:") for p in mixes[o.organ_id][k]]
                                        for k in JOBS if k in mixes.get(o.organ_id, {})}))
    notes = list(reading.notes)
    unknown = [s for s in mine if s != WHOLE and host_part(atlas, s) is None]
    if unknown:
        notes.append(f"portrait lines for machines no longer in my atlas: {', '.join(unknown)}")
    return PortraitV1(atlas_as_of=atlas.as_of, whole=whole,
                      cannot_yet=[f"{g.job}: {g.text}" for g in reading.gaps], machines=machines, organs=organs,
                      capacities=list(reading.capacities),
                      watts_read="read" if isinstance(watts, dict) else f"UNREAD: {watts}", notes=notes)


# --- the printout ---------------------------------------------------------------------------------

WRITE_PREFIX = "python -P -m orion.dig portrait write"


def _q(text: str, n: int = SHOWN_CHARS) -> str:
    return f'"{text}"' if len(text) <= n else f'"{text[: n - 1].rstrip()}…"'


def _day(at: str | None) -> str:
    return (at or "")[5:10]


def _mine(line: str, at: str | None, sentence: str | None, failed: list[str]) -> str:
    if line == "holds":
        return f"lived_record {_day(at)}, holds now: {_q(sentence or '', MACHINE_MAX_CHARS)}"
    return f"lived_record {_day(at)}, STALE ({'; '.join(failed)[:70]}): {_q(sentence or '', 70)}"


def render(p: PortraitV1) -> tuple[list[str], list[str]]:
    """(body lines, tail lines). The caller caps the body and keeps the tail whole."""
    lines = [f"portrait · atlas {p.atlas_as_of[:16]}Z · now = computed this read · "
             "lived_record = my words, checked now"]
    w = p.whole
    if w.line == "not written":
        lines.append("I am: not written yet (one paragraph, from the capacities below)")
    elif w.line == "holds":
        lines.append(f"I am (lived_record {_day(w.at)}, holds now): {_q(w.sentence or '', WHOLE_MAX_CHARS)}")
    else:
        lines.append(f"I am (lived_record {_day(w.at)}, STALE: {'; '.join(w.failed)[:90]}): {_q(w.sentence or '', 200)}")
    if p.cannot_yet:
        lines.append("I cannot yet (now; code's gap lines, not mine to remove; `capacities gaps`):")
        lines += [f"  {g[:SHOWN_CHARS]}" for g in p.cannot_yet[:4]]
        if len(p.cannot_yet) > 4:
            lines.append(f"  +{len(p.cannot_yet) - 4} more in the file")
    watts = "" if p.watts_read == "read" else f" · watts {p.watts_read[:80]}"
    lines.append(f"machines (now: status, role, watts, parts of mine there, organs, jobs){watts}:")
    idle = []
    for m in p.machines:
        if m.line == "cannot be checked":
            seen = f", last seen {m.last_seen_at[:10]}" if m.last_seen_at and m.status != "live" else ""
            idle.append(f"{m.name} ({m.role}, {m.status}{seen})")
            continue
        w = (f" · {m.watts['now']:.0f} W now, rest {m.watts['rest']:.0f}" if {"now", "rest"} <= set(m.watts) else "")
        organs = ", ".join(m.organs[:3]) + (f" +{len(m.organs) - 3}" if len(m.organs) > 3 else "")
        jobs = " ".join(m.jobs)
        lines.append(f"  {m.name} {m.status} {m.role}{w} · runs {len(m.reach.running)} of mine"
                     + (f" · organs {organs}" if organs else "") + (f" · jobs {jobs}" if jobs else ""))
        if m.line in ("holds", "stale"):
            lines.append(f"    {_mine(m.line, m.at, m.sentence, m.failed)}")
    if idle:
        lines.append("  nothing of mine runs on (facts only): " + "; ".join(idle))
    lines.append("capacities (lived_record; status now):")
    lines += [f"  {c.cap_id:<14} {c.text[:SHOWN_CHARS]}" for c in p.capacities]
    lines.append("organs (now: parts, where, gives most to, jobs; names mine):")
    for o in p.organs:
        label = f'"{o.name}"' if o.name else f"{o.organ_id} (unnamed)"
        on = " ".join(f"{h} {n}" for h, n in sorted(o.placement.items(), key=lambda kv: -kv[1]))
        lines.append(f"  {label} [{o.size}{'; ' + on if on else ''}]"
                     + (f" -> {', '.join(o.gives_to[:2])}" if o.gives_to else "")
                     + (f" · jobs {' '.join(o.jobs)}" if o.jobs else ""))
    lines += [f"note: {n[:90]}" for n in p.notes[:2]]
    tail: list[str] = []
    todo = [m for m in p.machines if m.line == "not written"]
    stale = [m for m in p.machines if m.line == "stale"] + ([p.whole] if p.whole.line == "stale" else [])
    if todo or p.whole.line == "not written":
        tail.append("Write now, without asking first, each line you can back; code checks each id against what "
                    "runs on that machine and what it feeds:")
        for m in todo[:4]:
            eg = ", ".join(m.reach.running[:2])
            tail.append(f'  {WRITE_PREFIX} {m.name} "<what {m.name} does and affects>" because "<ids>"  # runs: {eg}')
        if p.whole.line == "not written":
            tail.append(f'  {WRITE_PREFIX} whole "<I am ..., from the capacities>" because "<the cap: ids it rests on>"')
    if stale:
        tail.append(f"Rewrite {len(stale)} STALE line(s), naming only what holds now (same command, same subject).")
    return lines, tail

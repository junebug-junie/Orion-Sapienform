"""Rung 4, patch 1: which jobs of a viable system the body atlas shows a part doing, by rule.

Design: ``docs/superpowers/specs/2026-10-30-self-now-capacities-vsm-design.md`` (Stafford Beer's
Viable System Model: S1 operations, S2 coordination, S3 control, S3* audit, S4 intelligence,
S5 identity, the algedonic line). Three rules decide a role; the rest of the jobs print "no rule"
plus a gap line written here, never by Orion. Read-only: nothing here writes a file, a ledger or
a store. Claims, confirmations and capacity sentences are patch 2.

Every count uses **live parts only**: a ``service`` or ``endpoint`` part with status ``live``,
operator ids (``operator:...``) excluded. Transient smokes and probes are never live. A channel's
producers are the live sources of ``signals`` flows that carry it; its readers are the live
targets. A **reply** from R to P is a ``signals`` flow R -> P carrying at least one channel that
is not the census (a health heartbeat back is not an answer; a causal-only flow names no channel).

Rules (all knobs are the constants below):

* **R-boundary-in** (S1 outside contact): a live camera's ``sees_through`` flows, and every live
  endpoint whose id prefix (``http`` in ``http:anthropic``) is not another part's name, i.e. not
  a service's own sub-id on the bus (``orion-mesh-guardian:gpu-watch``).
* **R-matchmaker** (S2): a part is the only live reader of a non-census channel with
  >= FANIN_MIN_PRODUCERS live producers from >= FANIN_MIN_ORGANS organs and replies to at least
  one of them, AND is the only live reader of a second channel whose >= SECOND_MIN_PRODUCERS live
  producers share none with the first. Sitting between two disjoint crowds (requesters and the
  workers handed out) is what sets a coordinator apart from a busy server.
* **R-census-reader** (S3* candidate): reads the census and replies to < CENSUS_REPLY_MAX of its
  producers. It shows the part listens to all, not that it samples work.

Near misses (printed so "many call it" is not read as coordination): **R-server**, the
matchmaker's first half alone; **R-collector**, a fan-in reader (sole reader or not) with 0
replies. **R-source**: live, publishes non-census, receives none: an input candidate (the world
or a clock; the rule cannot tell).

**Census**: channels sorted by the share of live parts that publish them; the census is every
channel above the largest drop among the top CENSUS_TOP_N shares. No fixed percentage, no list.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.body_atlas.model import BodyAtlasV1

RULES_VERSION = "v1"
CENSUS_TOP_N = 10
FANIN_MIN_PRODUCERS = 5
FANIN_MIN_ORGANS = 2
SECOND_MIN_PRODUCERS = 3
CENSUS_REPLY_MAX = 0.10
OPERATOR_PREFIX = "operator:"

Job = Literal["S1", "S2", "S3", "S3*", "S4", "S5", "algedonic"]
JOBS: tuple[Job, ...] = ("S1", "S2", "S3", "S3*", "S4", "S5", "algedonic")
JOB_TITLES = {"S1": "operations", "S2": "coordination", "S3": "control", "S3*": "audit",
              "S4": "intelligence", "S5": "identity", "algedonic": "(pain line)"}


class RoleEvidenceV1(BaseModel):
    """One part holding one job by one rule. ``strength`` is ``rule`` (the rule decides) or
    ``candidate`` (the rule fits but cannot show the whole job)."""

    model_config = ConfigDict(extra="forbid")

    part_id: str
    role: Job
    rule: str
    strength: Literal["rule", "candidate"]
    evidence: list[str] = Field(default_factory=list)  # channels / flows, by full id
    counts: dict[str, int] = Field(default_factory=dict)


class NearMissV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    part_id: str
    rule: Literal["R-server", "R-collector"]
    channel: str
    producers: int
    replies: int
    readers: int


class CensusShareV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: str
    publishers: int
    share: float
    census: bool


class JobLineV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job: Job
    status: Literal["BY RULE", "CANDIDATE", "NO PART FOUND", "NO RULE"]
    text: str


class GapV1(BaseModel):
    """A job the atlas does not show any part doing. Written by code; Orion cannot remove one."""

    model_config = ConfigDict(extra="forbid")

    gap_id: str
    job: Job
    text: str


class OrganMixV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organ_id: str
    name: str | None = None
    size: int
    roles: dict[str, list[str]] = Field(default_factory=dict)  # job -> part ids


class CapacityReadingV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["capacity_reading.v1"] = "capacity_reading.v1"
    atlas_as_of: str
    rules_version: str = RULES_VERSION
    live_parts: int
    census: list[CensusShareV1]
    roles: list[RoleEvidenceV1]
    near_misses: list[NearMissV1]
    sources: list[str]  # R-source: input candidates
    silent_services: list[str]  # live services with no bus traffic but the census
    declared: list[str]  # declared facts printed as facts, not rule results (e.g. the runner)
    jobs: list[JobLineV1]
    gaps: list[GapV1]
    organ_mixes: list[OrganMixV1]


# --- the index the rules read -----------------------------------------------------------------

def _bare(part_id: str) -> str:
    return part_id.removeprefix("part:")


def live_part_ids(atlas: BodyAtlasV1) -> set[str]:
    return {p.part_id for p in atlas.parts if p.status == "live" and p.kind in ("service", "endpoint")
            and not p.name.startswith(OPERATOR_PREFIX)}


class _Index:
    def __init__(self, atlas: BodyAtlasV1) -> None:
        self.live = live_part_ids(atlas)
        self.organ = {m: o.organ_id for o in atlas.organs for m in o.members}
        self.producers: dict[str, set[str]] = defaultdict(set)
        self.readers: dict[str, set[str]] = defaultdict(set)
        # (source, target) -> channels on that signals flow (live ends only)
        self.pair_channels: dict[tuple[str, str], set[str]] = defaultdict(set)
        for f in atlas.flows:
            if f.relation != "signals" or f.source not in self.live or f.target not in self.live:
                continue
            for ch in f.channels:
                self.producers[ch].add(f.source)
                self.readers[ch].add(f.target)
                self.pair_channels[(f.source, f.target)].add(ch)
        self.census_rows, self.census = _census(self.producers, len(self.live))

    def replies(self, reader: str, producers: set[str]) -> set[str]:
        return {p for p in producers if self.pair_channels.get((reader, p), set()) - self.census}

    def fan_in(self, ch: str, reader: str) -> set[str] | None:
        """The channel's other live producers when they clear the fan-in floor, else None."""
        ps = self.producers[ch] - {reader}
        organs = {self.organ.get(p) for p in ps} - {None}
        if len(ps) >= FANIN_MIN_PRODUCERS and len(organs) >= FANIN_MIN_ORGANS:
            return ps
        return None


def _census(producers: dict[str, set[str]], n_live: int) -> tuple[list[CensusShareV1], set[str]]:
    if not n_live:
        return [], set()
    top = sorted(((len(ps) / n_live, ch) for ch, ps in producers.items()), key=lambda r: (-r[0], r[1]))
    top = top[:CENSUS_TOP_N]
    census: set[str] = set()
    if len(top) >= 2:
        drops = [top[i][0] - top[i + 1][0] for i in range(len(top) - 1)]
        cut = max(range(len(drops)), key=lambda i: (drops[i], -i))  # first of equal drops
        census = {ch for _, ch in top[: cut + 1]}
    rows = [CensusShareV1(channel=ch, publishers=len(producers[ch]), share=round(s, 3), census=ch in census)
            for s, ch in top]
    return rows, census


# --- rules ------------------------------------------------------------------------------------

def rule_boundary_in(atlas: BodyAtlasV1, idx: _Index) -> list[RoleEvidenceV1]:
    out = []
    # an endpoint whose prefix is a service's name is that service's own sub-id on the bus
    services = {p.name for p in atlas.parts if p.kind == "service"}
    sees: dict[str, list[str]] = defaultdict(list)
    for f in atlas.flows:
        if f.relation == "sees_through" and f.target in idx.live:
            sees[f.source].append(_bare(f.target))
    for p in atlas.parts:
        if p.kind == "sensor" and p.status == "live" and sees.get(p.part_id):
            out.append(RoleEvidenceV1(part_id=p.part_id, role="S1", rule="R-boundary-in", strength="rule",
                                      evidence=[f"sees_through -> {t}" for t in sorted(sees[p.part_id])],
                                      counts={"readers": len(sees[p.part_id])}))
        elif (p.part_id in idx.live and p.kind == "endpoint" and ":" in p.name
              and p.name.split(":", 1)[0] not in services):
            to = sorted({_bare(t) for (s, t) in idx.pair_channels if s == p.part_id})
            out.append(RoleEvidenceV1(part_id=p.part_id, role="S1", rule="R-boundary-in", strength="rule",
                                      evidence=[f"endpoint -> {t}" for t in to], counts={"targets": len(to)}))
    return out


def _sole_reader_channels(idx: _Index) -> dict[str, list[str]]:
    by_reader: dict[str, list[str]] = defaultdict(list)
    for ch, rs in idx.readers.items():
        if ch not in idx.census and len(rs) == 1:
            by_reader[next(iter(rs))].append(ch)
    return by_reader


def rule_matchmaker(idx: _Index, *, second_set: bool = True) -> tuple[list[RoleEvidenceV1], list[NearMissV1]]:
    """S2 hits, plus R-server near misses (first half only). ``second_set=False`` drops the
    disjoint-second-crowd clause: only the eval's mutation check uses it."""
    hits, servers = [], []
    for reader, chans in sorted(_sole_reader_channels(idx).items()):
        best = None
        for ch in sorted(chans):
            ps = idx.fan_in(ch, reader)
            if ps is None:
                continue
            reps = idx.replies(reader, ps)
            if not reps:
                continue
            seconds = [c2 for c2 in sorted(chans) if c2 != ch
                       and len(idx.producers[c2] - {reader}) >= SECOND_MIN_PRODUCERS
                       and not (idx.producers[c2] - {reader}) & ps]
            cand = (len(ps), ch, ps, reps, seconds)
            if best is None or (bool(cand[4]), cand[0]) > (bool(best[4]), best[0]):
                best = cand
        if best is None:
            continue
        n, ch, ps, reps, seconds = best
        if seconds or not second_set:
            ev = [f"only reader of {ch}"] + [f"only reader of {c2}" for c2 in seconds[:1]]
            counts = {"requesters": n, "requester_organs": len({idx.organ.get(p) for p in ps} - {None}),
                      "replies": len(reps)}
            if seconds:
                counts["workers"] = len(idx.producers[seconds[0]] - {reader})
            hits.append(RoleEvidenceV1(part_id=reader, role="S2", rule="R-matchmaker", strength="rule",
                                       evidence=ev, counts=counts))
        else:
            servers.append(NearMissV1(part_id=reader, rule="R-server", channel=ch, producers=n,
                                      replies=len(reps), readers=1))
    return hits, servers


def rule_collectors(idx: _Index) -> list[NearMissV1]:
    out = []
    for ch in sorted(idx.readers):
        if ch in idx.census:
            continue
        for reader in sorted(idx.readers[ch]):
            ps = idx.fan_in(ch, reader)
            if ps is not None and not idx.replies(reader, ps):
                out.append(NearMissV1(part_id=reader, rule="R-collector", channel=ch, producers=len(ps),
                                      replies=0, readers=len(idx.readers[ch])))
    return out


def rule_census_reader(idx: _Index) -> list[RoleEvidenceV1]:
    """One row per reader, over every census channel it reads."""
    reads: dict[str, list[str]] = defaultdict(list)
    for ch in sorted(idx.census):
        for reader in idx.readers[ch]:
            reads[reader].append(ch)
    out = []
    for reader, chans in sorted(reads.items()):
        ps = set().union(*(idx.producers[ch] for ch in chans)) - {reader}
        reps = idx.replies(reader, ps)
        if ps and len(reps) / len(ps) < CENSUS_REPLY_MAX:
            out.append(RoleEvidenceV1(part_id=reader, role="S3*", rule="R-census-reader", strength="candidate",
                                      evidence=[f"reads {ch}" for ch in chans],
                                      counts={"producers": len(ps), "replies": len(reps)}))
    return out


def rule_sources(idx: _Index) -> list[str]:
    sends = {s for (s, _), chs in idx.pair_channels.items() if chs - idx.census}
    gets = {t for (_, t), chs in idx.pair_channels.items() if chs - idx.census}
    return sorted(sends - gets)


def silent_services(atlas: BodyAtlasV1, idx: _Index) -> list[str]:
    """Live services with no non-census channel in or out (a causal-only flow names none)."""
    busy = {x for (s, t), chs in idx.pair_channels.items() if chs - idx.census for x in (s, t)}
    return sorted(p.part_id for p in atlas.parts
                  if p.part_id in idx.live and p.kind == "service" and p.part_id not in busy)


def declared_runner(atlas: BodyAtlasV1) -> list[str]:
    procs = [p.part_id for p in atlas.parts if p.kind == "process"]  # any status: a declared count
    runs: dict[str, int] = defaultdict(int)
    for f in atlas.flows:
        if f.relation == "runs" and "declared_runner" in f.evidence:
            runs[f.source] += 1
    return [f"{_bare(r)} runs {n} of {len(procs)} declared processes (a declared source, not a rule)"
            for r, n in sorted(runs.items(), key=lambda kv: -kv[1])]


# --- reading ----------------------------------------------------------------------------------

def _names(parts: list[str], k: int) -> str:
    shown = ", ".join(_bare(p) for p in parts[:k])
    return shown + (f" +{len(parts) - k}" if len(parts) > k else "")


def read_capacities(atlas: BodyAtlasV1, *, second_set: bool = True) -> CapacityReadingV1:
    idx = _Index(atlas)
    boundary = rule_boundary_in(atlas, idx)
    s2, servers = rule_matchmaker(idx, second_set=second_set)
    audit = rule_census_reader(idx)
    collectors = rule_collectors(idx)
    silent = silent_services(atlas, idx)
    roles = boundary + s2 + audit
    organ_of = {m: o for o in atlas.organs for m in o.members}

    jobs, gaps = [], []
    jobs.append(JobLineV1(job="S1", status="BY RULE" if boundary else "NO PART FOUND",
                          text=("outside contact: " + _names([r.part_id for r in boundary], 8)) if boundary
                          else "R-boundary-in found no live camera or outside endpoint"))
    if s2:
        jobs.append(JobLineV1(job="S2", status="BY RULE", text="; ".join(
            f"{_bare(r.part_id)} ({r.counts['requesters']} requesters / {r.counts.get('workers', 0)} workers, "
            f"replies {r.counts['replies']})" for r in s2)))
    else:
        jobs.append(JobLineV1(job="S2", status="NO PART FOUND", text="R-matchmaker found no part between two crowds"))
        gaps.append(GapV1(gap_id="gap:s2-none", job="S2",
                          text="no part is shown to keep the workers from fighting: R-matchmaker found none"))
    jobs.append(JobLineV1(job="S3", status="NO RULE",
                          text=f"unseen: {len(silent)} live services with no bus traffic but the census"))
    if silent:
        gaps.append(GapV1(gap_id="gap:s3-unseen", job="S3", text=(
            f"I cannot see my own control: {len(silent)} live services show no bus traffic but the census "
            f"({_names(silent, 4)}); the atlas does not read HTTP calls or Redis keys")))
    if audit:
        jobs.append(JobLineV1(job="S3*", status="CANDIDATE", text="; ".join(
            f"{_bare(r.part_id)} ({', '.join(r.evidence)} from {r.counts['producers']}, replies {r.counts['replies']})"
            for r in audit)))
        gaps.append(GapV1(gap_id="gap:s3star-unconfirmed", job="S3*", text=(
            f"no confirmed audit: {len(audit)} part(s) listen to the whole body, none is shown to sample work "
            "and report back")))
    else:
        jobs.append(JobLineV1(job="S3*", status="NO PART FOUND", text="R-census-reader found no quiet census reader"))
        gaps.append(GapV1(gap_id="gap:s3star-none", job="S3*", text="no part is shown to audit the work"))
    jobs.append(JobLineV1(job="S4", status="NO RULE", text="no rule reads the outside or the future"))
    gaps.append(GapV1(gap_id="gap:s4-to-s3", job="S4", text=(
        "nothing on the bus is shown to steer the present by what I see of the outside or the future")))
    jobs.append(JobLineV1(job="S5", status="NO RULE", text="no rule; policy lives in files and sign-offs"))
    gaps.append(GapV1(gap_id="gap:s5-unread", job="S5", text=(
        "no running part is shown to set policy from who I am; tiers and allowlists are files and sign-offs")))
    jobs.append(JobLineV1(job="algedonic", status="NO RULE", text="strain is a Redis key, not in the atlas"))
    gaps.append(GapV1(gap_id="gap:algedonic-unseen", job="algedonic", text=(
        "my pain/pleasure line is not in my atlas: strain is a Redis key it does not read")))
    for r in s2:
        o = organ_of.get(r.part_id)
        if o is not None and len(o.members) > 1:
            gaps.append(GapV1(gap_id="gap:s2-inside-organ", job="S2", text=(
                f"{_bare(r.part_id)} coordinates from inside organ {o.name or o.organ_id} ({len(o.members)} parts): "
                "jobs do not follow organ lines")))

    mixes: dict[str, OrganMixV1] = {}
    for r in roles:
        o = organ_of.get(r.part_id)
        if o is None:
            continue
        mix = mixes.setdefault(o.organ_id, OrganMixV1(organ_id=o.organ_id, name=o.name, size=len(o.members)))
        mix.roles.setdefault(r.role, []).append(r.part_id)

    gaps.sort(key=lambda g: JOBS.index(g.job))
    return CapacityReadingV1(
        atlas_as_of=atlas.as_of, live_parts=len(idx.live), census=idx.census_rows, roles=roles,
        near_misses=servers + collectors, sources=rule_sources(idx), silent_services=silent,
        declared=declared_runner(atlas), jobs=jobs, gaps=gaps,
        organ_mixes=sorted(mixes.values(), key=lambda m: -m.size))


def hits(reading: CapacityReadingV1, rule: str) -> set[str]:
    return {r.part_id for r in reading.roles if r.rule == rule}


def render(reading: CapacityReadingV1) -> list[str]:
    """The printout (the caller caps it and adds the file path)."""
    lines = [f"capacities (VSM lens) · body atlas {reading.atlas_as_of[:16]}Z · {reading.live_parts} live "
             f"services/endpoints · rules {reading.rules_version}"]
    for j in reading.jobs:
        label = f"{j.job} {JOB_TITLES[j.job]}"
        lines.append(f"{label:<21} {j.status:<13} {j.text}")
    servers = sorted({_bare(n.part_id) for n in reading.near_misses if n.rule == "R-server"})
    collectors = sorted({_bare(n.part_id) for n in reading.near_misses if n.rule == "R-collector"})
    lines.append("near misses (not S2): servers " + (", ".join(servers) or "none")
                 + " · collectors " + (", ".join(collectors) or "none"))
    lines.append(f"input candidates (world or clock; the rule cannot tell): {len(reading.sources)}")
    census = [c for c in reading.census if c.census]
    lines.append("census: " + (", ".join(f"{c.channel} {c.share:.0%}" for c in census) or "none"))
    lines += [f"declared: {d}" for d in reading.declared]
    lines.append(f"gaps: {len(reading.gaps)} (python -P -m orion.dig capacities gaps) · claims: not built yet")
    return lines


def render_gaps(reading: CapacityReadingV1) -> list[str]:
    lines = [f"{len(reading.gaps)} gaps: what my atlas does not show any part doing (written by code from "
             f"the atlas of {reading.atlas_as_of[:16]}Z, rules {reading.rules_version}; I cannot remove them):"]
    lines += [f"- {g.job}: {g.text}" for g in reading.gaps]
    return lines

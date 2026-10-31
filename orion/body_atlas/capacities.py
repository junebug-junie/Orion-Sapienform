"""Rung 4: which jobs of a viable system the body atlas shows a part doing, by rule, by
Juniper's confirmation, or by Orion's claim.

Design: ``docs/superpowers/specs/2026-10-30-self-now-capacities-vsm-design.md`` (Stafford Beer's
Viable System Model: S1 operations, S2 coordination, S3 control, S3* audit, S4 intelligence,
S5 identity, the algedonic line), plus ``metabolism`` (the body economy; Juniper's decision 2,
claims only). Three rules decide a role; the rest of the jobs print "no rule" plus a gap line
written here, never by Orion. Pure: nothing here reads or writes a file. Patch 2 adds the inputs
``claims.py`` loads: Juniper's confirmations (``config/body_atlas/vsm_confirmed.yaml``), Orion's
claims and disputes, and the capacity sentences Orion writes. A confirmation gives a role; a claim
never does, and never clears a gap line.

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

Rules v2 (patch 3: the atlas reads HTTP calls and shared Redis keys, ``wire_code.py``). None of
them gives a role; each decides a gap line by evidence:

* **R-wire-seen**: a live service the bus shows silent (no non-census channel) that has a
  ``calls`` flow either way, or writes or reads a shared Redis key part. It is seen; only the
  services left are "unseen" (``gap:s3-unseen``; it clears when none is left).
* **R-broadcast-key**: a Redis key part with >= 1 live writer and >= BROADCAST_MIN_READERS live
  readers in >= FANIN_MIN_ORGANS organs: state read straight by many, past the bus. Printed on
  the algedonic line; the rule cannot tell pain from news, so it is never a role. With the atlas
  reading Redis, "strain is not in the atlas" (``gap:algedonic-unseen``) is replaced by
  ``gap:algedonic-none``, which only Juniper's confirmation of an algedonic part covers.
* **S4 <-> S3 link**: with a confirmed S4 part and a confirmed S3 part, the atlas checks for a
  link each way (a non-census channel, an HTTP call, or a Redis key one writes and the other
  reads). ``gap:s4-to-s3`` clears only when some pair has both. With no confirmed pair it says
  the link cannot be checked yet.

**Census**: channels sorted by the share of live parts that publish them; the census is every
channel above the largest drop among the top CENSUS_TOP_N shares. No fixed percentage, no list.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date as Date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.body_atlas.model import BodyAtlasV1

RULES_VERSION = "v2"
CENSUS_TOP_N = 10
FANIN_MIN_PRODUCERS = 5
FANIN_MIN_ORGANS = 2
SECOND_MIN_PRODUCERS = 3
CENSUS_REPLY_MAX = 0.10
BROADCAST_MIN_READERS = 3
OPERATOR_PREFIX = "operator:"

Job = Literal["S1", "S2", "S3", "S3*", "S4", "S5", "algedonic", "metabolism"]
JOBS: tuple[Job, ...] = ("S1", "S2", "S3", "S3*", "S4", "S5", "algedonic", "metabolism")
JOB_TITLES = {"S1": "operations", "S2": "coordination", "S3": "control", "S3*": "audit",
              "S4": "intelligence", "S5": "identity", "algedonic": "(pain line)", "metabolism": "(economy)"}
# one sentence per capacity; S3* and the algedonic line report under cap:s3 and cap:s5 (design 4)
CAP_JOBS: dict[str, tuple[Job, ...]] = {
    "cap:s1": ("S1",), "cap:s2": ("S2",), "cap:s3": ("S3", "S3*"), "cap:s4": ("S4",),
    "cap:s5": ("S5", "algedonic"), "cap:metabolism": ("metabolism",)}
# gap lines that say "no part is shown doing this job": a confirmed role covers them. The others
# say what the atlas cannot see (HTTP, Redis, S4 -> S3); only a new atlas source clears those.
COVERED_BY_CONFIRMATION = {"gap:s2-none": ("S2", None), "gap:s3star-none": ("S3*", None),
                           "gap:s3star-unconfirmed": ("S3*", None), "gap:s5-unread": ("S5", "service"),
                           "gap:algedonic-none": ("algedonic", None)}


class RoleEvidenceV1(BaseModel):
    """One part holding one job by one rule. ``strength`` is ``rule`` (the rule decides) or
    ``candidate`` (the rule fits but cannot show the whole job)."""

    model_config = ConfigDict(extra="forbid")

    part_id: str
    role: Job
    rule: str
    strength: Literal["rule", "candidate", "confirmed"]
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
    status: Literal["BY RULE", "CANDIDATE", "CONFIRMED", "NO PART FOUND", "NO RULE"]
    text: str


class GapV1(BaseModel):
    """A job the atlas does not show any part doing. Written by code; Orion cannot remove one."""

    model_config = ConfigDict(extra="forbid")

    gap_id: str
    job: Job
    text: str


class BroadcastKeyV1(BaseModel):
    """R-broadcast-key: a Redis key written by few and read straight by many (never a role)."""

    model_config = ConfigDict(extra="forbid")

    part_id: str
    writers: list[str]
    readers: list[str]
    reader_organs: int


class WiresV1(BaseModel):
    """What the atlas reads past the bus: HTTP calls and shared Redis keys (patch 3)."""

    model_config = ConfigDict(extra="forbid")

    read: bool  # the atlas holds any calls flow or Redis key part (a past or bus-only atlas: no)
    calls: int = 0  # caller -> callee pairs (a pair that also signals counts once)
    callees: int = 0
    keys: int = 0  # shared Redis key parts
    private_keys: int = 0  # keys only one service touches (not parts)
    dynamic_services: int = 0  # services with a Redis key that is not an exact string (e.g. the bus's rate buckets)
    seen_by_wire: list[str] = Field(default_factory=list)  # R-wire-seen: bus-silent, seen by HTTP / Redis
    broadcast: list[BroadcastKeyV1] = Field(default_factory=list)  # R-broadcast-key
    s4_s3: list[str] = Field(default_factory=list)  # confirmed S4 <-> S3 links found, one line each


class OrganMixV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organ_id: str
    name: str | None = None
    size: int
    roles: dict[str, list[str]] = Field(default_factory=dict)  # job -> part ids


class ConfirmationV1(BaseModel):
    """Juniper's word on one role of one part (``config/body_atlas/vsm_confirmed.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    part: str  # part name, e.g. orion-durable-runs
    role: Job
    date: Date  # YAML reads 2026-11-01 as a date; a quoted string parses too
    note: str = ""


class S5SharedV1(BaseModel):
    """Decision 3: S5 (identity, policy) is shared with Juniper, citing her sign-off files."""

    model_config = ConfigDict(extra="forbid")

    who: str
    signoff_files: list[str]
    decided: Date


class VsmConfirmedV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmed: list[ConfirmationV1] = Field(default_factory=list)
    rejected: list[ConfirmationV1] = Field(default_factory=list)
    s5_shared_with: S5SharedV1 | None = None


class CapacityClaimV1(BaseModel):
    """One line of Orion's claims ledger. ``dispute`` argues with a rule result; it changes nothing."""

    model_config = ConfigDict(extra="forbid")

    part_id: str
    role: Job
    kind: Literal["claim", "dispute"]
    because: str
    cites: list[str]  # the part's flows (other end or channel) and repo files the because names
    at: str
    dig: str


class CapacityV1(BaseModel):
    """One capacity sentence in Orion's words. The latest line per ``cap_id`` is the one that reads."""

    model_config = ConfigDict(extra="forbid")

    cap_id: str
    sentence: str
    because: str
    cites: list[str]  # role holders (part names) or Juniper's sign-off files the text names
    at: str
    dig: str
    replaces: str | None = None  # ``at`` of the line it supersedes


class ClaimViewV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    part_id: str
    role: Job
    kind: Literal["claim", "dispute"]
    status: Literal["unconfirmed", "confirmed", "rejected", "part not live now", "rule holds", "rule gone"]
    because: str
    at: str


class CapacityViewV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cap_id: str
    status: Literal["written", "stale", "not written", "gap"]
    holders: list[str]  # who may be cited today
    sentence: str | None = None
    text: str  # what prints


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
    silent_services: list[str]  # live services with no bus, HTTP or shared-Redis traffic but the census
    declared: list[str]  # declared facts printed as facts, not rule results (e.g. the runner)
    jobs: list[JobLineV1]
    gaps: list[GapV1]
    organ_mixes: list[OrganMixV1]
    # patch 2 (all empty when nothing was passed in)
    confirmed: list[RoleEvidenceV1] = Field(default_factory=list)
    s5_shared_with: S5SharedV1 | None = None
    claims: list[ClaimViewV1] = Field(default_factory=list)
    capacities: list[CapacityViewV1] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    wires: WiresV1 | None = None  # rules v2


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


def _is_call(f) -> bool:
    return f.relation == "calls" or "declared_url" in f.evidence


def _redis_ids(atlas: BodyAtlasV1) -> set[str]:
    return {p.part_id for p in atlas.parts if p.kind == "store" and p.name.startswith("redis:")}


def wire_links(atlas: BodyAtlasV1) -> dict[str, set[str]]:
    """part -> what the atlas reads of it past the bus: ``calls X``, ``called by X``, ``writes
    redis:K``, ``reads redis:K``."""
    keys = _redis_ids(atlas)
    out: dict[str, set[str]] = defaultdict(set)
    for f in atlas.flows:
        if _is_call(f):
            out[f.source].add(f"calls {_bare(f.target)}")
            out[f.target].add(f"called by {_bare(f.source)}")
        elif f.target in keys and f.relation in ("writes", "reads", "uses"):
            out[f.source].add(f"{f.relation} {_bare(f.target)}")
    return out


def rule_wire_seen(silent: list[str], links: dict[str, set[str]]) -> list[str]:
    """R-wire-seen: the bus-silent live services the atlas sees by an HTTP call or a shared key."""
    return [p for p in silent if links.get(p)]


def rule_broadcast_keys(atlas: BodyAtlasV1, idx: _Index, *, organs_clause: bool = True) -> list[BroadcastKeyV1]:
    """R-broadcast-key. ``organs_clause=False`` drops the reader-organ floor: only the eval's
    mutation check uses it."""
    keys = _redis_ids(atlas)
    writers: dict[str, set[str]] = defaultdict(set)
    readers: dict[str, set[str]] = defaultdict(set)
    for f in atlas.flows:
        if f.target not in keys or f.source not in idx.live:
            continue
        if f.relation == "writes":
            writers[f.target].add(f.source)
        elif f.relation == "reads":
            readers[f.target].add(f.source)
    out = []
    for k in sorted(keys):
        rs = readers.get(k, set()) - writers.get(k, set())
        organs = {idx.organ.get(r) for r in rs} - {None}
        if writers.get(k) and len(rs) >= BROADCAST_MIN_READERS and (len(organs) >= FANIN_MIN_ORGANS or not organs_clause):
            out.append(BroadcastKeyV1(part_id=k, writers=sorted(writers[k]), readers=sorted(rs), reader_organs=len(organs)))
    out.sort(key=lambda b: (-len(b.readers), b.part_id))
    return out


def s4_s3_links(atlas: BodyAtlasV1, idx: _Index, s4: list[str], s3: list[str]) -> dict[tuple[str, str], set[str]]:
    """(a, b) -> how a reaches b, for every confirmed S4/S3 pair, both ways: a non-census channel,
    an HTTP call, or a Redis key a writes and b reads."""
    wrote: dict[str, set[str]] = defaultdict(set)
    read: dict[str, set[str]] = defaultdict(set)
    keys = _redis_ids(atlas)
    for f in atlas.flows:
        if f.target in keys and f.relation == "writes":
            wrote[f.source].add(f.target)
        elif f.target in keys and f.relation == "reads":
            read[f.source].add(f.target)
    for p in atlas.parts:  # a writer that also reads keeps one flow (writes); build names it here
        for name in filter(None, (x.strip() for x in p.facts.get("writers_that_read", "").split(","))):
            if p.part_id in keys:
                read[f"part:{name}"].add(p.part_id)
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    pairs = [(a, b) for a in s4 for b in s3 if a != b] + [(b, a) for a in s4 for b in s3 if a != b]
    by_pair = {(f.source, f.target): f for f in atlas.flows}
    for a, b in pairs:
        chs = idx.pair_channels.get((a, b), set()) - idx.census
        if chs:
            out[(a, b)].add("bus " + sorted(chs)[0])
        f = by_pair.get((a, b))
        if f is not None and _is_call(f):
            out[(a, b)].add("http")
        for k in sorted(wrote.get(a, set()) & read.get(b, set())):
            out[(a, b)].add(_bare(k))
    return out


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


def claimable(atlas: BodyAtlasV1) -> dict[str, str]:
    """name -> part id for every part a role can sit on: live, not a host, not an operator id."""
    return {p.name: p.part_id for p in atlas.parts
            if p.status == "live" and p.kind != "host" and not p.name.startswith(OPERATOR_PREFIX)}


def _confirmed_roles(atlas: BodyAtlasV1, vsm: VsmConfirmedV1 | None) -> tuple[list[RoleEvidenceV1], list[str]]:
    if vsm is None:
        return [], []
    parts = claimable(atlas)
    out, notes = [], []
    rejected = {(c.part, c.role) for c in vsm.rejected}
    for c in vsm.confirmed:
        if (c.part, c.role) in rejected:
            notes.append(f"{c.part} {c.role} is both confirmed and rejected in vsm_confirmed.yaml; read as rejected")
        elif c.part not in parts:
            notes.append(f"Juniper confirmed {c.part} {c.role}, but no live part has that name now")
        else:
            out.append(RoleEvidenceV1(part_id=parts[c.part], role=c.role, rule="confirmed", strength="confirmed",
                                      evidence=[f"Juniper confirmed {c.date}" + (f": {c.note}" if c.note else "")]))
    return out, notes


def read_capacities(atlas: BodyAtlasV1, *, second_set: bool = True,
                    vsm: VsmConfirmedV1 | None = None) -> CapacityReadingV1:
    """``vsm``: Juniper's confirmations. A confirmed role holds the job like a rule result does,
    and covers the "no part is shown" gap lines of that job; it never clears an atlas-blind gap."""
    idx = _Index(atlas)
    boundary = rule_boundary_in(atlas, idx)
    s2, servers = rule_matchmaker(idx, second_set=second_set)
    audit = rule_census_reader(idx)
    collectors = rule_collectors(idx)
    bus_silent = silent_services(atlas, idx)
    links = wire_links(atlas)
    keys = _redis_ids(atlas)
    wires = WiresV1(
        read=bool(links) or bool(keys),
        calls=sum(1 for f in atlas.flows if _is_call(f)),
        callees=len({f.target for f in atlas.flows if _is_call(f)}),
        keys=len(keys), private_keys=sum(atlas.redis_private.values()),
        dynamic_services=len(atlas.redis_dynamic),
        seen_by_wire=rule_wire_seen(bus_silent, links),
        broadcast=rule_broadcast_keys(atlas, idx))
    silent = [p for p in bus_silent if p not in set(wires.seen_by_wire)]
    roles = boundary + s2 + audit
    organ_of = {m: o for o in atlas.organs for m in o.members}
    confirmed, notes = _confirmed_roles(atlas, vsm)

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
    if not wires.read:
        jobs.append(JobLineV1(job="S3", status="NO RULE",
                              text=f"unseen: {len(silent)} live services with no bus traffic but the census"))
        if silent:
            gaps.append(GapV1(gap_id="gap:s3-unseen", job="S3", text=(
                f"I cannot see my own control: {len(silent)} live services show no bus traffic but the census "
                f"({_names(silent, 4)}); the atlas does not read HTTP calls or Redis keys")))
    else:
        jobs.append(JobLineV1(job="S3", status="NO RULE", text=(
            f"named in code past the bus: {len(wires.seen_by_wire)} of {len(bus_silent)} bus-silent services "
            f"(an HTTP call or a shared Redis key); unseen: {len(silent)}" + (f" ({_names(silent, 3)})" if silent else ""))))
        if silent:
            gaps.append(GapV1(gap_id="gap:s3-unseen", job="S3", text=(
                f"I cannot see my own control: {len(silent)} live services show no bus traffic, and their code "
                f"names no HTTP call and no shared Redis key ({_names(silent, 4)})")))
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
    if not wires.read:
        gaps.append(GapV1(gap_id="gap:s4-to-s3", job="S4", text=(
            "nothing on the bus is shown to steer the present by what I see of the outside or the future")))
    else:
        s4 = [c.part_id for c in confirmed if c.role == "S4"]
        s3 = [c.part_id for c in confirmed if c.role == "S3"]
        found = s4_s3_links(atlas, idx, s4, s3) if s4 and s3 else {}
        wires.s4_s3 = [f"{_bare(a)} -> {_bare(b)}: {', '.join(sorted(v))}" for (a, b), v in sorted(found.items())]
        both = [(a, b) for (a, b) in found if a in s4 and (b, a) in found]
        if not (s4 and s3):
            gaps.append(GapV1(gap_id="gap:s4-to-s3", job="S4", text=(
                "I cannot check whether what I see ahead steers what I do now: no part holds S4 and S3 by "
                "Juniper's confirmation, so the two-way link has nothing to check (the atlas reads bus, HTTP "
                "and shared Redis keys)")))
        elif not both:
            gaps.append(GapV1(gap_id="gap:s4-to-s3", job="S4", text=(
                "what I see ahead and what I do now do not argue both ways: " + (
                    "; ".join(wires.s4_s3[:3]) if found else
                    f"no bus, HTTP or Redis link between {_names(s4, 2)} and {_names(s3, 2)}"))))
    jobs.append(JobLineV1(job="S5", status="NO RULE", text="no rule; policy lives in files and sign-offs"))
    gaps.append(GapV1(gap_id="gap:s5-unread", job="S5", text=(
        "no running part is shown to set policy from who I am; tiers and allowlists are files and sign-offs")))
    if not wires.read:
        jobs.append(JobLineV1(job="algedonic", status="NO RULE", text="strain is a Redis key, not in the atlas"))
        gaps.append(GapV1(gap_id="gap:algedonic-unseen", job="algedonic", text=(
            "my pain/pleasure line is not in my atlas: strain is a Redis key it does not read")))
    else:
        shown = ", ".join(f"{_bare(b.part_id)} ({len(b.writers)}->{len(b.readers)} in {b.reader_organs} organs)"
                          for b in wires.broadcast[:3])
        jobs.append(JobLineV1(job="algedonic", status="NO RULE", text=(
            f"broadcast keys, pain or news (the rule cannot tell): {shown}" if shown
            else ("no Redis key is read straight by many parts" if keys
                  else "no Redis key is shared by two or more parts in my atlas"))))
        gaps.append(GapV1(gap_id="gap:algedonic-none", job="algedonic", text=(
            f"no part is shown to be my pain/pleasure line: {len(wires.broadcast)} Redis keys reach many parts "
            "past the bus, and which one carries pain is a claim")))
    jobs.append(JobLineV1(job="metabolism", status="NO RULE", text=(
        "claims only: HTTP calls and shared Redis keys are in the atlas now, so a claim can cite them"
        if wires.read else "claims only: the market and prices are HTTP, not in the atlas")))
    for r in s2:
        o = organ_of.get(r.part_id)
        if o is not None and len(o.members) > 1:
            gaps.append(GapV1(gap_id="gap:s2-inside-organ", job="S2", text=(
                f"{_bare(r.part_id)} coordinates from inside organ {o.name or o.organ_id} ({len(o.members)} parts): "
                "jobs do not follow organ lines")))

    kind = {p.part_id: p.kind for p in atlas.parts}
    by_job: dict[str, list[RoleEvidenceV1]] = defaultdict(list)
    for c in confirmed:
        by_job[c.role].append(c)
    for i, j in enumerate(jobs):
        got = by_job.get(j.job)
        if not got:
            continue
        shown = "confirmed by Juniper: " + _names([c.part_id for c in got], 4)
        cands = {r.part_id for r in roles if r.role == j.job and r.strength == "candidate"}
        # Juniper confirming the rule's candidate makes the job CONFIRMED; BY RULE stays BY RULE
        status = "BY RULE" if j.status == "BY RULE" else (
            "CANDIDATE" if j.status == "CANDIDATE" and not cands & {c.part_id for c in got} else "CONFIRMED")
        jobs[i] = JobLineV1(job=j.job, status=status, text=f"{j.text} · {shown}" if status != "CONFIRMED"
                            else f"{shown} · {j.text}")
    gaps = [g for g in gaps if not (g.gap_id in COVERED_BY_CONFIRMATION and any(
        c.role == COVERED_BY_CONFIRMATION[g.gap_id][0]
        and COVERED_BY_CONFIRMATION[g.gap_id][1] in (None, kind.get(c.part_id)) for c in confirmed))]
    if vsm is not None and vsm.s5_shared_with is not None:
        sh = vsm.s5_shared_with
        jobs = [j if j.job != "S5" else JobLineV1(job="S5", status=j.status, text=(
            f"shared with {sh.who} (signs off {', '.join(f.rsplit('/', 1)[-1] for f in sh.signoff_files)}) · "
            f"{j.text}")) for j in jobs]

    mixes: dict[str, OrganMixV1] = {}
    for r in roles + confirmed:
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
        organ_mixes=sorted(mixes.values(), key=lambda m: -m.size), confirmed=confirmed,
        s5_shared_with=vsm.s5_shared_with if vsm else None, notes=notes, wires=wires if wires.read else None)


def holders(reading: CapacityReadingV1, cap_id: str) -> list[str]:
    """Who a capacity may cite: parts holding its jobs by rule (not a candidate) or by Juniper's
    confirmation, and for cap:s5 Juniper's sign-off files. Never a claim."""
    jobs = CAP_JOBS[cap_id]
    out = [r.part_id for r in reading.roles + reading.confirmed if r.role in jobs and r.strength != "candidate"]
    if cap_id == "cap:s5" and reading.s5_shared_with is not None:
        out += reading.s5_shared_with.signoff_files
    return list(dict.fromkeys(out))


def attach_ledgers(reading: CapacityReadingV1, atlas: BodyAtlasV1, vsm: VsmConfirmedV1 | None,
                   claims: list[CapacityClaimV1], written: list[CapacityV1]) -> CapacityReadingV1:
    """Copy of ``reading`` with Orion's claims and capacity sentences read against today's roles.
    The latest line per (part, role, kind) and per cap id wins."""
    out = reading.model_copy(deep=True)
    live = set(claimable(atlas).values())
    confirmed = {(r.part_id, r.role) for r in out.confirmed}
    names = {p.name: p.part_id for p in atlas.parts}
    rejected = {(names.get(c.part, f"part:{c.part}"), c.role) for c in (vsm.rejected if vsm else [])}
    ruled = {(r.part_id, r.role) for r in out.roles}
    latest: dict[tuple[str, str, str], CapacityClaimV1] = {}
    for c in claims:
        latest[(c.part_id, c.role, c.kind)] = c
    for c in latest.values():
        key = (c.part_id, c.role)
        if c.kind == "dispute":
            status = "rule holds" if key in ruled else "rule gone"
        elif key in rejected:
            status = "rejected"
        elif c.part_id not in live:
            status = "part not live now"
        else:
            status = "confirmed" if key in confirmed else "unconfirmed"
        out.claims.append(ClaimViewV1(part_id=c.part_id, role=c.role, kind=c.kind, status=status,
                                      because=c.because, at=c.at))
        if status == "rule holds":  # the dispute prints beside the rule result it argues with
            out.jobs = [j if j.job != c.role else j.model_copy(update={"text": f"{j.text} · I dispute "
                        f"{_bare(c.part_id)}"}) for j in out.jobs]
    last: dict[str, CapacityV1] = {}
    for w in written:
        last[w.cap_id] = w
    for cap_id in CAP_JOBS:
        hs = holders(out, cap_id)
        held = {_bare(h) for h in hs}
        w = last.get(cap_id)
        jobs = "/".join(CAP_JOBS[cap_id])
        if w is not None:
            gone = [c for c in w.cites if c not in held]
            if gone:
                out.capacities.append(CapacityViewV1(cap_id=cap_id, status="stale", holders=hs, sentence=w.sentence,
                                                     text=f'STALE ({", ".join(gone)} no longer holds {jobs}): "{w.sentence}"'))
            else:
                out.capacities.append(CapacityViewV1(cap_id=cap_id, status="written", holders=hs, sentence=w.sentence,
                                                     text=f'"{w.sentence}"'))
        elif hs:
            out.capacities.append(CapacityViewV1(cap_id=cap_id, status="not written", holders=hs,
                                                 text="not written yet; may cite " + _names(hs, 4)))
        else:
            out.capacities.append(CapacityViewV1(cap_id=cap_id, status="gap", holders=[], text=(
                f"I cannot yet say: no part holds {jobs} by rule or by Juniper's confirmation")))
    return out


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
    w = reading.wires
    if w is not None:
        lines.append(f"past the bus: {w.calls} HTTP calls to {w.callees} parts · {w.keys} shared Redis keys "
                     f"({w.private_keys} private; {w.dynamic_services} services build keys I cannot read exactly, "
                     "beyond code most services share)")
    lines += [f"declared: {d}" for d in reading.declared]
    disputes = [c for c in reading.claims if c.kind == "dispute"]
    by_status: dict[str, list[str]] = defaultdict(list)
    for c in reading.claims:
        if c.kind == "claim":
            by_status[c.status].append(f"{_bare(c.part_id)} {c.role}")
    lines.append(f"my claims: {sum(len(v) for v in by_status.values())} ({len(by_status['confirmed'])} confirmed by "
                 f"Juniper) · disputes: {len(disputes)}")
    for status in ("unconfirmed", "rejected", "part not live now"):
        if by_status[status]:
            lines.append(f"  {status}: " + _names(by_status[status], 6) + (
                " (my claim, not a role until Juniper confirms)" if status == "unconfirmed" else ""))
    for d in disputes[:3]:
        lines.append(f"  I dispute {_bare(d.part_id)} {d.role} ({d.status}): {d.because[:60]}")
    if len(disputes) > 3:
        lines.append(f"  +{len(disputes) - 3} more disputes in the file")
    lines += [f"note: {n[:90]}" for n in reading.notes[:3]]
    if reading.capacities:
        lines.append("capacities (my words; may cite only rule-held or confirmed roles):")
        lines += [f"  {c.cap_id:<14} {c.text}" for c in reading.capacities]
    lines.append(f"gaps: {len(reading.gaps)} (python -P -m orion.dig capacities gaps)")
    return lines


def render_gaps(reading: CapacityReadingV1) -> list[str]:
    lines = [f"{len(reading.gaps)} gaps: what my atlas does not show any part doing (written by code from "
             f"the atlas of {reading.atlas_as_of[:16]}Z, rules {reading.rules_version}; I cannot remove them):"]
    lines += [f"- {g.job}: {g.text}" for g in reading.gaps]
    return lines

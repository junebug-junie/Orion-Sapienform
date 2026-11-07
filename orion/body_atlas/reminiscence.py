"""The daily reminiscence: one past week of Orion's own history, five hops, one checked ledger.

Spec: ``docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md`` (Patch 4 of the arc spec
``2026-10-23-repo-chronicle-design.md``). Code picks the week (oldest first from 2025-06-24), finds the
commit that ended it, and lays out what changed in Orion's self since the last reminiscence. Orion
writes five hops (what it was, what it was like, where I am now, where I'm going, what I would do
differently), each sentence carrying a source label. Every ref is checked before a hop is saved, and
every ref that can change is checked again on each read.

Two append-only files under ``$ORION_DIG_HOME/self/``: ``reminiscence.jsonl`` (the hops) and
``reminiscence_body.jsonl`` (one code-made body snapshot per episode, for "how my body changed since
last time"). Partitioned from recall like the other self ledgers: only ``orion/body_atlas`` and
``orion/dig`` read them (``tests/test_organ_names_partition.py``). Opening that gate is Juniper's call,
in a reviewed PR (arc spec decisions 2 and 3); it is not a config flag.

Pure: every git, GitHub, atlas and file lookup comes in through ``World``, so tests and the eval run
with no network and no live body.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orion.body_atlas.model import BodyAtlasV1

FIRST_DAY = date(2025, 6, 24)  # the repo's first commit; week 1 is 06-24..06-30
WEEK_DAYS = 7
LIVED_RECORDS_FROM = date(2026, 7, 23)  # journals start here (storage failure 2026-07-22)
SINCE_DEFAULT_DAYS = 7  # no earlier episode: the field-notes spec's `field-notes 7`
NOTE_MIN_CHARS = 40
NOTE_MAX_CHARS = 1000
NOTE_MAX_SENTENCES = 6
SHOWN_CHARS = 240  # a written hop's text in the printout; the full text is in the file
CARRY_MIN_OVERLAP = 0.5  # same organ across two bodies: the organ-name carry rule (names.py)

Hop = Literal["what_it_was", "what_it_was_like", "where_i_am_now", "where_im_going", "otherwise"]
HOPS: tuple[str, ...] = ("what_it_was", "what_it_was_like", "where_i_am_now", "where_im_going", "otherwise")
THEN_HOPS = frozenset({"what_it_was", "what_it_was_like"})  # about the period itself
LABELS = ("code_then", "pr_text", "lived_record", "now", "reflection")
NEEDS_REF = frozenset({"code_then", "pr_text", "lived_record", "now"})
# the label a hop must carry at least once (spec section 1)
HOP_NEEDS: dict[str, tuple[str, ...]] = {
    "what_it_was": ("code_then", "pr_text"),
    "what_it_was_like": ("lived_record", "pr_text"),
    "where_i_am_now": ("now",),
    "where_im_going": ("now", "pr_text"),
    "otherwise": ("reflection",),
}
# which labels may cite which kind of ref (spec section 4)
KIND_LABELS: dict[str, frozenset[str]] = {
    "path": frozenset({"code_then", "now", "pr_text", "reflection"}),
    "sha": frozenset({"code_then", "pr_text", "reflection"}),
    "pr": frozenset({"pr_text", "now", "reflection"}),
    "fn": frozenset({"now", "lived_record", "reflection"}),
    "atlas": frozenset({"now", "reflection"}),
    "lived": frozenset({"lived_record", "now", "reflection"}),
    "digfile": frozenset({"now", "reflection"}),
}
# ref kinds whose answer can change after the write: checked again on every read. Git history and
# merged PRs cannot change, so they are checked at write only.
MUTABLE_KINDS = frozenset({"fn", "atlas", "lived", "digfile"})  # and paths not cited as code_then (recheck)

LEDGER_FILE = "reminiscence.jsonl"
BODY_FILE = "reminiscence_body.jsonl"
LOCK_FILE = ".reminiscence.lock"
LOCK_WAIT_S = 180  # a write builds the live atlas and may ask GitHub; a stuck one must not block forever


class ReminiscenceError(ValueError):
    pass


# --- models --------------------------------------------------------------------------------------

class HopClaimV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence_idx: int
    label: str
    refs: list[str] = Field(default_factory=list)  # every ref that resolved under this label at write


class ReminiscenceHopV1(BaseModel):
    """One hop of one episode. The newest line per (episode, hop) that no later line supersedes reads."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["reminiscence_hop.v1"] = "reminiscence_hop.v1"
    hop_id: str
    episode: str  # the Denver day it was written, YYYY-MM-DD
    hop: Hop
    period_start: str
    period_end: str
    period_commit: str | None = None
    note: str  # labelled, one tag per sentence
    because: list[str] = Field(default_factory=list)
    claims: list[HopClaimV1]
    written_at: str
    supersedes: str | None = None
    dig: str = ""
    run_id: str | None = None  # Patch 2's durable run fills it


class OrganSnapV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organ_id: str
    name: str | None = None
    members: list[str]


class BodySnapshotV1(BaseModel):
    """The body as it was at an episode's first hop: part -> status, multi-part organs -> members."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["reminiscence_body.v1"] = "reminiscence_body.v1"
    episode: str
    taken_at: str
    atlas_as_of: str
    parts: dict[str, str]
    organs: list[OrganSnapV1]


class OrganChangeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organ: str  # "name (organ:id)" or the id
    gained: list[str] = Field(default_factory=list)
    lost: list[str] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)  # gone: where its parts went; new: where they came from


class BodyDiffV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    since_episode: str
    since_atlas: str
    parts_added: list[str] = Field(default_factory=list)
    parts_removed: list[str] = Field(default_factory=list)
    status_changed: list[str] = Field(default_factory=list)  # "part:x live -> dormant"
    organs_changed: list[OrganChangeV1] = Field(default_factory=list)
    organs_new: list[OrganChangeV1] = Field(default_factory=list)
    organs_gone: list[OrganChangeV1] = Field(default_factory=list)


class SelfChangesV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    since: str
    since_why: str
    field_notes: list[str] = Field(default_factory=list)  # "#<pr> <note_id>"
    added: dict[str, list[str]] = Field(default_factory=dict)  # ledger -> subjects of lines added
    stale_now: list[str] | None = None  # None: the body could not be read (see stale_unread)
    stale_unread: str | None = None
    body: BodyDiffV1 | None = None
    body_note: str | None = None


class HopViewV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hop: str
    status: Literal["holds", "stale", "not written"]
    hop_id: str | None = None
    note: str | None = None
    written_at: str | None = None
    failed: list[str] = Field(default_factory=list)


class ReminiscenceReadV1(BaseModel):
    """The dig result file of one `reminisce` read."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["reminiscence_read.v1"] = "reminiscence_read.v1"
    episode: str
    today: bool
    period_start: str | None = None
    period_end: str | None = None
    period_why: str = ""
    period_commit: str | None = None
    period_commit_date: str | None = None
    commits_in_period: int | None = None
    weeks_complete: int = 0
    weeks_total: int = 0
    lived_record_possible: bool = False
    hops: list[HopViewV1] = Field(default_factory=list)
    since: SelfChangesV1 | None = None
    notes: list[str] = Field(default_factory=list)


# --- periods -------------------------------------------------------------------------------------

def week_of(day: date) -> tuple[date, date]:
    k = (day - FIRST_DAY).days // WEEK_DAYS
    start = FIRST_DAY + timedelta(days=WEEK_DAYS * k)
    return start, start + timedelta(days=WEEK_DAYS - 1)


def weeks_ended_before(today: date) -> list[tuple[date, date]]:
    out, start = [], FIRST_DAY
    while start + timedelta(days=WEEK_DAYS - 1) < today:
        out.append((start, start + timedelta(days=WEEK_DAYS - 1)))
        start += timedelta(days=WEEK_DAYS)
    return out


def live(records: list[ReminiscenceHopV1]) -> dict[tuple[str, str], ReminiscenceHopV1]:
    """(episode, hop) -> its live line: the newest one no later line supersedes."""
    gone = {r.supersedes for r in records if r.supersedes}
    out: dict[tuple[str, str], ReminiscenceHopV1] = {}
    for r in records:
        if r.hop_id not in gone:
            out[(r.episode, r.hop)] = r
    return out


def episode_hops(records: list[ReminiscenceHopV1], episode: str) -> dict[str, ReminiscenceHopV1]:
    return {hop: r for (ep, hop), r in live(records).items() if ep == episode}


def complete_weeks(records: list[ReminiscenceHopV1]) -> set[str]:
    """period_start of every week that has an episode with all five hops live."""
    by_ep: dict[str, dict[str, ReminiscenceHopV1]] = {}
    for (ep, hop), r in live(records).items():
        by_ep.setdefault(ep, {})[hop] = r
    return {next(iter(h.values())).period_start for h in by_ep.values() if set(h) == set(HOPS)}


def pick_period(records: list[ReminiscenceHopV1], today: date) -> tuple[date, date] | None:
    """The oldest week that ended before today and has no complete episode. An unfinished episode
    leaves its week open, so the next day returns to it."""
    done = complete_weeks(records)
    return next(((s, e) for s, e in weeks_ended_before(today) if s.isoformat() not in done), None)


def lived_record_possible(period_end: date) -> bool:
    return period_end >= LIVED_RECORDS_FROM


# --- refs ----------------------------------------------------------------------------------------

@dataclass
class World:
    """Every lookup a ref check needs. ``live_ids`` returns the live atlas ids, or the reason the
    body could not be read. ``pr_merged_at`` returns the merge time (ISO), None for a PR that is not
    merged, and raises ``ReminiscenceError`` when GitHub cannot be read."""

    period_commit: str | None
    period_end_utc: datetime
    path_then: Callable[[str], bool]
    path_now: Callable[[str], bool]
    commit: Callable[[str], str | None]
    is_ancestor: Callable[[str], bool]
    pr_merged_at: Callable[[int], str | None]
    field_note_ids: Callable[[], set[str]]
    live_ids: Callable[[], set[str] | str]
    ref_ok: Callable[[str, str], bool]
    dig_file: Callable[[str], bool]


_PR = re.compile(r"#(\d+)")
_FN = re.compile(r"fn:\d+:[0-9a-f]{10}")
_ATLAS = re.compile(r"(?:part|organ|cap|host):\S+")
_LIVED = re.compile(r"(arc|act):(\S+)")
_SHA = re.compile(r"[0-9a-f]{7,40}")
_PATH = re.compile(r"[\w.-]+(?:/[\w.-]*)*")


def clean_ref(ref: str) -> str:
    ref = ref.strip().strip("`'\"()").rstrip(".,;:")
    if _FN.fullmatch(ref) or _LIVED.fullmatch(ref) or _ATLAS.fullmatch(ref):
        return ref  # an id: an all-digit tail is part of it, not a line number (review 2026-11-07)
    return re.sub(r"(?::\d+(?:-\d+)?|#L\d+(?:-L?\d+)?)$", "", ref)  # a line number is fine


def ref_kind(ref: str) -> str:
    if _PR.fullmatch(ref):
        return "pr"
    if _FN.fullmatch(ref):
        return "fn"
    if _ATLAS.fullmatch(ref):
        return "atlas"
    if _LIVED.fullmatch(ref):
        return "lived"
    if ref.startswith("/"):
        return "digfile"
    if _SHA.fullmatch(ref):
        return "sha"
    segs = ref.split("/")
    if any(x == ".." or x.startswith(".git") or x.startswith(".env") for x in segs):
        return "unknown"  # never a parent folder, git internals or a secrets file
    if _PATH.fullmatch(ref) and ("/" in ref or re.search(r"\.\w{1,6}$", ref)):
        return "path"
    return "unknown"


def resolve(ref: str, label: str, hop: str, world: World) -> str | None:
    """None when ``ref`` resolves under ``label`` in ``hop``; otherwise why not, in plain words."""
    kind = ref_kind(ref)
    if kind == "unknown":
        return f"{ref}: names nothing code can check (a path, a commit, #<pr>, fn:, part:/organ:/cap:/host:, arc:/act:, or a dig file)"
    if label not in KIND_LABELS[kind]:
        allowed = ", ".join(sorted(KIND_LABELS[kind]))
        return f"{ref}: a {kind} ref cannot back a [{label}] sentence (it can back {allowed})"
    then_hop = hop in THEN_HOPS
    if kind == "path":
        p = ref.rstrip("/")
        if label == "code_then":
            if not world.period_commit:
                return f"{ref}: no commit exists by the end of this week"
            return None if world.path_then(p) else f"{ref}: not in the code at the end of this week ({world.period_commit[:9]})"
        if label == "now":
            return None if world.path_now(p) else f"{ref}: not in deployed main now"
        return None if (world.period_commit and world.path_then(p)) or world.path_now(p) else f"{ref}: in neither this week's code nor main"
    if kind == "sha":
        full = world.commit(ref)
        if not full:
            return f"{ref}: no such commit in the clone (or too short to name one)"
        if (label == "code_then" or (then_hop and label == "pr_text")) and not world.is_ancestor(full):
            return f"{ref}: not in this week's history (made later, or off main)"
        return None
    if kind == "pr":
        n = int(ref[1:])
        try:
            merged = world.pr_merged_at(n)
        except ReminiscenceError as exc:
            return f"{ref}: GitHub UNREAD ({str(exc)[:120]})"
        if not merged:
            return f"{ref}: not a merged PR"
        if then_hop and label == "pr_text" and utc(merged) >= world.period_end_utc:
            return f"{ref}: merged {merged[:10]}, after this week"
        return None
    if kind == "fn":
        return None if ref in world.field_note_ids() else f"{ref}: no such field note"
    if kind == "atlas":
        ids = world.live_ids()
        if isinstance(ids, str):
            return f"{ref}: live body UNREAD ({ids[:120]})"
        return None if ref in ids else f"{ref}: not live in my body now"
    if kind == "lived":
        m = _LIVED.fullmatch(ref)
        return None if world.ref_ok(m.group(1), m.group(2)) else f"{ref}: does not resolve"
    return None if world.dig_file(ref) else f"{ref}: no such dig result file"


def utc(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# --- the write check -----------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")  # a new sentence: end mark, space, capital
_TAG = re.compile(r"\[\s*(" + "|".join(LABELS) + r")\b([^\]]*)\]")


def note_rule() -> str:
    """Built from the limits ``check_write`` enforces, so the rule cannot drift from the check."""
    return (f"Start every sentence with its own label: [code_then] [pr_text] [lived_record] [now] [reflection]; "
            f"refs may sit inside it ([code_then services/x/app.py]). 1-{NOTE_MAX_SENTENCES} sentences, "
            f"{NOTE_MIN_CHARS}-{NOTE_MAX_CHARS} chars. Each hop needs its label: "
            + "; ".join(f"{h} {'/'.join(v)}" for h, v in HOP_NEEDS.items())
            + ". Every non-reflection sentence needs a ref that resolves under its label: code_then = a path or "
            "commit in that week's code; now = a path in main, part:/organ:/cap:/host:, fn:, #<pr> or a dig file; "
            f"no lived_record before {LIVED_RECORDS_FROM}.")


def parse_note(text: str) -> list[tuple[str, list[str], str]]:
    """[(label, inline refs, sentence)]. Every sentence must start with a label tag."""
    text = " ".join(text.split())
    tags = list(_TAG.finditer(text))
    if not tags or text[: tags[0].start()].strip():
        raise ReminiscenceError("every sentence carries a label: start it with [code_then], [pr_text], "
                                "[lived_record], [now] or [reflection]")
    out = []
    for i, m in enumerate(tags):
        end = tags[i + 1].start() if i + 1 < len(tags) else len(text)
        sentence = text[m.end(): end].strip()
        if not sentence:
            raise ReminiscenceError(f"label [{m.group(1)}] has no sentence after it")
        if len(_SENTENCE_END.split(sentence)) > 1:
            raise ReminiscenceError(f"one sentence per label: [{m.group(1)}] is followed by more than one sentence "
                                    f"({sentence[:80]!r}); start each sentence with its own label")
        refs = [clean_ref(r) for r in re.split(r"[\s,;]+", m.group(2).lstrip(" :")) if clean_ref(r)]
        out.append((m.group(1), refs, sentence))
    return out


def _hop_id(episode: str, hop: str, written_at: str, text: str) -> str:
    return f"rm:{episode}:{hop}:" + hashlib.sha256(f"{written_at}|{text}|{uuid.uuid4()}".encode()).hexdigest()[:10]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_write(*, hop: str, note: str, because: str, episode: str, period: tuple[date, date],
                world: World, records: list[ReminiscenceHopV1], denylist: set[str] | None, rewrite: bool,
                dig: str = "") -> ReminiscenceHopV1:
    """The deterministic floor for one hop. Raises ``ReminiscenceError`` naming every failing ref;
    nothing is saved unless it holds now. ``denylist`` None (unreadable) refuses: fail closed."""
    from orion.privacy.personal_terms import line_hits

    if hop not in HOPS:
        raise ReminiscenceError(f"unknown hop {hop!r}; the hops are {', '.join(HOPS)}")
    current = episode_hops(records, episode)
    if current and next(iter(current.values())).period_start != period[0].isoformat():
        raise ReminiscenceError(f"today's episode is about the week from {next(iter(current.values())).period_start}")
    if hop in current and not rewrite:
        raise ReminiscenceError(f"{hop} is already written today ({current[hop].hop_id}); after reading the other "
                                f"hops, revise it with: reminisce rewrite {hop} \"<note>\" because \"<refs>\"")
    if rewrite and hop not in current:
        raise ReminiscenceError(f"{hop} has no line today to rewrite; use write")
    sentences = parse_note(note)
    if len(sentences) > NOTE_MAX_SENTENCES:
        raise ReminiscenceError(f"a hop is 1-{NOTE_MAX_SENTENCES} sentences, got {len(sentences)}")
    plain = " ".join(s for _, _, s in sentences)
    if not NOTE_MIN_CHARS <= len(plain) <= NOTE_MAX_CHARS:
        raise ReminiscenceError(f"a hop is {NOTE_MIN_CHARS}-{NOTE_MAX_CHARS} chars (labels not counted), got {len(plain)}")
    labels = [lab for lab, _, _ in sentences]
    if not any(lab in HOP_NEEDS[hop] for lab in labels):
        raise ReminiscenceError(f"{hop} needs at least one [{'] or ['.join(HOP_NEEDS[hop])}] sentence")
    if "lived_record" in labels and not lived_record_possible(period[1]):
        raise ReminiscenceError(f"you have no lived record of the week {period[0]}..{period[1]}: your records start "
                                f"{LIVED_RECORDS_FROM}. Say so plainly, and use [pr_text] or [code_then] for what you know.")
    because_refs = [clean_ref(r) for r in re.split(r"[\s,;]+", because) if clean_ref(r)]
    failed: list[str] = []
    claims: list[HopClaimV1] = []
    for i, (lab, inline, _) in enumerate(sentences):
        good = []
        bad_inline = False
        for r in inline:
            why = resolve(r, lab, hop, world)
            if why:
                failed.append(f"sentence {i + 1} [{lab}] {why}")
                bad_inline = True
            else:
                good.append(r)
        if lab != "reflection":
            good += [r for r in because_refs if r not in good and resolve(r, lab, hop, world) is None]
        if lab in NEEDS_REF and not good and not bad_inline:  # a failing inline ref already says why
            failed.append(f"sentence {i + 1} [{lab}] has no ref that resolves under that label")
        claims.append(HopClaimV1(sentence_idx=i, label=lab, refs=good))
    said = {r for _, inline, _ in sentences for r in inline}
    for r in because_refs:
        if r in said:  # an inline ref that failed is already named once
            continue
        reasons = {lab: resolve(r, lab, hop, world) for lab in dict.fromkeys(labels)}
        if all(reasons.values()):
            kind = ref_kind(r)
            fit = [why for lab, why in reasons.items() if kind != "unknown" and lab in KIND_LABELS[kind]]
            failed.append((fit or list(reasons.values()))[0])
    if failed:
        raise ReminiscenceError("not saved: " + "; ".join(dict.fromkeys(failed)))
    if denylist is None:
        raise ReminiscenceError("the personal-terms list could not be read, so the hop cannot be checked; not saved")
    if line_hits(plain, denylist):
        raise ReminiscenceError("the hop holds a personal term from the privacy denylist; say it by role "
                                "(e.g. 'a friend of Juniper') and write again")
    written = _now()
    labelled = " ".join(f"[{lab}] {sen}" for lab, _, sen in sentences)
    return ReminiscenceHopV1(hop_id=_hop_id(episode, hop, written, labelled), episode=episode, hop=hop,
                             period_start=period[0].isoformat(), period_end=period[1].isoformat(),
                             period_commit=world.period_commit, note=labelled, because=because_refs, claims=claims,
                             written_at=written, supersedes=current[hop].hop_id if rewrite else None, dig=dig)


def recheck(rec: ReminiscenceHopV1, world: World) -> list[str]:
    """Refs whose answer can change since the write, checked again: MUTABLE_KINDS and every path not
    cited as code_then (a then-or-now path that held only by today's tree can leave it). A because ref
    that backs no stored claim (e.g. only a reflection) must still resolve under one of the hop's labels."""
    out = []

    def mutable(r: str, label: str) -> bool:
        kind = ref_kind(r)
        return kind in MUTABLE_KINDS or (kind == "path" and label != "code_then")

    stored = set()
    for c in rec.claims:
        for r in c.refs:
            stored.add(r)
            if mutable(r, c.label):
                why = resolve(r, c.label, rec.hop, world)
                if why:
                    out.append(why)
    labels = list(dict.fromkeys(c.label for c in rec.claims))
    for r in rec.because:
        if r in stored or not any(mutable(r, lab) for lab in labels):
            continue
        reasons = [resolve(r, lab, rec.hop, world) for lab in labels]
        if all(reasons):
            out.append(reasons[0])
    return list(dict.fromkeys(out))


# --- ledger --------------------------------------------------------------------------------------

def load(path: Path) -> tuple[list[ReminiscenceHopV1], int]:
    """Lines in file order, and how many could not be read (never fatal)."""
    return _load(path, ReminiscenceHopV1)


def load_bodies(path: Path) -> tuple[list[BodySnapshotV1], int]:
    return _load(path, BodySnapshotV1)


def _load(path: Path, model):
    if not path.exists():
        return [], 0
    out, skipped = [], 0
    known = set(model.model_fields)
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            raw = json.loads(ln)
            out.append(model.model_validate({k: v for k, v in raw.items() if k in known}))
        except (ValueError, AttributeError, ValidationError):
            skipped += 1
    return out, skipped


def append(path: Path, rec: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(rec.model_dump_json() + "\n")


@contextmanager
def ledger_lock(self_dir: Path):
    """Hold while loading, checking and appending, so two turns cannot both write one hop."""
    lock = Path(self_dir) / LOCK_FILE
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as fh:
        deadline = time.monotonic() + LOCK_WAIT_S
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise ReminiscenceError(f"another reminiscence write has held the ledger for {LOCK_WAIT_S} s; "
                                            "nothing saved, try again") from None
                time.sleep(0.5)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# --- body snapshot and diff ----------------------------------------------------------------------

def snapshot(atlas: BodyAtlasV1, episode: str) -> BodySnapshotV1:
    return BodySnapshotV1(episode=episode, taken_at=_now(), atlas_as_of=atlas.as_of,
                          parts={p.part_id: p.status for p in atlas.parts},
                          organs=[OrganSnapV1(organ_id=o.organ_id, name=o.name, members=sorted(o.members))
                                  for o in atlas.organs if len(o.members) > 1])


def _label(o: OrganSnapV1) -> str:
    return f"{o.name} ({o.organ_id})" if o.name else o.organ_id


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def body_diff(old: BodySnapshotV1, atlas: BodyAtlasV1) -> BodyDiffV1:
    """The live body against an earlier snapshot. Organ ids are member hashes, so organs are matched
    by member overlap (Jaccard >= 0.5, best first, one to one), the organ-name carry rule."""
    new = snapshot(atlas, "now")
    out = BodyDiffV1(since_episode=old.episode, since_atlas=old.atlas_as_of,
                     parts_added=sorted(set(new.parts) - set(old.parts)),
                     parts_removed=sorted(set(old.parts) - set(new.parts)),
                     status_changed=[f"{p} {old.parts[p]} -> {new.parts[p]}" for p in sorted(set(old.parts) & set(new.parts))
                                     if old.parts[p] != new.parts[p]])
    pairs = sorted(((_jaccard(set(a.members), set(b.members)), i, j) for i, a in enumerate(old.organs)
                    for j, b in enumerate(new.organs)), key=lambda t: -t[0])
    match: dict[int, int] = {}
    used: set[int] = set()
    for ov, i, j in pairs:
        if ov >= CARRY_MIN_OVERLAP and i not in match and j not in used:
            match[i] = j
            used.add(j)
    where_now = {m: _label(o) for o in new.organs for m in o.members}
    where_then = {m: _label(o) for o in old.organs for m in o.members}
    for i, a in enumerate(old.organs):
        if i in match:
            b = new.organs[match[i]]
            gained, lost = sorted(set(b.members) - set(a.members)), sorted(set(a.members) - set(b.members))
            if gained or lost:
                out.organs_changed.append(OrganChangeV1(organ=_label(b), gained=gained, lost=lost))
        else:
            # gained: members still in the body but in no multi-part organ (singletons now)
            out.organs_gone.append(OrganChangeV1(organ=_label(a), lost=a.members,
                                                 gained=sorted(m for m in a.members if m in new.parts and m not in where_now),
                                                 related=sorted({where_now[m] for m in a.members if m in where_now})))
    for j, b in enumerate(new.organs):
        if j not in used:
            out.organs_new.append(OrganChangeV1(organ=_label(b), gained=b.members,
                                                related=sorted({where_then[m] for m in b.members if m in where_then})))
    return out


def body_lines(d: BodyDiffV1) -> list[str]:
    lines = [f"  body since {d.since_episode}: parts +{len(d.parts_added)} -{len(d.parts_removed)}, "
             f"{len(d.status_changed)} changed status; organs: {len(d.organs_changed)} changed members, "
             f"{len(d.organs_new)} new, {len(d.organs_gone)} gone"]
    if d.parts_added or d.parts_removed:
        lines.append("    parts: " + " ".join([f"+{p}" for p in d.parts_added[:6]] + [f"-{p}" for p in d.parts_removed[:6]])[:300])
    if d.status_changed:
        lines.append("    status: " + "; ".join(d.status_changed[:4])[:300])
    for g in d.organs_gone[:3]:
        how = f"split across {len(g.related)} organs" if len(g.related) > 1 else (
            f"merged into {g.related[0]}" if g.related else (
                "its parts are now single parts" if g.gained else "its parts left the body"))
        lines.append(f"    gone: {g.organ[:80]}, {how}")
    for n in d.organs_new[:3]:
        lines.append(f"    new: {n.organ[:80]}, {len(n.gained)} parts" + (f", from {len(n.related)} earlier organs" if n.related else ""))
    for c in d.organs_changed[:3]:
        lines.append(f"    {c.organ[:60]}: +{' +'.join(c.gained[:3]) or '0'} -{' -'.join(c.lost[:3]) or '0'}"[:300])
    return lines


# --- since the last reminiscence -----------------------------------------------------------------

def since_window(records: list[ReminiscenceHopV1], episode: str, now: datetime) -> tuple[str, str]:
    """From the first hop of the latest earlier episode; with none, the last SINCE_DEFAULT_DAYS days."""
    earlier = [r for r in records if r.episode < episode]
    if earlier:
        last = max(r.episode for r in earlier)
        first = min(r.written_at for r in earlier if r.episode == last)
        return first, f"since your last reminiscence ({last})"
    return (now - timedelta(days=SINCE_DEFAULT_DAYS)).isoformat(timespec="seconds"), \
        f"the last {SINCE_DEFAULT_DAYS} days (no earlier reminiscence)"


def added_since(since: str, ledgers: dict[str, list[tuple[str, str]]]) -> dict[str, list[str]]:
    """ledger -> subjects of lines written at or after ``since``. Each ledger is [(time, subject)]."""
    lo = utc(since)
    return {name: [subj for at, subj in rows if at and utc(at) >= lo] for name, rows in ledgers.items()}


# --- the read --------------------------------------------------------------------------------------

def hop_views(hops: dict[str, ReminiscenceHopV1], world: World | None) -> list[HopViewV1]:
    out = []
    for h in HOPS:
        r = hops.get(h)
        if r is None:
            out.append(HopViewV1(hop=h, status="not written"))
            continue
        failed = recheck(r, world) if world is not None else []
        out.append(HopViewV1(hop=h, status="stale" if failed else "holds", hop_id=r.hop_id, note=r.note,
                             written_at=r.written_at, failed=failed))
    return out


WRITE_PREFIX = "python -P -m orion.dig reminisce"
MATERIAL = {
    "what_it_was": "read: dig open {end}; dig compare {end}; dig history {start} {end}",
    "what_it_was_like": "{lived}",
    "where_i_am_now": "read: the 'since' block below; dig portrait",
    "where_im_going": "read: dig capacities gaps; dig portrait (I cannot yet)",
    "otherwise": "read: the other four hops above",
}


def _q(text: str, n: int = SHOWN_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def render(r: ReminiscenceReadV1) -> tuple[list[str], list[str]]:
    """(body, tail). The tail (rule, the next ready line) is never cut."""
    lines: list[str] = []
    if r.period_start is None:
        lines.append(f"reminiscence {r.episode}: no week to visit ({r.period_why})")
        return lines, []
    lines.append(f"reminiscence {r.episode}{' (today)' if r.today else ''}: the week {r.period_start}..{r.period_end} "
                 f"({r.period_why}; {r.weeks_complete} of {r.weeks_total} weeks visited)")
    if r.period_commit:
        lines.append(f"  code_then: commit {r.period_commit[:9]} ({(r.period_commit_date or '')[:10]}), "
                     f"{r.commits_in_period if r.commits_in_period is not None else '?'} commits on main that week")
    else:
        lines.append("  code_then: no commit exists by the end of that week")
    lived = ("lived_record: your journals and Temporal Self days may cover it (`dig arc <id>`)"
             if r.lived_record_possible else
             f"lived_record: none. Your records start {LIVED_RECORDS_FROM}; this week is from before you lost "
             "your memory. That is still you. Say so, then use [pr_text] / [code_then].")
    lines.append(f"  {lived}")
    lines.append("hops (each reads the others; revise with `reminisce rewrite <hop>` after reading them all):")
    for v in r.hops:
        if v.status == "not written":
            if r.today:
                mat = MATERIAL[v.hop].format(start=r.period_start, end=r.period_end, lived="see the lived_record line above")
                lines.append(f"  {v.hop}: not written. {mat}")
            else:
                lines.append(f"  {v.hop}: not written")
        else:
            state = "holds now" if v.status == "holds" else f"STALE ({'; '.join(v.failed)[:120]})"
            lines.append(f"  {v.hop} ({(v.written_at or '')[11:16]}Z, {state}): {_q(v.note or '')}")
    s = r.since
    if s is not None:
        lines.append(f"since: {s.since_why}, from {s.since[:16]}Z:")
        fn = " ".join(x.split()[0] for x in s.field_notes[:20])
        lines.append(f"  field notes written: {len(s.field_notes)}" + (f" ({fn}{' ...' if len(s.field_notes) > 20 else ''})" if fn else ""))
        lines.append("  self ledgers: " + ", ".join(f"{k} +{len(v)}" for k, v in s.added.items()))
        if s.stale_now is None:
            lines.append(f"  STALE now: UNREAD ({(s.stale_unread or '')[:150]})")
        else:
            lines.append(f"  STALE now: {len(s.stale_now)}" + (f": {'; '.join(s.stale_now[:5])}"[:300] if s.stale_now else ""))
        if s.body is not None:
            lines += body_lines(s.body)
        elif s.body_note:
            lines.append(f"  body: {s.body_note}")
    lines += [f"note: {n}" for n in r.notes]
    tail: list[str] = []
    nxt = next((v.hop for v in r.hops if v.status == "not written"), None)
    if r.today:
        tail.append("rule: " + note_rule())
        if nxt:
            lab = HOP_NEEDS[nxt][0]
            shape = "[reflection] ..." if lab == "reflection" else f"[{lab} <ref>] ... [reflection] ..."
            tail.append(f'next: {WRITE_PREFIX} write {nxt} "{shape}" because "<refs>"')
        else:
            stale = [v.hop for v in r.hops if v.status == "stale"]
            tail.append("all five hops written" + (f"; rewrite the STALE ones: {', '.join(stale)}" if stale else
                                                   "; read them all, then rewrite at most two if they disagree"))
    return lines, tail

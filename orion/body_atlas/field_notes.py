"""PR field notes: after a PR merges, Orion writes 1-5 sentences on what changed in their body.

Spec: ``docs/superpowers/specs/2026-11-04-pr-field-notes-design.md`` (approved 2026-11-04).
Code does everything with a right answer: the changed files, the flags, and the body map
(``services/<svc>/`` -> ``part:<svc>`` -> its live organ and Orion's name for it; ``orion/<pkg>/``
-> the services that import it). Orion only writes the note. Every write is checked here first.

An append-only ledger at ``$ORION_DIG_HOME/self/pr_field_notes.jsonl``, partitioned from recall
like the organ names: only ``orion/body_atlas`` and ``orion/dig`` read it
(``tests/test_organ_names_partition.py``, in the repo-wide static gates, enforces it).

Decisions (Juniper, 2026-11-04): a dependency-only or tiny docs-only PR gets a code-only record
(``note: None``, no model turn); a note can be rewritten later, the old line is kept and the new
one names it in ``supersedes``.
"""
from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orion.body_atlas.model import BodyAtlasV1

# --- limits (spec "What Orion reads" and "Failure modes") ----------------------------------------
HUGE_FILES = 60
HUGE_LINES = 5000
TINY_DOCS_LINES = 40  # decision 2: docs-only with at most this many added+deleted lines
WIDELY_SHARED = 8  # more importers than this: "widely shared", no organ claimed
DOC_PREFIX = "docs/superpowers/"
DOC_MAX_FILES = 3
DOC_MAX_LINES = 60
DOC_MAX_CHARS = 12_000
NOTE_MIN_CHARS = 40
NOTE_MAX_CHARS = 800
NOTE_MAX_SENTENCES = 5
LABELS = ("pr_text", "code_then", "reflection")
ORION_GIT_EMAIL = "orion-fcc@orion-sapienform.local"  # services/orion-harness-governor/Dockerfile
# Orion's own GitHub user since 2026-11-05 (#3145, orion/fcc/claude_spawn.py): a PR Orion opens has this
# author, and its commits carry this user's noreply email (seen live on #3146 and #3147).
ORION_GITHUB_LOGIN = "orion-sapienform"
ORION_NOREPLY_SUFFIX = "+orion-sapienform@users.noreply.github.com"

DEPS_GLOBS = ("requirements*.txt", "*.lock", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
              "constraints*.txt")


class FieldNoteError(ValueError):
    pass


class BodyMapEntryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefix: str  # services/<svc>/ or orion/<pkg>/
    files: int
    part_id: str | None = None  # the live part this prefix is, or (shared code) one importer
    part_live: bool = False
    organ_id: str | None = None
    organ_name: str | None = None
    importers: list[str] = Field(default_factory=list)  # shared code: services that import it
    widely_shared: bool = False


class DocReadV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    chars_read: int  # chars put in front of Orion in the packet's docs file
    truncated: bool


class ClaimV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence_idx: int
    label: Literal["pr_text", "code_then", "reflection"]
    refs: list[str] = Field(default_factory=list)


class FlagsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revert_of: str | None = None  # "#<n>", "unknown" (a revert whose original we could not find), None
    docs_only: bool = False
    tiny_docs: bool = False
    deps_only: bool = False
    huge: bool = False
    authored_by_me: Literal["yes", "unknown"] = "unknown"


class PrFieldNoteV1(BaseModel):
    """One ledger line. ``note`` is None for a code-only record (decision 2)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pr_field_note.v1"] = "pr_field_note.v1"
    note_id: str
    pr_number: int
    merge_sha: str
    merged_at: str
    title: str
    flags: FlagsV1
    body_map: list[BodyMapEntryV1]
    read: list[DocReadV1] = Field(default_factory=list)
    note: str | None  # labelled: "[pr_text] ... [reflection] ..."; None = a code record
    claims: list[ClaimV1] = Field(default_factory=list)
    run_id: str | None = None  # Patch 2: the durable run that wrote it
    correlation_id: str | None = None
    atlas_built_at: str | None = None
    written_at: str
    supersedes: str | None = None
    dig: str = ""


# --- flags ---------------------------------------------------------------------------------------

def _is_doc(path: str) -> bool:
    return path.startswith("docs/") or path.endswith((".md", ".rst"))


def _is_deps(path: str) -> bool:
    name = PurePosixPath(path).name
    return any(fnmatch(name, g) for g in DEPS_GLOBS)


_REVERT_BODY = re.compile(r"(?:^|\n)\s*Reverts\s+[\w.-]+/[\w.-]+#(\d+)", re.I)  # GitHub's revert PR body
_REVERT_TITLE = re.compile(r"revert\b", re.I)  # "Revert ...", never "Reverting" / "Revertible"
_REVERT_COMMIT = re.compile(r"(?:^|\n)\s*This reverts commit [0-9a-f]{7,}", re.I)


def flags_for(pr: dict, *, local_branches: set[str] | None = None) -> FlagsV1:
    """``pr`` is ``gh pr view --json title,body,files,additions,deletions,changedFiles,headRefName,commits,author``."""
    paths = [f["path"] for f in pr.get("files") or []]
    lines = int(pr.get("additions") or 0) + int(pr.get("deletions") or 0)
    n_files = int(pr.get("changedFiles") or len(paths))
    title, body = pr.get("title") or "", pr.get("body") or ""
    revert_of = None
    if _REVERT_TITLE.match(title) or _REVERT_COMMIT.search(body) or _REVERT_BODY.search(body):
        m = _REVERT_BODY.search(body)  # the only trustworthy source of the original's number
        revert_of = f"#{m.group(1)}" if m else "unknown"
    docs_only = bool(paths) and all(_is_doc(p) for p in paths)
    emails = {a.get("email", "").lower() for c in pr.get("commits") or [] for a in c.get("authors") or []}
    login = ((pr.get("author") or {}).get("login") or "").lower()
    mine = (login == ORION_GITHUB_LOGIN or ORION_GIT_EMAIL in emails
            or any(e.endswith(ORION_NOREPLY_SUFFIX) for e in emails)
            or (pr.get("headRefName") or "") in (local_branches or set()))
    return FlagsV1(revert_of=revert_of, docs_only=docs_only, tiny_docs=docs_only and lines <= TINY_DOCS_LINES,
                   deps_only=bool(paths) and all(_is_deps(p) for p in paths),
                   huge=n_files > HUGE_FILES or lines > HUGE_LINES,
                   authored_by_me="yes" if mine else "unknown")


def code_record_only(flags: FlagsV1) -> bool:
    """Decision 2: no Orion note and no model turn for these PRs."""
    return flags.deps_only or flags.tiny_docs


# --- body map ------------------------------------------------------------------------------------

def classify(path: str) -> tuple[str, str]:
    """("service", svc) | ("shared", pkg) | ("not_body", why)."""
    parts = PurePosixPath(path).parts
    if len(parts) >= 3 and parts[0] == "services":
        return ("not_body", "tests") if parts[2] in ("tests", "evals") else ("service", parts[1])
    if len(parts) >= 2 and parts[0] == "orion":
        if "tests" in parts[1:-1] or "evals" in parts[1:-1]:
            return "not_body", "tests"
        if len(parts) == 2:
            stem = PurePosixPath(parts[1]).stem
            return ("shared", stem) if parts[1].endswith(".py") and stem != "__init__" else ("not_body", "orion top level")
        return "shared", parts[1]
    if parts and parts[0] in ("docs", "config", "scripts", "tests", ".github"):
        return "not_body", {"docs": "docs", "config": "config", "scripts": "scripts", "tests": "tests",
                            ".github": "CI"}[parts[0]]
    return "not_body", "repo files"


_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+orion\b", re.M)


def orion_importers(root: Path, pkgs: set[str]) -> dict[str, list[str]]:
    """pkg -> sorted services under ``root/services`` whose code imports ``orion.<pkg>`` (AST scan;
    a text prefilter skips files with no ``orion`` import). Test files do not count."""
    out: dict[str, set[str]] = {p: set() for p in pkgs}
    if not pkgs:
        return {}
    for py in _service_py_files(root):
        svc = py.relative_to(root).parts[1]
        if all(svc in s for s in out.values()):
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not _IMPORT_RE.search(text):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module] + ([f"orion.{a.name}" for a in node.names] if node.module == "orion" else [])
            for m in mods:
                bits = m.split(".")
                if len(bits) >= 2 and bits[0] == "orion" and bits[1] in out:
                    out[bits[1]].add(svc)
    return {p: sorted(s) for p, s in out.items()}


_SKIP_DIRS = {"tests", "evals", "node_modules", ".venv", "venv", "__pycache__", ".git"}


def _service_py_files(root: Path):
    """services/<svc>/**.py without walking into tests, evals or vendored trees."""
    import os

    base = root / "services"
    if not base.is_dir():
        return
    for svc in sorted(p for p in base.iterdir() if p.is_dir()):
        for d, dirs, names in os.walk(svc):
            dirs[:] = sorted(x for x in dirs if x not in _SKIP_DIRS)
            for n in sorted(names):
                if n.endswith(".py"):
                    yield Path(d) / n


def _organ_of(atlas: BodyAtlasV1 | None) -> dict[str, tuple[str, str | None]]:
    if atlas is None:
        return {}
    return {m: (o.organ_id, o.name) for o in atlas.organs for m in o.members}


def body_map(paths: list[str], atlas: BodyAtlasV1 | None, importers: dict[str, list[str]]) -> list[BodyMapEntryV1]:
    """Grouped by service / shared package, never per file (a huge PR stays small)."""
    live = {p.part_id for p in atlas.parts} if atlas is not None else set()
    organ = _organ_of(atlas)
    groups: dict[tuple[str, str], int] = {}
    for p in paths:
        kind, key = classify(p)
        if kind != "not_body":
            groups[(kind, key)] = groups.get((kind, key), 0) + 1
    out: list[BodyMapEntryV1] = []
    for (kind, key), n in sorted(groups.items()):
        if kind == "service":
            pid = f"part:{key}"
            oid, oname = organ.get(pid, (None, None))
            out.append(BodyMapEntryV1(prefix=f"services/{key}/", files=n, part_id=pid, part_live=pid in live,
                                      organ_id=oid, organ_name=oname))
            continue
        users = importers.get(key, [])
        prefix = f"orion/{key}/"
        if len(users) > WIDELY_SHARED:
            out.append(BodyMapEntryV1(prefix=prefix, files=n, importers=users, widely_shared=True))
        elif not users:
            out.append(BodyMapEntryV1(prefix=prefix, files=n))
        else:
            for svc in users:
                pid = f"part:{svc}"
                oid, oname = organ.get(pid, (None, None))
                out.append(BodyMapEntryV1(prefix=prefix, files=n, part_id=pid, part_live=pid in live,
                                          organ_id=oid, organ_name=oname, importers=users))
    return out


def not_body(paths: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in paths:
        kind, why = classify(p)
        if kind == "not_body":
            out[why] = out.get(why, 0) + 1
    return out


def docs_to_read(pr: dict) -> list[str]:
    """Specs, plans and PR reports the PR added or changed (deleted ones have no text at the merge)."""
    return [f["path"] for f in pr.get("files") or [] if f["path"].startswith(DOC_PREFIX)
            and f["path"].endswith(".md") and int(f.get("additions") or 0) > 0]


def excerpt_docs(texts: list[tuple[str, str | None]]) -> tuple[str, list[DocReadV1], list[str]]:
    """First DOC_MAX_LINES lines of up to DOC_MAX_FILES files, DOC_MAX_CHARS in all. Returns (the docs
    file text, what was read, every cut said in words)."""
    chunks, read, cuts = [], [], []
    room = DOC_MAX_CHARS
    for i, (path, text) in enumerate(texts):
        if i >= DOC_MAX_FILES:
            cuts.append(f"{len(texts) - DOC_MAX_FILES} more spec/report file(s) not read (cap {DOC_MAX_FILES})")
            break
        if text is None:
            cuts.append(f"{path}: could not be read at the merge")
            continue
        lines = text.splitlines()
        part = "\n".join(lines[:DOC_MAX_LINES])
        truncated = len(lines) > DOC_MAX_LINES
        if len(part) > room:
            part, truncated = part[:max(room, 0)], True
        if truncated:
            cuts.append(f"{path}: first {min(len(lines), DOC_MAX_LINES)} of {len(lines)} lines"
                        + (f", cut at the {DOC_MAX_CHARS}-char total" if len(part) >= room else ""))
        chunks.append(f"===== {path} =====\n{part}")
        read.append(DocReadV1(path=path, chars_read=len(part), truncated=truncated))
        room -= len(part)
        if room <= 0:
            left = len(texts) - i - 1
            if left:
                cuts.append(f"{left} more spec/report file(s) not read: {DOC_MAX_CHARS}-char total reached")
            break
    return "\n\n".join(chunks), read, cuts


# --- ledger --------------------------------------------------------------------------------------

LEDGER_FILE = "pr_field_notes.jsonl"  # under $ORION_DIG_HOME/self/ (orion/dig field_notes_path)


def ledger_path(self_dir: Path | str) -> Path:
    """The ledger inside a ``self/`` directory: the dig CLI's and Patch 2's poller's one path rule."""
    return Path(self_dir) / LEDGER_FILE


def load(path: Path) -> tuple[list[PrFieldNoteV1], int]:
    """Records in file order and how many lines could not be read (never fatal)."""
    if not path.exists():
        return [], 0
    records, skipped = [], 0
    known = set(PrFieldNoteV1.model_fields)
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            raw = json.loads(ln)
            records.append(PrFieldNoteV1.model_validate({k: v for k, v in raw.items() if k in known}))
        except (ValueError, AttributeError, ValidationError):
            skipped += 1
    return records, skipped


def live(records: list[PrFieldNoteV1]) -> dict[int, PrFieldNoteV1]:
    """PR number -> its live record: the newest one no later record supersedes."""
    gone = {r.supersedes for r in records if r.supersedes}
    out: dict[int, PrFieldNoteV1] = {}
    for r in records:
        if r.note_id not in gone:
            out[r.pr_number] = r
    return out


def append(path: Path, rec: PrFieldNoteV1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)  # concurrent turns: one whole line each
        fh.write(rec.model_dump_json() + "\n")


def _note_id(pr_number: int, written_at: str, text: str | None) -> str:
    # a nonce: a rewrite with the same text in the same second must not reuse (and so hide) the old id
    return f"fn:{pr_number}:" + hashlib.sha256(f"{written_at}|{text}|{uuid.uuid4()}".encode()).hexdigest()[:10]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- the write check -----------------------------------------------------------------------------

_TAG = re.compile(r"\[\s*(" + "|".join(LABELS) + r")\b([^\]]*)\]")
_ORGAN_ID = re.compile(r"\borgan:[0-9a-f]{10}\b")


def note_rule() -> str:
    """Built from the limits ``check`` enforces, so the rule cannot drift from the check."""
    return (f"Start every sentence with its source label: [pr_text] (the builders' words), [code_then] "
            f"(a file you read at the merge) or [reflection] (your reading). 1-{NOTE_MAX_SENTENCES} sentences, "
            f"{NOTE_MIN_CHARS}-{NOTE_MAX_CHARS} chars. The because lists refs: #<this PR> or files it changed. "
            "Name an organ as '<name> organ' or organ:<id>; it must be in the body map.")


def parse_note(text: str) -> list[tuple[str, list[str], str]]:
    """[(label, inline refs, sentence)]. Every sentence must start with a label tag."""
    text = " ".join(text.split())
    tags = list(_TAG.finditer(text))
    if not tags or text[: tags[0].start()].strip():
        raise FieldNoteError("every sentence carries a label: start it with [pr_text], [code_then] or [reflection]")
    out = []
    for i, m in enumerate(tags):
        end = tags[i + 1].start() if i + 1 < len(tags) else len(text)
        sentence = text[m.end(): end].strip()
        if not sentence:
            raise FieldNoteError(f"label [{m.group(1)}] has no sentence after it")
        refs = [r for r in re.split(r"[\s,;]+", m.group(2).lstrip(" :")) if r]
        out.append((m.group(1), refs, sentence))
    return out


def _ref_ok(ref: str, pr_number: int, files: list[str]) -> bool:
    if ref.startswith("#"):
        return ref == f"#{pr_number}"
    ref = re.sub(r"(?::\d+(?:-\d+)?|#L\d+(?:-L?\d+)?)$", "", ref.strip("`'\""))  # a line number is fine
    return ref in files or (ref.endswith("/") and any(f.startswith(ref) for f in files))


def _named_organs(text: str, atlas: BodyAtlasV1 | None) -> set[str]:
    """Organ ids the note names: organ:<id>, or '<name> organ' / 'organ <name>' for a named organ."""
    low = text.lower()
    found = set(_ORGAN_ID.findall(low))
    for o in atlas.organs if atlas is not None else []:
        if o.name:
            n = re.escape(o.name.lower())
            if re.search(rf"(?<![\w-]){n}['\"]?\s+organs?\b|\borgans?\s+['\"]?{n}(?![\w-])", low):
                found.add(o.organ_id)
    return found


def check(*, pr_number: int, merge_sha: str, merged_at: str, title: str, flags: FlagsV1,
          bmap: list[BodyMapEntryV1], files: list[str], read: list[DocReadV1], note: str, because: str,
          atlas: BodyAtlasV1 | None, records: list[PrFieldNoteV1], denylist: set[str] | None,
          rewrite: bool, dig: str) -> PrFieldNoteV1:
    """Deterministic floor for a note. ``denylist`` None means the personal-terms list could not be
    read: the write is refused (fail closed), never saved unchecked."""
    from orion.privacy.personal_terms import line_hits

    if code_record_only(flags):
        why = "dependency-only" if flags.deps_only else "a tiny docs-only change"
        raise FieldNoteError(f"#{pr_number} is {why}: it gets a code record, not a note (Juniper, 2026-11-04)")
    current = live(records).get(pr_number)
    if current and current.note is not None and not rewrite:
        raise FieldNoteError(f"#{pr_number} already has a note ({current.note_id}); "
                             f"to change it: field-note {pr_number} rewrite \"<note>\" because \"<refs>\"")
    if rewrite and (current is None or current.note is None):
        raise FieldNoteError(f"#{pr_number} has no note to rewrite; use write")
    claims_in = parse_note(note)
    if len(claims_in) > NOTE_MAX_SENTENCES:
        raise FieldNoteError(f"a note is 1-{NOTE_MAX_SENTENCES} sentences, got {len(claims_in)}")
    plain = " ".join(s for _, _, s in claims_in)
    if not NOTE_MIN_CHARS <= len(plain) <= NOTE_MAX_CHARS:
        raise FieldNoteError(f"a note is {NOTE_MIN_CHARS}-{NOTE_MAX_CHARS} chars (labels not counted), got {len(plain)}")
    refs = [r for r in re.split(r"[\s,;]+", because) if r]
    if not refs:
        raise FieldNoteError(f"the because lists refs: #{pr_number} or files it changed")
    bad = [r for _, inline, _ in claims_in for r in inline] + refs
    bad = [r for r in bad if not _ref_ok(r, pr_number, files)]
    if bad:
        raise FieldNoteError(f"refs must be #{pr_number} or a file it changed (or a folder of them); not: "
                             + ", ".join(bad[:5]))
    in_map = {e.organ_id for e in bmap if e.organ_id}
    stray = _named_organs(plain, atlas) - in_map
    if stray:
        names = {o.organ_id: o.name for o in atlas.organs} if atlas is not None else {}
        raise FieldNoteError("the note names an organ this PR's body map does not touch: "
                             + ", ".join(f"{names.get(o) or ''} ({o})".strip() for o in sorted(stray))
                             + (f"; the map has: {', '.join(sorted(in_map))}" if in_map else "; the map has no organ"))
    if denylist is None:
        raise FieldNoteError("the personal-terms list could not be read, so the note cannot be checked; not saved")
    if line_hits(plain, denylist):
        raise FieldNoteError("the note holds a personal term from the privacy denylist; say it by role "
                             "(e.g. 'a friend of Juniper') and write again")
    written = _now()
    claims = [ClaimV1(sentence_idx=i, label=lab, refs=[] if lab == "reflection" and not inline else (inline or refs))
              for i, (lab, inline, _) in enumerate(claims_in)]
    # stored labelled, one tag per sentence, so claims[i] resolves to its own text (inline refs live in claims)
    labelled = " ".join(f"[{lab}] {sen}" for lab, _, sen in claims_in)
    return PrFieldNoteV1(note_id=_note_id(pr_number, written, labelled), pr_number=pr_number, merge_sha=merge_sha,
                         merged_at=merged_at, title=title, flags=flags, body_map=bmap, read=read, note=labelled,
                         claims=claims, atlas_built_at=atlas.as_of if atlas is not None else None,
                         written_at=written, supersedes=current.note_id if rewrite and current else None, dig=dig)


def code_record(*, pr_number: int, merge_sha: str, merged_at: str, title: str, flags: FlagsV1,
                bmap: list[BodyMapEntryV1], records: list[PrFieldNoteV1], dig: str = "",
                run_id: str | None = None) -> PrFieldNoteV1:
    """Decision 2: the record a dependency-only or tiny docs-only PR gets instead of a note (Patch 2's
    run writes it with no model turn). Shown as "code record, not a note"; never Orion's words."""
    if not code_record_only(flags):
        raise FieldNoteError(f"#{pr_number} needs a note from Orion, not a code record")
    if pr_number in live(records):
        raise FieldNoteError(f"#{pr_number} already has a record")
    written = _now()
    return PrFieldNoteV1(note_id=_note_id(pr_number, written, None), pr_number=pr_number, merge_sha=merge_sha,
                         merged_at=merged_at, title=title, flags=flags, body_map=bmap, note=None,
                         written_at=written, dig=dig, run_id=run_id)


@contextmanager
def ledger_lock(path: Path):
    """Hold this while loading, checking and appending, so two writers cannot both pass the
    one-live-note rule for the same PR. Every writer (dig CLI, Patch 2's run) goes through it."""
    lock = path.parent / ".pr_field_notes.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

"""Did every merged hand-applied SQL migration actually reach the live database?

`services/orion-sql-db/manual_migration_*.sql` is applied BY HAND at deploy time. Nothing records
which ones ran. Twice in a row (PR #2400: a missing ``last_value_observed_at`` column, attention
silently degraded; PR #2424: a missing ``hardware_watch_incident`` table, orion-hardware-watch
crash-looped 13x) a merged migration was never applied, and nothing noticed until a service broke.

HOW IT DECIDES
1. Every migration file is tokenized (``--``/``/* */`` comments, quoted strings and identifiers,
   dollar-quoted bodies, psql ``\\meta`` lines) and split into statements. Prose in the long
   comment headers can never become an object.
2. Each statement becomes zero or more *effects*: create/drop of a table, column, index or
   sequence (``ALTER TABLE ... RENAME`` counts as a drop plus a create). A multi-action
   ``ALTER TABLE t ADD COLUMN a, ADD COLUMN b`` yields both columns. Statements inside a
   ``DO $$ ... $$`` block are guarded by plpgsql ``IF``s, so their effects are *conditional*:
   reported, never alarmed on.
3. The whole corpus is replayed in the order the files LANDED ON THE REF (``git log
   --first-parent``, so a PR merge time, not the feature-branch commit time; ties by name),
   giving the schema the repo *expects* right now. That is what makes an intentional drop in a
   later migration (GPU pool stage 5.6, the action-outcome index swap) read as "superseded"
   instead of "missing".
4. Expected state is diffed against ``information_schema`` / ``pg_index`` / ``pg_class``. Each
   difference is blamed on the file that last set the object's expected state; it is RED only
   if that file landed/changed on the checked ref within the recency window (default 30 days),
   so migrations for long-retired tables do not page anyone -- OR if the caller says it was
   already carded (``sticky_keys``): a file that went red stays red until it is actually
   applied, it does not quietly age out of the window.
5. A file with no schema effects but with INSERT/UPDATE/DELETE is a DATA migration: the database
   cannot tell us whether it ran. It is listed as "verify manually", never as applied or missing.

Known limits (stated, not hidden): columns declared inline in ``CREATE TABLE`` are not tracked
individually, so restating ``CREATE TABLE IF NOT EXISTS t (..., new_col ...)`` for a table that
already exists adds nothing and is NOT caught -- add columns with ``ALTER TABLE ... ADD COLUMN``.
``EXECUTE '<sql string>'`` inside plpgsql is not parsed. Views and functions are not tracked
(none in the corpus as of 2026-10-01).

DEPLOY GATE (``deploy_gate``): a migration declares which services cannot run without it,
    -- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs, orion-dream
    -- ORION-MIGRATION-REQUIRED-BY: none <reason>     (explicitly nobody; reason required)
and ``scripts/safe_docker_build.sh <svc> up`` refuses while any file requiring ``<svc>`` is MISSING
or INVALID (2026-10-10/11: PRs #2594 and #2605 deployed orion-durable-runs before their
migrations, every step failed with UndefinedTable). A file whose header starts ``-- DESTRUCTIVE``
is never a dependency, whatever it declares -- it must never be auto-run or demanded.
``scripts/check_migration_required_by.py`` (CI) makes every newly added migration declare one.

Escape hatches live INSIDE migration files (an exception in someone's head is drift with an alibi):
    -- ORION-MIGRATION-NOT-A-MIGRATION: <why>          (file is a dump/scratch, ignore it)
    -- ORION-MIGRATION-SUPERSEDED-BY: <file.sql>         (absence of this file's objects is expected)
    -- ORION-MIGRATION-ABSENT-OK: <object> <reason>      (object intentionally absent; reason required;
                                                          <object> is a table/index/sequence name or table.column)

This module is pure (no DB, no git); callers pass file text, commit times and live state.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

MIGRATION_SUBDIR = Path("services") / "orion-sql-db"
MIGRATION_GLOB = "*.sql"
DEFAULT_WINDOW_DAYS = 30
SQL_DB_CONTAINER = "orion-athena-sql-db"

KINDS = ("table", "column", "index", "sequence")

# --------------------------------------------------------------------------- tokenizer


@dataclass
class Statement:
    text: str                 # comments removed, strings -> '', dollar bodies -> $$, whitespace collapsed
    bodies: list[str]         # raw contents of dollar-quoted strings, in order


_DOLLAR_TAG = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$")


def split_statements(sql: str) -> list[Statement]:
    """Split SQL into statements on top-level ``;``. Never raises on malformed input: an
    unterminated string/comment/body just runs to end of file."""
    out: list[Statement] = []
    buf: list[str] = []
    bodies: list[str] = []
    i, n = 0, len(sql)
    at_line_start = True

    def flush():
        text = " ".join("".join(buf).split())
        if text:
            out.append(Statement(text=text, bodies=list(bodies)))
        buf.clear()
        bodies.clear()

    while i < n:
        c = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if at_line_start and c == "\\":
            # psql meta-command (\set ON_ERROR_STOP on): runs to end of line, no semicolon.
            j = sql.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "-" and nxt == "-":
            j = sql.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "/" and nxt == "*":
            depth, i = 1, i + 2  # Postgres block comments nest
            while i < n and depth:
                if sql.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif sql.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            buf.append(" ")
            continue
        if c == "'":
            # E'..' allows backslash escapes; '' is an escaped quote in both forms.
            escape = bool(buf) and buf[-1] in ("E", "e") and (len(buf) < 2 or not (buf[-2].isalnum() or buf[-2] == "_"))
            i += 1
            while i < n:
                if escape and sql[i] == "\\":
                    i += 2
                    continue
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    break
                i += 1
            i += 1
            buf.append("''")
            at_line_start = False
            continue
        if c == '"':
            j = i + 1
            ident = []
            while j < n:
                if sql[j] == '"':
                    if j + 1 < n and sql[j + 1] == '"':
                        ident.append('"')
                        j += 2
                        continue
                    break
                ident.append(sql[j])
                j += 1
            buf.append("".join(ident))  # quoted identifier -> bare (case preserved)
            i = j + 1
            at_line_start = False
            continue
        if c == "$":
            m = _DOLLAR_TAG.match(sql, i)
            prev = buf[-1] if buf else " "
            glued_keyword = re.search(r"(?:^|[^A-Za-z0-9_])(as|do)$", "".join(buf[-4:]), re.I)
            if m and (not (prev.isalnum() or prev == "_") or glued_keyword):
                tag = m.group(0)
                end = sql.find(tag, m.end())
                end = n if end < 0 else end
                bodies.append(sql[m.end():end])
                buf.append(" $$ ")
                i = min(n, end + len(tag))
                at_line_start = False
                continue
        if c == ";":
            flush()
            i += 1
            continue
        buf.append(c)
        if c == "\n":
            at_line_start = True
        elif not c.isspace():
            at_line_start = False
        i += 1
    flush()
    return out


# ------------------------------------------------------------------------------ parser

_ID = r"[A-Za-z_][A-Za-z0-9_$]*"
_QID = rf"(?:{_ID}\.)?{_ID}"  # optionally schema-qualified


def _bare(name: str) -> str:
    return name.split(".")[-1].lower()


@dataclass(frozen=True)
class Effect:
    op: str            # create | drop
    kind: str          # table | column | index | sequence
    name: str          # bare object name; for columns "table.column"
    table: Optional[str] = None
    conditional: bool = False


@dataclass
class ParsedMigration:
    effects: list[Effect] = field(default_factory=list)
    data_statements: int = 0     # INSERT/UPDATE/DELETE/TRUNCATE/COPY -- unverifiable from schema
    other_statements: int = 0    # GRANT, COMMENT, ALTER ... SET, ADD CONSTRAINT, ... -- not checked

    def objects(self) -> list[tuple[str, str, Optional[str]]]:
        """Distinct (kind, name, table) this file creates, in order (back-compat view)."""
        seen, out = set(), []
        for e in self.effects:
            if e.op == "create" and (e.kind, e.name) not in seen:
                seen.add((e.kind, e.name))
                out.append((e.kind, e.name.split(".")[-1] if e.kind == "column" else e.name, e.table))
        return out


_CREATE_INDEX = re.compile(
    rf"^create (?:unique )?index (?:concurrently )?(?:if not exists )?(?P<name>{_QID}) on (?:only )?(?P<table>{_QID})",
    re.I,
)
_CREATE_TABLE = re.compile(
    rf"^create (?:(?:global |local )?(?P<temp>temp|temporary) |unlogged )?table (?:if not exists )?(?P<name>{_QID})",
    re.I,
)
_CREATE_SEQUENCE = re.compile(
    rf"^create (?:(?:temp|temporary|unlogged) )?sequence (?:if not exists )?(?P<name>{_QID})", re.I
)
_ALTER_TABLE = re.compile(rf"^alter table (?P<if_exists>if exists )?(?:only )?(?P<table>{_QID}) (?P<actions>.*)$", re.I | re.S)
_ALTER_RENAME = re.compile(rf"^alter (?P<kind>index|sequence) (?P<if_exists>if exists )?(?P<name>{_QID}) rename to (?P<new>{_ID})$", re.I)
_DROP_MANY = re.compile(
    r"^drop (?P<kind>table|index|sequence) (?:concurrently )?(?:if exists )?(?P<names>.*?)(?: (?:cascade|restrict))?$",
    re.I | re.S,
)
_DATA = re.compile(r"^(?:insert|update|delete|truncate|copy|merge)\b", re.I)
_DATA_CTE = re.compile(r"^with\b.*\b(?:insert|update|delete)\b", re.I | re.S)
_NOISE = re.compile(
    r"^(?:begin|commit|end|rollback|start transaction|abort|set|reset|select|show|explain|values|"
    r"savepoint|release|declare|perform|raise|return|null|if|else|elsif|loop|exit|continue)\b",
    re.I,
)
_ADD = re.compile(rf"^add (?:column )?(?:if not exists )?(?P<col>{_ID})", re.I)
_DROPCOL = re.compile(rf"^drop (?:column )?(?:if exists )?(?P<col>{_ID})", re.I)
_RENAME_COL = re.compile(rf"^rename (?:column )?(?P<old>{_ID}) to (?P<new>{_ID})$", re.I)
_RENAME_TABLE = re.compile(rf"^rename to (?P<new>{_ID})$", re.I)
_NOT_A_COLUMN = {"constraint", "primary", "unique", "check", "foreign", "exclude"}

# Inside a DO body a schema statement is preceded by plpgsql control flow
# ("BEGIN IF NOT EXISTS (...) THEN CREATE TABLE ..."), so find the statement start.
_PLPGSQL_STMT_START = re.compile(
    r"\b(?:create (?:unique )?index|create (?:temp |temporary |unlogged )?table|create sequence|"
    r"alter table|drop (?:table|index|sequence)|insert into|update |delete from)\b",
    re.I,
)


def _split_top_level_commas(s: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _classify(text: str, conditional: bool, pm: ParsedMigration) -> bool:
    """Append effects for one statement. Returns True if the statement was recognised."""
    m = _CREATE_INDEX.match(text)
    if m:
        pm.effects.append(Effect("create", "index", _bare(m["name"]), _bare(m["table"]), conditional))
        return True
    m = _CREATE_TABLE.match(text)
    if m:
        if not m["temp"]:
            pm.effects.append(Effect("create", "table", _bare(m["name"]), None, conditional))
        return True
    m = _CREATE_SEQUENCE.match(text)
    if m:
        pm.effects.append(Effect("create", "sequence", _bare(m["name"]), None, conditional))
        return True
    m = _DROP_MANY.match(text)
    if m:
        kind = m["kind"].lower()
        for raw in m["names"].split(","):
            raw = raw.strip()
            if re.fullmatch(_QID, raw):
                pm.effects.append(Effect("drop", kind, _bare(raw), None, conditional))
        return True
    m = _ALTER_RENAME.match(text)
    if m:
        cond = conditional or bool(m["if_exists"])
        kind = m["kind"].lower()
        pm.effects.append(Effect("drop", kind, _bare(m["name"]), None, cond))
        pm.effects.append(Effect("create", kind, m["new"].lower(), None, cond))
        return True
    m = _ALTER_TABLE.match(text)
    if m:
        table = _bare(m["table"])
        # ALTER TABLE IF EXISTS: the author expected the table might be absent, so nothing it
        # adds can be required.
        conditional = conditional or bool(m["if_exists"])
        recognised = False
        for action in _split_top_level_commas(m["actions"]):
            a = _ADD.match(action)
            if a and a["col"].lower() not in _NOT_A_COLUMN:
                col = a["col"].lower()
                pm.effects.append(Effect("create", "column", f"{table}.{col}", table, conditional))
                recognised = True
                continue
            d = _DROPCOL.match(action)
            if d and d["col"].lower() not in _NOT_A_COLUMN:
                col = d["col"].lower()
                pm.effects.append(Effect("drop", "column", f"{table}.{col}", table, conditional))
                recognised = True
                continue
            r = _RENAME_COL.match(action)
            if r and r["old"].lower() not in _NOT_A_COLUMN:
                pm.effects.append(Effect("drop", "column", f"{table}.{r['old'].lower()}", table, conditional))
                pm.effects.append(Effect("create", "column", f"{table}.{r['new'].lower()}", table, conditional))
                recognised = True
                continue
            t = _RENAME_TABLE.match(action)
            if t:
                pm.effects.append(Effect("drop", "table", table, None, conditional))
                pm.effects.append(Effect("create", "table", t["new"].lower(), None, conditional))
                recognised = True
                continue
        if not recognised:
            pm.other_statements += 1
        return True
    if _DATA.match(text) or _DATA_CTE.match(text):
        pm.data_statements += 1
        return True
    return False


def _parse_into(sql: str, conditional: bool, pm: ParsedMigration) -> None:
    for st in split_statements(sql):
        text = st.text
        if conditional:
            # plpgsql: strip leading control flow down to the first SQL statement keyword.
            m = _PLPGSQL_STMT_START.search(text)
            if not m:
                continue
            text = text[m.start():]
        if re.match(r"^do\b", text, re.I):
            for body in st.bodies:
                _parse_into(body, True, pm)
            continue
        if re.match(r"^create (?:or replace )?function\b|^create (?:or replace )?procedure\b", text, re.I):
            pm.other_statements += 1  # function bodies do not run at migration time
            continue
        if _classify(text, conditional, pm):
            continue
        if not _NOISE.match(text):
            pm.other_statements += 1


def parse_migration(sql: str) -> ParsedMigration:
    pm = ParsedMigration()
    _parse_into(sql, False, pm)
    return pm


# --------------------------------------------------------------------------- markers

_SUPERSEDED = re.compile(r"--\s*ORION-MIGRATION-SUPERSEDED-BY:\s*(?P<file>\S+)", re.I)
_NOT_A_MIGRATION = re.compile(r"--\s*ORION-MIGRATION-NOT-A-MIGRATION:\s*(?P<why>.+)", re.I)
_ABSENT_OK = re.compile(r"--\s*ORION-MIGRATION-ABSENT-OK:[ \t]*(?P<obj>[^\s]*)[ \t]*(?P<why>[^\n]*)", re.I)
_HEADER_SCRIPT = re.compile(r"\bscripts/[\w./-]+\.(?:sh|py)\b")
_REQUIRED_BY = re.compile(r"^[ \t]*--[ \t]*ORION-MIGRATION-REQUIRED-BY:[ \t]*(?P<val>[^\n]*)$", re.I | re.M)
_REQUIRED_NONE = re.compile(r"^none\b[\s:;,()\-\u2013\u2014]*(?P<why>.*)$", re.I)
_DESTRUCTIVE = re.compile(r"^[ \t]*--[ \t]*DESTRUCTIVE\b", re.M)
_SERVICE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass
class RequiredBy:
    """Parsed ``ORION-MIGRATION-REQUIRED-BY`` marker(s) of one migration file."""
    services: tuple[str, ...] = ()
    none_reason: Optional[str] = None
    errors: list[str] = field(default_factory=list)


def parse_required_by(text: str) -> Optional[RequiredBy]:
    """None when the file declares nothing. Several marker lines are unioned."""
    matches = list(_REQUIRED_BY.finditer(text))
    if not matches:
        return None
    rb = RequiredBy()
    services: list[str] = []
    for m in matches:
        val = m["val"].strip()
        none = _REQUIRED_NONE.match(val)
        if none:
            why = none["why"].strip().rstrip(")").strip()
            if not why:
                rb.errors.append("REQUIRED-BY: none needs a reason after it")
            rb.none_reason = why or rb.none_reason
            continue
        names = [n.strip() for n in val.split(",") if n.strip()]
        if not names:
            rb.errors.append("REQUIRED-BY names no service (write 'none <reason>' if nothing needs it)")
        for n in names:
            if _SERVICE_NAME.match(n):
                services.append(n)
            else:
                rb.errors.append(f"REQUIRED-BY: {n!r} is not a service directory name")
    rb.services = tuple(dict.fromkeys(services))
    if rb.services and rb.none_reason is not None:
        rb.errors.append("REQUIRED-BY says both 'none' and names services")
    return rb


def is_destructive(text: str) -> bool:
    """A ``-- DESTRUCTIVE`` header line: operator-approved only, never a deploy dependency."""
    return bool(_DESTRUCTIVE.search(text))


def required_for(files: Iterable[MigrationFile], service: str) -> list[MigrationFile]:
    """Files a deploy of ``service`` needs applied. DESTRUCTIVE and NOT-A-MIGRATION files never
    count, whatever they declare."""
    out = []
    for f in files:
        if is_destructive(f.text) or _NOT_A_MIGRATION.search(f.text):
            continue
        rb = parse_required_by(f.text)
        if rb and service in rb.services:
            out.append(f)
    return out


@dataclass
class DeployGate:
    service: str
    required: list[str]
    blocking: list[FileResult]       # MISSING / INVALID: refuse the deploy
    unverifiable: list[FileResult]   # DATA / UNKNOWN / CONDITIONAL: the schema cannot say; warn only

    @property
    def ok(self) -> bool:
        return not self.blocking


def deploy_gate(report: DriftReport, required: Iterable[str], service: str) -> DeployGate:
    """Window-free verdict for the files ``service`` requires: a requirement does not age out."""
    names = list(dict.fromkeys(required))
    by_name = {f.name: f for f in report.files}
    blocking, unverifiable = [], []
    for n in names:
        r = by_name.get(n)
        if r is None:
            continue
        if r.broken:
            blocking.append(r)
        elif r.status in ("DATA", "UNKNOWN", "CONDITIONAL"):
            unverifiable.append(r)
    return DeployGate(service=service, required=names, blocking=blocking, unverifiable=unverifiable)


# --------------------------------------------------------------------------- replay


@dataclass
class MigrationFile:
    name: str                    # basename
    text: str
    added_at: datetime           # first commit on the ref (replay order)
    changed_at: datetime         # last commit on the ref (recency window)


@dataclass
class LiveState:
    tables: set[str]
    columns: set[str]            # "table.column"
    indexes: dict[str, bool]     # name -> indisvalid
    sequences: set[str]

    def has(self, kind: str, name: str) -> bool:
        if kind == "table":
            return name in self.tables
        if kind == "column":
            return name in self.columns
        if kind == "index":
            return name in self.indexes
        if kind == "sequence":
            return name in self.sequences
        raise ValueError(kind)


@dataclass
class ObjectFinding:
    kind: str
    name: str
    table: Optional[str]
    status: str        # missing | invalid | drop_not_applied
    file: str          # blamed migration
    detail: str = ""


@dataclass
class FileResult:
    name: str
    in_window: bool
    changed_at: datetime
    status: str        # APPLIED | MISSING | INVALID | DATA | UNKNOWN | SUPERSEDED | SKIPPED | CONDITIONAL
    problems: list[ObjectFinding] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    objects_checked: int = 0
    data_statements: int = 0
    marker_error: Optional[str] = None
    header_script: Optional[str] = None
    sticky: bool = False   # already carded: stays red past the window until actually applied

    @property
    def broken(self) -> bool:
        return self.status in ("MISSING", "INVALID")

    @property
    def red(self) -> bool:
        return self.broken and (self.in_window or self.sticky)

    @property
    def key(self) -> str:
        return f"migration:{self.name}"

    def apply_command(self, repo_rel_dir: str = str(MIGRATION_SUBDIR)) -> str:
        cmd = (f"docker exec -i {SQL_DB_CONTAINER} psql -U postgres -d conjourney -v ON_ERROR_STOP=1 "
               f"< {repo_rel_dir}/{self.name}")
        if self.header_script:
            return f"{self.header_script} (the file's header says to apply it through this; raw form: {cmd})"
        return cmd

    def summary(self) -> str:
        if self.marker_error:
            return f"{self.name}: {self.marker_error}"
        objs = ", ".join(
            f"{p.kind} {p.name}" + (" (present but INVALID)" if p.status == "invalid" else "")
            + (" (should have been dropped)" if p.status == "drop_not_applied" else "")
            for p in self.problems
        )
        return f"{self.name}: not applied to the live database -- {objs}"


@dataclass
class DriftReport:
    files: list[FileResult]
    window_days: Optional[int]

    @property
    def red_files(self) -> list[FileResult]:
        return [f for f in self.files if f.red]

    @property
    def red(self) -> bool:
        return bool(self.red_files)

    def red_keys(self) -> list[str]:
        return sorted(f.key for f in self.red_files)

    def green_keys(self) -> list[str]:
        """Files verifiably fine this tick (any window): forgetting a delivered key needs proof
        it was applied, never merely that it aged out."""
        return sorted(f.key for f in self.files if not f.broken)

    def old_broken(self) -> list[FileResult]:
        """Drift outside the window that was never carded: reported, not alarmed."""
        return [f for f in self.files if f.broken and not f.red]

    def verify_manually(self) -> list[FileResult]:
        return [f for f in self.files if f.in_window and f.status == "DATA"]


def evaluate(
    files: list[MigrationFile],
    live: LiveState,
    *,
    now: Optional[datetime] = None,
    window_days: Optional[int] = DEFAULT_WINDOW_DAYS,
    sticky_keys: Iterable[str] = (),
) -> DriftReport:
    """Replay the corpus in commit order, diff the expected schema against ``live``."""
    now = now or datetime.now(timezone.utc)
    cutoff = None if not window_days else now - timedelta(days=window_days)
    names = {f.name for f in files}

    parsed: dict[str, ParsedMigration] = {}
    skipped: dict[str, str] = {}
    superseded: dict[str, str] = {}
    marker_errors: dict[str, str] = {}
    absent_ok: dict[tuple[str, str], str] = {}
    header_scripts: dict[str, str] = {}
    for f in files:
        nam = _NOT_A_MIGRATION.search(f.text)
        if nam:
            skipped[f.name] = nam["why"].strip()
            continue
        sup = _SUPERSEDED.search(f.text)
        if sup:
            target = Path(sup["file"].strip()).name
            if target in names:
                superseded[f.name] = target
            else:
                marker_errors[f.name] = f"SUPERSEDED-BY names {target!r}, which does not exist"
        for m in _ABSENT_OK.finditer(f.text):
            obj, why = m["obj"].strip().lower(), m["why"].strip()
            if not obj or not why:
                marker_errors[f.name] = "ABSENT-OK marker needs both an object name and a reason"
                continue
            for kind in KINDS:
                absent_ok[(kind, obj)] = f"{f.name}: {why}"
        header = "\n".join(line for line in f.text.splitlines()[:60] if line.lstrip().startswith("--"))
        hs = _HEADER_SCRIPT.search(header)
        if hs:
            header_scripts[f.name] = hs.group(0)
        parsed[f.name] = parse_migration(f.text)

    # Replay. final[(kind,name)] = (op, seq, file, conditional, table)
    ordered = sorted((f for f in files if f.name in parsed), key=lambda f: (f.added_at, f.name))
    final: dict[tuple[str, str], tuple[str, int, str, bool, Optional[str]]] = {}
    table_drops: dict[str, list[tuple[int, str]]] = {}
    seq = 0
    for f in ordered:
        for e in parsed[f.name].effects:
            seq += 1
            key = (e.kind, e.name)
            if e.conditional and key in final and not final[key][3]:
                # A guarded (DO-block IF / ALTER ... IF EXISTS) effect may not have run, so it
                # cannot overturn an unconditional expectation either way.
                continue
            final[key] = (e.op, seq, f.name, e.conditional, e.table)
            if e.kind == "table" and e.op == "drop" and not e.conditional:
                table_drops.setdefault(e.name, []).append((seq, f.name))

    problems: dict[str, list[ObjectFinding]] = {}
    conditional_notes: dict[str, list[str]] = {}
    dropped_later_by: dict[tuple[str, str], str] = {}
    for (kind, name), (op, s, blamed, conditional, table) in final.items():
        expect_present = op == "create"
        if expect_present and table and any(ds > s for ds, _ in table_drops.get(table, [])):
            # The whole table was dropped after this column/index was declared.
            expect_present = False
            blamed = max(table_drops[table])[1]
        if not expect_present:
            dropped_later_by[(kind, name)] = blamed
        present = live.has(kind, name)
        finding: Optional[ObjectFinding] = None
        if expect_present and not present:
            finding = ObjectFinding(kind, name, table, "missing", blamed)
        elif expect_present and kind == "index" and not live.indexes.get(name, True):
            finding = ObjectFinding(
                kind, name, table, "invalid", blamed,
                "exists but indisvalid=false -- an interrupted CREATE INDEX CONCURRENTLY; "
                "IF NOT EXISTS will not rebuild it. DROP INDEX and re-run.",
            )
        elif not expect_present and present and op == "drop":
            finding = ObjectFinding(kind, name, table, "drop_not_applied", blamed)
        if finding is None:
            continue
        if conditional:
            conditional_notes.setdefault(blamed, []).append(
                f"{kind} {name} is {finding.status.replace('_', ' ')}, but it sits inside a DO block "
                "guarded by IF -- verify manually"
            )
            continue
        if (kind, name) in absent_ok and finding.status == "missing":
            continue
        problems.setdefault(blamed, []).append(finding)

    sticky = set(sticky_keys)
    results: list[FileResult] = []
    for f in sorted(files, key=lambda f: f.name):
        in_window = cutoff is None or f.changed_at >= cutoff
        r = FileResult(name=f.name, in_window=in_window, changed_at=f.changed_at, status="APPLIED")
        r.sticky = f"migration:{f.name}" in sticky
        if f.name in skipped:
            r.status = "SKIPPED"
            r.info.append(skipped[f.name])
            results.append(r)
            continue
        pm = parsed[f.name]
        r.header_script = header_scripts.get(f.name)
        r.objects_checked = len({(e.kind, e.name) for e in pm.effects})
        r.data_statements = pm.data_statements
        r.problems = sorted(problems.get(f.name, []), key=lambda p: (p.kind, p.name))
        r.info += conditional_notes.get(f.name, [])
        if f.name in marker_errors:
            r.marker_error = marker_errors[f.name]
            r.status = "MISSING"
        elif r.problems and f.name in superseded:
            r.status = "SUPERSEDED"
            r.info.append(f"superseded by {superseded[f.name]}")
            r.problems = []
        elif any(p.status == "invalid" for p in r.problems):
            r.status = "INVALID"
        elif r.problems:
            r.status = "MISSING"
        elif not pm.effects:
            r.status = "DATA" if pm.data_statements else "UNKNOWN"
        elif _all_dropped_elsewhere(f.name, pm, dropped_later_by):
            # Everything this file created was intentionally dropped by a later migration.
            r.status = "SUPERSEDED"
            r.info.append("everything it created was dropped later by "
                          + ", ".join(sorted({dropped_later_by[(e.kind, e.name)] for e in pm.effects
                                              if e.op == "create"})))
        elif all(e.conditional for e in pm.effects):
            r.status = "CONDITIONAL"
        if r.status == "APPLIED":
            later = sorted({dropped_later_by[(e.kind, e.name)] for e in pm.effects
                            if e.op == "create" and dropped_later_by.get((e.kind, e.name)) not in (None, f.name)})
            if later:
                r.info.append("some of what it created was intentionally dropped later by " + ", ".join(later))
        if pm.effects and pm.data_statements and r.status == "APPLIED":
            r.info.append(f"+{pm.data_statements} data statement(s) the schema cannot confirm")
        results.append(r)
    return DriftReport(files=results, window_days=window_days)


def _all_dropped_elsewhere(name: str, pm: ParsedMigration, dropped_later_by: dict[tuple[str, str], str]) -> bool:
    creates = {(e.kind, e.name) for e in pm.effects if e.op == "create"}
    return bool(creates) and all(dropped_later_by.get(k) not in (None, name) for k in creates)


# ------------------------------------------------------------------------- IO helpers


def commit_times(repo: Path, ref: str = "HEAD", subdir: Path = MIGRATION_SUBDIR) -> dict[str, tuple[datetime, datetime]]:
    """basename -> (first landed, last changed) on ``ref``. One git call (~0.2s).

    ``--first-parent --diff-merges=first-parent``: this repo merges PRs with merge commits, and a
    plain ``git log`` reports the feature-branch commit time -- up to 10 days before the file
    reached main on the real corpus, which would shrink or erase its alarm window and misorder
    the replay. First-parent times mean "when it landed on ``ref``"."""
    out = subprocess.run(
        ["git", "-C", str(repo), "log", "--first-parent", "--diff-merges=first-parent",
         "--format=@%ct", "--name-only", ref, "--", str(subdir)],
        capture_output=True, text=True, check=True, timeout=60,
    ).stdout
    times: dict[str, tuple[datetime, datetime]] = {}
    ts: Optional[datetime] = None
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("@"):
            ts = datetime.fromtimestamp(int(line[1:]), tz=timezone.utc)
            continue
        if ts is None:
            continue
        base = Path(line).name
        if base in times:
            times[base] = (ts, times[base][1])   # log is newest-first: keep pushing "first" back
        else:
            times[base] = (ts, ts)
    return times


def is_rollback_file(name: str) -> bool:
    """A ``*_rollback.sql`` file undoes another migration and is applied only when backing that
    migration out, so it is never part of the schema the repo expects. Replaying it made the
    watch demand the forward migration's objects be dropped (RED on every run, 2026-10-04..07)."""
    return name.endswith("_rollback.sql")


def load_files(repo: Path, times: dict[str, tuple[datetime, datetime]], now: Optional[datetime] = None,
               *, include_uncommitted: bool = True) -> list[MigrationFile]:
    """Every *.sql in the migration dir. A file git has never seen on the ref (uncommitted) is
    treated as brand new -- newest in replay order and inside any window -- unless
    ``include_uncommitted`` is False (the watch: it alarms on MERGED migrations only)."""
    now = now or datetime.now(timezone.utc)
    files = []
    for p in sorted((repo / MIGRATION_SUBDIR).glob(MIGRATION_GLOB)):
        if is_rollback_file(p.name):
            continue
        if p.name not in times and not include_uncommitted:
            continue
        added, changed = times.get(p.name, (now, now))
        files.append(MigrationFile(p.name, p.read_text(errors="replace"), added, changed))
    return files


# pg_catalog, not information_schema: information_schema filters by the caller's privileges, so a
# non-superuser DSN would hide tables/columns and every one would read as MISSING.
LIVE_STATE_SQL = {
    "tables": "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
              "WHERE c.relkind IN ('r','p','v','m','f') "
              "AND n.nspname NOT IN ('pg_catalog','information_schema') AND n.nspname NOT LIKE 'pg_toast%'",
    "columns": "SELECT c.relname, a.attname FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
               "JOIN pg_namespace n ON n.oid = c.relnamespace "
               "WHERE a.attnum > 0 AND NOT a.attisdropped AND c.relkind IN ('r','p','v','m','f') "
               "AND n.nspname NOT IN ('pg_catalog','information_schema')",
    "indexes": "SELECT c.relname, i.indisvalid FROM pg_class c JOIN pg_index i ON i.indexrelid = c.oid "
               "JOIN pg_namespace n ON n.oid = c.relnamespace "
               "WHERE n.nspname NOT IN ('pg_catalog','information_schema')",
    "sequences": "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                 "WHERE c.relkind = 'S' AND n.nspname NOT IN ('pg_catalog','information_schema')",
}


def load_live_state(conn) -> LiveState:
    """Read-only catalog reads. ``conn`` is any DB-API connection (psycopg2)."""
    with conn.cursor() as cur:
        cur.execute(LIVE_STATE_SQL["tables"])
        tables = {r[0].lower() for r in cur.fetchall()}
        cur.execute(LIVE_STATE_SQL["columns"])
        columns = {f"{r[0].lower()}.{r[1].lower()}" for r in cur.fetchall()}
        cur.execute(LIVE_STATE_SQL["indexes"])
        indexes = {r[0].lower(): bool(r[1]) for r in cur.fetchall()}
        cur.execute(LIVE_STATE_SQL["sequences"])
        sequences = {r[0].lower() for r in cur.fetchall()}
    return LiveState(tables=tables, columns=columns, indexes=indexes, sequences=sequences)


def check_repo(conn, repo: Path, *, ref: str = "HEAD", window_days: Optional[int] = DEFAULT_WINDOW_DAYS,
               now: Optional[datetime] = None, sticky_keys: Iterable[str] = (),
               include_uncommitted: bool = False) -> DriftReport:
    times = commit_times(repo, ref)
    files = load_files(repo, times, now, include_uncommitted=include_uncommitted)
    return evaluate(files, load_live_state(conn), now=now, window_days=window_days, sticky_keys=sticky_keys)


def alert_lines(report: DriftReport) -> list[str]:
    if not report.red_files:
        return []
    lines = ["A merged SQL migration has not been applied to the live database:"]
    for f in report.red_files:
        lines.append(f"- {f.summary()}")
        lines.append(f"  apply: {f.apply_command()}")
    lines.append("Services that read these objects run degraded or crash until it is applied.")
    return lines


def iter_human(report: DriftReport, quiet: bool = False) -> Iterable[str]:
    for f in report.files:
        if quiet and not f.red and not f.broken and not (f.in_window and f.status in ("DATA", "CONDITIONAL")):
            continue
        if not f.in_window and not quiet and f.status in ("APPLIED", "SUPERSEDED", "SKIPPED", "UNKNOWN"):
            continue
        label = {"DATA": "verify manually"}.get(f.status, f.status.lower())
        mark = "RED " if f.red else ("old " if not f.in_window else "    ")
        yield f"{mark}[{label:>15}] {f.name}  (last changed {f.changed_at:%Y-%m-%d})"
        if f.marker_error:
            yield f"            {f.marker_error}"
        for p in f.problems:
            where = f" on {p.table}" if p.table and p.kind != "column" else ""
            yield f"            {p.status.upper()} {p.kind} {p.name}{where}"
            if p.detail:
                yield f"              {p.detail}"
        for line in f.info:
            yield f"            note: {line}"
        if f.red:
            yield f"            apply: {f.apply_command()}"
        if f.status == "DATA" and f.in_window:
            yield "            data-only migration: the schema cannot show whether it ran -- verify manually"

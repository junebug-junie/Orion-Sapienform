#!/usr/bin/env python3
"""Apply the hand-written SQL migrations a `git pull` brought in, before the rebuilds run.

Why: ``scripts/safe_docker_build.sh <svc> up`` refuses a service while a migration it declares
(``-- ORION-MIGRATION-REQUIRED-BY: <svc>``) is unapplied. A pull that carries a new migration
starts the post-merge rebuilds at once, so the consumers were refused before anyone could apply
it (Juniper, 2026-10-15: "the second I git pull i cant run that migration ... happens all the
time"). She asked for this step to run inside the pull, before the rebuilds.

``scripts/rebuild_services_from_git_diff.sh`` calls this with the changed
``services/orion-sql-db/*.sql`` paths.

WHERE IT RUNS -- the same place a live deploy may run from, and only there:
  * the primary checkout (not a linked worktree), on ``main``, clean, HEAD == ``origin/main``
    (safe_docker_build.sh's deploy rule; a pull on a feature branch must never write to the
    live database). ``--dry-run`` skips this check: it writes nothing.
  * a host with the database: the classifier lists migrations only where the host's service list
    includes orion-sql-db (circe gets none), and the default apply command execs into the local
    ``orion-athena-sql-db`` container, which must be running (this is what stops hosts with no
    service list, e.g. hecate).

WHAT IT APPLIES -- a changed file the drift checker (orion/sql_migration_drift.py, the same check
the deploy gate runs) reads as NOT APPLIED (MISSING), and only when ALL of these hold:
  * not a ``*_rollback.sql`` file, no ``-- DESTRUCTIVE`` header line, no NOT-A-MIGRATION marker;
  * a valid ``-- ORION-MIGRATION-REQUIRED-BY:`` header (services or ``none <reason>``);
  * ``drift.auto_apply_refusals`` is empty: every statement is on the additive-DDL allowlist
    (CREATE TABLE/INDEX/SEQUENCE, ALTER TABLE ADD COLUMN/ADD CONSTRAINT/SET DEFAULT, COMMENT ON,
    INSERT ... VALUES ... ON CONFLICT DO NOTHING, own balanced BEGIN/COMMIT) and the rest of it is
    inert: no function call that could run per existing row, no GENERATED, CASCADE, EXECUTE,
    SELECT, SET, quoted identifier, DO/function body or psql meta-command; and on a table that
    already exists live, indexes are CONCURRENTLY and constraints NOT VALID (no long write lock);
  * its header does not name a script to apply it through (``-- ORION-MIGRATION-APPLY-WITH:``, or
    an apply verb right before the path; a script merely mentioned in prose does not count),
    and no marker error.
Every other file is left alone with the reason. APPLIED files are skipped. DATA/CONDITIONAL files
(the schema cannot show whether they ran) are never auto-applied.

ORDER AND STOPS: filename order. A file that needs a human -- refused, DESTRUCTIVE, or one whose
live state cannot be verified -- holds every later file in the set, and so does the first failed
apply. Rollback, APPLIED and superseded files hold nothing.

HOW: ``psql -X -v ON_ERROR_STOP=1`` with ``lock_timeout=10s`` (a pg_dump backup can hold locks for
minutes; fail fast rather than hang the pull) and a 15 min statement_timeout backstop, inside
``--single-transaction`` unless the file has CREATE INDEX CONCURRENTLY or its own BEGIN/COMMIT.
The exact text that was scanned is what psql receives. After applying, the live state is read
again; a file only counts as applied when the checker agrees. Every decision that is not a plain
skip is appended to the durable log.

BACKUPS (incident 2026-10-17 ~03:56 UTC: the nightly pg_dump held AccessShareLocks for ~45 min, the
apply of manual_migration_vision_motion_story_strip_v2.sql hit lock_timeout, and its consumers stayed
refused until it was applied by hand). lock_timeout is never raised. Instead:
  * before each file, a read-only look at pg_stat_activity for a backup session
    (application_name pg_dump / pg_dumpall / pg_basebackup, on this database). While one runs,
    wait: poll every 30 s with a progress line, at most ORION_MIGRATION_BACKUP_MAX_WAIT_S in
    total for the run (default 5400 = 90 min; ``--max-backup-wait``). Past that the file fails and
    holds the rest, as a failed apply does. If the look itself fails, apply anyway (lock_timeout
    still bounds it), as before this step existed.
  * while psql runs, a watcher reads pg_blocking_pids() for the apply session (application_name
    ``orion-post-pull-migrate-<pid>``). A lock_timeout failure is retried (3 times, backoff
    30/60/120 s, each after waiting for the backup again) ONLY when every blocker seen was a backup
    and the file ran in a single transaction (a failure rolled all of it back). Any other failure,
    including a lock_timeout behind a non-backup session, fails and holds as before.
  * ``--defer-if-backup`` (the rebuild script's first pass) never waits: a file that would wait or
    retry is DEFERRED with every later file, exit 3. The rebuild script then rebuilds everything
    else, runs this again without the flag (the wait), and re-runs the rebuild of the deferred
    files' consumers that failed (the deploy gate refused them) once the files are applied.
Every wait, retry and deferral is appended to the durable log.

ENV
    ORION_POST_PULL_APPLY_MIGRATIONS=0   kill switch (default on); nothing is applied, report only
    ORION_APPLIED_MIGRATIONS_LOG         log path (default ~/.orion/applied_migrations.log)
    ORION_MIGRATION_APPLY_PSQL           psql command prefix replacing the docker exec (tests point
                                         it at a scratch server; it also skips the container check)
    ORION_PG_HOST/PORT/USER/PASSWORD/DB  read-only live-state connection (as the checker); USER and
                                         DB also pick the role/database the default command writes
    ORION_MIGRATION_BACKUP_MAX_WAIT_S    longest total wait for a running backup (default 5400);
                                         ``--max-backup-wait`` overrides it

Exit: 0 = nothing left that this step should have applied; 1 = a file was refused, held, failed,
or did not verify; 2 = not allowed here, or live state UNKNOWN (no answer is not "applied");
3 = (``--defer-if-backup`` only) nothing failed, but a backup is running and files were deferred. The
caller runs the rebuilds regardless; the deploy gate still refuses every service whose migration
is not applied.
"""
from __future__ import annotations

import argparse
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from orion import sql_migration_drift as drift  # noqa: E402
from check_sql_migrations_applied import open_connection  # noqa: E402

TAG = "post-pull migrations"
KILL_SWITCH = "ORION_POST_PULL_APPLY_MIGRATIONS"
LOCK_TIMEOUT = "10s"
STATEMENT_TIMEOUT = "15min"
PGOPTIONS = f"-c lock_timeout={LOCK_TIMEOUT} -c statement_timeout={STATEMENT_TIMEOUT}"
APPLY_TIMEOUT_S = 20 * 60

# A running backup (2026-10-17 incident). Never raise lock_timeout; wait for the backup instead.
BACKUP_APP_NAMES = ("pg_dump", "pg_dumpall", "pg_basebackup")
MAX_WAIT_ENV = "ORION_MIGRATION_BACKUP_MAX_WAIT_S"
DEFAULT_MAX_WAIT_S = 90 * 60
BACKUP_POLL_S = 30.0
RETRY_BACKOFF_S = (30.0, 60.0, 120.0)   # one entry per retry after a backup-blocked lock timeout
WATCH_INTERVAL_S = 1.0                  # blocker watcher poll; lock_timeout gives it ~10 polls
APP_NAME = f"orion-post-pull-migrate-{os.getpid()}"
LOCK_TIMEOUT_TEXT = "canceling statement due to lock timeout"
EXIT_DEFERRED = 3

# Read-only. Backup sessions that can hold locks in this database (pg_basebackup's walsender has no
# datname). Its own session is excluded.
BACKUP_SQL = (
    "SELECT pid, application_name, "
    "coalesce(extract(epoch FROM now() - backend_start)::bigint, 0) "
    "FROM pg_stat_activity WHERE application_name = ANY(%s) AND pid <> pg_backend_pid() "
    "AND (datname IS NULL OR datname = current_database()) ORDER BY pid"
)
# Read-only. Who blocks the apply session right now (its application_name is APP_NAME).
BLOCKER_SQL = (
    "SELECT DISTINCT b.pid, coalesce(b.application_name, '') "
    "FROM pg_stat_activity w CROSS JOIN LATERAL unnest(pg_blocking_pids(w.pid)) AS bp(pid) "
    "JOIN pg_stat_activity b ON b.pid = bp.pid WHERE w.application_name = %s"
)


@dataclass
class Item:
    rel: str
    name: str
    text: str
    action: str = "skip"         # apply | skip | refuse | operator
    reason: str = ""
    manual: str = ""             # manual command to print, "" when none is safe to print
    holds: bool = False          # a human must act on this before any later file is applied
    no_txn: list[str] = field(default_factory=list)
    outcome: str = ""            # applied | failed | held | would-apply | unverified


def enabled(env=os.environ) -> bool:
    return env.get(KILL_SWITCH, "1").strip().lower() not in ("0", "false", "no", "off")


def log_path(env=os.environ) -> Path:
    return Path(env.get("ORION_APPLIED_MIGRATIONS_LOG") or Path.home() / ".orion" / "applied_migrations.log")


def max_backup_wait(env=os.environ) -> float:
    raw = env.get(MAX_WAIT_ENV, "").strip()
    try:
        return max(0.0, float(raw)) if raw else float(DEFAULT_MAX_WAIT_S)
    except ValueError:
        return float(DEFAULT_MAX_WAIT_S)


def default_psql(env=os.environ) -> list[str]:
    """docker exec into the database container, writing the same role/database the read uses."""
    return ["docker", "exec", "-i", "-e", f"PGOPTIONS={PGOPTIONS}", "-e", f"PGAPPNAME={APP_NAME}",
            drift.SQL_DB_CONTAINER, "psql", "-U", env.get("ORION_PG_USER", "postgres"), "-d", env.get("ORION_PG_DB", "conjourney")]


def psql_prefix(env=os.environ) -> list[str]:
    raw = env.get("ORION_MIGRATION_APPLY_PSQL", "").strip()
    return shlex.split(raw) if raw else default_psql(env)


def classify(rel: str, text: str, result: drift.FileResult | None,
             live_tables: set[str] | None = None) -> Item:
    """Decide what to do with one changed file. Pure; see the module docstring for the rules."""
    name = Path(rel).name
    it = Item(rel=rel, name=name, text=text)
    cmd = result.apply_command() if result is not None else ""
    if drift.is_rollback_file(name):
        it.action, it.reason = "operator", "rollback file: undoes another migration; never applied on pull"
        return it
    if drift.is_destructive(text):
        it.action, it.reason = "operator", "-- DESTRUCTIVE file: operator-approved only; read it and decide by hand"
        it.holds = True
        return it
    if drift._NOT_A_MIGRATION.search(text):
        it.reason = "NOT-A-MIGRATION marker"
        return it
    if result is None:
        it.action, it.reason = "refuse", "the checker did not load this file; cannot tell its live state"
        return it
    if result.status == "APPLIED":
        it.reason = "APPLIED"
        return it
    if result.status in ("SUPERSEDED", "SKIPPED"):
        it.reason = result.status
        return it
    if result.status in ("DATA", "CONDITIONAL", "UNKNOWN"):
        it.reason = (f"UNVERIFIABLE ({result.status.lower()}: the schema cannot show whether it ran); "
                     "never auto-applied -- apply by hand if it has not run")
        it.manual, it.holds = cmd, True
        return it
    # MISSING / INVALID from here: the deploy gate refuses its consumers until it is fixed.
    it.manual = cmd
    if result.status == "INVALID":
        it.action, it.reason = "refuse", "INVALID index (interrupted CONCURRENTLY build): " + result.summary()
        return it
    if result.marker_error:
        it.action, it.reason = "refuse", f"marker error: {result.marker_error}"
        return it
    rb = drift.parse_required_by(text)
    if rb is None:
        it.action, it.reason = "refuse", "no -- ORION-MIGRATION-REQUIRED-BY: header"
        return it
    if rb.errors:
        it.action, it.reason = "refuse", "invalid REQUIRED-BY header: " + "; ".join(rb.errors)
        return it
    refusals = drift.auto_apply_refusals(text, live_tables)
    if refusals:
        it.action, it.reason = "refuse", "not safe to auto-apply: " + "; ".join(refusals)
        return it
    if result.header_script:
        it.action, it.reason = "refuse", f"its header says to apply it through {result.header_script}"
        return it
    it.action, it.reason = "apply", "NOT APPLIED: " + result.summary()
    it.no_txn = drift.transaction_incompatible(text)
    return it


def _git(repo: Path, *args: str) -> str | None:
    """stdout, or None when git failed (a failure must never read as "clean")."""
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def checkout_problem(repo: Path) -> str:
    """Why this checkout may not write to the live database ("" = it may). Same rule as a live
    deploy in safe_docker_build.sh: primary checkout, main, clean, synchronized with origin/main
    (as of the fetch the pull just did)."""
    git_dir = _git(repo, "rev-parse", "--absolute-git-dir")
    common = _git(repo, "rev-parse", "--git-common-dir")
    if common and not os.path.isabs(common):  # relative to the -C directory
        common = str((repo / common).resolve())
    if not git_dir or not common or Path(git_dir).resolve() != Path(common).resolve():
        return "this is a linked worktree, not the primary checkout"
    branch = _git(repo, "branch", "--show-current")
    if branch != "main":
        return f"branch is {branch or '(detached)'!r}, not main"
    status = _git(repo, "status", "--porcelain")
    if status is None:
        return "git status failed"
    if status:
        return "the checkout has uncommitted changes"
    head, origin = _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "--verify", "origin/main")
    if not origin:
        return "origin/main is unknown"
    if head != origin:
        return f"main ({head[:9]}) is not origin/main ({origin[:9]})"
    return ""


def running_backups(connect_fn) -> list[tuple[int, str, int]]:
    """(pid, application_name, seconds since its session started) of every backup session on this
    database. Raises when the database cannot be asked."""
    conn = connect_fn()
    try:
        with conn.cursor() as cur:
            cur.execute(BACKUP_SQL, (list(BACKUP_APP_NAMES),))
            return [(int(r[0]), str(r[1]), int(r[2] or 0)) for r in cur.fetchall()]
    finally:
        conn.close()


class BlockerWatch:
    """While psql runs, record who blocks the apply session (pg_blocking_pids). Best effort: a
    watcher that cannot connect sees nobody, and nobody is not a backup -- the failure stands."""

    def __init__(self, connect_fn, interval: float = WATCH_INTERVAL_S):
        self.connect_fn, self.interval = connect_fn, interval
        self.blockers: dict[int, str] = {}
        self.error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="migration-blocker-watch", daemon=True)

    def _run(self) -> None:
        try:
            conn = self.connect_fn()
        except Exception as exc:  # noqa: BLE001
            self.error = " ".join(str(exc).split()) or type(exc).__name__
            return
        try:
            while True:
                with conn.cursor() as cur:
                    cur.execute(BLOCKER_SQL, (APP_NAME,))
                    for pid, name in cur.fetchall():
                        self.blockers[int(pid)] = str(name)
                if self._stop.wait(self.interval):
                    return
        except Exception as exc:  # noqa: BLE001
            self.error = " ".join(str(exc).split()) or type(exc).__name__
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)
        return False

    def only_backups(self) -> bool:
        return bool(self.blockers) and all(n in BACKUP_APP_NAMES for n in self.blockers.values())

    def describe(self) -> str:
        if not self.blockers:
            return "no blocker seen" + (f" (watcher: {self.error})" if self.error else "")
        return ", ".join(f"{n or '?'} pid={p}" for p, n in sorted(self.blockers.items()))


def apply_one(it: Item, prefix: list[str], env=os.environ) -> tuple[bool, str]:
    """Run one file's scanned text through psql. (ok, combined output tail)."""
    cmd = list(prefix) + ["-X", "-q", "-v", "ON_ERROR_STOP=1"]
    if not it.no_txn:
        cmd.append("--single-transaction")
    cmd += ["-f", "-"]
    run_env = dict(env)
    run_env["PGOPTIONS"] = PGOPTIONS  # a local psql reads it; the docker form passes it with -e
    run_env["PGAPPNAME"] = APP_NAME   # so the blocker watcher can find this session
    try:
        proc = subprocess.run(cmd, input=it.text, capture_output=True, text=True, env=run_env,
                              timeout=APPLY_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, f"psql did not finish within {APPLY_TIMEOUT_S}s"
    except OSError as exc:
        return False, f"could not run {cmd[0]}: {exc}"
    out = "\n".join(x for x in (proc.stdout.strip(), proc.stderr.strip()) if x)
    tail = "\n".join(out.splitlines()[-12:])
    return proc.returncode == 0, tail


def write_log(path: Path, line: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        print(f"{TAG}: could not write {path}: {exc}", file=sys.stderr)


def container_running(env=os.environ) -> tuple[bool, str]:
    """Default prefix only: is the database container up on THIS host, and is it the database the
    checker reads (ORION_PG_HOST local)? Otherwise the write and the verification could be two
    different databases."""
    if env.get("ORION_MIGRATION_APPLY_PSQL", "").strip():
        return True, ""
    read_host = env.get("ORION_PG_HOST", "localhost").strip()
    if read_host not in ("", "localhost", "127.0.0.1", "::1"):
        return False, (f"ORION_PG_HOST={read_host} is not this host, but the apply command writes the "
                       f"local {drift.SQL_DB_CONTAINER}; refusing to write one database and verify another")
    try:
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", drift.SQL_DB_CONTAINER],
                           capture_output=True, text=True, timeout=20, env=dict(env))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"cannot ask docker about {drift.SQL_DB_CONTAINER}: {exc}"
    if r.returncode != 0 or r.stdout.strip() != "true":
        return False, f"{drift.SQL_DB_CONTAINER} is not running on this host"
    return True, ""


def read_report(repo: Path, ref: str, connect_fn):
    times = drift.commit_times(repo, ref)
    files = drift.load_files(repo, times)
    conn = connect_fn()
    try:
        live = drift.load_live_state(conn)
    finally:
        conn.close()
    return files, drift.evaluate(files, live, window_days=None), live


def run(paths: list[str], *, repo: Path = REPO_ROOT, ref: str = "HEAD", dry_run: bool = False,
        env=os.environ, connect_fn=None, out=print, defer_if_backup: bool = False,
        max_wait_s: float | None = None, sleep=time.sleep, clock=time.monotonic,
        poll_s: float = BACKUP_POLL_S, backoff_s: tuple[float, ...] = RETRY_BACKOFF_S,
        watch_interval_s: float = WATCH_INTERVAL_S) -> int:
    connect_fn = connect_fn or open_connection
    max_wait_s = max_backup_wait(env) if max_wait_s is None else max(0.0, max_wait_s)
    rels = sorted(dict.fromkeys(p for p in paths if (repo / p).is_file()), key=lambda p: Path(p).name)
    if not rels:
        return 0
    if not enabled(env):
        out(f"{TAG}: auto-apply OFF ({KILL_SWITCH}=0) -- nothing applied; see the report below")
        return 0
    if not dry_run:
        why = checkout_problem(repo)
        if why:
            out(f"{TAG}: not applying anything -- {why}. Only the clean primary main checkout, "
                "synchronized with origin/main, writes to the live database (the deploy rule).")
            return 2
    try:
        files, report, live = read_report(repo, ref, connect_fn)
    except Exception as exc:  # noqa: BLE001 -- any failure is UNKNOWN, never a verdict
        out(f"{TAG}: live state UNKNOWN ({' '.join(str(exc).split()) or type(exc).__name__}); "
            "applying nothing -- no answer is not 'applied'")
        return 2
    by_name = {f.name: f for f in report.files}
    texts = {f.name: f.text for f in files}
    items = [classify(rel, texts.get(Path(rel).name) or (repo / rel).read_text(errors="replace"),
                      by_name.get(Path(rel).name), live.tables) for rel in rels]

    todo = [it for it in items if it.action == "apply"]
    if todo and not dry_run:
        ok, why = container_running(env)
        if not ok:
            out(f"{TAG}: {why}; applying nothing")
            for it in todo:
                out(f"  - {it.rel}: NOT APPLIED, left for a human -- apply: {it.manual}")
            return 2

    lp = log_path(env)
    sha = _git(repo, "rev-parse", "--short", "HEAD") or "?"
    host = socket.gethostname()
    prefix = psql_prefix(env)

    def log(kind: str, it: Item, extra: str = "") -> None:
        if dry_run:
            return
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_log(lp, f"{stamp} {kind} {it.rel} head={sha} host={host}" + (f" {extra}" if extra else ""))

    deadline = clock() + max_wait_s

    def wait_for_backup(it: Item) -> str:
        """"" when no backup is running (or the look failed); "defer" (--defer-if-backup);
        "exceeded" when the run's wait budget is spent with a backup still running."""
        started, waited_logged = clock(), False
        while True:
            try:
                backups = running_backups(connect_fn)
            except Exception as exc:  # noqa: BLE001 -- not knowing is today's behaviour: apply
                out(f"      could not check for a running backup ({' '.join(str(exc).split())}); "
                    f"applying anyway, lock_timeout={LOCK_TIMEOUT} still bounds it")
                return ""
            desc = ", ".join(f"{n} pid={p} running {s // 60}m" for p, n, s in backups)
            if not backups:
                if waited_logged:
                    secs = int(clock() - started)
                    out(f"      backup finished after a {secs}s wait")
                    log("BACKUP-WAIT-DONE", it, f"waited={secs}s")
                return ""
            if defer_if_backup:
                out(f"      a backup is running ({desc}); DEFERRED until it finishes")
                log("DEFERRED", it, f"backup={desc}")
                return "defer"
            left = deadline - clock()
            if left <= 0:
                out(f"      a backup is still running ({desc}) after waiting {int(max_wait_s)}s "
                    f"({MAX_WAIT_ENV}); not applied")
                log("BACKUP-WAIT-EXCEEDED", it, f"max_wait={int(max_wait_s)}s backup={desc}")
                return "exceeded"
            if not waited_logged:
                log("BACKUP-WAIT", it, f"backup={desc} max_wait={int(max_wait_s)}s")
                waited_logged = True
            out(f"      waiting for a running backup ({desc}); waited {int(clock() - started)}s, "
                f"{int(left)}s of the wait budget left")
            sleep(min(poll_s, left))

    stopped_by = ""
    deferred_by = ""
    applied: list[Item] = []
    rc = 0
    out(f"{TAG}: {len(rels)} changed file(s), checked against the live database"
        + (" [dry-run: nothing is applied]" if dry_run else ""))
    for it in items:
        if it.action in ("skip", "operator"):
            out(f"  - {it.rel}: {it.reason}")  # the report printed after this names any command
            if it.holds:
                rc = 1
                stopped_by = stopped_by or it.name
                log("NEEDS-HUMAN", it, f"reason={it.reason}")
            continue
        if it.action == "refuse":
            rc = 1
            out(f"  - {it.rel}: NOT auto-applied -- {it.reason}")
            if it.manual:
                out(f"      apply by hand after reading it: {it.manual}")
            log("REFUSED", it, f"reason={it.reason[:400]}")
            stopped_by = stopped_by or it.name
            continue
        mode = ("no transaction wrapper: " + ", ".join(it.no_txn)) if it.no_txn else "single transaction"
        if deferred_by:
            it.outcome = "deferred"
            out(f"  - {it.rel}: DEFERRED behind {deferred_by} (a backup is running)")
            log("DEFERRED", it, f"behind={deferred_by}")
            continue
        if stopped_by:
            rc = 1
            it.outcome = "held"
            out(f"  - {it.rel}: HELD, not applied -- {stopped_by} comes first in filename order and "
                f"needs a human; apply by hand afterwards: {it.manual}")
            log("HELD", it, f"behind={stopped_by}")
            continue
        if dry_run:
            it.outcome = "would-apply"
            out(f"  - {it.rel}: {it.reason}\n      would apply ({mode}, lock_timeout={LOCK_TIMEOUT})")
            continue
        out(f"  - {it.rel}: applying ({mode}, lock_timeout={LOCK_TIMEOUT}) ...")
        attempt, ok, tail, gate = 0, False, "", ""
        while True:
            gate = wait_for_backup(it)
            if gate:
                break
            with BlockerWatch(connect_fn, watch_interval_s) as watch:
                ok, tail = apply_one(it, prefix, env)
            if ok or LOCK_TIMEOUT_TEXT not in tail:
                break
            if not watch.only_backups():
                out(f"      lock timeout; blocker: {watch.describe()} -- not a backup, not retried")
                break
            if it.no_txn:
                out(f"      lock timeout behind a backup ({watch.describe()}), but the file ran without a "
                    "transaction wrapper, so a retry could re-run statements that committed; not retried")
                break
            if defer_if_backup:
                out(f"      lock timeout behind a backup ({watch.describe()}); DEFERRED until it finishes")
                log("DEFERRED", it, f"lock_timeout blocker={watch.describe()}")
                gate = "defer"
                break
            if attempt >= len(backoff_s):
                out(f"      lock timeout behind a backup ({watch.describe()}) after {attempt} "
                    "retries; giving up")
                break
            pause = backoff_s[attempt]
            attempt += 1
            out(f"      lock timeout behind a backup ({watch.describe()}); retry {attempt}/"
                f"{len(backoff_s)} in {int(pause)}s, after the backup finishes")
            log("RETRY", it, f"attempt={attempt}/{len(backoff_s)} backoff={int(pause)}s "
                             f"blocker={watch.describe()}")
            sleep(pause)
        if gate == "defer":
            it.outcome = "deferred"
            deferred_by = it.name
            continue
        if gate == "exceeded":
            rc = 1
            it.outcome = "failed"
            stopped_by = it.name
            out("      later files are held; apply by hand once the backup is done: " + it.manual)
            continue
        if ok:
            it.outcome = "applied"
            applied.append(it)
            log("APPLIED", it, f"mode={mode}" + (f" retries={attempt}" if attempt else ""))
            out(f"      applied at {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
        else:
            rc = 1
            it.outcome = "failed"
            stopped_by = it.name
            log("FAILED", it, f"mode={mode} error={' '.join(tail.split())[:400]}")
            out("      FAILED; later files are held. psql said:")
            for line in tail.splitlines():
                out(f"        {line}")
            if it.no_txn:
                out("      (ran without a transaction wrapper, so earlier statements may have "
                    "committed; the report below shows what is live)")

    if applied:
        try:
            _files, after, _live = read_report(repo, ref, connect_fn)
        except Exception as exc:  # noqa: BLE001
            out(f"{TAG}: could not re-read the live state after applying ({exc}); UNVERIFIED")
            return 2
        after_by = {f.name: f for f in after.files}
        for it in applied:
            r = after_by.get(it.name)
            if r is not None and r.status == "APPLIED":
                out(f"  verified: {it.rel} is APPLIED")
            else:
                rc = 1
                it.outcome = "unverified"
                summary = r.summary() if r is not None else "not in the checker's report"
                log("UNVERIFIED", it, f"checker={summary}")
                out(f"  NOT verified: psql succeeded but the checker still says {summary}")
        out(f"{TAG}: log: {lp}")
    if deferred_by:
        out(f"{TAG}: deferred from {deferred_by} on: a backup is running; run again without "
            "--defer-if-backup to wait for it (the rebuild script does this after the rebuilds)")
        if rc == 0:
            return EXIT_DEFERRED
    return rc


def consumers(repo: Path, paths: list[str]) -> list[str]:
    """Services named in the REQUIRED-BY headers of these files (sorted, unique)."""
    names: set[str] = set()
    for rel in paths:
        f = repo / rel
        if not f.is_file():
            continue
        rb = drift.parse_required_by(f.read_text(errors="replace"))
        if rb is not None and not rb.errors:
            names.update(rb.services)
    return sorted(names)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", help="changed migration paths, repo-relative")
    ap.add_argument("--dry-run", action="store_true", help="say what would be applied; apply nothing")
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--repo", default=str(REPO_ROOT))
    ap.add_argument("--defer-if-backup", action="store_true",
                    help="never wait for a running backup: defer those files and exit 3")
    ap.add_argument("--max-backup-wait", type=float, default=None, metavar="SECONDS",
                    help=f"longest total wait for a running backup (default ${MAX_WAIT_ENV} or "
                         f"{DEFAULT_MAX_WAIT_S})")
    ap.add_argument("--required-by", action="store_true",
                    help="print the services named in the files' REQUIRED-BY headers; apply nothing")
    args = ap.parse_args(argv)
    if args.required_by:
        for name in consumers(Path(args.repo), args.paths):
            print(name)
        return 0
    return run(args.paths, repo=Path(args.repo), ref=args.ref, dry_run=args.dry_run,
               defer_if_backup=args.defer_if_backup, max_wait_s=args.max_backup_wait)


if __name__ == "__main__":
    raise SystemExit(main())

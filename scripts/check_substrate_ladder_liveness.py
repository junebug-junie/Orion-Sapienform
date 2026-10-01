#!/usr/bin/env python3
"""Substrate ladder liveness gate: every rung still writing, every strict-schema
consumer running a schema at least as new as the one being written.

Detection gate for the 2026-09-20 incident (see
``orion/substrate_ladder_liveness.py``'s docstring): a forbid-model field was
added to FieldStateV1, only the producer was redeployed, and attention/proposal
silently stopped writing for ~48h while every container was "Up".

Read-only everywhere:
- Postgres: bounded ``max(ts)`` per rung in a read-only session with a
  statement timeout, plus catalog reads (information_schema / pg_index /
  pg_class) for the migrations section: every merged hand-applied
  ``services/orion-sql-db/*.sql`` changed in the last ``--migration-days``
  (default 30) must have its tables/columns/indexes/sequences live -- the
  PR #2400 (missing column) and PR #2424 (missing table, 13x crash loop)
  incidents. Logic and how it decides: ``orion/sql_migration_drift.py``.
- docker: ``docker ps`` / ``docker inspect`` / ``docker image inspect``, plus
  one ``docker exec <c> python -c`` per consumer container that hashes its copy
  of the schema file (no writes, no restarts).
- git: ``git log --first-parent`` / ``git show`` on ``--ref`` (default
  origin/main, else HEAD) -- used to report producer-vs-main drift and as the
  timestamp fallback when a container's schema bytes cannot be read.

Alerting (``--notify``): one Hub Pending Attention card per new failing
rung/container via orion-notify ``/attention/request`` -- the same path
disk-threshold-watchdog and postgres-headroom-watch already use from host cron,
which is the path a human already reads. Debounced per failing key; a key that
fails to deliver is retried next tick; a key that recovers is forgotten so a
recurrence alerts again.

If the debounce state cannot be read/written, red keys are carded anyway with
no dedupe (one card per tick) -- repeated cards beat silence. That was the
2026-09-26 incident: a root-owned state dir failed every run, 204 red runs
(~34h) raised zero cards, and the only trace was a log line nobody read.

Exit codes: 0 green, 1 at least one red rung or skewed consumer,
2 could not complete a check (DB or docker unreachable) and nothing was red,
4 --notify could not escalate: dedupe state unusable, or orion-notify did not
accept a card. Wins over 1/2 because a red nobody was told about is the worse
failure; the last stdout line still says RED / GREEN / CANNOT CHECK.

Usage:
    python scripts/check_substrate_ladder_liveness.py
    python scripts/check_substrate_ladder_liveness.py --json
    python scripts/check_substrate_ladder_liveness.py --notify
    python scripts/check_substrate_ladder_liveness.py --list-consumers
    python scripts/check_substrate_ladder_liveness.py --test-escalation   # ONE labelled test card
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from orion import schema_skew_discovery as ssd  # noqa: E402
from orion import sql_migration_drift as drift  # noqa: E402
from orion import substrate_ladder_liveness as ll  # noqa: E402

EXIT_OK = 0
EXIT_RED = 1
EXIT_CANNOT_CHECK = 2
# 3 is the disk watchdog's "the script itself crashed"; kept distinct everywhere.
EXIT_ESCALATION_FAILED = 4

DEFAULT_NOTIFY_BASE_URL = os.getenv("NOTIFY_BASE_URL", "http://localhost:7140")
STATEMENT_TIMEOUT_MS = 20_000
DOCKER_TIMEOUT_SEC = 30


# --------------------------------------------------------------------- postgres


def connection_params(explicit: Optional[str] = None):
    """Same host-side defaults as check_postgres_connection_headroom.py."""
    dsn = explicit or os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL")
    if dsn:
        return dsn
    return dict(
        host=os.environ.get("ORION_PG_HOST", "localhost"),
        port=int(os.environ.get("ORION_PG_PORT", "55432")),
        user=os.environ.get("ORION_PG_USER", "postgres"),
        password=os.environ.get("ORION_PG_PASSWORD", os.environ.get("PGPASSWORD", "postgres")),
        dbname=os.environ.get("ORION_PG_DB", "conjourney"),
    )


def read_freshness(conn, lookback_sec: float) -> tuple[dict[str, Optional[datetime]], list[int], list[str]]:
    newest: dict[str, Optional[datetime]] = {}
    errors: list[str] = []
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
        for rung in ll.RUNGS:
            params: list[Any] = [lookback_sec]
            if rung.lane_column:
                params.append(rung.lane_value)
            try:
                cur.execute(ll.freshness_sql(rung), params)
                newest[rung.name] = cur.fetchone()[0]
            except Exception as exc:  # noqa: BLE001 - one bad rung must not hide the rest
                errors.append(f"{rung.name}: {exc.__class__.__name__}: {str(exc).strip()}")
        motif_counts: list[int] = []
        try:
            cur.execute(ll.CONSOLIDATION_MOTIF_SQL, [6 * 3600, ll.CONSOLIDATION_EMPTY_RUN])
            motif_counts = [int(r[0]) for r in cur.fetchall()]
        except Exception as exc:  # noqa: BLE001
            errors.append(f"consolidation:motifs: {exc.__class__.__name__}: {str(exc).strip()}")
    return newest, motif_counts, errors


# ----------------------------------------------------------------------- docker


def _run(cmd: list[str], timeout: int = DOCKER_TIMEOUT_SEC) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    return out.stdout


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value or value.startswith("0001-"):
        return None
    v = value.strip().replace("Z", "+00:00")
    # docker emits nanoseconds; fromisoformat takes at most microseconds.
    if "." in v:
        head, rest = v.split(".", 1)
        frac = rest[: len(rest) - len(rest.lstrip("0123456789"))]
        tz = rest[len(frac):]
        v = f"{head}.{frac[:6]}{tz}"
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return None


# Locate the top-level ``orion`` package without importing it (find_spec on a
# top-level name only searches sys.path), then read the requested files by
# path and print them as one JSON object. Nothing from the repo executes
# inside the production container; one exec per container covers every
# schema file that container's service reads or writes.
_READ_SNIPPET = (
    "import importlib.util as u,json,os,sys\n"
    "s=u.find_spec('orion')\n"
    "b=s.submodule_search_locations[0] if s and s.submodule_search_locations else None\n"
    "out={'__orion__':b is not None}\n"
    "for rel in json.loads(sys.argv[1]):\n"
    "    try:\n"
    "        out[rel]=open(os.path.join(b,*rel.split('/')[1:]),encoding='utf-8').read() if b else None\n"
    "    except (OSError,UnicodeDecodeError):\n"
    "        out[rel]=None\n"
    "sys.stdout.write(json.dumps(out))\n"
)


NO_ORION = "no-orion"


def container_sources(container: str, paths: list[str]):
    """``{repo-relative path: file text or None}`` as ``container`` sees them.

    ``NO_ORION`` when the container has no Python or its Python has no
    ``orion`` package (a sidecar such as redis/grafana in the same compose
    file; it cannot be running a reader or writer). ``None`` when the read
    itself failed (timeout, restarting container, daemon error): the caller
    reports that as could-not-check rather than guessing.
    """
    arg = json.dumps(sorted(paths))
    no_python = 0
    for py in ("python", "python3"):
        try:
            out = _run(["docker", "exec", container, py, "-c", _READ_SNIPPET, arg], timeout=DOCKER_TIMEOUT_SEC)
        except subprocess.CalledProcessError as exc:
            # docker exec exits 127 (126) when the command is not found (not
            # executable) in the image: no such interpreter (redis, grafana,
            # bus-core). The message goes to the terminal, not stderr, so the
            # exit code is the signal. Anything else -- a restarting
            # container, a daemon error -- is a failed read.
            if exc.returncode in (126, 127):
                no_python += 1
            continue
        except (subprocess.SubprocessError, OSError):
            continue
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if not data.pop("__orion__", False):
            return NO_ORION
        return {p: (v if isinstance(v, str) else None) for p, v in data.items()}
    return NO_ORION if no_python == 2 else None


class ContainerMeta(NamedTuple):
    name: str
    service_dir: str
    image_created: Optional[datetime]
    started_at: Optional[datetime]


def list_containers(services: set[str]) -> list[ContainerMeta]:
    names = [n for n in _run(["docker", "ps", "--format", "{{.Names}}"]).split() if n]
    if not names:
        return []
    inspected = json.loads(_run(["docker", "inspect", *names]))
    image_ids = sorted({c.get("Image") for c in inspected if c.get("Image")})
    created: dict[str, Optional[datetime]] = {}
    if image_ids:
        for img in json.loads(_run(["docker", "image", "inspect", *image_ids])):
            created[img.get("Id")] = _parse_ts(img.get("Created"))
    out = []
    for c in inspected:
        labels = (c.get("Config") or {}).get("Labels") or {}
        svc = ll.service_dir_from_compose_files(labels.get("com.docker.compose.project.config_files"))
        if svc not in services:
            continue
        out.append(
            ContainerMeta(
                name=(c.get("Name") or "").lstrip("/"),
                service_dir=svc,
                image_created=created.get(c.get("Image")),
                started_at=_parse_ts((c.get("State") or {}).get("StartedAt")),
            )
        )
    return out


def read_all_sources(
    metas: list[ContainerMeta], paths_by_service: dict[str, set[str]], workers: int
) -> dict[str, Optional[dict[str, Optional[str]]]]:
    """container name -> sources, one exec per container, run in parallel."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            m.name: pool.submit(container_sources, m.name, sorted(paths_by_service.get(m.service_dir, ())))
            for m in metas
        }
        return {name: f.result() for name, f in futs.items()}


def _sha(text: Optional[str]) -> Optional[str]:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None


# -------------------------------------------------------------------------- git


def default_ref(repo: str) -> str:
    try:
        subprocess.run(["git", "-C", repo, "rev-parse", "--verify", "-q", "origin/main"], capture_output=True, check=True)
        return "origin/main"
    except subprocess.CalledProcessError:
        return "HEAD"


def schema_commit(repo: str, ref: str, path: str) -> tuple[datetime, str, str]:
    """(commit time, short sha, sha256 of the file on ref)."""
    # --first-parent: the time the change LANDED on ref (its merge), not the
    # side-branch commit's own date, which can be hours or days earlier.
    log = _run(["git", "-C", repo, "log", "-1", "--first-parent", "--format=%cI %h", ref, "--", path]).split()
    if len(log) != 2:
        raise RuntimeError(f"no commit touches {path} on {ref}")
    body = subprocess.run(["git", "-C", repo, "show", f"{ref}:{path}"], capture_output=True, check=True).stdout
    return datetime.fromisoformat(log[0]), log[1], hashlib.sha256(body).hexdigest()


def ref_file_shas(repo: str, ref: str, paths: list[str]) -> dict[str, Optional[str]]:
    """sha256 of each file on ``ref`` in one ``git cat-file --batch`` call."""
    if not paths:
        return {}
    req = "".join(f"{ref}:{p}\n" for p in paths).encode()
    raw = subprocess.run(["git", "-C", repo, "cat-file", "--batch"], input=req, capture_output=True, check=True).stdout
    out: dict[str, Optional[str]] = {}
    pos = 0
    for p in paths:
        nl = raw.index(b"\n", pos)
        header = raw[pos:nl].split()
        pos = nl + 1
        if len(header) == 3 and header[1] == b"blob":
            size = int(header[2])
            out[p] = hashlib.sha256(raw[pos : pos + size]).hexdigest()
            pos += size + 1
        else:
            out[p] = None
    return out


# ----------------------------------------------------------------------- notify


def default_state_file() -> str:
    root = os.getenv("TELEMETRY_ROOT", "/mnt/telemetry")
    project = os.getenv("PROJECT", "orion-athena")
    return os.path.join(root, project, "substrate-ladder-liveness", "state.json")


def _load_state(path: str) -> dict[str, Any]:
    """A missing, unparseable, or wrong-shaped file reads as empty: the red is
    re-carded and the file is rewritten, rather than a TypeError killing the
    send every tick."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    keys = data.get("notified_keys", [])
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        data["notified_keys"] = []
    return data


def _save_state(path: str, state: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def keys_to_notify(red_keys: list[str], notified: list[str]) -> list[str]:
    """Pure debounce: red keys not already confirmed-delivered."""
    done = set(notified)
    return [k for k in red_keys if k not in done]


def carry_notified(notified: list[str], green_keys: list[str]) -> list[str]:
    """Forget a delivered key only once its check ran and came back green.

    A key that is merely absent (its query errored, docker was unreachable) is
    kept, so a flaky read cannot re-send a card a human already has.
    """
    green = set(green_keys)
    return [k for k in notified if k not in green]


class Escalation(NamedTuple):
    """What ``--notify`` actually achieved this tick.

    ``sent``: True/False when a card was attempted, None when nothing needed one.
    ``state_error``: the dedupe state could not be read/written (no memory, so
    red cards go out undeduped every tick until it is fixed).
    ``notify_error``: orion-notify did not accept the card (down, 5xx, bad token).
    """

    sent: Optional[bool] = None
    state_error: Optional[str] = None
    notify_error: Optional[str] = None
    skipped_locked: bool = False

    @property
    def failed(self) -> bool:
        return self.state_error is not None or self.sent is False


class _StateUnusable(Exception):
    pass


def _send_card(report: ll.LadderReport, *, new: list[str], client, base_url: str, token: Optional[str],
               state_error: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """One attention_request. Never raises: (accepted, failure detail)."""
    red = report.red_keys()
    message = report.alert_message()
    if state_error is not None:
        message = (
            "The ladder watch cannot remember which cards it already sent "
            f"({state_error}), so this card repeats every tick until that is fixed.\n\n" + message
        )
    try:
        if client is None:
            from orion.notify.client import NotifyClient

            client = NotifyClient(base_url=base_url, api_token=token, timeout=10)
        accepted = client.attention_request(
            message=message,
            severity=report.severity(),
            require_ack=True,
            context={
                "source_service": "check_substrate_ladder_liveness",
                "reason": "substrate_ladder_liveness",
                "red_keys": red,
                "new_keys": new,
                "dedupe_state_error": state_error,
            },
        )
    except Exception as exc:  # noqa: BLE001 - a client bug is a failed send, not a crash
        return False, f"{exc.__class__.__name__}: {exc}"
    if bool(getattr(accepted, "ok", False)):
        return True, None
    return False, str(getattr(accepted, "detail", None) or "orion-notify returned ok=False")


def notify(report: ll.LadderReport, *, state_file: str, base_url: str, token: Optional[str], client=None) -> Escalation:
    """Debounced card for new red keys. Never silent about its own failure.

    If the dedupe state cannot be used (the 2026-09-26 incident: a root-owned
    telemetry dir made every run fail before sending anything, for ~34h of red),
    fall back to carding every current red key with no dedupe: a card per tick
    is bounded noise a human will act on; no card is silence nobody sees.
    """
    try:
        os.makedirs(os.path.dirname(state_file) or ".", exist_ok=True)
        lock = open(f"{state_file}.lock", "w")
    except OSError as exc:
        return _notify_stateless(report, f"{exc.__class__.__name__}: {exc}", base_url=base_url, token=token, client=client)
    with lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                return Escalation(skipped_locked=True)
            # Not contention (ENOLCK, NFS...): skipping would be silent every tick.
            return _notify_stateless(report, f"flock {exc.__class__.__name__}: {exc}", base_url=base_url, token=token, client=client)
        state = _load_state(state_file)
        red = report.red_keys()
        # Forget keys that verifiably recovered, so a recurrence alerts again.
        notified = carry_notified(list(state.get("notified_keys", [])), report.green_keys())
        new = keys_to_notify(red, notified)
        sent: Optional[bool] = None
        notify_error: Optional[str] = None
        if new:
            sent, notify_error = _send_card(report, new=new, client=client, base_url=base_url, token=token)
            if sent:
                notified = sorted(set(notified) | set(new))
        state.update(
            {
                "notified_keys": notified,
                "last_red_keys": red,
                "last_run_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        try:
            _save_state(state_file, state)
        except OSError as exc:
            # The card (if any) already went out this tick; do not send a
            # second one. Next tick has no memory and will re-card.
            return Escalation(sent=sent, state_error=f"{exc.__class__.__name__}: {exc}", notify_error=notify_error)
        return Escalation(sent=sent, notify_error=notify_error)


def _notify_stateless(report: ll.LadderReport, state_error: str, *, base_url: str, token: Optional[str], client) -> Escalation:
    red = report.red_keys()
    if not red:
        return Escalation(state_error=state_error)
    sent, detail = _send_card(report, new=red, client=client, base_url=base_url, token=token, state_error=state_error)
    return Escalation(sent=sent, state_error=state_error, notify_error=detail)


TEST_CARD_TITLE = "TEST: substrate ladder watch escalation check -- safe to dismiss"


def send_test_card(*, base_url: str, token: Optional[str], client=None) -> tuple[bool, Optional[str], Optional[str]]:
    """One labelled card through the real orion-notify path; no state touched.

    Returns (accepted, notification_id, failure detail).
    """
    try:
        if client is None:
            from orion.notify.client import NotifyClient

            client = NotifyClient(base_url=base_url, api_token=token, timeout=10)
        accepted = client.attention_request(
            message=(
                f"{TEST_CARD_TITLE}\n\nThis is a one-off proof that the substrate ladder watch "
                "can reach Hub Pending Attention through orion-notify. Nothing is wrong with the ladder."
            ),
            severity="warning",
            require_ack=True,
            context={
                "source_service": "check_substrate_ladder_liveness",
                "reason": "TEST substrate_ladder_liveness escalation check (safe to dismiss)",
                "test": True,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{exc.__class__.__name__}: {exc}"
    ok = bool(getattr(accepted, "ok", False))
    nid = getattr(accepted, "notification_id", None)
    return ok, (str(nid) if nid else None), (None if ok else str(getattr(accepted, "detail", None) or "ok=False"))


# ------------------------------------------------------------------------- main


def build_report(args) -> ll.LadderReport:
    report = ll.LadderReport()
    now = datetime.now(timezone.utc)

    if not args.skip_db:
        try:
            import psycopg2

            params = connection_params(args.dsn)
            conn = psycopg2.connect(params, connect_timeout=10) if isinstance(params, str) else psycopg2.connect(connect_timeout=10, **params)
            conn.set_session(readonly=True, autocommit=True)
            try:
                newest, motifs, errors = read_freshness(conn, args.lookback_hours * 3600)
                report.rungs = ll.evaluate_ladder(newest, now, consolidation_motif_counts=motifs)
                report.cannot_check += errors
                if not args.skip_migrations:
                    check_migrations(conn, args, report)
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            report.cannot_check.append(f"postgres: {exc.__class__.__name__}: {str(exc).strip()}")

    if not args.skip_docker:
        try:
            skew, notes = check_skew(args)
            report.skew += skew
            report.cannot_check += notes
        except Exception as exc:  # noqa: BLE001
            report.cannot_check.append(f"skew: {exc.__class__.__name__}: {str(exc).strip()}")
    return report


def _delivered_migration_keys(state_file: str) -> list[str]:
    """Migration keys already carded (read-only, no lock). A carded file stays red until it is
    actually applied instead of silently ageing out of the window with its card muted. An
    unreadable state degrades to plain window behaviour; notify() reports the state problem."""
    try:
        with open(state_file, encoding="utf-8") as fh:
            keys = json.load(fh).get("notified_keys", [])
        return [k for k in keys if isinstance(k, str) and k.startswith("migration:")]
    except Exception:  # noqa: BLE001
        return []


def check_migrations(conn, args, report: ll.LadderReport) -> None:
    """Merged hand-applied SQL migrations vs the live schema (PR #2400 / #2424 incidents).

    Same read-only connection; git history of the checkout this runs from (cron: the primary
    checkout on main, i.e. what was deployed). A failure here is CANNOT CHECK for this section
    only -- it never hides the rungs or skew results.
    """
    try:
        report.migrations = drift.check_repo(
            conn, Path(args.repo), ref="HEAD", window_days=args.migration_days or None,
            sticky_keys=_delivered_migration_keys(getattr(args, "state_file", None) or default_state_file()),
        )
    except Exception as exc:  # noqa: BLE001 - one bad section must not hide the rest
        report.cannot_check.append(f"migrations: {exc.__class__.__name__}: {str(exc).strip()}")


def check_skew(args) -> tuple[list[ll.SkewResult], list[str]]:
    repo = args.repo
    ref = args.ref or default_ref(repo)
    schemas, disc = ll.discovered_schemas(Path(repo))
    if not args.include_loose:
        schemas = [s for s in schemas if s.strict]
    notes = [f"skew: unresolved writer for {u.key} (declare it in DECLARED_WRITERS)" for u in disc.unresolved]

    paths_by_service: dict[str, set[str]] = {}
    for sc in schemas:
        files = {sc.path, *sc.deps}
        for svc in {sc.producer_service, *(r for r, _ in sc.reader_models)}:
            paths_by_service.setdefault(svc, set()).update(files)
    metas = list_containers(set(paths_by_service))
    sources = read_all_sources(metas, paths_by_service, args.docker_workers)
    skipped = sorted(m.name for m in metas if sources.get(m.name) == NO_ORION)
    failed = sorted(m.name for m in metas if sources.get(m.name) is None)
    notes += [f"skew: could not read schema files in container {n} (docker exec failed)" for n in failed]
    metas = [m for m in metas if isinstance(sources.get(m.name), dict)]
    if args.verbose and skipped:
        print(f"skew: {len(skipped)} containers have no orion package (sidecars), skipped: {', '.join(skipped)}", file=sys.stderr)
    shapes = {m.name: ssd.shapes_from_sources({p: t for p, t in sources[m.name].items() if t is not None}) for m in metas}
    ref_shas = ref_file_shas(repo, ref, sorted({sc.path for sc in schemas}))
    commit_cache: dict[str, Optional[datetime]] = {}

    def commit_time(path: str) -> Optional[datetime]:
        if path not in commit_cache:
            try:
                commit_cache[path] = schema_commit(repo, ref, path)[0]
            except Exception:  # noqa: BLE001
                commit_cache[path] = None
        return commit_cache[path]

    results: list[ll.SkewResult] = []
    for sc in schemas:
        svcs = {sc.producer_service, *(r for r, _ in sc.reader_models)}
        files = (sc.path, *sc.deps)
        containers = []
        for m in metas:
            if m.service_dir not in svcs:
                continue
            src = sources[m.name]
            complete = all(src.get(f) is not None for f in files)
            containers.append(
                ll.RunningContainer(
                    name=m.name,
                    service_dir=m.service_dir,
                    image_created=m.image_created,
                    started_at=m.started_at,
                    schema_sha256=_sha(src.get(sc.path)),
                    shapes=shapes[m.name] if complete else None,
                )
            )
        needs_time = any(c.shapes is None and c.schema_sha256 is None for c in containers)
        results += ll.evaluate_skew(
            sc,
            schema_commit_time=commit_time(sc.path) if needs_time else None,
            schema_sha256_on_ref=ref_shas.get(sc.path),
            consumer_services=[r for r, _ in sc.reader_models],
            containers=containers,
        )
    if args.verbose:
        print(
            f"skew: {len(schemas)} (file, writer) pairs over {len({s.path for s in schemas})} files, "
            f"{sum(len(s.reader_models) for s in schemas)} writer->reader pairs, {len(metas)} containers read",
            file=sys.stderr,
        )
    return results, notes


def print_human(report: ll.LadderReport, verbose: bool = False) -> None:
    for r in report.rungs:
        mark = "RED " if r.red else "ok  "
        if r.rung == "consolidation:motifs":
            print(f"{mark}consolidation:motifs: {'last %d frames empty' % int(r.max_age_sec) if r.red else 'motifs present'}")
        else:
            print(f"{mark}{r.summary()}")
    counts: dict[str, int] = {}
    for s in report.skew:
        counts[s.status] = counts.get(s.status, 0) + 1
        # ok / not_running / no_writer are thousands of rows on a healthy host.
        if s.status in ("ok", "not_running", "no_writer") and not verbose:
            continue
        mark = "RED " if s.red else ("ok  " if s.status == "ok" else "warn")
        print(f"{mark}skew {s.schema} {s.producer or '?'} -> {s.service_dir} [{s.container or '-'}] {s.status}: {s.detail}")
    if report.skew:
        print("skew rows: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if report.migrations is not None:
        m = report.migrations
        in_window = [f for f in m.files if f.in_window]
        for f in m.red_files:
            print(f"RED migration {f.summary()}")
            print(f"    apply: {f.apply_command()}")
        for f in m.old_broken():
            print(f"warn old migration {f.summary()} (last changed {f.changed_at:%Y-%m-%d}, "
                  f"outside the {m.window_days}-day window, never carded)")
        for f in m.verify_manually():
            print(f"warn migration {f.name}: data-only (no schema objects) -- verify manually")
        for f in in_window:
            for note in f.info:
                if "DO block" in note:
                    print(f"warn migration {f.name}: {note}")
        if not m.red_files:
            print(f"ok   migrations: {len(in_window)} file(s) changed in the last {m.window_days or 'all'} days, "
                  "every declared schema object present")
    for c in report.cannot_check:
        print(f"CANNOT CHECK {c}")
    print("RED" if report.red else ("CANNOT CHECK" if report.cannot_check else "GREEN"))


def report_escalation(esc: Escalation, *, red: bool) -> bool:
    """Print what escalation did; True when it failed (exit EXIT_ESCALATION_FAILED)."""
    if esc.skipped_locked:
        print("escalation skipped: another ladder watch run holds the state lock", file=sys.stderr)
    if esc.sent is True:
        print("attention card sent" + (" (UNDEDUPED fallback)" if esc.state_error else ""), file=sys.stderr)
    if esc.state_error:
        print(
            f"ESCALATION FAILED: dedupe state unusable ({esc.state_error}). "
            + ("Red cards are sent every tick with no dedupe until this is fixed."
               if red else "Nothing is red now, but a red tick will card every run until this is fixed."),
            file=sys.stderr,
        )
    if esc.sent is False:
        print(
            f"ESCALATION FAILED: {'RED and ' if red else ''}orion-notify did not accept the attention card "
            f"({esc.notify_error}); retrying next tick. No human has been told.",
            file=sys.stderr,
        )
    return esc.failed


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=None, help="Postgres DSN (default $POSTGRES_URI, else localhost:55432)")
    ap.add_argument("--lookback-hours", type=float, default=ll.DEFAULT_LOOKBACK.total_seconds() / 3600)
    ap.add_argument("--repo", default=_REPO_ROOT)
    ap.add_argument("--ref", default=None, help="git ref the schema is compared against (default origin/main, else HEAD)")
    ap.add_argument("--skip-db", action="store_true")
    ap.add_argument("--skip-docker", action="store_true")
    ap.add_argument("--skip-migrations", action="store_true",
                    help="skip the merged-SQL-migration-applied section")
    ap.add_argument("--migration-days", type=int, default=drift.DEFAULT_WINDOW_DAYS,
                    help="alarm only on migrations changed in the last N days (0 = all)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--list-candidates", "--list-consumers", dest="list_candidates", action="store_true",
        help="print discovered (schema file, writer, readers) triples and exit",
    )
    ap.add_argument("--include-loose", action=argparse.BooleanOptionalAction, default=True,
                    help="also check non-forbid models (silent field drops; never red unless a required field is missing)")
    ap.add_argument("--docker-workers", type=int, default=8)
    ap.add_argument("--notify", action="store_true", help="raise a Hub Pending Attention card on new red (debounced)")
    ap.add_argument("--notify-base-url", default=DEFAULT_NOTIFY_BASE_URL)
    ap.add_argument("--notify-api-token", default=os.getenv("NOTIFY_API_TOKEN"))
    ap.add_argument("--state-file", default=None)
    ap.add_argument("--test-escalation", action="store_true",
                    help=f"send ONE card titled '{TEST_CARD_TITLE}' through orion-notify and exit (no checks run)")
    args = ap.parse_args(argv)

    if args.test_escalation:
        ok, nid, detail = send_test_card(base_url=args.notify_base_url, token=args.notify_api_token)
        if ok:
            print(f"test card accepted by orion-notify at {args.notify_base_url} (notification_id={nid})")
            return EXIT_OK
        print(f"ESCALATION FAILED: test card not accepted by orion-notify at {args.notify_base_url} ({detail})", file=sys.stderr)
        return EXIT_ESCALATION_FAILED

    if args.list_candidates:
        schemas, disc = ll.discovered_schemas(Path(args.repo))
        for sc in schemas:
            cls = "forbid" if sc.strict else "loose"
            for reader, models in sc.reader_models:
                print(f"{cls}\t{sc.path}\t{sc.producer_service}\t{reader}\t{','.join(models)}")
        for u in disc.unresolved:
            print(f"UNRESOLVED\t{u.path}\t?\t{','.join(u.readers)}\t{u.key}")
        strict = [s for s in schemas if s.strict]
        print(
            f"# {disc.strict_models} forbid models; {len({s.path for s in strict})} files / "
            f"{sum(len(s.reader_models) for s in strict)} writer->reader pairs cross services (forbid); "
            f"{sum(len(s.reader_models) for s in schemas if not s.strict)} loose pairs; "
            f"{len(disc.unresolved)} unresolved",
            file=sys.stderr,
        )
        return EXIT_OK

    report = build_report(args)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_human(report, verbose=args.verbose)

    escalation_failed = False
    if args.notify:
        try:
            esc = notify(
                report,
                state_file=args.state_file or default_state_file(),
                base_url=args.notify_base_url,
                token=args.notify_api_token,
            )
        except Exception as exc:  # noqa: BLE001 - escalation must never mask the result
            # A bug in the dedupe path is treated like unusable state: still card the red.
            esc = _notify_stateless(
                report, f"unexpected {exc.__class__.__name__}: {exc}",
                base_url=args.notify_base_url, token=args.notify_api_token, client=None,
            )
        escalation_failed = report_escalation(esc, red=report.red)

    if escalation_failed:
        return EXIT_ESCALATION_FAILED
    if report.red:
        return EXIT_RED
    if report.cannot_check:
        return EXIT_CANNOT_CHECK
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

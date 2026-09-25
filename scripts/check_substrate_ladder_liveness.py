#!/usr/bin/env python3
"""Substrate ladder liveness gate: every rung still writing, every strict-schema
consumer running a schema at least as new as the one being written.

Detection gate for the 2026-09-20 incident (see
``orion/substrate_ladder_liveness.py``'s docstring): a forbid-model field was
added to FieldStateV1, only the producer was redeployed, and attention/proposal
silently stopped writing for ~48h while every container was "Up".

Read-only everywhere:
- Postgres: bounded ``max(ts)`` per rung in a read-only session with a
  statement timeout.
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

Exit codes: 0 green, 1 at least one red rung or skewed consumer,
2 could not complete a check (DB or docker unreachable) and nothing was red.

Usage:
    python scripts/check_substrate_ladder_liveness.py
    python scripts/check_substrate_ladder_liveness.py --json
    python scripts/check_substrate_ladder_liveness.py --notify
    python scripts/check_substrate_ladder_liveness.py --list-consumers
"""

from __future__ import annotations

import argparse
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
from orion import substrate_ladder_liveness as ll  # noqa: E402

EXIT_OK = 0
EXIT_RED = 1
EXIT_CANNOT_CHECK = 2

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


def container_sources(container: str, paths: list[str]) -> Optional[dict[str, Optional[str]]]:
    """``{repo-relative path: file text or None}`` as ``container`` sees them;
    ``None`` when the container has no Python or no ``orion`` package at all
    (a sidecar such as redis/grafana in the same compose file), since such a
    container cannot be running a reader."""
    arg = json.dumps(sorted(paths))
    for py in ("python", "python3"):
        try:
            out = _run(["docker", "exec", container, py, "-c", _READ_SNIPPET, arg], timeout=DOCKER_TIMEOUT_SEC)
        except (subprocess.SubprocessError, OSError):
            continue
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or not data.pop("__orion__", False):
            return None
        return {p: (v if isinstance(v, str) else None) for p, v in data.items()}
    return None


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
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


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


def notify(report: ll.LadderReport, *, state_file: str, base_url: str, token: Optional[str], client=None) -> Optional[bool]:
    """Returns True/False when a card was attempted, None when nothing was new."""
    os.makedirs(os.path.dirname(state_file) or ".", exist_ok=True)
    with open(f"{state_file}.lock", "w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return None
        state = _load_state(state_file)
        red = report.red_keys()
        # Forget keys that verifiably recovered, so a recurrence alerts again.
        notified = carry_notified(list(state.get("notified_keys", [])), report.green_keys())
        new = keys_to_notify(red, notified)
        sent: Optional[bool] = None
        if new:
            if client is None:
                from orion.notify.client import NotifyClient

                client = NotifyClient(base_url=base_url, api_token=token, timeout=10)
            accepted = client.attention_request(
                message=report.alert_message(),
                severity=report.severity(),
                require_ack=True,
                context={
                    "source_service": "check_substrate_ladder_liveness",
                    "reason": "substrate_ladder_liveness",
                    "red_keys": red,
                    "new_keys": new,
                },
            )
            sent = bool(getattr(accepted, "ok", False))
            if sent:
                notified = sorted(set(notified) | set(new))
        state.update(
            {
                "notified_keys": notified,
                "last_red_keys": red,
                "last_run_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _save_state(state_file, state)
        return sent


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
            finally:
                conn.close()
            report.rungs = ll.evaluate_ladder(newest, now, consolidation_motif_counts=motifs)
            report.cannot_check += errors
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
    skipped = sorted(m.name for m in metas if sources.get(m.name) is None)
    metas = [m for m in metas if sources.get(m.name) is not None]
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
    for c in report.cannot_check:
        print(f"CANNOT CHECK {c}")
    print("RED" if report.red else ("CANNOT CHECK" if report.cannot_check else "GREEN"))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=None, help="Postgres DSN (default $POSTGRES_URI, else localhost:55432)")
    ap.add_argument("--lookback-hours", type=float, default=ll.DEFAULT_LOOKBACK.total_seconds() / 3600)
    ap.add_argument("--repo", default=_REPO_ROOT)
    ap.add_argument("--ref", default=None, help="git ref the schema is compared against (default origin/main, else HEAD)")
    ap.add_argument("--skip-db", action="store_true")
    ap.add_argument("--skip-docker", action="store_true")
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
    args = ap.parse_args(argv)

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

    if args.notify:
        try:
            sent = notify(
                report,
                state_file=args.state_file or default_state_file(),
                base_url=args.notify_base_url,
                token=args.notify_api_token,
            )
            if sent is not None:
                print(f"attention card {'sent' if sent else 'FAILED to send (will retry next tick)'}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - escalation must never mask the result
            print(f"escalation failed ({exc.__class__.__name__}: {exc})", file=sys.stderr)

    if report.red:
        return EXIT_RED
    if report.cannot_check:
        return EXIT_CANNOT_CHECK
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

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
from typing import Any, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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
# top-level name only searches sys.path), then hash the file by path. Nothing
# from the repo executes inside the production container.
_HASH_SNIPPET = (
    "import hashlib,importlib.util as u,os,sys;"
    "s=u.find_spec('orion');"
    "p=os.path.join(s.submodule_search_locations[0],*sys.argv[1].split('/')[1:]);"
    "print(hashlib.sha256(open(p,'rb').read()).hexdigest())"
)


def container_schema_sha(container: str, schema_path: str) -> Optional[str]:
    for py in ("python", "python3"):
        try:
            out = _run(["docker", "exec", container, py, "-c", _HASH_SNIPPET, schema_path]).strip()
        except (subprocess.SubprocessError, OSError):
            continue
        if len(out) == 64:
            return out
    return None


def read_containers(services: set[str], schema_path: str) -> list[ll.RunningContainer]:
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
        name = (c.get("Name") or "").lstrip("/")
        out.append(
            ll.RunningContainer(
                name=name,
                service_dir=svc,
                image_created=created.get(c.get("Image")),
                started_at=_parse_ts((c.get("State") or {}).get("StartedAt")),
                schema_sha256=container_schema_sha(name, schema_path),
            )
        )
    return out


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
        repo = args.repo
        ref = args.ref or default_ref(repo)
        for schema in ll.STRICT_SCHEMAS:
            try:
                commit_time, commit_sha, ref_sha = schema_commit(repo, ref, schema.path)
                consumers = set(ll.schema_consumer_services(Path(repo), schema))
                containers = read_containers(consumers | {schema.producer_service}, schema.path)
                report.skew += ll.evaluate_skew(
                    schema,
                    schema_commit_time=commit_time,
                    schema_sha256_on_ref=ref_sha,
                    consumer_services=consumers,
                    containers=containers,
                )
                if args.verbose:
                    print(f"{schema.path} @ {ref} last changed {commit_sha} {commit_time.isoformat()}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                report.cannot_check.append(f"skew:{schema.symbol}: {exc.__class__.__name__}: {str(exc).strip()}")
    return report


def print_human(report: ll.LadderReport) -> None:
    for r in report.rungs:
        mark = "RED " if r.red else "ok  "
        if r.rung == "consolidation:motifs":
            print(f"{mark}consolidation:motifs: {'last %d frames empty' % int(r.max_age_sec) if r.red else 'motifs present'}")
        else:
            print(f"{mark}{r.summary()}")
    for s in report.skew:
        mark = "RED " if s.red else ("ok  " if s.status == "ok" else "warn")
        print(f"{mark}skew {s.schema} {s.service_dir} [{s.container or '-'}] {s.status}: {s.detail}")
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
    ap.add_argument("--list-consumers", action="store_true", help="print derived schema consumers and exit")
    ap.add_argument("--notify", action="store_true", help="raise a Hub Pending Attention card on new red (debounced)")
    ap.add_argument("--notify-base-url", default=DEFAULT_NOTIFY_BASE_URL)
    ap.add_argument("--notify-api-token", default=os.getenv("NOTIFY_API_TOKEN"))
    ap.add_argument("--state-file", default=None)
    args = ap.parse_args(argv)

    if args.list_consumers:
        for schema in ll.STRICT_SCHEMAS:
            for svc, files in ll.schema_consumer_services(Path(args.repo), schema).items():
                print(f"{schema.symbol}\t{svc}\t{', '.join(files[:3])}{' ...' if len(files) > 3 else ''}")
        return EXIT_OK

    report = build_report(args)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_human(report)

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

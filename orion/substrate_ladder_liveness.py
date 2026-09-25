"""Substrate ladder liveness: is every rung of the cognition ladder still writing,
and is every consumer of a strict cross-service schema running code at least as
new as that schema?

Why this exists (the incident it must catch, 2026-09-20 21:58Z .. 09-23):
commit 586faf93b added ``queue_contention_*`` fields to ``FieldStateV1``
(``orion/schemas/field_state.py``, ``extra="forbid"``) and only
orion-field-digester was redeployed. orion-attention-runtime (image 09-08) and
orion-proposal-runtime (image 09-14) kept the old schema, raised
``extra_forbidden`` on every field-state row they read, and wrote nothing.
``substrate_proposal_frames`` went 2d05h without a row (attention frames the
same; those rows have since aged out of retention) while every container read
"Up" and every flag was on. Policy/dispatch/feedback idled behind it and
consolidation kept writing hourly frames with zero motif observations. Nobody
noticed for ~48h.

Two deterministic checks, both pure over already-fetched readings so they can
be replayed from a fixture:

1. Ladder freshness -- ``evaluate_rung`` / ``evaluate_consolidation_empty``.
2. Schema/image skew -- ``schema_consumer_services`` (derived from an import
   scan, not a hand list) + ``evaluate_skew``.

IO (Postgres, docker, git, notify) lives in
``scripts/check_substrate_ladder_liveness.py``. Stdlib only, so the CI static
gates job can import it without service dependencies.
"""

from __future__ import annotations

import ast
import collections
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# 1. Ladder freshness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rung:
    """One rung of the ladder: a table (optionally one lane of it) that must keep
    receiving rows.

    ``max_age`` is set from the observed worst normal gap over 7-14 days of live
    data (see the PR report), with headroom, so a quiet-but-alive rung does not
    page anyone.
    """

    name: str
    table: str
    ts_column: str
    max_age: timedelta
    # Equality filter on one column (lane). Column name is a module constant,
    # the value is passed as a bound parameter.
    lane_column: Optional[str] = None
    lane_value: Optional[str] = None


_TICK = timedelta(minutes=15)
_CHAIN = timedelta(minutes=45)

RUNGS: tuple[Rung, ...] = (
    # Raw input lanes (the three grammar lanes the substrate consumes with a
    # dedicated partial index; see idx_grammar_events_*_consume).
    Rung("grammar:orion-bus", "grammar_events", "created_at", _TICK, "source_service", "orion-bus"),
    Rung("grammar:orion-biometrics", "grammar_events", "created_at", _TICK, "source_service", "orion-biometrics"),
    Rung("grammar:orion-cortex-exec", "grammar_events", "created_at", timedelta(minutes=60), "source_service", "orion-cortex-exec"),
    # Reducers with a steady cadence. Receipts expire after 30 minutes and are
    # pruned, so "no row in the window" is the normal way a dead reducer shows up.
    # Activity-driven reducers (execution_trajectory, node_pressure,
    # route_arbitration) are deliberately excluded: quiet is legitimate for them.
    Rung("receipts:transport_bus_reducer", "substrate_reduction_receipts", "created_at", timedelta(minutes=20), "reducer_name", "transport_bus_reducer"),
    Rung("receipts:biometrics_node_reducer", "substrate_reduction_receipts", "created_at", timedelta(minutes=20), "reducer_name", "biometrics_node_reducer"),
    Rung("receipts:substrate.bus_synaptic", "substrate_reduction_receipts", "created_at", timedelta(minutes=20), "reducer_name", "substrate.bus_synaptic"),
    Rung("receipts:substrate.perception", "substrate_reduction_receipts", "created_at", timedelta(minutes=20), "reducer_name", "substrate.perception"),
    Rung("receipts:substrate.vision_channel", "substrate_reduction_receipts", "created_at", timedelta(minutes=20), "reducer_name", "substrate.vision_channel"),
    # The ladder proper, bottom to top.
    Rung("field_state", "substrate_field_state", "generated_at", _TICK),
    Rung("attention", "substrate_attention_frames", "generated_at", _TICK),
    Rung("proposal", "substrate_proposal_frames", "generated_at", _CHAIN),
    Rung("policy", "substrate_policy_decision_frames", "generated_at", _CHAIN),
    Rung("dispatch", "substrate_execution_dispatch_frames", "generated_at", _CHAIN),
    Rung("feedback", "substrate_feedback_frames", "generated_at", _CHAIN),
    Rung("consolidation", "substrate_consolidation_frames", "generated_at", timedelta(hours=2)),
)

#: Every freshness query is bounded to this window so Postgres does an index
#: range scan. An unbounded max(generated_at) on the attention/field tables has
#: timed out at 60s+ on this host. No row inside the window reads as stale.
DEFAULT_LOOKBACK = timedelta(days=3)


def freshness_sql(rung: Rung) -> str:
    """SQL for the newest timestamp of one rung. Parameters: lookback seconds,
    then the lane value when the rung has a lane."""
    where = f"{rung.ts_column} > now() - make_interval(secs => %s)"
    if rung.lane_column:
        where += f" AND {rung.lane_column} = %s"
    return f"SELECT max({rung.ts_column}) FROM {rung.table} WHERE {where}"


@dataclass(frozen=True)
class RungResult:
    rung: str
    status: str  # "fresh" | "stale"
    newest: Optional[datetime]
    age_sec: Optional[float]
    max_age_sec: float

    @property
    def red(self) -> bool:
        return self.status != "fresh"

    def summary(self) -> str:
        if self.newest is None:
            return f"{self.rung}: no row in the lookback window"
        return (
            f"{self.rung}: newest {self.newest.isoformat()} "
            f"({_fmt_age(self.age_sec)} old, limit {_fmt_age(self.max_age_sec)})"
        )


def _as_utc(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _fmt_age(sec: Optional[float]) -> str:
    if sec is None:
        return "?"
    if sec < 120:
        return f"{sec:.0f}s"
    if sec < 7200:
        return f"{sec / 60:.0f}m"
    return f"{sec / 3600:.1f}h"


def evaluate_rung(rung: Rung, newest: Optional[datetime], now: datetime) -> RungResult:
    max_age_sec = rung.max_age.total_seconds()
    if newest is None:
        return RungResult(rung.name, "stale", None, None, max_age_sec)
    age = (_as_utc(now) - _as_utc(newest)).total_seconds()
    status = "fresh" if age <= max_age_sec else "stale"
    return RungResult(rung.name, status, _as_utc(newest), round(age, 1), max_age_sec)


#: Consolidation keeps writing hourly frames even when everything under it is
#: dead -- it just writes them empty. Over 60 days of live frames the longest
#: run of consecutive empty frames outside the incident was 1; the incident was
#: 52. Three in a row is therefore both quiet in normal operation and catches an
#: outage within ~3 hours.
CONSOLIDATION_EMPTY_RUN = 3

CONSOLIDATION_MOTIF_SQL = (
    "SELECT jsonb_array_length(COALESCE(consolidation_frame_json->'motif_observations', '[]'::jsonb)) "
    "FROM substrate_consolidation_frames "
    "WHERE generated_at > now() - make_interval(secs => %s) "
    "ORDER BY generated_at DESC LIMIT %s"
)


def evaluate_consolidation_empty(
    recent_motif_counts: Sequence[int], run: int = CONSOLIDATION_EMPTY_RUN
) -> RungResult:
    """``recent_motif_counts``: motif_observations lengths, newest first."""
    empty = len(recent_motif_counts) >= run and all(n == 0 for n in recent_motif_counts[:run])
    return RungResult(
        "consolidation:motifs",
        "empty" if empty else "fresh",
        None,
        None,
        float(run),
    )


# ---------------------------------------------------------------------------
# 2. Schema / image skew
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrictSchema:
    """A forbid-model that one service writes and other services validate."""

    path: str  # repo-relative file
    symbol: str

    @property
    def module(self) -> str:
        return self.path[: -len(".py")].replace("/", ".")


STRICT_SCHEMAS: tuple[StrictSchema, ...] = (
    StrictSchema("orion/schemas/field_state.py", "FieldStateV1"),
)

#: ``orion.schemas.registry`` imports every schema for name lookup, and ~60
#: services import the registry. A service only validates a schema through the
#: registry if that schema travels on the bus; when ``orion/bus/channels.yaml``
#: does not name it, the registry is not a real consumption path and following
#: it would flag every service in the repo.
REGISTRY_MODULE = "orion.schemas.registry"


def _is_test_path(rel: Path) -> bool:
    return rel.name.startswith("test_") or any(p in ("tests", "evals", "test") for p in rel.parts)


def _module_name(repo_root: Path, path: Path) -> str:
    parts = list(path.relative_to(repo_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _scan_file(path: Path, symbol: str) -> tuple[set[str], bool]:
    """Absolute ``orion.*`` imports in a file, and whether it names ``symbol``."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return set(), False
    # Exact prefilter: with neither token in the text, the AST walk below can
    # find nothing. Skips most of the ~3.8k files a repo scan touches.
    if "orion" not in text and symbol not in text:
        return set(), False
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set(), False
    imports: set[str] = set()
    uses = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name for a in node.names if a.name.split(".")[0] == "orion")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] != "orion":
                continue
            imports.add(node.module)
            imports.update(f"{node.module}.{a.name}" for a in node.names)
            if any(a.name == symbol for a in node.names):
                uses = True
        elif isinstance(node, ast.Name) and node.id == symbol:
            uses = True
        elif isinstance(node, ast.Attribute) and node.attr == symbol:
            uses = True
    return imports, uses


def schema_consumer_services(repo_root: Path, schema: StrictSchema) -> dict[str, list[str]]:
    """Service directories whose non-test code reaches ``schema.symbol``.

    A service is a consumer if one of its own files names the symbol, or imports
    (transitively, through the ``orion`` package) a module that does. Returns
    ``{service_dir_name: [evidence file, ...]}``. Includes the producer; the
    caller decides which one that is.
    """
    repo_root = Path(repo_root)
    channels = repo_root / "orion" / "bus" / "channels.yaml"
    try:
        bus_carries = schema.symbol in channels.read_text(encoding="utf-8")
    except OSError:
        bus_carries = False
    excluded = set() if bus_carries else {REGISTRY_MODULE}

    imports_of: dict[str, set[str]] = {}
    tainted: set[str] = set()
    for path in (repo_root / "orion").rglob("*.py"):
        if _is_test_path(path.relative_to(repo_root)):
            continue
        mod = _module_name(repo_root, path)
        imps, uses = _scan_file(path, schema.symbol)
        imports_of[mod] = imps
        if uses and mod not in excluded:
            tainted.add(mod)

    importers: dict[str, set[str]] = collections.defaultdict(set)
    for mod, imps in imports_of.items():
        for imp in imps:
            if imp in imports_of:
                importers[imp].add(mod)
    stack = list(tainted)
    while stack:
        for parent in importers[stack.pop()]:
            if parent not in tainted and parent not in excluded:
                tainted.add(parent)
                stack.append(parent)

    out: dict[str, list[str]] = {}
    services = repo_root / "services"
    for svc in sorted(p for p in services.iterdir() if p.is_dir()):
        hits = []
        for path in sorted(svc.rglob("*.py")):
            rel = path.relative_to(svc)
            if _is_test_path(rel):
                continue
            imps, uses = _scan_file(path, schema.symbol)
            if uses or (imps & tainted):
                hits.append(str(rel))
        if hits:
            out[svc.name] = hits
    return out


@dataclass(frozen=True)
class RunningContainer:
    name: str
    service_dir: Optional[str]  # e.g. "orion-attention-runtime", from compose labels
    image_created: Optional[datetime]
    started_at: Optional[datetime] = None
    # sha256 of the schema file as the container sees it; None when unreadable.
    schema_sha256: Optional[str] = None


@dataclass(frozen=True)
class SkewResult:
    schema: str
    service_dir: str
    container: Optional[str]
    status: str  # "ok" | "skew" | "content_differs" | "not_running" | "unknown"
    detail: str

    @property
    def red(self) -> bool:
        return self.status == "skew"


def evaluate_skew(
    schema: StrictSchema,
    *,
    schema_commit_time: datetime,
    schema_sha256_on_ref: Optional[str],
    consumer_services: Iterable[str],
    containers: Sequence[RunningContainer],
) -> list[SkewResult]:
    """Flag consumer containers running a copy of the schema older than the ref.

    Rules, per running container of a consumer service:

    - schema file inside the container matches the ref byte-for-byte -> ok,
      whatever the image date (content beats timestamps);
    - it differs and the image predates the last schema commit -> skew (red):
      this is the 09-20 shape;
    - it differs but the image is newer -> content_differs (reported, not red):
      usually a deploy from a branch ahead of main;
    - content unreadable -> fall back to the timestamp rule alone.
    """
    commit = _as_utc(schema_commit_time)
    by_service: dict[str, list[RunningContainer]] = collections.defaultdict(list)
    for c in containers:
        if c.service_dir:
            by_service[c.service_dir].append(c)

    results: list[SkewResult] = []
    for svc in sorted(set(consumer_services)):
        running = by_service.get(svc, [])
        if not running:
            results.append(SkewResult(schema.symbol, svc, None, "not_running", "no running container for this service"))
            continue
        for c in running:
            if c.image_created is None:
                results.append(SkewResult(schema.symbol, svc, c.name, "unknown", "image creation time unreadable"))
                continue
            built = _as_utc(c.image_created)
            older = built < commit
            ages = f"image built {built.isoformat()}, {schema.path} last changed {commit.isoformat()}"
            content_known = c.schema_sha256 is not None and schema_sha256_on_ref is not None
            if content_known and c.schema_sha256 == schema_sha256_on_ref:
                results.append(SkewResult(schema.symbol, svc, c.name, "ok", f"schema file matches; {ages}"))
            elif older:
                why = "schema file differs" if content_known else "schema file not readable"
                results.append(SkewResult(schema.symbol, svc, c.name, "skew", f"{why}; {ages}"))
            elif content_known:
                results.append(SkewResult(schema.symbol, svc, c.name, "content_differs", f"schema file differs but image is newer; {ages}"))
            else:
                results.append(SkewResult(schema.symbol, svc, c.name, "ok", ages))
    return results


def service_dir_from_compose_files(config_files: Optional[str]) -> Optional[str]:
    """``services/<dir>/docker-compose.yml`` -> ``<dir>`` from the compose label
    (which may point into any worktree)."""
    if not config_files:
        return None
    for entry in config_files.split(","):
        parts = Path(entry.strip()).parts
        for i in range(len(parts) - 2, -1, -1):
            if parts[i] == "services" and i + 1 < len(parts) - 1:
                return parts[i + 1]
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class LadderReport:
    rungs: list[RungResult] = field(default_factory=list)
    skew: list[SkewResult] = field(default_factory=list)
    cannot_check: list[str] = field(default_factory=list)

    @property
    def red_rungs(self) -> list[RungResult]:
        return [r for r in self.rungs if r.red]

    @property
    def red_skew(self) -> list[SkewResult]:
        return [s for s in self.skew if s.red]

    @property
    def red(self) -> bool:
        return bool(self.red_rungs or self.red_skew)

    def red_keys(self) -> list[str]:
        """Stable identifiers for debounce: one per failing rung / skewed container."""
        keys = [f"rung:{r.rung}" for r in self.red_rungs]
        keys += [f"skew:{s.schema}:{s.container}" for s in self.red_skew]
        return sorted(keys)

    def alert_message(self) -> str:
        lines = []
        if self.red_rungs:
            lines.append("Part of the substrate ladder has stopped writing:")
            lines += [f"- {r.summary()}" if r.rung != "consolidation:motifs" else
                      f"- consolidation: last {int(r.max_age_sec)} frames have zero motif observations (running, but on nothing)"
                      for r in self.red_rungs]
        if self.red_skew:
            lines.append("Containers are running a strict schema older than the one being written:")
            lines += [f"- {s.container} ({s.service_dir}) vs {s.schema}: {s.detail}" for s in self.red_skew]
            lines.append("Rebuild/redeploy those consumers; a forbid-model change is a consumer-first migration.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "red": self.red,
            "red_keys": self.red_keys(),
            "cannot_check": list(self.cannot_check),
            "rungs": [
                {
                    "rung": r.rung,
                    "status": r.status,
                    "newest": r.newest.isoformat() if r.newest else None,
                    "age_sec": r.age_sec,
                    "max_age_sec": r.max_age_sec,
                }
                for r in self.rungs
            ],
            "skew": [
                {"schema": s.schema, "service_dir": s.service_dir, "container": s.container, "status": s.status, "detail": s.detail}
                for s in self.skew
            ],
        }


def evaluate_ladder(
    newest_by_rung: Mapping[str, Optional[datetime]],
    now: datetime,
    *,
    consolidation_motif_counts: Optional[Sequence[int]] = None,
    rungs: Sequence[Rung] = RUNGS,
) -> list[RungResult]:
    out = [evaluate_rung(r, newest_by_rung.get(r.name), now) for r in rungs if r.name in newest_by_rung]
    if consolidation_motif_counts is not None:
        out.append(evaluate_consolidation_empty(consolidation_motif_counts))
    return out

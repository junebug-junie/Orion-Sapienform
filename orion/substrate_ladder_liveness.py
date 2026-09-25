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
2. Schema/image skew -- every (schema file, writer, readers) triple found by
   ``orion.schema_skew_discovery.discover`` (an AST scan of call sites, not a
   hand list) + ``evaluate_skew`` over each container's own copy of the file.

IO (Postgres, docker, git, notify) lives in
``scripts/check_substrate_ladder_liveness.py``. Stdlib only, so the CI static
gates job can import it without service dependencies.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from orion import schema_skew_discovery as ssd

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
    "SELECT CASE WHEN jsonb_typeof(consolidation_frame_json->'motif_observations') = 'array' "
    "THEN jsonb_array_length(consolidation_frame_json->'motif_observations') ELSE 0 END "
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
    """One (schema file, writer service) pair whose models other services
    validate. Built from ``orion.schema_skew_discovery.discover`` for the live
    check; ``STRICT_SCHEMAS`` below pins the ones that must stay discovered.

    ``models``: class names in ``path`` the writer writes and some reader reads.
    ``reader_models``: per reader service, the subset it actually reads.
    ``strict``: at least one of ``models`` is extra="forbid" (the 09-20 shape);
    otherwise the pair is the lower-severity silent-drop class.
    """

    path: str  # repo-relative file
    symbol: str  # label used in keys/messages
    producer_service: str
    models: tuple[str, ...] = ()
    reader_models: tuple[tuple[str, tuple[str, ...]], ...] = ()
    strict: bool = True
    deps: tuple[str, ...] = ()  # files the models' inherited fields live in
    writer_evidence: str = ""  # first writer call site, so a card can be triaged

    @property
    def module(self) -> str:
        mod = self.path[: -len(".py")].replace("/", ".")
        return mod[: -len(".__init__")] if mod.endswith(".__init__") else mod

    def models_for(self, reader: str) -> tuple[str, ...]:
        return dict(self.reader_models).get(reader, self.models)

    @classmethod
    def from_candidate(cls, c: "ssd.Candidate", deps: Sequence[str] = ()) -> "StrictSchema":
        return cls(
            path=c.path,
            symbol=c.path,
            producer_service=c.writer,
            models=tuple(c.models),
            reader_models=tuple(sorted((k, tuple(v)) for k, v in c.reader_models.items())),
            strict=c.strict,
            deps=tuple(deps),
            writer_evidence=next(iter(c.evidence.get(c.writer, ())), ""),
        )


#: Pinned: (file, class, writer) facts discovery must keep finding. A test
#: fails if any of these falls out of ``discover()``'s output.
STRICT_SCHEMAS: tuple[StrictSchema, ...] = (
    # orion-field-digester writes substrate_field_state; attention, proposal,
    # feedback runtimes and the hub read it back with model_validate.
    StrictSchema("orion/schemas/field_state.py", "FieldStateV1", "orion-field-digester", models=("FieldStateV1",)),
)


def schemas_from_discovery(disc: "ssd.Discovery") -> list[StrictSchema]:
    return [StrictSchema.from_candidate(c, disc.dependency_files.get(c.path, ())) for c in disc.candidates]


def discovered_schemas(repo_root: Path) -> tuple[list[StrictSchema], "ssd.Discovery"]:
    disc = ssd.discover(Path(repo_root))
    return schemas_from_discovery(disc), disc


@dataclass(frozen=True)
class RunningContainer:
    name: str
    service_dir: Optional[str]  # e.g. "orion-attention-runtime", from compose labels
    image_created: Optional[datetime]
    started_at: Optional[datetime] = None
    # sha256 of the schema file as the container sees it; None when unreadable.
    schema_sha256: Optional[str] = None
    # Model shapes parsed from the container's own copy of the schema file
    # (+ inherited-field files), ``module:Class`` -> shape; None when any of
    # those files could not be read.
    shapes: Optional[Mapping[str, "ssd.ModelShape"]] = None


@dataclass(frozen=True)
class SkewResult:
    schema: str
    service_dir: str
    container: Optional[str]
    # "ok" | "skew" (red) | "drops_fields" | "forbidden_unless_stripped" | "content_differs"
    # | "producer_differs_from_main" | "not_running" | "no_writer" | "unknown"
    status: str
    detail: str
    producer: Optional[str] = None

    @property
    def red(self) -> bool:
        return self.status == "skew"

    @property
    def key(self) -> str:
        return f"skew:{self.schema}:{self.container or self.service_dir}"


def _shape_verdict(schema: StrictSchema, svc: str, writer: RunningContainer, reader: RunningContainer) -> Optional[tuple[str, str]]:
    """Field-level verdict from both containers' own schema copies, or None
    when either side's shapes are unavailable."""
    if writer.shapes is None or reader.shapes is None:
        return None
    diffs = ssd.compare_models(schema.module, schema.models_for(svc), writer.shapes, reader.shapes)
    if diffs is None:
        return None
    broken = [d for d in diffs if d.breaks]
    if broken:
        parts = []
        for d in broken:
            if d.extra_forbidden:
                parts.append(f"{d.model}: writer sends {', '.join(d.extra_forbidden)} (reader forbids extra fields)")
            if d.missing_required:
                parts.append(f"{d.model}: reader requires {', '.join(d.missing_required)} (writer lacks them)")
        return "skew", "; ".join(parts)
    stripped = [d for d in diffs if d.maybe_stripped]
    if stripped:
        return "forbidden_unless_stripped", "; ".join(
            f"{d.model}: writer sends {', '.join(d.maybe_stripped)}; reader forbids extras but has a before-validator that may strip them"
            for d in stripped
        )
    dropped = [d for d in diffs if d.dropped]
    if dropped:
        return "drops_fields", "; ".join(f"{d.model}: reader silently drops {', '.join(d.dropped)}" for d in dropped)
    return "ok", "writer and reader field sets compatible"


def evaluate_skew(
    schema: StrictSchema,
    *,
    schema_commit_time: Optional[datetime],
    schema_sha256_on_ref: Optional[str],
    consumer_services: Iterable[str],
    containers: Sequence[RunningContainer],
) -> list[SkewResult]:
    """Flag reader containers that cannot read what the writer is actually
    writing.

    The reference is the writer's running container, not main: a writer
    deployed from a branch ahead of main is exactly the 09-20 shape, and a
    change merged to main but not yet deployed to the writer breaks nothing.
    Per reader container:

    - both containers' copies parse -> field-level comparison of the models
      this reader reads (``ssd.compare_models``): a writer field a forbid
      reader lacks, or a reader-required field the writer lacks -> skew (red),
      whatever the image ages; writer fields a non-forbid reader lacks ->
      drops_fields (reported); otherwise ok, even when the bytes differ;
    - else same schema file bytes as the writer -> ok;
    - different bytes and the reader image is older than the writer image
      -> skew (red); newer -> content_differs (reported);
    - bytes unreadable on either side -> timestamp fallback: red when the
      reader image predates the schema's last change on main
      (``schema_commit_time``) and the writer image does not.

    The writer gets its own row: ok when it matches main, otherwise
    producer_differs_from_main (reported, not red).
    """
    commit = _as_utc(schema_commit_time) if schema_commit_time else None
    by_service: dict[str, list[RunningContainer]] = collections.defaultdict(list)
    for c in containers:
        if c.service_dir:
            by_service[c.service_dir].append(c)

    producers = by_service.get(schema.producer_service, [])
    producer = producers[0] if producers else None
    ref_sha = producer.schema_sha256 if producer and producer.schema_sha256 else None
    ref_built = _as_utc(producer.image_created) if producer and producer.image_created else None
    pw = schema.producer_service

    results: list[SkewResult] = []
    for svc in sorted(set(consumer_services) | {schema.producer_service}):
        running = by_service.get(svc, [])
        if not running:
            results.append(SkewResult(schema.symbol, svc, None, "not_running", "no running container for this service", pw))
            continue
        for c in running:
            if svc == schema.producer_service:
                if c.schema_sha256 is None or schema_sha256_on_ref is None:
                    results.append(SkewResult(schema.symbol, svc, c.name, "unknown", "producer schema file not readable", pw))
                elif c.schema_sha256 == schema_sha256_on_ref:
                    results.append(SkewResult(schema.symbol, svc, c.name, "ok", "producer schema matches main", pw))
                else:
                    built = f"image {_as_utc(c.image_created).isoformat()}" if c.image_created else "image time unknown"
                    results.append(SkewResult(
                        schema.symbol, svc, c.name, "producer_differs_from_main",
                        f"producer ({built}) writes a {schema.path} that differs from main", pw,
                    ))
                continue
            if producer is None:
                # Nothing is being written right now, so nothing can be rejected.
                results.append(SkewResult(schema.symbol, svc, c.name, "no_writer", "writer has no running container", pw))
                continue
            verdict = _shape_verdict(schema, svc, producer, c)
            if verdict is not None:
                status, why = verdict
                where = f"; written at {schema.writer_evidence}" if schema.writer_evidence and status != "ok" else ""
                results.append(SkewResult(schema.symbol, svc, c.name, status, f"{why} (writer {producer.name}{where})", pw))
                continue
            if c.image_created is None:
                results.append(SkewResult(schema.symbol, svc, c.name, "unknown", "image creation time unreadable", pw))
                continue
            built = _as_utc(c.image_created)
            if ref_sha is not None and c.schema_sha256 is not None:
                vs = f"image built {built.isoformat()}, producer image {ref_built.isoformat() if ref_built else '?'}"
                if c.schema_sha256 == ref_sha:
                    results.append(SkewResult(schema.symbol, svc, c.name, "ok", f"schema file matches producer; {vs}", pw))
                elif ref_built is None or built < ref_built:
                    results.append(SkewResult(schema.symbol, svc, c.name, "skew", f"schema file differs from the producer's; {vs}", pw))
                else:
                    results.append(SkewResult(schema.symbol, svc, c.name, "content_differs", f"schema file differs from the producer's but image is newer; {vs}", pw))
                continue
            if commit is None:
                results.append(SkewResult(schema.symbol, svc, c.name, "unknown", "schema bytes unreadable and no commit time", pw))
                continue
            producer_new = ref_built is None or ref_built >= commit
            ages = f"image built {built.isoformat()}, {schema.path} last changed on main {commit.isoformat()} (schema bytes unreadable)"
            if built < commit and producer_new:
                results.append(SkewResult(schema.symbol, svc, c.name, "skew", ages, pw))
            else:
                results.append(SkewResult(schema.symbol, svc, c.name, "ok", ages, pw))
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
        keys += [s.key for s in self.red_skew]
        return sorted(set(keys))

    def green_keys(self) -> list[str]:
        """Keys whose check actually ran this tick and came back not-red.

        A rung whose query errored is absent from ``rungs`` entirely, and a
        container that could not be inspected is ``unknown``/``not_running``;
        neither counts as recovered, so a flaky DB/docker read cannot re-arm a
        card that was already delivered.
        """
        keys = [f"rung:{r.rung}" for r in self.rungs if not r.red]
        # A key is green only if no row for it is red this tick: one reader
        # container is compared against every writer of the file, and the
        # same key can be red against one writer and ok against another.
        # Known flap: if the only red writer stops running (no_writer) while
        # another writer's row is ok, the key re-arms and fires again when
        # that writer returns -- acceptable, it is again unreadable then.
        red = {s.key for s in self.skew if s.red}
        keys += sorted({
            s.key for s in self.skew
            if not s.red and s.status not in ("unknown", "not_running", "no_writer") and s.key not in red
        })
        return sorted(keys)

    def severity(self) -> str:
        """critical when anything that stops cognition is red; warning when the
        only finding is empty consolidation (a symptom, usually of a red rung)."""
        return "warning" if self.red_keys() == ["rung:consolidation:motifs"] else "critical"

    def alert_message(self) -> str:
        lines = []
        if self.red_rungs:
            lines.append("Part of the substrate ladder has stopped writing:")
            lines += [f"- {r.summary()}" if r.rung != "consolidation:motifs" else
                      f"- consolidation: last {int(r.max_age_sec)} frames have zero motif observations (running, but on nothing)"
                      for r in self.red_rungs]
        if self.red_skew:
            lines.append("Containers cannot read a schema another service is writing (their copy is older or incompatible):")
            seen = set()
            for s in self.red_skew:
                if s.key in seen:
                    continue
                seen.add(s.key)
                lines.append(f"- {s.container} ({s.service_dir}) reading {s.schema} from {s.producer or '?'}: {s.detail}")
            lines.append("Rebuild/redeploy those readers; a forbid-model change is a consumer-first migration.")
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
                {
                    "schema": s.schema,
                    "producer": s.producer,
                    "service_dir": s.service_dir,
                    "container": s.container,
                    "status": s.status,
                    "detail": s.detail,
                }
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

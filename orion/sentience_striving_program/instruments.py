"""Load the instrument manifest and join it to live repo + database state.

This module is the *reducer* in the manifest's event->schema->reducer chain. It
owns no facts of its own: mechanical facts come from `orion/metrics/` (the
existing metric semantic layer) and from live Postgres; editorial facts come from
instruments.yaml. The join is the whole contribution.

Read-only throughout. Every database read goes through
`orion.metrics.liveness.open_readonly_connection`, the same connection helper the
metric layer's own liveness pass uses, so this cannot write to production.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).resolve().parent / "instruments.yaml"

# Claim kinds the manifest may declare. `sql` is checked against live Postgres;
# `absent_from_repo` is checked with ripgrep (it asserts a *deletion* held);
# `manual` records a human-run check that has no cheap automated form -- it is
# reported, never silently treated as passing.
CLAIM_KINDS = frozenset({"sql", "absent_from_repo", "manual"})

# Files whose own text is allowed to mention an `absent_from_repo` target: the
# manifest declares the string, and the program README narrates the retirement.
# Without this the gate would fire on its own declaration -- the same
# self-matching trap that has bitten detector work in this repo before.
_ABSENCE_SCAN_EXCLUDES = (
    "orion/sentience_striving_program/",
    "docs/",
    "CLAUDE.md",
    "AGENTS.md",
)


@dataclass(frozen=True)
class Claim:
    """One falsifiable statement the program currently rests on."""

    id: str
    question: str
    kind: str
    recorded: Any
    recorded_at: str
    sql: str | None = None
    target: str | None = None
    blocks: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class Storage:
    """Where an instrument's output actually lands, and what bounds its history."""

    kind: str
    table: str | None = None
    ts_column: str | None = None
    graph: str | None = None
    graph_host_env: str | None = None
    retention_setting: str | None = None
    retention_service: str | None = None


@dataclass(frozen=True)
class Instrument:
    id: str
    title: str
    theory: str
    program_ref: str
    module: str
    outcome: str
    unlock: str
    last_reviewed: str
    storage: Storage
    claims: tuple[Claim, ...]
    entrypoint: str | None = None
    metrics: tuple[str, ...] = ()


@dataclass
class ClaimResult:
    """A claim re-run against live state."""

    claim: Claim
    observed: Any = None
    status: str = "UNKNOWN"  # HOLDS | DRIFTED | MANUAL | ERROR
    detail: str = ""

    @property
    def drifted(self) -> bool:
        return self.status == "DRIFTED"


@dataclass
class InstrumentState:
    """Everything the board renders for one instrument."""

    instrument: Instrument
    claims: list[ClaimResult] = field(default_factory=list)
    module_exists: bool = True
    entrypoint_exists: bool | None = None
    review_age_days: int | None = None
    review_stale: bool = False
    # Live storage shape -- None when the instrument has no table (kind: none,
    # graph) or when the database was not reachable.
    row_count: int | None = None
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    history_hours: float | None = None
    retention_hours: float | None = None
    retention_source: str = ""
    storage_note: str = ""
    consumers: list[str] = field(default_factory=list)
    consumer_note: str = ""


def _claim_from_raw(raw: dict[str, Any]) -> Claim:
    # A claim carrying `sql` is a sql claim; the manifest need not restate it.
    kind = raw.get("kind") or ("sql" if raw.get("sql") else "manual")
    if kind not in CLAIM_KINDS:
        raise ValueError(f"claim {raw.get('id')!r}: unknown kind {kind!r}")
    return Claim(
        id=raw["id"],
        question=raw["question"],
        kind=kind,
        recorded=raw.get("recorded"),
        recorded_at=str(raw.get("recorded_at", "")),
        sql=raw.get("sql"),
        target=raw.get("target"),
        blocks=raw.get("blocks"),
        note=raw.get("note"),
    )


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Parse instruments.yaml into outcomes + Instrument objects.

    Raises ValueError on a manifest that is internally inconsistent (an unknown
    outcome id, a duplicate instrument id, an unknown claim kind) -- these are
    authoring mistakes and should fail loudly rather than render as an empty
    panel.
    """
    raw = yaml.safe_load((path or MANIFEST_PATH).read_text())
    outcomes = raw["outcomes"]
    instruments: list[Instrument] = []
    seen: set[str] = set()
    for item in raw["instruments"]:
        if item["id"] in seen:
            raise ValueError(f"duplicate instrument id {item['id']!r}")
        seen.add(item["id"])
        if item["outcome"] not in outcomes:
            raise ValueError(
                f"instrument {item['id']!r} claims unknown outcome {item['outcome']!r}"
            )
        claims = tuple(_claim_from_raw(c) for c in item.get("claims") or ())
        for claim in claims:
            if claim.blocks and claim.blocks not in outcomes:
                raise ValueError(
                    f"claim {claim.id!r} blocks unknown outcome {claim.blocks!r}"
                )
        instruments.append(
            Instrument(
                id=item["id"],
                title=item["title"],
                theory=item["theory"],
                program_ref=item["program_ref"],
                module=item["module"],
                outcome=item["outcome"],
                unlock=item["unlock"].strip(),
                last_reviewed=str(item["last_reviewed"]),
                storage=Storage(**(item.get("storage") or {"kind": "none"})),
                claims=claims,
                entrypoint=item.get("entrypoint"),
                metrics=tuple(item.get("metrics") or ()),
            )
        )
    return {
        "version": raw["version"],
        "outcomes": outcomes,
        "review_max_age_days": int(raw.get("review_max_age_days", 90)),
        "instruments": instruments,
    }


def _parse_day(value: str) -> date | None:
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def check_repo_presence(inst: Instrument, root: Path | None = None) -> tuple[bool, bool | None]:
    """Does the module this instrument names still exist, with its entrypoint?

    This is the cheap half of the gate and the half that already caught two
    stale paths while the manifest was being written (`novelty_for_target` had
    moved to scoring.py; capability_policy.py lives under orion/autonomy/).
    """
    root = root or REPO_ROOT
    target = root / inst.module
    if not target.exists():
        return False, None
    if not inst.entrypoint:
        return True, None
    if target.is_dir():
        return True, None
    return True, f"def {inst.entrypoint}" in target.read_text()


def _rg_hit_count(pattern: str, root: Path) -> int:
    """Count non-excluded files mentioning `pattern`. 0 means the deletion held."""
    proc = subprocess.run(
        ["rg", "--files-with-matches", "--fixed-strings", pattern, "."],
        cwd=root,
        capture_output=True,
        text=True,
    )
    # rg exits 1 for "no matches" -- that is the passing case here, not an error.
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"rg failed for {pattern!r}: {proc.stderr.strip()}")
    hits = [
        line
        for line in proc.stdout.splitlines()
        if line.strip()
        and not any(line.lstrip("./").startswith(x) for x in _ABSENCE_SCAN_EXCLUDES)
    ]
    return len(hits)


def evaluate_claim(claim: Claim, conn: Any | None, root: Path | None = None) -> ClaimResult:
    """Re-run one claim against live state and compare to its recorded value."""
    root = root or REPO_ROOT
    result = ClaimResult(claim=claim)

    if claim.kind == "manual":
        result.status = "MANUAL"
        result.observed = claim.recorded
        result.detail = "human-run check; reported as recorded, never auto-passed"
        return result

    try:
        if claim.kind == "absent_from_repo":
            result.observed = _rg_hit_count(claim.target or "", root)
        else:
            if conn is None:
                result.status = "ERROR"
                result.detail = "no database connection"
                return result
            with conn.cursor() as cur:
                cur.execute(claim.sql)
                row = cur.fetchone()
            result.observed = row[0] if row else None
    except Exception as exc:  # noqa: BLE001 -- a broken claim must not kill the board
        result.status = "ERROR"
        result.detail = str(exc)
        return result

    recorded = claim.recorded
    # Compare numerically when both sides are numbers, so a YAML int and a
    # Postgres bigint do not read as drift.
    try:
        same = float(recorded) == float(result.observed)
    except (TypeError, ValueError):
        same = recorded == result.observed
    result.status = "HOLDS" if same else "DRIFTED"
    if not same:
        result.detail = f"recorded {recorded!r} on {claim.recorded_at}, live reads {result.observed!r}"
    return result


def _storage_state(inst: Instrument, conn: Any, state: InstrumentState) -> None:
    """Fill in row count, history span, and the retention ceiling bounding it."""
    st = inst.storage
    if st.kind in ("none", "graph") or not st.table:
        state.storage_note = f"no SQL table (kind: {st.kind})"
        return
    ts = st.ts_column or "created_at"
    # Deliberately COUNT(*), not pg_stat_user_tables.n_live_tup: on 2026-09-02
    # n_live_tup read 0 for substrate_attention_self_model while the real count
    # was 19,774. A stats-view read would have rendered a live instrument as dead.
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*), min({ts}), max({ts}) FROM {st.table}")  # noqa: S608
        count, first, last = cur.fetchone()
    state.row_count = count
    state.first_seen = first
    state.last_seen = last
    if first and last:
        state.history_hours = (last - first).total_seconds() / 3600.0
    if st.kind == "singleton":
        state.storage_note = (
            "singleton upsert -- current value only, no history is recoverable "
            "from this table by construction"
        )


def resolve_retention_hours(
    inst: Instrument, root: Path | None = None
) -> tuple[float | None, str]:
    """Read the retention ceiling bounding this instrument's recoverable history.

    Returns (hours, source). Prefers the service `.env` -- the value actually in
    force -- and falls back to `.env_example`.

    The fallback is load-bearing, not defensive: `.env` is gitignored, so it does
    not exist in a fresh worktree or in CI. Without it this returned None there
    and the board silently rendered no ceiling at all, which is the exact failure
    it was built to expose. `source` is reported so a reader can tell a live
    value from a template default rather than having to trust the number.
    """
    root = root or REPO_ROOT
    st = inst.storage
    if not st.retention_setting or not st.retention_service:
        return None, ""
    base = root / "services" / st.retention_service
    for name in (".env", ".env_example"):
        env_path = base / name
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{st.retention_setting}="):
                try:
                    return float(line.split("=", 1)[1].strip()), name
                except ValueError:
                    return None, f"{name} (unparseable)"
    return None, "not found"


def resolve_consumers(inst: Instrument) -> tuple[list[str], str]:
    """Blast radius for this instrument's metrics, from the metric semantic layer.

    Reuses orion.metrics rather than rescanning. `ScanResult.consumers_for()`
    already excludes tests, the registry-of-origin, and low-confidence access
    kinds, and is the same call `check_metric_lineage.py --metric` renders -- so
    the board and that CLI cannot disagree about who consumes what.

    Returns (consumers, note). The note carries the reason whenever the list is
    empty, so an unreachable or failed scan is never rendered as the much
    stronger claim "nothing consumes this."
    """
    if not inst.metrics:
        return [], "instrument declares no metric URNs"
    try:
        from orion.metrics.consumers import scan_repo
        from orion.metrics.lineage import build_graph
    except Exception as exc:  # noqa: BLE001
        return [], f"metric layer unavailable: {exc}"

    try:
        graph = build_graph()
        tokens = {
            node.scan_token
            for urn in inst.metrics
            for node in [graph.nodes.get(urn)]
            if node is not None
        }
        unknown = [u for u in inst.metrics if u not in graph.nodes]
        if unknown:
            # A URN the metric layer cannot resolve is an authoring error in the
            # manifest, not an empty result -- say so rather than report zero.
            return [], f"unresolved metric URN(s): {', '.join(unknown)}"
        scan = scan_repo(tokens)
    except Exception as exc:  # noqa: BLE001
        return [], f"metric layer scan failed: {exc}"

    out: list[str] = []
    for token in sorted(tokens):
        for hit in scan.consumers_for(token):
            loc = f"{hit.path}:{hit.line}"
            if loc not in out:
                out.append(loc)
    return out, "" if out else "metric layer resolved no non-test consumers"


def build_state(
    manifest: dict[str, Any] | None = None,
    conn: Any | None = None,
    root: Path | None = None,
    with_consumers: bool = True,
) -> list[InstrumentState]:
    """Join manifest + repo + live database into what the board renders."""
    manifest = manifest or load_manifest()
    root = root or REPO_ROOT
    max_age = manifest["review_max_age_days"]
    today = datetime.now(timezone.utc).date()

    states: list[InstrumentState] = []
    for inst in manifest["instruments"]:
        state = InstrumentState(instrument=inst)
        state.module_exists, state.entrypoint_exists = check_repo_presence(inst, root)

        reviewed = _parse_day(inst.last_reviewed)
        if reviewed:
            state.review_age_days = (today - reviewed).days
            state.review_stale = state.review_age_days > max_age

        state.retention_hours, state.retention_source = resolve_retention_hours(
            inst, root
        )

        if conn is not None:
            try:
                _storage_state(inst, conn, state)
            except Exception as exc:  # noqa: BLE001
                state.storage_note = f"storage read failed: {exc}"

        state.claims = [evaluate_claim(c, conn, root) for c in inst.claims]

        if with_consumers:
            state.consumers, state.consumer_note = resolve_consumers(inst)

        states.append(state)
    return states

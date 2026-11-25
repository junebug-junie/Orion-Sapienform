"""Site-routing rule for the heartbeat's tensor-network substrate.

WHERE (site_index): `GrammarProvenanceV1.source_service`. Each seat is one
service; a seat cannot drift because the emitter decides it.

HOW (operator_kind): `GrammarAtomV1.atom_type` picks the gate generator.

HOW HARD (level): **2026-11-24 overhaul (#3838).** The old rule read every
atom's confidence/salience/uncertainty as gate strength. A live audit of 3 days
of `grammar_events` (PR report, "Input audit") found that most of those values
are constants the emitter stamps per atom type, not measurements. So each
(service, semantic_role) now has an explicit input rule:

- ``measured``: one field really varies on live data and can read calm on a
  quiet night. Its value becomes ``level`` in [0, 1]; 0 is rest, and a gate at
  level 0 does nothing.
- ``presence``: the atom is event-driven, but its fields are stamped
  constants. Only *that it happened* is real, so it gates at the fixed
  ``PRESENCE_LEVEL``. The seat reads calm by going silent.
- ``skip``: fixed-cadence clockwork with constant fields, a duplicate of
  another atom, a gauge with a structural floor that can never read calm, or
  a value derived from another seat. It carries neither timing nor value, so
  it is not absorbed. Skips are counted per reason on ``/health``.

An unlisted role on a routed service falls back to its service default and is
counted (``inputs_unlisted_role`` on ``/health``), so a new emitter role shows
up in the audit instead of silently gating.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import math
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional

OperatorKind = Literal["amplitude", "phase", "rotation", "projection"]
InputKind = Literal["measured", "presence", "skip"]

# Seats, one per boundary site. A seat is named after its main service.
# 2026-11-24: orion-cortex-orch (about 70 atoms a day, every field stamped)
# folds into the execution seat; orion-harness-governor (the agent harness,
# about 18,500 event-driven atoms a day) takes the freed site 4. N_SITES and
# the boundary/bulk cut are unchanged, so the verdict bands and the probe
# geometry still apply.
SEAT_ORDER: tuple[str, ...] = (
    "orion-hub",  # chat turns (includes Orion's own curiosity/day-letter turns)
    "orion-biometrics",  # body telemetry
    "orion-cortex-exec",  # execution (cortex-exec + cortex-orch)
    "orion-bus",  # transport
    "orion-harness-governor",  # agent harness turns
)
SITE_ORGAN_MAP: dict[int, str] = dict(enumerate(SEAT_ORDER))

# source_service -> site. More than one service may share a seat.
ORGAN_SITE_MAP: dict[str, int] = {
    "orion-hub": 0,
    "orion-biometrics": 1,
    "orion-cortex-exec": 2,
    "orion-cortex-orch": 2,
    "orion-bus": 3,
    "orion-harness-governor": 4,
}

# Catalogued grammar producers that are deliberately not routed, with the
# reason. Read by /health so the gap is named, not hidden. Evidence: the
# 2026-11-24 input audit (PR report).
UNROUTED_PRODUCERS: dict[str, str] = {
    "orion-gpu-pool": "live but small (~90 event atoms/day, stamped fields); no free boundary seat without changing N_SITES",
    "orion-llm-gateway": "fixed-cadence window atoms with stamped fields; real counts exist only in summary text",
    "orion-sql-writer": "fixed-cadence window atoms with stamped fields; real counts exist only in summary text",
    "orion-vision-frame-router": "fixed-cadence window atoms with stamped fields; real counts exist only in summary text",
    "orion-substrate-runtime": "all-null bounded-input snapshots plus an audio spike derived from the biometrics cabinet sensor (not independent)",
    "orion-actions": "catalogued, no atoms in the 3-day ledger",
    "orion-dream": "catalogued, no atoms in the 3-day ledger",
    "orion-durable-runs": "catalogued, no atoms in the 3-day ledger",
    "orion-embodiment": "catalogued, no atoms in the 3-day ledger",
    "orion-mind": "catalogued, no atoms in the 3-day ledger",
    "orion-thought": "catalogued, no atoms in the 3-day ledger",
    "orion-topic-foundry": "catalogued, no atoms in the 3-day ledger",
}

N_SITES = 10
BOUNDARY_SITES: tuple[int, ...] = (0, 1, 2, 3, 4)
BULK_SITES: tuple[int, ...] = (5, 6, 7, 8, 9)
BOUNDARY_BULK_CUT = 5  # mps.entropy(BOUNDARY_BULK_CUT) is the single, cheap,
# native MPS bipartite-entanglement measurement of exactly this cut -- the
# central H1 v0 number.

BOND_DIM = 4  # unchanged from the charter -- kept conservative on purpose,
# see design doc: a larger bond dimension trivially makes reconstruction/
# entanglement easier, which would bias any redundancy finding toward a false
# positive.
PHYS_DIM = 4  # unchanged from the charter.

# atom_type -> operator_kind. Every AtomType from orion/schemas/grammar.py is
# covered explicitly (no silent catch-all) so a future grammar.py addition
# fails loud (KeyError) here rather than being routed by accident.
ATOM_TYPE_OPERATOR_KIND: dict[str, OperatorKind] = {
    "raw_span": "amplitude",
    "observation": "amplitude",
    "signal": "amplitude",
    "entity": "amplitude",
    "claim": "rotation",
    "affective_cue": "phase",
    "salience_marker": "phase",
    "uncertainty_marker": "rotation",
    "memory_claim": "rotation",
    "stance_influence": "phase",
    "reasoning_step": "rotation",
    "action_candidate": "projection",
    "spoken_output": "amplitude",
    "scene_state": "amplitude",
    "compaction": "projection",
    "projection": "projection",
}


# Gate level for an event-driven atom whose fields are stamped constants.
# Equals the old neutral reading (confidence 1.0 x salience 0.5), so a
# presence atom pushes about as hard as a typical atom did before.
PRESENCE_LEVEL = 0.5

# Calm floor of a single |z| reading: for a calm (standard normal) z, E|z| is
# sqrt(2/pi) ~= 0.798 (half-normal mean), not 0. The same floor is
# subtracted in orion/substrate/prediction_error.py for the same reason
# (CLAUDE.md metric gate, 2026-07-26 incident).
ABS_Z_CALM_FLOOR = math.sqrt(2.0 / math.pi)
# orion/grammar/atom_signals.py::uncertainty_from_abs_zscore saturates at
# |z| / 3.0, so uncertainty * 3.0 recovers |z| exactly below saturation.
ABS_Z_SATURATION = 3.0


class SkippedInputError(ValueError):
    """The (service, role) rule says this atom carries no signal. ``reason``
    is the rule's reason, used as the per-reason skip counter key."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _num(atom: Mapping[str, Any], key: str) -> Optional[float]:
    value = atom.get(key)
    if value is None:
        return None
    out = float(value)  # ValueError/TypeError on junk -> caller counts it malformed
    if math.isnan(out):
        return None
    return out


def _salience(atom: Mapping[str, Any]) -> Optional[float]:
    value = _num(atom, "salience")
    return None if value is None else _clamp01(value)


def _inverted_salience(atom: Mapping[str, Any]) -> Optional[float]:
    # stability_signal: high salience means calm (orion.field.pressure
    # HIGHER_IS_BETTER_CHANNELS), so the push is 1 - salience.
    value = _num(atom, "salience")
    return None if value is None else _clamp01(1.0 - value)


def _bus_activity_level(atom: Mapping[str, Any]) -> Optional[float]:
    """services/orion-bus/app/grammar_emit.py::record_bus_activity_zscore_computed.

    The emitter stamps confidence 0.9 when the EWMA tracker has a baseline and
    0.5 when it has none (uncertainty is then a fixed 0.75, not a reading).
    With a baseline, uncertainty = min(1, |z| / 3), so |z| = 3 * uncertainty.
    The level subtracts the calm floor, so a calm bus reads 0 on just over
    half of its ticks (P(|z| < 0.798) = 0.575 for a calm z).

    Known transient: on the tracker's second observation orion/bus/ewma.py
    has no variance yet (1e-6 floor), so |z| saturates and this tick reads
    level 1.0 once after a bus restart.
    """
    confidence = _num(atom, "confidence")
    uncertainty = _num(atom, "uncertainty")
    if confidence is None or uncertainty is None:
        return None
    if confidence < 0.9:
        raise SkippedInputError("no baseline yet: bus activity tracker is warming up")
    abs_z = ABS_Z_SATURATION * _clamp01(uncertainty)
    return _clamp01((abs_z - ABS_Z_CALM_FLOOR) / (ABS_Z_SATURATION - ABS_Z_CALM_FLOOR))


@dataclass(frozen=True)
class InputRule:
    kind: InputKind
    reason: str
    level_fn: Optional[Callable[[Mapping[str, Any]], Optional[float]]] = None


_PRESENCE = InputRule("presence", "event-driven; fields are stamped constants")
_CLOCKWORK = InputRule("skip", "clockwork: fixed cadence and constant fields")


def _measured(reason: str, fn: Callable[[Mapping[str, Any]], Optional[float]] = _salience) -> InputRule:
    return InputRule("measured", reason, fn)


# Fallback rule per routed service, for any role not listed below.
SERVICE_DEFAULT_RULES: dict[str, InputRule] = {
    "orion-hub": _PRESENCE,
    "orion-biometrics": _CLOCKWORK,
    "orion-cortex-exec": _PRESENCE,
    "orion-cortex-orch": _PRESENCE,
    "orion-bus": _CLOCKWORK,
    "orion-harness-governor": _PRESENCE,
}

# (source_service, semantic_role) -> rule. Evidence: 2026-11-24 input audit
# over 3 days of grammar_events (PR report table). "night" = 07-13 UTC.
ROLE_RULES: dict[tuple[str, str], InputRule] = {
    # --- hub (event-driven: one set per chat turn) ---
    ("orion-hub", "repair_pressure_reading"): _measured("repair pressure level; 404 distinct values, night p05 0.02"),
    ("orion-hub", "repair_signal"): _measured("repair pressure level on a repair turn; 34 distinct values"),
    ("orion-hub", "stance_disposition"): _measured("two real stance states (0.3 / 1.0)"),
    ("orion-hub", "user_utterance"): InputRule(
        "presence", "salience is word count, saturated at 1.0 on 78% of turns; presence only"
    ),
    ("orion-hub", "session_context"): InputRule(
        "skip", "duplicate: fires once per turn with user_utterance, all fields stamped"
    ),
    # --- biometrics (fixed cadence, ~16 s; only gauges that can read calm) ---
    ("orion-biometrics", "gpu_pressure_signal"): _measured("gpu util; reads 0 on 36% of night ticks"),
    ("orion-biometrics", "thermal_pressure_signal"): _measured("thermal pressure; reads 0 on 34% of night ticks"),
    ("orion-biometrics", "disk_pressure_signal"): _measured("disk io pressure; reads ~0 on 49% of night ticks"),
    ("orion-biometrics", "stability_signal"): _measured(
        "volatility (1 - stability); reads ~0 on 41% of night ticks", _inverted_salience
    ),
    ("orion-biometrics", "power_pressure_signal"): InputRule(
        "skip", "structural floor: night p05 0.10, at rest on 3% of night ticks"
    ),
    ("orion-biometrics", "memory_pressure_signal"): InputRule("skip", "structural floor: never below 0.207"),
    ("orion-biometrics", "fan_pressure_signal"): InputRule("skip", "structural floor: night p05 0.36"),
    ("orion-biometrics", "disk_capacity_pressure_signal"): InputRule("skip", "fill gauge pinned at ~0.81, not activity"),
    ("orion-biometrics", "body_state"): InputRule("skip", "strain composite of the pressure gauges (not independent); floor 0.12"),
    ("orion-biometrics", "cabinet_climate_activity_signal"): InputRule("skip", "structural floor: night p05 0.075"),
    ("orion-biometrics", "cabinet_proximity_activity_signal"): InputRule("skip", "structural floor: night p05 0.16"),
    ("orion-biometrics", "cabinet_ambient_audio_activity_signal"): InputRule("skip", "structural floor: night p05 0.11"),
    ("orion-biometrics", "cabinet_em_activity_signal"): InputRule("skip", "structural floor: night p05 0.11"),
    ("orion-biometrics", "cabinet_vibration_activity_signal"): InputRule("skip", "structural floor: night p05 0.12"),
    ("orion-biometrics", "cabinet_uv_activity_signal"): InputRule("skip", "pinned at 0.0 for 3 days"),
    ("orion-biometrics", "cabinet_sensor_staleness_signal"): InputRule("skip", "alarm pinned at 0.0 for 3 days"),
    ("orion-biometrics", "cabinet_ambient_audio_staleness_signal"): InputRule("skip", "alarm pinned at 0.0 for 3 days"),
    # Salience stamped; the only varying field is the per-tick telemetry
    # uncertainty that the emitter stamps on every atom of the tick.
    ("orion-biometrics", "node_context"): _CLOCKWORK,
    ("orion-biometrics", "telemetry_sample"): _CLOCKWORK,
    ("orion-biometrics", "capability_surface"): _CLOCKWORK,
    ("orion-biometrics", "node_availability"): _CLOCKWORK,
    # --- bus (fixed cadence, ~11 s) ---
    ("orion-bus", "bus_activity_zscore_computed"): _measured(
        "mesh publish-rate |z| minus its calm floor; reads 0 on 53% of night ticks", _bus_activity_level
    ),
    ("orion-bus", "rpc_transport_timeout"): _PRESENCE,
    ("orion-bus", "bus_census_computed"): InputRule("skip", "clockwork: drift fraction spans 0-0.023 (9 values)"),
    ("orion-bus", "bus_observer_tick_started"): _CLOCKWORK,
    ("orion-bus", "bus_observer_tick_completed"): _CLOCKWORK,
    ("orion-bus", "bus_health_observed"): _CLOCKWORK,
    # --- execution seat: cortex-exec + cortex-orch (event-driven, stamped) ---
    **{
        ("orion-cortex-exec", role): _PRESENCE
        for role in (
            "exec_request_received",
            "exec_plan_started",
            "exec_recall_gate_observed",
            "exec_step_started",
            "exec_step_completed",
            "exec_step_failed",
            "exec_result_assembled",
            "exec_result_emitted",
        )
    },
    ("orion-cortex-orch", "route_arbitration_decided"): InputRule(
        "presence", "about 70 a day; uncertainty stamped 0.42 on all 207 in 3 days"
    ),
    # --- agent harness seat (event-driven, every field stamped or null) ---
    **{
        ("orion-harness-governor", role): _PRESENCE
        for role in (
            "harness_fcc_step",
            "exec_request_received",
            "exec_plan_started",
            "exec_recall_gate_observed",
            "exec_step_started",
            "exec_step_completed",
            "exec_step_failed",
            "exec_result_assembled",
            "exec_result_emitted",
            "exec_tool_provenance_mismatch",
        )
    },
}


@dataclass(frozen=True)
class SiteAssignment:
    """Where and how hard one atom gates the substrate."""

    site_index: int
    operator_kind: OperatorKind
    level: float  # [0, 1]; 0 = rest (no gate)
    input_kind: InputKind
    source_service: str = ""
    semantic_role: str = ""


class UnroutableOrganError(ValueError):
    """source_service is not on any seat. Callers skip and count it."""


class UnroutableAtomTypeError(ValueError):
    """atom_type is not in ATOM_TYPE_OPERATOR_KIND -- a fail-loud guard
    against a future orion/schemas/grammar.py addition."""


GRAMMAR_EVENT_CHANNEL = "orion:grammar:event"
SELF_SOURCE_SERVICE = "orion-heartbeat"


def catalog_grammar_producers() -> frozenset[str]:
    """orion:grammar:event's producer_services from orion/bus/channels.yaml.

    Used to NAME the producers heartbeat does not route (see
    UNROUTED_PRODUCERS for why each is left out). Empty set if the catalog
    can't be read; callers must treat that as "unknown", not "no producers".
    """
    from orion.core.bus.catalog_loader import load_channel_catalog

    entry = load_channel_catalog().get(GRAMMAR_EVENT_CHANNEL) or {}
    producers = entry.get("producer_services") or []
    return frozenset(str(p) for p in producers if isinstance(p, str) and p and p != "*")


def input_rule(source_service: str, semantic_role: str) -> tuple[InputRule, bool]:
    """(rule, listed). ``listed`` is False when the service default applied."""
    rule = ROLE_RULES.get((source_service, semantic_role))
    if rule is not None:
        return rule, True
    return SERVICE_DEFAULT_RULES[source_service], False


def route_atom(*, source_service: str, atom: Mapping[str, Any]) -> SiteAssignment:
    """WHERE + HOW + HOW HARD for one GrammarAtomV1 (as a dict).

    Raises UnroutableOrganError (not on a seat), UnroutableAtomTypeError
    (unknown atom_type), SkippedInputError (rule says skip, or a measured
    field is missing), or ValueError/TypeError (a non-numeric field).
    """
    if source_service not in ORGAN_SITE_MAP:
        raise UnroutableOrganError(source_service)
    atom_type = str(atom.get("atom_type") or "")
    if atom_type not in ATOM_TYPE_OPERATOR_KIND:
        raise UnroutableAtomTypeError(atom_type)
    role = str(atom.get("semantic_role") or "")
    rule, _listed = input_rule(source_service, role)
    if rule.kind == "skip":
        raise SkippedInputError(rule.reason)
    if rule.kind == "presence":
        level = PRESENCE_LEVEL
    else:
        assert rule.level_fn is not None
        measured = rule.level_fn(atom)
        if measured is None:
            raise SkippedInputError("measured field missing")
        level = measured
    return SiteAssignment(
        site_index=ORGAN_SITE_MAP[source_service],
        operator_kind=ATOM_TYPE_OPERATOR_KIND[atom_type],
        level=level,
        input_kind=rule.kind,
        source_service=source_service,
        semantic_role=role,
    )


def _fn_shape(fn: Callable[..., Any]) -> str:
    """The function's code as an AST dump: comments, docstring and line numbers do not count,
    so a comment edit does not change the routing version, but any change in logic does."""
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    except (OSError, TypeError, SyntaxError, IndexError):
        return f"{getattr(fn, '__name__', fn)}:nosource"
    body = getattr(node, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        node.body = body[1:]
    return ast.dump(node)


def routing_version() -> str:
    """Short hash of everything that decides a seat's level: the seat map, the operator map, the
    per-(service, role) rule kinds and level functions, the constants, and the code of the level
    functions and route_atom. Rule reason strings do not count. Stored on every seat rollup row
    (issue #3886) so a reader never mixes sums made under different rules."""
    parts = [
        repr(SEAT_ORDER),
        repr(sorted(ORGAN_SITE_MAP.items())),
        repr(sorted(ATOM_TYPE_OPERATOR_KIND.items())),
        repr((PRESENCE_LEVEL, ABS_Z_CALM_FLOOR, ABS_Z_SATURATION)),
    ]
    fns: dict[str, Callable[..., Any]] = {}
    for key, rule in sorted(SERVICE_DEFAULT_RULES.items()) + sorted(ROLE_RULES.items()):
        name = rule.level_fn.__name__ if rule.level_fn is not None else ""
        parts.append(f"{key}|{rule.kind}|{name}")
        if rule.level_fn is not None:
            fns[name] = rule.level_fn
    for fn in (_num, _clamp01, *[fns[k] for k in sorted(fns)], input_rule, route_atom):
        parts.append(_fn_shape(fn))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]

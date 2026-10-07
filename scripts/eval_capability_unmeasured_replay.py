#!/usr/bin/env python3
"""Replay real field ticks: what each "unmeasured capability" encoding does downstream.

Read-only. Input is a JSONL dump of real `substrate_field_state.field_json` rows:

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \\
      "select field_json::text from (select field_json, row_number() over \\
       (order by generated_at) rn from substrate_field_state where generated_at \\
       > now()-interval '72 hours') x where rn % 60 = 0" > /tmp/field_sample.jsonl
    python scripts/eval_capability_unmeasured_replay.py --dump /tmp/field_sample.jsonl

For every tick and scenario it re-runs reconcile + the capability decay step,
then runs BOTH diffusion implementations on that same pre-diffusion state:

  legacy  `_legacy_apply_diffusion` below: a frozen copy of origin/main's
          apply_diffusion before 2026-10-07 (unmeasured target -> 0.0; derived
          confidence/available_capacity -> 1.0 when pressure is unmeasured).
          Option (c) "keep values + a measured:false marker" is numerically
          identical to legacy for every consumer (none reads a marker), so it
          is reported as legacy.
  a       this branch's apply_diffusion: key absent.
  b0      overlay on legacy: confidence = available_capacity = 0.0 for a
          capability whose pressure nothing measured.
  b1      b0, and every unmeasured target channel reads 1.0 (unknown = alarm).

Learned cap->cap weights (FIELD_PLASTICITY_ENABLED) are switched off for both
implementations, so they read the same designed weights.

Scenarios: natural (as recorded), and simulated outages that drop the live
input channels the way decay.py's expire_unrefreshed_channels() would.

Downstream readers measured (the real functions, not copies):
  - orion.field.pressure.field_pressures_with_provenance: resource_pressure /
    reliability_pressure value and winning source (proposal arena, feedback)
  - orion.field.credit_integrity.channel_write_backed: feedback credit gate
  - merged confidence / available_capacity (corpus, anomaly scorer, glossary)
  - orion.attention.field_attention.selectors._current_pressure_proxy: which
    capability reads most urgent to the attention selector

Fidelity check: `legacy_reconstruction_vs_recorded` compares the rebuilt
legacy vectors with what production actually stored. On the 2026-10-04..07
sample it matched 1,949 / 2,051 ticks; every mismatch is capability:
orchestration, whose llm_inference -> orchestration edge reads the PREVIOUS
tick's llm_inference pressure (not stored in the row), ratio ~1.000.
`a_measured_channel_mismatch` counts ticks where a channel BOTH
implementations wrote differs -- a regression on measured channels, which a
value-only "a vs legacy" diff could not see.

Result on that sample (2,051 ticks, every 60th of 123,099): `a` changed no
downstream reading vs legacy in any scenario. `b0` made capability:vision the
attention selector's most urgent capability on 2,051 / 2,051 vision-outage
ticks and pinned merged confidence/available_capacity at 0.0. `b1` pinned
resource_pressure at 1.0 on every all-outage tick and flipped feedback credit.
Natural data: 0 ticks with an unmeasured capability channel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIGESTER = REPO / "services" / "orion-field-digester"
for p in (str(DIGESTER), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.digestion.decay import CAPABILITY_DECAY_CHANNELS  # noqa: E402
from app.digestion.diffusion import apply_diffusion  # noqa: E402
from app.graph.lattice import load_lattice  # noqa: E402
from app.tensor.reconcile import reconcile_field_state_with_lattice  # noqa: E402

from orion.attention.field_attention.selectors import _current_pressure_proxy  # noqa: E402
from orion.field.credit_integrity import channel_write_backed  # noqa: E402
from orion.field.pressure import collect_field_channel_pressures, field_pressures_with_provenance  # noqa: E402
from orion.schemas.field_state import FieldStateV1  # noqa: E402

TOPOLOGY = REPO / "config" / "field" / "orion_field_topology.v1.yaml"
SCENARIOS: dict[str, dict[str, tuple[str, ...]]] = {
    "natural": {},
    "outage:vision": {"node:substrate.vision_organ": ("vision_frame_staleness", "vision_processing_failure_pressure")},
    "outage:storage": {"node:substrate.storage_write": ("write_failure_pressure",)},
    "outage:rpc": {"node:substrate.rpc_delivery": ("rpc_timeout_pressure",)},
}
SCENARIOS["outage:all"] = {k: v for s in list(SCENARIOS.values()) for k, v in s.items()}
VARIANTS = ("legacy", "a", "b0", "b1")
DIMS = ("resource_pressure", "reliability_pressure")
DERIVED = ("confidence", "available_capacity")
CREDIT_STALENESS_SEC = 120.0
DECAY_RATE = 0.92  # BIOMETRICS_FIELD_DECAY_RATE, live value


def _decay_capabilities(state: FieldStateV1) -> None:
    """apply_decay()'s capability loop, which runs between reconcile and
    diffusion in the real tick. Node vectors are NOT re-decayed: the recorded
    ones are already this tick's post-decay values. Without this, the
    capability->capability edge (llm_inference -> orchestration) reads an
    undecayed source and the replay drifts from what production stored."""
    for vec in state.capability_vectors.values():
        for ch in CAPABILITY_DECAY_CHANNELS:
            if ch in vec:
                vec[ch] = vec[ch] * DECAY_RATE


def _pressure_targets(state: FieldStateV1) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for e in state.edges:
        out.setdefault(e.target_id, set()).update(e.channel_map.values())
    return out


def _clamp01_legacy(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _legacy_apply_diffusion(state: FieldStateV1, *, diffusion_rate: float) -> None:
    """Frozen copy of origin/main bbc51703a apply_diffusion (pre-2026-10-07). Do not edit."""
    best_contribution: dict[tuple[str, str], float] = {}
    best_source: dict[tuple[str, str], str] = {}
    measured_zero_source: dict[tuple[str, str], str] = {}
    possible_targets: dict[str, set[str]] = {}

    for edge in state.edges:
        src = state.node_vectors.get(edge.source_id) or state.capability_vectors.get(edge.source_id, {})
        effective_weight = edge.weight
        for src_ch, tgt_ch in edge.channel_map.items():
            possible_targets.setdefault(edge.target_id, set()).add(tgt_ch)
            src_measured = src_ch in src
            src_val = float(src.get(src_ch, 0.0))
            contribution = _clamp01_legacy(src_val * effective_weight * diffusion_rate)
            key = (edge.target_id, tgt_ch)
            if contribution > 0.0 and contribution >= best_contribution.get(key, 0.0):
                best_contribution[key] = contribution
                best_source[key] = edge.source_id
            elif src_measured:
                measured_zero_source.setdefault(key, edge.source_id)

    for target_id, channels in possible_targets.items():
        tgt = state.capability_vectors.setdefault(target_id, {})
        provenance = state.capability_provenance.setdefault(target_id, {})
        for tgt_ch in channels:
            key = (target_id, tgt_ch)
            tgt[tgt_ch] = best_contribution.get(key, 0.0)
            attributed = best_source.get(key) or measured_zero_source.get(key)
            if attributed is not None:
                provenance[tgt_ch] = attributed
            else:
                provenance.pop(tgt_ch, None)

        if "pressure" in tgt:
            if (target_id, "available_capacity") not in best_source:
                tgt["available_capacity"] = max(0.0, 1.0 - tgt.get("pressure", 0.0))
            if (target_id, "confidence") not in best_source:
                tgt["confidence"] = max(0.0, 1.0 - 0.5 * tgt.get("pressure", 0.0))


def _unmeasured(state: FieldStateV1) -> dict[str, set[str]]:
    """Target channels with no provenance after diffusion = nothing measured them."""
    out: dict[str, set[str]] = {}
    for cap, chans in _pressure_targets(state).items():
        prov = state.capability_provenance.get(cap, {})
        miss = {ch for ch in chans if ch not in prov}
        if miss:
            out[cap] = miss
    return out


def build_variants(pre: FieldStateV1) -> dict[str, FieldStateV1]:
    """Run both real implementations on copies of the same pre-diffusion state,
    then lay the b-options over legacy."""
    a = pre.model_copy(deep=True)
    apply_diffusion(a, diffusion_rate=1.0)
    legacy = pre.model_copy(deep=True)
    _legacy_apply_diffusion(legacy, diffusion_rate=1.0)
    out = {"legacy": legacy, "a": a}
    unmeasured = _unmeasured(legacy)
    for name in ("b0", "b1"):
        s = legacy.model_copy(deep=True)
        for cap, chans in unmeasured.items():
            vec = s.capability_vectors[cap]
            if "pressure" in chans:
                for d in DERIVED:
                    vec[d] = 0.0
            if name == "b1":
                for ch in chans:
                    vec[ch] = 1.0
        out[name] = s
    return out


def readout(state: FieldStateV1) -> dict:
    dims, detail = field_pressures_with_provenance(state)
    merged, prov = collect_field_channel_pressures(state)
    proxies = {cap: _current_pressure_proxy(vec) for cap, vec in state.capability_vectors.items()}
    top = max(proxies, key=lambda c: (proxies[c], c)) if proxies else None
    return {
        "dims": {d: (round(dims.get(d, -1.0), 6), detail[d].winning_source_id if d in detail else None) for d in DIMS},
        "backed": {d: channel_write_backed(state, d, max_staleness_seconds=CREDIT_STALENESS_SEC) for d in DIMS},
        "merged": {d: (round(merged.get(d, -1.0), 6), prov.get(d)) for d in (*DERIVED, "pressure", "reliability_pressure")},
        "top_cap": top,
        "proxies": proxies,
    }


def run(dump: Path, limit: int | None = None) -> dict:
    os.environ.pop("FIELD_PLASTICITY_ENABLED", None)
    lattice = load_lattice(TOPOLOGY)
    stats: dict[str, Counter] = {s: Counter() for s in SCENARIOS}
    legacy_matches_recorded = Counter()
    n = 0
    with dump.open() as fh:
        for line in fh:
            if limit is not None and n >= limit:
                break
            line = line.strip()
            if not line:
                continue
            recorded = FieldStateV1.model_validate_json(line)
            n += 1
            for scen, drops in SCENARIOS.items():
                st = recorded.model_copy(deep=True)
                for node, chans in drops.items():
                    for ch in chans:
                        st.node_vectors.get(node, {}).pop(ch, None)
                st = reconcile_field_state_with_lattice(st, lattice=lattice)
                _decay_capabilities(st)
                variants = build_variants(st)
                reads = {v: readout(s) for v, s in variants.items()}
                c = stats[scen]
                c["ticks"] += 1
                a_state, leg_state = variants["a"], variants["legacy"]
                c["ticks_with_unmeasured_capability_channel"] += int(bool(_unmeasured(leg_state)))
                c["a_measured_channel_mismatch"] += int(any(
                    abs(val - leg_state.capability_vectors.get(cap, {}).get(ch, -9.0)) > 1e-12
                    for cap, vec in a_state.capability_vectors.items()
                    for ch, val in vec.items()
                ))
                base = reads["legacy"]
                for v in VARIANTS:
                    r = reads[v]
                    for d in DIMS:
                        c[f"{v}:{d}:value_differs"] += int(r["dims"][d][0] != base["dims"][d][0])
                        c[f"{v}:{d}:winner_differs"] += int(r["dims"][d][1] != base["dims"][d][1])
                        c[f"{v}:{d}:credit_backed"] += int(r["backed"][d] is True)
                        c[f"{v}:{d}:credit_differs"] += int(r["backed"][d] != base["backed"][d])
                    for d in r["merged"]:
                        c[f"{v}:merged_{d}:value_differs"] += int(r["merged"][d][0] != base["merged"][d][0])
                        c[f"{v}:merged_{d}:winner_differs"] += int(r["merged"][d][1] != base["merged"][d][1])
                        c[f"{v}:merged_{d}:zero"] += int(r["merged"][d][0] == 0.0)
                    c[f"{v}:attention_top_cap_differs"] += int(r["top_cap"] != base["top_cap"])
                    outage_caps = {
                        "outage:vision": "capability:vision",
                        "outage:storage": "capability:storage",
                        "outage:rpc": "capability:transport",
                    }
                    oc = outage_caps.get(scen)
                    if oc:
                        c[f"{v}:outage_cap_is_attention_top"] += int(r["top_cap"] == oc)
                        c[f"{v}:outage_cap_proxy_sum"] += r["proxies"].get(oc, 0.0)
                    if scen == "outage:all":
                        c[f"{v}:resource_pressure_is_1.0"] += int(r["dims"]["resource_pressure"][0] == 1.0)
                if scen == "natural":
                    # Validates the legacy reconstruction against what production stored.
                    same = all(
                        abs(variants["legacy"].capability_vectors.get(cap, {}).get(ch, -9) - val) < 1e-9
                        for cap, vec in recorded.capability_vectors.items()
                        for ch, val in vec.items()
                    )
                    legacy_matches_recorded["match" if same else "mismatch"] += 1
    return {"ticks": n, "legacy_reconstruction_vs_recorded": dict(legacy_matches_recorded), "scenarios": {k: dict(v) for k, v in stats.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    print(json.dumps(run(args.dump, args.limit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

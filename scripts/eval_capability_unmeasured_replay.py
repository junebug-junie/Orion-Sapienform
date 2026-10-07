#!/usr/bin/env python3
"""Replay real field ticks: what each "unmeasured capability" encoding does downstream.

Read-only. Input is a JSONL dump of real `substrate_field_state.field_json` rows:

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \\
      "select field_json::text from (select field_json, row_number() over \\
       (order by generated_at) rn from substrate_field_state where generated_at \\
       > now()-interval '72 hours') x where rn % 60 = 0" > /tmp/field_sample.jsonl
    python scripts/eval_capability_unmeasured_replay.py --dump /tmp/field_sample.jsonl

For every tick and scenario it re-runs reconcile + apply_diffusion (this branch:
an unmeasured capability channel is DROPPED), then rebuilds the alternatives
from that result:

  legacy  main before 2026-10-07: unmeasured target -> 0.0; derived
          confidence/available_capacity -> 1.0 when pressure is unmeasured.
          Option (c) "keep values + a measured:false marker" is numerically
          identical to legacy for every consumer (none reads a marker), so it
          is reported as legacy.
  a       this branch: key absent.
  b0      legacy, but confidence = available_capacity = 0.0 when unmeasured.
  b1      b0, and the unmeasured pressure channels read 1.0 (unknown = alarm).

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

Result on that sample (2,051 ticks, every 60th of 123,099): `a` changed no
downstream reading vs legacy in any scenario. `b0` made capability:vision the
attention selector's most urgent capability on 2,051 / 2,051 vision-outage
ticks and pinned merged confidence/available_capacity at 0.0. `b1` pinned
resource_pressure at 1.0 on every all-outage tick and flipped feedback credit.
Natural data: 0 ticks with an unmeasured capability channel.
"""
from __future__ import annotations

import argparse
import copy
import json
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


def build_variants(state_a: FieldStateV1) -> dict[str, FieldStateV1]:
    """state_a is this branch's output; rebuild the alternatives from it."""
    targets = _pressure_targets(state_a)
    out = {"a": state_a}
    for name in ("legacy", "b0", "b1"):
        s = state_a.model_copy(deep=True)
        for cap, chans in targets.items():
            vec = s.capability_vectors.setdefault(cap, {})
            missing = {ch for ch in chans if ch not in vec}
            for ch in missing:
                vec[ch] = 1.0 if name == "b1" else 0.0
            if "pressure" in chans:
                for d in DERIVED:
                    if d not in vec:
                        vec[d] = 1.0 if name == "legacy" else 0.0
                if name == "b1" and "pressure" in missing:
                    # legacy derivation from the alarm pressure would also be 0.5/0.0;
                    # b1 keeps b0's explicit zeros.
                    pass
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
        "merged": {d: (round(merged.get(d, -1.0), 6), prov.get(d)) for d in DERIVED},
        "top_cap": top,
        "proxies": proxies,
    }


def run(dump: Path, limit: int | None = None) -> dict:
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
                apply_diffusion(st, diffusion_rate=1.0)
                variants = build_variants(st)
                reads = {v: readout(s) for v, s in variants.items()}
                c = stats[scen]
                c["ticks"] += 1
                unmeasured = sum(
                    1
                    for cap, chans in _pressure_targets(st).items()
                    for ch in chans
                    if ch not in st.capability_vectors.get(cap, {})
                )
                c["ticks_with_unmeasured_capability_channel"] += int(unmeasured > 0)
                base = reads["legacy"]
                for v in VARIANTS:
                    r = reads[v]
                    for d in DIMS:
                        c[f"{v}:{d}:value_differs"] += int(r["dims"][d][0] != base["dims"][d][0])
                        c[f"{v}:{d}:winner_differs"] += int(r["dims"][d][1] != base["dims"][d][1])
                        c[f"{v}:{d}:credit_backed"] += int(r["backed"][d] is True)
                        c[f"{v}:{d}:credit_differs"] += int(r["backed"][d] != base["backed"][d])
                    for d in DERIVED:
                        c[f"{v}:merged_{d}:value_differs"] += int(r["merged"][d][0] != base["merged"][d][0])
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

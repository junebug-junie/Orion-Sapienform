#!/usr/bin/env python3
"""Replay real field ticks through two code trees: D3 rename, #2534 decisions 1, 2, 4.

Read-only. Runs the SAME dump through a frozen tree (origin/main, before) and
this branch (after), then diffs what every downstream reader saw. "Before" is
main's real code, not a reconstruction (the #2534 review lesson).

Dumps (bounded 72 h window, every 60th tick plus the tick before it, so
novelty has a real consecutive prior):

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "
    with ids as (select tick_id, row_number() over (order by generated_at) rn
      from substrate_field_state where generated_at > '<t0>' and generated_at <= '<t1>')
    select json_build_object('rn', ids.rn, 'field', s.field_json)::text from ids
      join substrate_field_state s using (tick_id) where ids.rn % 60 in (59, 0)
      order by ids.rn" > field_pairs.jsonl

Feedback pairs: real feedback frames, before = the dispatch's source field
tick, after = the first field tick within 30 s of the dispatch (the feedback
runtime's own load_latest_field_after rule). See the PR report for the SQL.

Usage:
    python scripts/eval_field_decisions_replay.py run --tree <repo> --pairs field_pairs.jsonl --out main.jsonl
    python scripts/eval_field_decisions_replay.py run --tree . --pairs field_pairs.jsonl --out branch.jsonl
    python scripts/eval_field_decisions_replay.py compare main.jsonl branch.jsonl
    python scripts/eval_field_decisions_replay.py feedback --tree . --pairs feedback_pairs.jsonl

Per tick and scenario the tree's own reconcile -> capability decay ->
apply_diffusion runs on the stored pre-diffusion state. Scenarios drop the live
input channels on the AFTER tick only, the way decay.py's
expire_unrefreshed_channels would when a producer stops:
natural, outage:vision, outage:storage, outage:rpc, outage:all.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from collections import Counter
from pathlib import Path

SCENARIOS: dict[str, dict[str, tuple[str, ...]]] = {
    "natural": {},
    "outage:vision": {"node:substrate.vision_organ": ("vision_frame_staleness", "vision_processing_failure_pressure")},
    "outage:storage": {"node:substrate.storage_write": ("write_failure_pressure",)},
    "outage:rpc": {"node:substrate.rpc_delivery": ("rpc_timeout_pressure",)},
}
SCENARIOS["outage:all"] = {k: v for s in list(SCENARIOS.values()) for k, v in s.items()}
CREDIT_DIMS = ("resource_pressure", "reliability_pressure", "execution_pressure")
DECAY_RATE = 0.92  # BIOMETRICS_FIELD_DECAY_RATE, live value
# Names this branch changed on purpose: compared after mapping, not as diffs.
RENAMED = {"contract_pressure": "catalog_drift_pressure"}
RETIRED = {"observer_failure_pressure"}


def _load_tree(tree: Path):
    os.environ["FIELD_PLASTICITY_ENABLED"] = "false"
    for p in (str(tree / "services" / "orion-field-digester"), str(tree)):
        sys.path.insert(0, p)
    from app.digestion.decay import CAPABILITY_DECAY_CHANNELS
    from app.digestion.diffusion import apply_diffusion
    from app.graph.lattice import load_lattice
    from app.tensor.reconcile import reconcile_field_state_with_lattice
    from orion.attention.field_attention.builder import build_attention_frame
    from orion.attention.field_attention.policy import load_attention_policy
    import orion.field.credit_integrity as ci
    from orion.field.pressure import collect_field_channel_pressures, field_pressures_with_provenance
    from orion.schemas.field_state import FieldStateV1

    lattice = load_lattice(tree / "config" / "field" / "orion_field_topology.v1.yaml")
    policy = load_attention_policy(tree / "config" / "attention" / "field_attention_policy.v1.yaml")
    takes_prev_field = "previous_field" in inspect.signature(build_attention_frame).parameters

    def digest(raw: dict, drop: dict[str, tuple[str, ...]]) -> FieldStateV1:
        state = FieldStateV1.model_validate(raw)
        for node, chans in drop.items():
            vec = state.node_vectors.get(node) or {}
            stamps = state.node_vector_updated_at.get(node) or {}
            for ch in chans:
                vec.pop(ch, None)
                stamps.pop(ch, None)
        state = reconcile_field_state_with_lattice(state, lattice=lattice)
        for vec in state.capability_vectors.values():
            for ch in CAPABILITY_DECAY_CHANNELS:
                if ch in vec:
                    vec[ch] = vec[ch] * DECAY_RATE
        apply_diffusion(state, diffusion_rate=1.0)
        return state

    def attention(prev, cur):
        prev_frame = build_attention_frame(field=prev, policy=policy, now=prev.generated_at)
        kwargs = {"previous_field": prev} if takes_prev_field else {}
        return build_attention_frame(field=cur, policy=policy, previous_frame=prev_frame, now=cur.generated_at, **kwargs)

    return {
        "FieldStateV1": FieldStateV1,
        "digest": digest,
        "attention": attention,
        "merge": collect_field_channel_pressures,
        "dims": field_pressures_with_provenance,
        "ci": ci,
    }


def _r(x: float) -> float:
    return round(float(x), 9)


def _targets(frame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for bucket in ("dominant_targets", "node_targets", "capability_targets", "suppressed_targets"):
        for t in getattr(frame, bucket):
            out.setdefault(t.target_id, {"kind": t.target_kind, "novelty": _r(t.novelty_score), "pressure": _r(t.pressure_score), "salience": _r(t.salience_score)})
    return out


def _measure(T, prev, cur) -> dict:
    merged, _prov = T["merge"](cur)
    dims, detail = T["dims"](cur)
    frame = T["attention"](prev, cur)
    targets = _targets(frame)
    caps = {k: v for k, v in targets.items() if v["kind"] == "capability"}
    rec = {
        "dims": {k: _r(v) for k, v in dims.items()},
        "winners": {k: [d.winning_channel, d.winning_source_id] for k, d in detail.items()},
        "merged": {k: _r(v) for k, v in merged.items()},
        "cap_vectors": {c: {k: _r(v) for k, v in vec.items()} for c, vec in cur.capability_vectors.items()},
        "targets": targets,
        "top_capability": max(caps, key=lambda k: (caps[k]["salience"], k)) if caps else None,
        "dominant": [t.target_id for t in frame.dominant_targets],
        "transport_reliability_measured": "reliability_pressure" in (cur.capability_vectors.get("capability:transport") or {}),
        "backed": {d: T["ci"].channel_write_backed(cur, d, max_staleness_seconds=120.0) for d in CREDIT_DIMS},
    }
    guard = getattr(T["ci"], "before_winner_went_unmeasured", None)
    if guard is not None:
        rec["guard"] = {d: guard(prev, cur, d, max_staleness_seconds=120.0) for d in CREDIT_DIMS}
    return rec


def cmd_run(args) -> int:
    T = _load_tree(Path(args.tree).resolve())
    rows = [json.loads(line) for line in Path(args.pairs).read_text().splitlines() if line.strip()]
    by_rn = {r["rn"]: r["field"] for r in rows}
    n = 0
    with open(args.out, "w") as out:
        for rn in sorted(by_rn):
            if rn % 60 != 0 or (rn - 1) not in by_rn:
                continue
            for name, drop in SCENARIOS.items():
                prev = T["digest"](by_rn[rn - 1], {})
                cur = T["digest"](by_rn[rn], drop)
                out.write(json.dumps({"rn": rn, "scenario": name, **_measure(T, prev, cur)}) + "\n")
            n += 1
    print(f"{n} tick pairs x {len(SCENARIOS)} scenarios -> {args.out}")
    return 0


def _rename(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if k in RETIRED:
            continue
        out[RENAMED.get(k, k)] = v
    return out


def cmd_compare(args) -> int:
    def load(p):
        return {(r["rn"], r["scenario"]): r for r in map(json.loads, Path(p).read_text().splitlines())}

    a, b = load(args.before), load(args.after)
    c: Counter = Counter()
    examples: dict[str, list] = {}
    only_a, only_b = set(a) - set(b), set(b) - set(a)
    print(f"rows: both={len(set(a) & set(b))} only_before={len(only_a)} only_after={len(only_b)}")
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        sc = key[1]
        c[(sc, "ticks")] += 1

        def diff(label, left, right):
            if left != right:
                c[(sc, label)] += 1
                examples.setdefault(f"{sc}:{label}", []).append((key[0], left, right))

        diff("dims", x["dims"], y["dims"])
        diff("dim_winner_channel", {k: v[0] for k, v in x["winners"].items()}, {k: v[0] for k, v in y["winners"].items()})
        diff("dim_winner_source", {k: v[1] for k, v in x["winners"].items()}, {k: v[1] for k, v in y["winners"].items()})
        # The merge is by channel NAME across node and capability levels. On
        # main, contract_pressure (0.85 x drift) is its own key; on the branch
        # that capability value joins the node-level catalog_drift_pressure
        # key, where the max() must still pick the node reading. So: drop the
        # old key and the retired one, compare everything else byte for byte.
        mx = {k: v for k, v in x["merged"].items() if k not in RENAMED and k not in RETIRED}
        my = {k: v for k, v in y["merged"].items() if k not in RETIRED}
        diff("merged_values", mx, my)
        # Key sets after the intended rename/retirement: any remaining
        # difference is a channel that appeared or vanished unexpectedly.
        diff(
            "merged_keys_unexpected",
            sorted(k for k in x["merged"] if k not in RENAMED and k not in RETIRED),
            sorted(k for k in y["merged"] if k not in RETIRED),
        )
        cvx = {cap: _rename(v) for cap, v in x["cap_vectors"].items()}
        cvy = {cap: _rename(v) for cap, v in y["cap_vectors"].items()}
        diff("cap_vectors_after_rename", cvx, cvy)
        diff("attention_top_capability", x["top_capability"], y["top_capability"])
        diff("attention_dominant", x["dominant"], y["dominant"])
        nx = {k: v["novelty"] for k, v in x["targets"].items()}
        ny = {k: v["novelty"] for k, v in y["targets"].items()}
        diff("novelty_any", nx, ny)
        for k in set(nx) | set(ny):
            if nx.get(k) != ny.get(k):
                c[(sc, f"novelty:{k}")] += 1
        px = {k: v["pressure"] for k, v in x["targets"].items()}
        py = {k: v["pressure"] for k, v in y["targets"].items()}
        diff("attention_pressure_proxy", px, py)
        diff("credit_backed", x["backed"], y["backed"])
        if x["transport_reliability_measured"]:
            c[(sc, "transport_reliability_measured_before")] += 1
        if y["transport_reliability_measured"]:
            c[(sc, "transport_reliability_measured_after")] += 1
        for d, fired in (y.get("guard") or {}).items():
            if fired:
                c[(sc, f"guard_fired:{d}")] += 1
    for sc in SCENARIOS:
        row = {lab: n for (s, lab), n in sorted(c.items()) if s == sc}
        print(sc, json.dumps(row, sort_keys=True))
    if args.examples:
        for k, ex in sorted(examples.items()):
            print(k, json.dumps(ex[: args.examples])[:2000])
    # Gate: on natural data nothing a cognition consumer reads may change.
    # Expected natural differences: none, except capability:transport
    # reliability going unmeasured on a tick where the RPC bridge expired
    # (decision 4) -- that is cap_vectors_after_rename and is reported, not gated.
    gated = (
        "dims", "dim_winner_channel", "dim_winner_source", "merged_values",
        "merged_keys_unexpected", "attention_top_capability", "attention_dominant",
        "novelty_any", "attention_pressure_proxy", "credit_backed",
    )
    bad = {lab: c[("natural", lab)] for lab in gated if c[("natural", lab)]}
    bad.update({f"guard_fired:{d}": c[("natural", f"guard_fired:{d}")] for d in CREDIT_DIMS if c[("natural", f"guard_fired:{d}")]})
    if only_a or only_b:
        bad["unpaired_rows"] = len(only_a) + len(only_b)
    if bad:
        print("GATE FAIL (natural data moved):", json.dumps(bad, sort_keys=True))
        return 1
    print("GATE PASS: natural data -- no consumer-visible change beyond the rename/retirement")
    return 0


def cmd_feedback(args) -> int:
    """Decision 1 on real feedback frames: how often would the guard withhold?"""
    T = _load_tree(Path(args.tree).resolve())
    FS = T["FieldStateV1"]
    ci = T["ci"]
    c: Counter = Counter()
    for line in Path(args.pairs).read_text().splitlines():
        r = json.loads(line)
        if not r.get("before") or not r.get("after"):
            c["missing_window"] += 1
            continue
        c["frames"] += 1
        for mode in ("stored", "redigested"):
            if mode == "stored":
                before, after = FS.model_validate(r["before"]), FS.model_validate(r["after"])
            else:
                before, after = T["digest"](r["before"], {}), T["digest"](r["after"], {})
            for d in CREDIT_DIMS:
                backed = ci.channel_write_backed(after, d, max_staleness_seconds=120.0)
                if backed is not True:
                    c[f"{mode}:r5b_withheld:{d}"] += 1
                elif ci.before_winner_went_unmeasured(before, after, d, max_staleness_seconds=120.0):
                    c[f"{mode}:guard_withheld:{d}"] += 1
                    holder = ci.dimension_winner_holder(before, d)
                    c[f"{mode}:guard_holder:{d}:{holder}"] += 1
    print(json.dumps(dict(sorted(c.items())), indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tree", required=True)
    r.add_argument("--pairs", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--examples", type=int, default=0)
    f = sub.add_parser("feedback")
    f.add_argument("--tree", required=True)
    f.add_argument("--pairs", required=True)
    args = ap.parse_args()
    return {"run": cmd_run, "compare": cmd_compare, "feedback": cmd_feedback}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

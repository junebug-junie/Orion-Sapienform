"""Are the atlas's HTTP calls and Redis keys true, and do the v2 gap rules find what a reviewer
expects? (patch 3, ``orion/body_atlas/wire_code.py`` and ``capacities.py`` rules v2)

1. **Scanner facts** on this checkout, hand-checked with the line that proves each (read
   2026-10-31). ``TRUE_*``: what the code really does; every one the scan still finds must carry
   that direction, and recall is printed. ``FALSE_*``: what an earlier draft claimed and is wrong
   (a reply or a dict read as a Redis key, a model's URL default read as a call by everyone who
   imports the model). None may come back. Gate: precision 1.0 on the judged set.
2. **Rules on two frozen bodies.** ``whole_body_2026-10-29_wires.json`` (held out: the 10-29 body
   with the HTTP calls and Redis keys of main at its export time, no live Redis read; no knob was
   set on it) and ``whole_body_2026-10-31_wires.json`` (live, with Redis key liveness). For each:
   R-wire-seen and R-broadcast-key hit exactly the reviewed sets below (precision and recall 1.0),
   and the organ gate holds (``run_organ_structure_eval.evaluate``).
3. **Mutations that must fail:** the scan with no Redis-handle check (any ``.get("x")`` is a key);
   R-broadcast-key with no reader floor; R-wire-seen counting any flow (placement, the census).

The expected sets are the only hand-written truth here; the rules never read them.

    python orion/body_atlas/evals/run_wire_eval.py
    python orion/body_atlas/evals/run_wire_eval.py --make-held-out CHECKOUT_OF_MAIN_AT_10-29T21:05Z
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from orion.body_atlas import capacities as K  # noqa: E402
from orion.body_atlas import wire_code as W  # noqa: E402
from orion.body_atlas.build import build_atlas  # noqa: E402
from orion.body_atlas.sources import BodyInputs, apply_wire  # noqa: E402

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import run_organ_structure_eval as _structure  # noqa: E402

HELD_OUT_BASE = _HERE / "fixtures" / "whole_body_2026-10-29.json"
HELD_OUT = _HERE / "fixtures" / "whole_body_2026-10-29_wires.json"
TODAY = _HERE / "fixtures" / "whole_body_2026-10-31_wires.json"
HELD_OUT_COMMIT = "94f9043536c33eaa5a1e100a0b7d793a3fe70b86"  # main at the 10-29 export (21:05Z)

# (service, URL): read 2026-10-31
TRUE_CALLS = [
    # services/orion-proposal-runtime/app/worker.py:96 urlopen(self._settings.hardware_watch_health_url)
    ("orion-proposal-runtime", "http://hardware-watch:8131/health"),
    # services/orion-proposal-runtime/app/worker.py:162 urlopen(baseline_policy.thought_url + "/visual-chain/activity")
    ("orion-proposal-runtime", "http://orion-athena-thought:7155"),
    # services/orion-graphiti-adapter/app/main.py:148 embed_url=settings.CRYSTALLIZER_EMBED_HOST_URL
    ("orion-graphiti-adapter", "http://orion-athena-vector-host:8320/embedding"),
    # services/orion-thought/app/physiology_shadow.py:62 orion.physiology.client.abid -> DEFAULT_URL
    ("orion-thought", "http://orion-athena-durable-runs:8121/physiology/bid"),
    # services/orion-notify-digest/app/main.py: NOTIFY_URL default
    ("orion-notify-digest", "http://notify:7140"),
    # services/orion-thought/app/mind_enrichment.py:395 getattr(settings, "mind_base_url", "")
    ("orion-thought", "http://mind:6611"),
]
FALSE_CALLS = [
    # orion/policy/evaluator.py imports validate_eligibility, which names VisualBaselinePolicy; its
    # thought_url default is config. orion-policy-runtime never reads .thought_url
    ("orion-policy-runtime", "http://orion-athena-thought:7155"),
    ("orion-execution-dispatch-runtime", "http://orion-athena-thought:7155"),
    # services/orion-world-pulse/app/settings.py:157 actions_url: no code reads the field
    ("orion-world-pulse", "http://actions:8110"),
    # ConceptSettings.embedding_base_url is read only by the concept inducer, not by cortex-exec
    ("orion-cortex-exec", "http://orion-athena-vector-host:8320"),
]
# (service, key, relation): read 2026-10-31
TRUE_KEYS = [
    ("orion-durable-runs", "orion:regulation:latest", "writes"),  # app/main.py:307 rpc_bus.redis.setex
    ("orion-dream", "orion:regulation:latest", "reads"),  # orion/regulation/arousal_read.py client.get
    ("orion-proposal-runtime", "orion:regulation:latest", "reads"),
    ("orion-durable-runs", "equilibrium:state", "reads"),  # app/main.py:674 hgetall(EQUILIBRIUM_STATE_REDIS_KEY)
    ("orion-durable-runs", "orion:situation:latest", "writes"),  # app/main.py:229 setex(SITUATION_STATE_REDIS_KEY)
    ("orion-dream", "orion:drive:rest:latest", "writes"),  # app/main.py:236 setex(REST_DRIVE_REDIS_KEY)
]
FALSE_KEYS = [
    # persistence = await redis.info("persistence"); persistence.get("rdb_current_bgsave_time_sec"): a reply
    ("orion-mesh-guardian", "rdb_current_bgsave_time_sec", "reads"),
    ("orion-mesh-guardian", "save", "reads"),  # cfg = await redis.config_get("save"); cfg.get("save")
    ("orion-mind", "data", "reads"),  # raw = fields.get(...); raw.get("data"): a pub/sub message
    ("orion-hub", "data", "reads"),
    ("orion-actions", "channel", "reads"),  # the bus chassis: msg.get("channel") in every service
    ("orion-mind", "signal", "reads"),
]

EXPECTED = {
    HELD_OUT.name: {
        "R-wire-seen": {"part:notify-digest", "part:orion-graphiti-adapter", "part:orion-pageindex",
                        "part:orion-proposal-runtime", "part:orion-self-experiments"},
        "R-broadcast-key": {"part:redis:orion:regulation:latest"},
    },
    TODAY.name: {
        "R-wire-seen": {"part:notify-digest", "part:orion-graphiti-adapter", "part:orion-pageindex",
                        "part:orion-proposal-runtime", "part:orion-self-experiments"},
        "R-broadcast-key": {"part:redis:orion:regulation:latest"},
    },
}


def score(got: set, want: set) -> tuple[float, float]:
    tp = len(got & want)
    return (tp / len(got) if got else 1.0, tp / len(want) if want else 1.0)


def scanner_failures(root: Path, *, verbose: bool = True) -> list[str]:
    refs = W.code_wire_refs(root)
    calls = {(r.service, r.value) for r in refs.http}
    keys = {(r.service, k, r.relation) for k, rs in refs.redis.items() for r in rs}
    f = []
    back = sorted(set(FALSE_CALLS) & calls) + sorted(set(FALSE_KEYS) & keys)
    if back:
        f.append(f"false claims came back: {back}")
    judged_calls = calls & (set(TRUE_CALLS) | set(FALSE_CALLS))
    judged_keys = keys & (set(TRUE_KEYS) | set(FALSE_KEYS))
    judged = len(judged_calls) + len(judged_keys)
    right = len(judged_calls & set(TRUE_CALLS)) + len(judged_keys & set(TRUE_KEYS))
    precision = right / judged if judged else 1.0
    if precision < 1.0:
        f.append(f"precision {precision:.2f} on {judged} judged claims")
    missed = sorted(set(TRUE_CALLS) - calls) + sorted(set(TRUE_KEYS) - keys)
    if verbose:
        print(f"scanner on {root}: {len(calls)} service->URL, {len(refs.redis)} keys; precision {precision:.2f} "
              f"over {judged} judged; recall {(len(TRUE_CALLS) + len(TRUE_KEYS) - len(missed))}/"
              f"{len(TRUE_CALLS) + len(TRUE_KEYS)}" + (f" (missed {missed})" if missed else ""))
    return f


def rule_failures(path: Path, *, verbose: bool = True) -> list[str]:
    catalog, observed, now, body = _structure._fixture_inputs(path)
    atlas = build_atlas(catalog, observed, now=now, body=body)
    reading = K.read_capacities(atlas)
    f = list(_structure.evaluate(atlas))
    w = reading.wires
    got = {"R-wire-seen": set(w.seen_by_wire) if w else set(),
           "R-broadcast-key": {b.part_id for b in w.broadcast} if w else set()}
    for rule, want in EXPECTED[path.name].items():
        prec, rec = score(got[rule], want)
        if verbose:
            print(f"  {rule}: precision {prec:.2f} recall {rec:.2f} ({len(want)} expected)")
        if prec < 1.0 or rec < 1.0:
            f.append(f"{rule}: precision {prec:.2f} recall {rec:.2f} (extra {sorted(got[rule] - want)}, "
                     f"missed {sorted(want - got[rule])})")
    text = "\n".join(K.render(reading))
    if len(text.splitlines()) > 39 or len(text) > 3900:
        f.append(f"printout too big: {len(text.splitlines())} lines, {len(text)} chars")
    if verbose:
        s = atlas.stats
        print(f"  organs {s.organs} ({s.singleton_organs} singletons), Q={s.modularity}, largest "
              f"{s.largest_organ_share:.1%}; unseen {len(reading.silent_services)}")
    return f


def _any_flow_links(atlas):
    """The R-wire-seen mutant: every flow counts, placement and the census included."""
    out: dict[str, set[str]] = {}
    for fl in atlas.flows:
        out.setdefault(fl.source, set()).add(fl.relation)
        out.setdefault(fl.target, set()).add(fl.relation)
    return out


def mutations(root: Path) -> dict[str, bool]:
    """name -> caught (the eval fails with the mutation in place)."""
    caught = {}
    real = W._is_handle_expr
    W._is_handle_expr = lambda expr, handles: True
    try:
        caught["scan with no Redis-handle check"] = bool(scanner_failures(root, verbose=False))
    finally:
        W._is_handle_expr = real
    floor = K.BROADCAST_MIN_READERS
    K.BROADCAST_MIN_READERS = 1
    try:
        caught["R-broadcast-key with no reader floor"] = all(rule_failures(p, verbose=False) for p in (HELD_OUT, TODAY))
    finally:
        K.BROADCAST_MIN_READERS = floor
    links = K.wire_links
    K.wire_links = _any_flow_links
    try:
        caught["R-wire-seen counting any flow"] = all(rule_failures(p, verbose=False) for p in (HELD_OUT, TODAY))
    finally:
        K.wire_links = links
    return caught


def make_held_out(checkout: Path) -> None:
    """The 10-29 body plus the HTTP calls and Redis keys that main's code named at its export.
    No live Redis read existed that day: every key part is declared_only."""
    data = json.loads(HELD_OUT_BASE.read_text(encoding="utf-8"))
    body = BodyInputs.model_validate(data["body"])
    body.stores.extend(apply_wire(body, W.code_wire_refs(checkout)))
    data["body"] = body.model_dump(mode="json")
    data["wires_from_commit"] = HELD_OUT_COMMIT
    HELD_OUT.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {HELD_OUT}: {len(body.calls)} calls, {sum(1 for s in body.stores if s.store == 'redis')} shared keys")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=_REPO)
    ap.add_argument("--make-held-out", type=Path, metavar="CHECKOUT")
    args = ap.parse_args(argv)
    if args.make_held_out:
        make_held_out(args.make_held_out)
        return 0
    failed = 0
    f = scanner_failures(args.root)
    for x in f:
        print(f"FAIL scanner: {x}")
    print("scanner facts: " + ("PASS" if not f else f"FAIL ({len(f)})"))
    failed += bool(f)
    for path in (HELD_OUT, TODAY):
        print(f"== {path.name}" + (" (held out)" if path == HELD_OUT else ""))
        f = rule_failures(path)
        for x in f:
            print(f"FAIL {x}")
        print(f"wire rules ({path.name}): " + ("PASS" if not f else f"FAIL ({len(f)})"))
        failed += bool(f)
    for name, ok in mutations(args.root).items():
        print(f"mutation ({name}): " + ("CAUGHT" if ok else "NOT CAUGHT"))
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

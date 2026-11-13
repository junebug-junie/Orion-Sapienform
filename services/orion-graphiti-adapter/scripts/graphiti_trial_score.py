"""Score the Graphiti trial graph (``graphiti_trial``) against the spec's go/no-go checks.

    python graphiti_trial_score.py dump            # graph -> /tmp/graphiti-trial/graph_dump.json
    python graphiti_trial_score.py sample          # 50 seeded entities -> dup_sample.json (to judge by hand)
    python graphiti_trial_score.py report          # dump + judgments + results.jsonl -> report.md
                                                   #   + report.redacted.md (safe to commit)

Needs graphiti_trial_terms.local.json (gitignored; see the .example.json) for the real names.

The checks are read off fact edges (RELATES_TO: fact, valid_at, invalid_at, expired_at) and entity
nodes. A fact is CURRENT when it has no invalid_at and no expired_at. Matching is regex over fact
text; every edge a check looked at is listed in the report so a human can overrule it.

Read-only against FalkorDB (ro_query). Pure scoring functions take plain dicts so they are tested
without a graph.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUT_DIR = Path(os.environ.get("GRAPHITI_TRIAL_OUT", "/tmp/graphiti-trial"))
TRIAL_GRAPH = "graphiti_trial"
DUP_SAMPLE_SIZE = 50
DUP_PASS_RATE = 0.10

LIVES_RE = re.compile(r"\b(lives?|living|resides?|residing|home|house|located|based)\b", re.I)
STAY_RE = re.compile(r"\b(stay(?:s|ing|ed)?|hotel|trip|visit(?:s|ing|ed)?|meeting|travel(?:s|ing|ed)?|fl(?:y|ew|ies|ying)|"
                     r"flight|night|return(?:s|ed|ing)?|back|from|wake ?up)\b", re.I)
DOG_RE = re.compile(r"\bdogs?\b|\bpupp(?:y|ies)\b", re.I)
NOT_CAT_RE = re.compile(r"\bnot\s+(?:a\s+)?cats?\b|\bisn't\s+(?:a\s+)?cats?\b", re.I)

# The real names these checks are about (home city, spouse, hotel...) are personal terms the public
# repo's privacy gate refuses (tests/test_no_personal_terms.py). They live in a gitignored local file
# next to this script; the committed example holds placeholders with the same keys.
TERMS_LOCAL = Path(__file__).resolve().parent / "graphiti_trial_terms.local.json"
TERMS_EXAMPLE = Path(__file__).resolve().parent / "graphiti_trial_terms.example.json"
TERM_KEYS = ("subject", "home_city", "trip_city", "trip_hotel", "spouse", "spouse_alias", "pet",
             "trip_start", "trip_end")


def load_terms(path: Path | None = None) -> dict[str, str]:
    p = path or (TERMS_LOCAL if TERMS_LOCAL.exists() else None)
    if p is None:
        raise SystemExit(f"missing {TERMS_LOCAL.name}: copy {TERMS_EXAMPLE.name} and fill in the real values")
    terms = json.loads(Path(p).read_text())
    missing = [k for k in TERM_KEYS if not str(terms.get(k) or "").strip()]
    if missing:
        raise SystemExit(f"{p}: missing keys {missing}")
    return terms


def word(term: str) -> str:
    return r"\b" + re.escape(term) + r"\b"


def negated_before(place: str) -> re.Pattern:
    """Negation that governs ``place``: "not <city>", "no longer in <city>", "instead of <city>".
    A negation elsewhere in the fact ("lives in <home>, not <trip city>") does not negate <home>."""
    return re.compile(r"(\bnot\b|n't\b|\bno longer\b|\binstead of\b|\brather than\b)[^.,;]{0,30}" + word(place), re.I)


def _dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    s = str(value).replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def is_current(edge: dict[str, Any]) -> bool:
    return _dt(edge.get("invalid_at")) is None and _dt(edge.get("expired_at")) is None


def _edge_line(e: dict[str, Any]) -> str:
    return (f"[{'CURRENT' if is_current(e) else 'invalid'}] {e.get('source')} -[{e.get('name')}]-> "
            f"{e.get('target')}: \"{e.get('fact')}\" valid_at={e.get('valid_at')} "
            f"invalid_at={e.get('invalid_at')} expired_at={e.get('expired_at')}")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    evidence: list[str] = field(default_factory=list)


def _mentions(e: dict[str, Any], pattern: re.Pattern) -> bool:
    return bool(pattern.search(" ".join(str(e.get(k) or "") for k in ("fact", "source", "target"))))


def check_home_current(edges: list[dict[str, Any]], t: dict[str, str]) -> Check:
    home = t["home_city"]
    hits = [e for e in edges if _mentions(e, re.compile(word(home), re.I)) and LIVES_RE.search(str(e.get("fact")))]
    current = [e for e in hits if is_current(e) and not negated_before(home).search(str(e.get("fact")))]
    return Check("lives in [home city] is current", bool(current),
                 f"{len(current)} current of {len(hits)} [home city]-residence facts", [_edge_line(e) for e in hits])


def check_trip_city_not_home(edges: list[dict[str, Any]], t: dict[str, str]) -> Check:
    """Fail when a CURRENT fact says the subject lives in the trip city (a stay is not a home)."""
    city = t["trip_city"]
    hits = [e for e in edges if _mentions(e, re.compile(word(city), re.I))]
    homes = [e for e in hits if LIVES_RE.search(str(e.get("fact")))
             and not STAY_RE.search(str(e.get("fact"))) and not negated_before(city).search(str(e.get("fact")))]
    bad = [e for e in homes if is_current(e)]
    return Check("lives in [trip city] absent or invalidated", not bad,
                 f"{len(bad)} current [trip city]-home facts ({len(homes)} home-shaped, {len(hits)} [trip city] facts)",
                 [_edge_line(e) for e in hits])


def check_trip_stay(edges: list[dict[str, Any]], t: dict[str, str]) -> Check:
    """Pass when a trip-city stay fact starts inside the trip window and is closed (invalid_at)
    inside it. The window carries a day of slack each side for local-time vs UTC dating."""
    lo, hi = _dt(t["trip_start"]), _dt(t["trip_end"])
    place = re.compile(word(t["trip_city"]) + "|" + word(t["trip_hotel"]), re.I)
    stays = [e for e in edges if _mentions(e, place) and STAY_RE.search(str(e.get("fact")))]
    dated = [e for e in stays if (v := _dt(e.get("valid_at"))) and lo <= v <= hi]
    closed = [e for e in dated if (i := _dt(e.get("invalid_at"))) and lo <= i <= hi]
    return Check(f"[trip city] stay dated as a stay (window {t['trip_start']}..{t['trip_end']})", bool(closed),
                 f"{len(closed)} stay facts opened and closed inside the window; "
                 f"{len(dated)} opened inside it; {len(stays)} stay-shaped [trip city] facts",
                 [_edge_line(e) for e in stays])


def check_pet_dog(edges: list[dict[str, Any]], t: dict[str, str]) -> Check:
    pet_re = re.compile(word(t["pet"]), re.I)
    is_cat = re.compile(word(t["pet"]) + r"[^.;]{0,20}\b(?:is|was|being)\s+(?:[\w']+\s+){0,2}cat\b", re.I)
    pet = [e for e in edges if _mentions(e, pet_re)]
    dog_current = [e for e in pet if is_current(e) and DOG_RE.search(str(e.get("fact")))
                   and not re.search(r"\bnot\s+a\s+dog\b", str(e.get("fact")), re.I)]
    cat_current = [e for e in pet if is_current(e) and is_cat.search(str(e.get("fact")))
                   and not NOT_CAT_RE.search(str(e.get("fact")))]
    return Check("[pet] is a dog current, cat absent or invalidated", bool(dog_current) and not cat_current,
                 f"{len(dog_current)} current dog facts, {len(cat_current)} current cat facts, {len(pet)} [pet] facts",
                 [_edge_line(e) for e in pet])


def check_one_entity(nodes: list[dict[str, Any]], pattern: re.Pattern, label: str) -> Check:
    hits = [n for n in nodes if pattern.search(str(n.get("name") or ""))]
    return Check(f"{label} is one entity", len(hits) == 1,
                 f"{len(hits)} entity nodes match: {[n.get('name') for n in hits]}",
                 [f"{n.get('name')} ({n.get('uuid')}): {str(n.get('summary') or '')[:200]}" for n in hits])


def duplicate_sample(nodes: list[dict[str, Any]], size: int = DUP_SAMPLE_SIZE, seed: int = 20261113) -> list[dict[str, Any]]:
    """Seeded sample; each pick carries the 5 entities whose names share a word with it, as the
    candidate duplicates a human judges against."""
    pool = sorted(nodes, key=lambda n: str(n.get("uuid")))
    picks = random.Random(seed).sample(pool, k=min(size, len(pool)))
    out = []
    for n in picks:
        words = {w for w in re.findall(r"[a-z0-9]+", str(n.get("name")).lower()) if len(w) > 2}
        near = [m for m in pool if m.get("uuid") != n.get("uuid")
                and words & set(re.findall(r"[a-z0-9]+", str(m.get("name")).lower()))][:5]
        out.append({"uuid": n.get("uuid"), "name": n.get("name"), "summary": str(n.get("summary") or "")[:300],
                    "candidates": [{"uuid": m.get("uuid"), "name": m.get("name"),
                                    "summary": str(m.get("summary") or "")[:200]} for m in near],
                    "duplicate": None, "note": ""})
    return out


def duplicate_rate(judged: list[dict[str, Any]]) -> tuple[float | None, int, int]:
    """(rate, duplicates, judged). Unjudged entries (duplicate=None) are not counted."""
    done = [j for j in judged if j.get("duplicate") is not None]
    for j in done:
        if not isinstance(j["duplicate"], bool):
            raise ValueError(f"judgment for {j.get('name')!r} must be true/false, got {j['duplicate']!r}")
    dups = sum(1 for j in done if j["duplicate"])
    return (dups / len(done) if done else None), dups, len(done)


def cost_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Cost over EVERY attempt (a failed-then-retried episode costs twice); ok/failed counts over
    the last attempt per episode."""
    last: dict[str, dict[str, Any]] = {}
    for r in results:
        last[r["uuid"]] = r
    ok = [r for r in last.values() if r.get("ok")]
    walls = [float(r.get("wall_sec") or 0) for r in results]
    leases = [float(r.get("lease_sec") or 0) for r in results]
    llm_s = [float((r.get("llm") or {}).get("llm_seconds") or 0) for r in results]
    calls = sum(int((r.get("llm") or {}).get("calls") or 0) for r in results)
    return {
        "episodes": len(last), "attempts": len(results), "ok": len(ok), "failed": len(last) - len(ok),
        "wall_sec_total": round(sum(walls), 1),
        "wall_sec_per_episode_mean": round(statistics.mean(walls), 1) if walls else None,
        "wall_sec_per_episode_median": round(statistics.median(walls), 1) if walls else None,
        "lease_sec_per_episode_mean": round(statistics.mean(leases), 1) if leases else None,
        "llm_call_sec_summed_per_episode_mean": round(statistics.mean(llm_s), 1) if llm_s else None,
        "llm_calls": calls, "llm_calls_per_episode": round(calls / len(last), 1) if last else None,
        "llm_failures": sum(int((r.get("llm") or {}).get("failures") or 0) for r in results),
        "graphiti_retries": sum(int((r.get("llm") or {}).get("retries") or 0) for r in results),
        "schema_fallbacks": sum(int((r.get("llm") or {}).get("schema_fallbacks") or 0) for r in results),
        "completion_tokens": sum(int((r.get("llm") or {}).get("completion_tokens") or 0) for r in results),
    }


def run_checks(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], judged: list[dict[str, Any]],
               t: dict[str, str]) -> list[Check]:
    checks = [
        check_home_current(edges, t),
        check_trip_city_not_home(edges, t),
        check_trip_stay(edges, t),
        check_pet_dog(edges, t),
        check_one_entity(nodes, re.compile(word(t["pet"]), re.I), "[pet] (all name forms)"),
        check_one_entity(nodes, re.compile(word(t["spouse"]) + "|" + word(t["spouse_alias"]), re.I),
                         "[spouse]/my [spouse alias]"),
    ]
    rate, dups, n = duplicate_rate(judged)
    checks.append(Check(f"duplicate entities under {int(DUP_PASS_RATE * 100)}% (hand-judged sample)",
                        rate is not None and n >= min(DUP_SAMPLE_SIZE, len(nodes)) and rate < DUP_PASS_RATE,
                        f"{dups}/{n} judged duplicates" + (f" = {rate:.0%}" if rate is not None else " (not judged)")))
    return checks


# --- graph I/O --------------------------------------------------------------------------------

def dump_graph(host: str, port: int) -> dict[str, Any]:
    from falkordb import FalkorDB  # noqa: PLC0415
    g = FalkorDB(host=host, port=port).select_graph(TRIAL_GRAPH)
    nodes = [dict(zip(("uuid", "name", "summary", "created_at"), r)) for r in g.ro_query(
        "MATCH (n:Entity) RETURN n.uuid, n.name, n.summary, n.created_at").result_set]
    edges = [dict(zip(("uuid", "source", "name", "target", "fact", "valid_at", "invalid_at", "expired_at",
                       "created_at", "episodes"), r)) for r in g.ro_query(
        "MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity) RETURN e.uuid, a.name, e.name, b.name, e.fact, "
        "e.valid_at, e.invalid_at, e.expired_at, e.created_at, e.episodes").result_set]
    episodes = g.ro_query("MATCH (e:Episodic) RETURN count(e)").result_set[0][0]
    return {"nodes": nodes, "edges": edges, "episodes": int(episodes)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render_report(dump: dict[str, Any], checks: list[Check], cost: dict[str, Any], plan: dict[str, Any],
                  runs: list[dict[str, Any]], judged: list[dict[str, Any]]) -> str:
    passed = all(c.passed for c in checks)
    lines = ["# Graphiti full-engine trial: scored", "",
             f"Verdict by the checks: **{'GO' if passed else 'NO-GO'}**", "",
             f"Graph `{TRIAL_GRAPH}`: {dump['episodes']} episodes, {len(dump['nodes'])} entities, "
             f"{len(dump['edges'])} fact edges ({sum(is_current(e) for e in dump['edges'])} current).",
             f"Planned {plan.get('planned')} of {plan.get('source_total')} source episodes "
             f"(sample={plan.get('sample')}, key rows={plan.get('key_rows')}).", "", "## Checks", ""]
    for c in checks:
        lines.append(f"- **{'PASS' if c.passed else 'FAIL'}** {c.name}: {c.detail}")
    lines += ["", "## Cost", "", "```json", json.dumps(cost, indent=1), "```", ""]
    if runs:
        lines += ["Runs:", ""] + [f"- run {r.get('run_id')}: done {r.get('done')}, errors {r.get('errors')}, "
                                  f"wall {r.get('wall_sec')} s, lease {r.get('lease_sec')} s, isolation "
                                  f"{(r.get('isolation') or {}).get('isolated', 'not recorded')}" for r in runs] + [""]
    lines += ["## Evidence (every edge each check looked at)", ""]
    for c in checks:
        if c.evidence:
            lines += [f"### {c.name}", ""] + [f"- {x}" for x in c.evidence[:60]] + [""]
    lines += ["## Duplicate sample (hand-judged)", ""]
    for j in judged:
        mark = {True: "DUP", False: "ok", None: "??"}[j.get("duplicate")]
        cands = ", ".join(c["name"] for c in j.get("candidates", []))
        lines.append(f"- [{mark}] {j.get('name')}" + (f" (vs: {cands})" if cands else "")
                     + (f" -- {j['note']}" if j.get("note") else ""))
    return "\n".join(lines) + "\n"


PLACEHOLDERS = {"subject": "[subject]", "home_city": "[home city]", "trip_city": "[trip city]",
                "trip_hotel": "[trip hotel]", "spouse": "[spouse]", "spouse_alias": "[spouse alias]", "pet": "[pet]"}


def _denylist_hashes() -> tuple[set[str], Any]:
    """The repo's hashed personal-term denylist (config/privacy), or empty when not mounted."""
    try:
        import orion  # noqa: PLC0415
        from orion.privacy import personal_terms as pt  # noqa: PLC0415
    except ImportError:
        return set(), None
    path = Path(orion.__file__).resolve().parents[1] / pt.DENYLIST_REL
    if not path.exists():
        return set(), pt
    return pt.load_denylist(path)[0], pt


def redact(text: str, t: dict[str, str], hashes: set[str] | None = None, pt: Any = None) -> str:
    """Term values -> role placeholders; then any other word on the denylist -> [name]."""
    for key in sorted(PLACEHOLDERS, key=lambda k: -len(str(t.get(k) or ""))):
        val = str(t.get(key) or "").strip()
        if val:
            text = re.sub(word(val), PLACEHOLDERS[key], text, flags=re.I)
    if hashes and pt is not None:
        text = pt._ascii_fold(text)
        text = pt._TOKEN.sub(lambda m: "[name]" if pt._h(m.group(0).lower()) in hashes else m.group(0), text)
    return text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("dump", "sample", "report"))
    p.add_argument("--falkor-host", default="orion-athena-falkordb")
    p.add_argument("--falkor-port", type=int, default=6379)
    p.add_argument("--judgments", default=str(OUT_DIR / "dup_sample.json"))
    p.add_argument("--force", action="store_true", help="sample: overwrite a judgments file that has judgments")
    p.add_argument("--terms", default="", help=f"terms file (default {TERMS_LOCAL.name})")
    a = p.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dump_path = OUT_DIR / "graph_dump.json"
    if a.cmd == "sample" and Path(a.judgments).exists() and not a.force and any(
            j.get("duplicate") is not None for j in json.loads(Path(a.judgments).read_text())):
        print(f"refusing: {a.judgments} already holds hand judgments (use --force)", file=sys.stderr)
        return 1
    if a.cmd in ("dump", "sample") or not dump_path.exists():
        dump = dump_graph(a.falkor_host, a.falkor_port)
        dump_path.write_text(json.dumps(dump, indent=1, default=str))
    dump = json.loads(dump_path.read_text())
    print(f"graph dump {dump_path} ({dump['episodes']} episodes, {len(dump['nodes'])} entities, "
          f"written {datetime.fromtimestamp(dump_path.stat().st_mtime, tz=timezone.utc).isoformat(timespec='seconds')})",
          file=sys.stderr)
    if a.cmd == "sample":
        Path(a.judgments).write_text(json.dumps(duplicate_sample(dump["nodes"]), indent=1))
        print(f"wrote {a.judgments}: set each 'duplicate' to true/false by hand, then run report")
        return 0
    if a.cmd == "report":
        judged = json.loads(Path(a.judgments).read_text()) if Path(a.judgments).exists() else []
        results = _read_jsonl(OUT_DIR / "results.jsonl")
        plan = json.loads((OUT_DIR / "plan.json").read_text()) if (OUT_DIR / "plan.json").exists() else {}
        runs = [json.loads(f.read_text()) for f in sorted(OUT_DIR.glob("run-*.json"))]
        terms = load_terms(Path(a.terms) if a.terms else None)
        checks = run_checks(dump["nodes"], dump["edges"], judged, terms)
        report = render_report(dump, checks, cost_summary(results), plan, runs, judged)
        (OUT_DIR / "report.md").write_text(report)
        hashes, pt = _denylist_hashes()
        red = redact(report, terms, hashes, pt)
        left = [n for n, line in enumerate(red.splitlines(), 1) if pt is not None and pt.line_hits(line, hashes)]
        (OUT_DIR / "report.redacted.md").write_text(red)
        print(red)
        if not hashes:
            print("WARNING: denylist not found; report.redacted.md only has term-file redaction", file=sys.stderr)
        if left:
            print(f"redaction left personal terms on lines {left}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

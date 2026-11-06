"""Are Orion's PR field notes true to the PR they describe? (spec
docs/superpowers/specs/2026-11-04-pr-field-notes-design.md, acceptance check 7.)

The write check (``field_notes.check``) runs once, on the packet the turn read. This eval re-reads every
live note from the ledger and checks it again against the PR as GitHub has it now, so a drifted check,
a hand-edited line, or a flag rule that changed shows up. Per note:

* **refs**: every ``pr_text`` and ``code_then`` claim's refs resolve to ``#<this PR>`` or a file the PR
  changed (or a folder of them). The match rate is printed per label.
* **organs**: an organ the note names ("<name> organ", "organ <name>" or ``organ:<id>``) is in that note's
  own ``body_map``. The candidate names are the organ names found in ANY note's body map, so the check
  needs no live atlas and no word list.
* **read**: when the PR added or changed a spec, plan or PR report, ``read`` is not empty.
* **numbers** (printed, not gated): a figure of 2+ digits in a ``pr_text`` sentence appears in the PR's
  title, body, its line/file counts, or the part of the docs the note read (the packet's own cut). A low rate means a pr_text sentence carries figures the
  builders never wrote.
* **flags** (printed, not gated): the stored flags equal ``flags_for`` run again on today's PR data. A
  note written before a flag rule changed disagrees, which is history, not a bad note.

    python orion/body_atlas/evals/run_field_note_truth_eval.py            # CI: built-in cases, no network
    python orion/body_atlas/evals/run_field_note_truth_eval.py --live     # the real ledger + gh (read-only)

Gate: refs, organs and read are 1.0 (the write check makes anything else impossible, so a miss is a
bug). In CI mode the planted bad notes must each fail exactly their own check. Exit 0 pass, 1 fail,
2 when the live inputs could not be read.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from orion.body_atlas import field_notes as fn  # noqa: E402

GH_FIELDS = ("number,title,body,mergedAt,mergeCommit,files,additions,deletions,changedFiles,headRefName,"
             "commits,author")
CHECKS = ("refs", "organs", "read")
_NUM = re.compile(r"(?<![\w.])#?\d[\d,]*(?:\.\d+)?")


@dataclass
class NoteResult:
    pr: int
    note_id: str
    failed: list[str] = field(default_factory=list)  # gated checks that failed, with a reason
    claims: dict[str, list[int]] = field(default_factory=dict)  # label -> [resolved, total]
    numbers: list[int] = field(default_factory=lambda: [0, 0])  # [grounded, total]
    ungrounded: list[str] = field(default_factory=list)
    flag_diff: list[str] = field(default_factory=list)


def _norm_num(tok: str) -> str:
    return tok.lstrip("#").replace(",", "")


def _sentences(note: str) -> list[tuple[str, str]]:
    return [(lab, sen) for lab, _, sen in fn.parse_note(note)]


def _named(text: str, names: set[str]) -> tuple[set[str], set[str]]:
    """(organ ids written as organ:<id>, organ names written as '<name> organ' / 'organ <name>').
    ``names``: every lowercased organ name in any note's body map. Organ ids change when the atlas is
    rebuilt (the same "substrate" organ has had several), so a name is matched by name, never by id."""
    low = text.lower()
    ids = set(re.findall(r"\borgan:[0-9a-f]{10}\b", low))
    found = set()
    for name in names:
        n = re.escape(name)
        if re.search(rf"(?<![\w-]){n}['\"]?\s+organs?\b|\borgans?\s+['\"]?{n}(?![\w-])", low):
            found.add(name)
    return ids, found


def known_organ_names(records: list[fn.PrFieldNoteV1]) -> set[str]:
    return {e.organ_name.lower() for r in records for e in r.body_map if e.organ_name}


def judge(rec: fn.PrFieldNoteV1, pr: dict, docs_text: str, names: set[str],
          local_branches: set[str] | None = None) -> NoteResult:
    res = NoteResult(pr=rec.pr_number, note_id=rec.note_id)
    files = [f["path"] for f in pr.get("files") or []]
    sentences = _sentences(rec.note or "")
    bad_refs = []
    for c in rec.claims:
        if c.label == "reflection":
            continue
        ok = bool(c.refs) and all(fn._ref_ok(r, rec.pr_number, files) for r in c.refs)
        tally = res.claims.setdefault(c.label, [0, 0])
        tally[0] += ok
        tally[1] += 1
        if not ok:
            bad_refs += [r for r in c.refs if not fn._ref_ok(r, rec.pr_number, files)] or ["(no refs)"]
    if bad_refs:
        res.failed.append("refs: " + ", ".join(bad_refs[:4]))
    ids, named = _named(" ".join(s for _, s in sentences), names)
    stray = sorted((ids - {e.organ_id for e in rec.body_map if e.organ_id})
                   | (named - {e.organ_name.lower() for e in rec.body_map if e.organ_name}))
    if stray:
        res.failed.append("organs: names " + ", ".join(stray) + " not in its body map")
    if fn.docs_to_read(pr) and not rec.read:
        res.failed.append(f"read: the PR changed {len(fn.docs_to_read(pr))} spec/report file(s), none read")
    source = " ".join([pr.get("title") or "", pr.get("body") or "", docs_text])
    # the packet's own counts ("files: 1 (+944 -0)") are sources too: the note may quote them
    counts = {str(int(pr.get(k) or 0)) for k in ("additions", "deletions", "changedFiles")}
    counts.add(str(int(pr.get("additions") or 0) + int(pr.get("deletions") or 0)))
    have = {_norm_num(t) for t in _NUM.findall(source)} | {str(rec.pr_number)} | counts
    for lab, sen in sentences:
        if lab != "pr_text":
            continue
        for tok in _NUM.findall(sen):
            if len(_norm_num(tok).split(".")[0]) < 2:
                continue
            res.numbers[1] += 1
            if _norm_num(tok) in have:
                res.numbers[0] += 1
            else:
                res.ungrounded.append(tok)
    now = fn.flags_for(pr, local_branches=local_branches).model_dump()
    then = rec.flags.model_dump()
    res.flag_diff = [f"{k}: {then[k]} -> {now[k]}" for k in then if then[k] != now[k]]
    return res


def report(results: list[NoteResult]) -> tuple[list[str], bool]:
    lines = [f"field notes judged: {len(results)}"]
    for label in ("pr_text", "code_then"):
        ok = sum(r.claims.get(label, [0, 0])[0] for r in results)
        tot = sum(r.claims.get(label, [0, 0])[1] for r in results)
        lines.append(f"  {label:10s} claims whose refs resolve: {ok}/{tot}" + (f" ({ok / tot:.0%})" if tot else ""))
    for check in CHECKS:
        bad = [r for r in results if any(f.startswith(check + ":") for f in r.failed)]
        lines.append(f"  {check:10s} notes passing: {len(results) - len(bad)}/{len(results)}")
    g = sum(r.numbers[0] for r in results)
    t = sum(r.numbers[1] for r in results)
    lines.append(f"  numbers in pr_text sentences found in the PR text or the docs read: {g}/{t}"
                 + (f" ({g / t:.0%})" if t else "") + "  (printed, not gated)")
    for r in results:
        if r.ungrounded:
            lines.append(f"    #{r.pr}: not in its sources: {', '.join(r.ungrounded[:6])}")
    diff = [r for r in results if r.flag_diff]
    lines.append(f"  flags equal to today's flag rule: {len(results) - len(diff)}/{len(results)}  (printed, not gated)")
    for r in diff:
        lines.append(f"    #{r.pr}: " + "; ".join(r.flag_diff))
    for r in results:
        for f in r.failed:
            lines.append(f"  FAIL #{r.pr} {r.note_id}: {f}")
    return lines, not any(r.failed for r in results)


# --- CI mode: built-in cases ---------------------------------------------------------------------

def _pr(n: int, paths: list[str], *, body: str = "", login: str = "junebug-junie") -> dict:
    return {"number": n, "title": f"feat: change {n}", "body": body, "mergedAt": "2026-11-05T00:00:00Z",
            "files": [{"path": p, "additions": 5, "deletions": 1} for p in paths], "additions": 5 * len(paths),
            "deletions": len(paths), "changedFiles": len(paths), "headRefName": f"feat/x{n}",
            "commits": [{"authors": [{"email": "athena@orion"}]}], "author": {"login": login}}


def _rec(pr: dict, note: str, refs: list[list[str]], *, bmap=None, read=None) -> fn.PrFieldNoteV1:
    labels = [lab for lab, _ in _sentences(note)]
    return fn.PrFieldNoteV1(
        note_id=f"fn:{pr['number']}:case", pr_number=pr["number"], merge_sha="a" * 40, merged_at=pr["mergedAt"],
        title=pr["title"], flags=fn.flags_for(pr), body_map=bmap or [], read=read or [], note=note,
        claims=[fn.ClaimV1(sentence_idx=i, label=lab, refs=r) for i, (lab, r) in enumerate(zip(labels, refs))],
        written_at="2026-11-05T01:00:00+00:00")


HUB = fn.BodyMapEntryV1(prefix="services/orion-hub/", files=1, part_id="part:orion-hub", part_live=True,
                        organ_id="organ:aaaaaaaaaa", organ_name="substrate")
SQLW = fn.BodyMapEntryV1(prefix="services/orion-sql-writer/", files=1, part_id="part:orion-sql-writer",
                         part_live=True, organ_id="organ:bbbbbbbbbb", organ_name="cortex")
SPEC = "docs/superpowers/specs/2026-11-05-x.md"
SPEC_READ = [fn.DocReadV1(path=SPEC, chars_read=900, truncated=False)]


def cases() -> list[tuple[str, fn.PrFieldNoteV1, dict, str, str | None]]:
    """(name, note, pr, docs text, the one check it must fail or None)."""
    good_pr = _pr(1, ["services/orion-hub/app/x.py", SPEC], body="Adds 3 tiles and a 900 s budget.")
    good = _rec(good_pr, "[pr_text] The hub part in my substrate organ gains a 900 s budget. "
                         "[reflection] I notice the change.", [["#1", "services/orion-hub/app/x.py"], []],
                bmap=[HUB], read=SPEC_READ)
    other_pr = _pr(5, ["services/orion-sql-writer/app/y.py"])
    other = _rec(other_pr, "[code_then] The cortex organ writes one more table.",
                 [["services/orion-sql-writer/app/y.py"]], bmap=[SQLW])
    bad_ref = _rec(_pr(2, ["services/orion-hub/app/x.py"]), "[pr_text] The hub changed in a small way.",
                   [["services/orion-hub/app/not_in_pr.py"]], bmap=[HUB])
    bad_organ = _rec(_pr(3, ["services/orion-hub/app/x.py"]), "[pr_text] My cortex organ changed today.",
                     [["#3"]], bmap=[HUB])
    unread_pr = _pr(4, ["services/orion-hub/app/x.py", SPEC])
    unread = _rec(unread_pr, "[pr_text] The hub part changed and a spec came with it.", [["#4"]], bmap=[HUB])
    return [("clean note", good, good_pr, "", None), ("clean code_then", other, other_pr, "", None),
            ("ref outside the PR", bad_ref, _pr(2, ["services/orion-hub/app/x.py"]), "", "refs"),
            ("organ not in its map", bad_organ, _pr(3, ["services/orion-hub/app/x.py"]), "", "organs"),
            ("spec changed, nothing read", unread, unread_pr, "", "read")]


def run_cases() -> int:
    cs = cases()
    names = known_organ_names([c[1] for c in cs])
    ok = True
    for name, rec, pr, docs, must_fail in cs:
        res = judge(rec, pr, docs, names)
        got = sorted({f.split(":")[0] for f in res.failed})
        want = [must_fail] if must_fail else []
        good = got == want and not res.flag_diff
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name}: failed {got or 'nothing'}, want {want or 'nothing'}"
              + (f"; flag diff {res.flag_diff}" if res.flag_diff else ""))
    # authorship drift: a record stored before the GitHub-user rule shows a flag diff, and fails nothing
    mine = _pr(6, ["services/orion-hub/app/x.py"], login="orion-sapienform")
    old = _rec(mine, "[pr_text] The hub part changed in a small way.", [["#6"]], bmap=[HUB])
    old.flags.authored_by_me = "unknown"
    drift = judge(old, mine, "", names)
    if drift.failed or drift.flag_diff != ["authored_by_me: unknown -> yes"]:
        ok = False
        print(f"  FAIL authorship drift: failed {drift.failed}, flag diff {drift.flag_diff}")
    else:
        print("  ok   authorship drift: shown as a flag diff, no gated check fails")
    clean = judge(cs[0][1], cs[0][2], cs[0][3], names)
    if clean.numbers != [1, 1]:  # "900" is in the PR body
        ok = False
        print(f"  FAIL number grounding: {clean.numbers}, want [1, 1]")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


# --- live mode -----------------------------------------------------------------------------------

def _gh_repo() -> str:
    return os.environ.get("ORION_DIG_GITHUB_REPO", "junebug-junie/Orion-Sapienform")


def gh_pr(n: int) -> dict:
    proc = subprocess.run(["gh", "pr", "view", str(n), "--repo", _gh_repo(), "--json", GH_FIELDS],
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"gh pr view {n}: {proc.stderr.strip()[:200]}")
    return json.loads(proc.stdout)


def gh_file(sha: str, path: str) -> str:
    proc = subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github.raw",
                           f"repos/{_gh_repo()}/contents/{quote(path)}?ref={sha}"],
                          capture_output=True, text=True, timeout=60)
    return proc.stdout if proc.returncode == 0 else ""


def run_live(ledger: Path, sample: int | None, sandbox: Path) -> int:
    records, skipped = fn.load(ledger)
    notes = [r for r in fn.live(records).values() if r.note]
    notes.sort(key=lambda r: r.pr_number)
    if sample:
        notes = notes[-sample:]
    if not notes:
        print(f"no live notes in {ledger} (skipped lines: {skipped})")
        return 2
    try:
        branches = set(subprocess.run(["git", "-C", str(sandbox), "for-each-ref", "refs/heads",
                                       "--format=%(refname:short)"], capture_output=True, text=True,
                                      timeout=30).stdout.split())
    except (OSError, subprocess.TimeoutExpired):
        branches = set()
    names = known_organ_names(records)
    code_diff = []
    for rec in sorted((r for r in fn.live(records).values() if r.note is None), key=lambda r: r.pr_number):
        try:
            now = fn.flags_for(gh_pr(rec.pr_number), local_branches=branches).model_dump()
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
            print(f"cannot read #{rec.pr_number} from GitHub: {exc}")
            return 2
        then = rec.flags.model_dump()
        code_diff.append((rec.pr_number, [f"{k}: {then[k]} -> {now[k]}" for k in then if then[k] != now[k]]))
    results = []
    for rec in notes:
        try:
            pr = gh_pr(rec.pr_number)
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
            print(f"cannot read #{rec.pr_number} from GitHub: {exc}")
            return 2
        # the same cut the writer saw (first lines of each doc, the char cap), not the whole files
        docs = fn.excerpt_docs([(d.path, gh_file(rec.merge_sha, d.path)) for d in rec.read])[0]
        results.append(judge(rec, pr, docs, names, local_branches=branches))
    lines, ok = report(results)
    print(f"ledger {ledger} ({skipped} unreadable lines)")
    print("\n".join(lines))
    same = [n for n, d in code_diff if not d]
    print(f"  code records (no note; flags only): {len(same)}/{len(code_diff)} equal to today's flag rule"
          "  (printed, not gated)")
    for n, d in code_diff:
        if d:
            print(f"    #{n}: " + "; ".join(d))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--live", action="store_true", help="judge the real ledger against GitHub (read-only)")
    ap.add_argument("--ledger", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=None, help="only the newest N notes")
    ap.add_argument("--sandbox", type=Path, default=Path("/mnt/orion-fcc/repo"))
    args = ap.parse_args(argv)
    if not args.live:
        return run_cases()
    home = Path(os.environ.get("ORION_DIG_HOME", "/mnt/orion-fcc"))
    return run_live(args.ledger or fn.ledger_path(home / "self"), args.sample, args.sandbox)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Refuse a commit that adds a NEW reference to a metric already confirmed
unclean (dead / never_produced / ratchet_suspect) by phase-5 computed
liveness (orion/metrics/liveness.py).

Why a commit gate and not just the existing edit-time nudge: the informational
PreToolUse hook (scripts/hooks/metric_lineage_nudge.py, wired on Edit/Write)
prints a lineage card and a pointer to `check_metric_lineage.py --metric` --
but it is informational only, fails open by design, and nothing stops an
agent from reading it and building on top of a known-dead metric anyway. Per
CLAUDE.md's own rule: "The right fix for forgotten env sync is not a louder
prompt. The right fix is a failing gate." This is that gate, for the same
failure mode applied to metrics: don't ask agents to notice, make it
mechanically impossible to commit the mistake silently. Confirmed live
2026-08-20 that the failure mode is real, not hypothetical: a real session
working an unrelated task got the graphify PreToolUse "MANDATORY" reminder on
nearly every tool call and acted on it exactly once before routing around it
for the rest of the session.

Deliberately narrow, matching the two-candidates scoping of phase 5 itself:
only the handful of tokens that CURRENTLY have a registered liveness source
(orion.metrics.liveness.has_registered_source()) can ever trigger this --
today that's the 5 attention_self_model.v1 scalar fields and l7_l11_ladder.
Every other metric name in the codebase is silently ignored here, honestly,
not asserted clean. Token set is read live off build_graph() (~1s, no repo-
wide consumer scan -- that's orion.metrics.consumers.scan_repo(), NOT called
here) rather than hand-duplicated, because this exact module already had one
real bug from two independent copies of routing logic drifting apart
(orion/metrics/liveness.py's _resolve_source_kind() docstring).

Detection is AST-based, not regex, reusing `orion.metrics.consumers.
_MetricVisitor` -- the exact classifier `scan_repo()` already uses to build
the metric-lineage cache -- rather than a hand-rolled token-boundary regex.
Found by code review 2026-08-20: an earlier regex version matched any bare
occurrence of a covered token's name in ANY added line, including inside
comments, log strings, and docstrings, and one covered token
(`confidence`) is a common enough word/attribute name to make that a real
false-positive risk once that metric ever actually goes unclean. AST
matching only counts hits in `orion.metrics.consumers.HIGH_CONFIDENCE_KINDS`
(attribute read, subscript read, dict key, `.get()`, collection member --
NOT a bare literal/compare, and NOT a WRITE_KINDS kwarg: writing a new value
for a dead metric is how you'd revive it, not a wiring mistake) and,
critically, requires an exact string-constant match rather than substring
containment -- a comment like `# TODO: confidence is still wired here`
cannot match at all (comments aren't in the AST), and a docstring reading
'confidence check' cannot either (its constant value is the whole
multi-word string, not the bare token). This is exactly the "No regex
swamp" distinction CLAUDE.md's own architectural rules draw: regex is fine
as a narrow sensor (parsing hunk line numbers below, which really is just
text), not as the actual classification logic for "is this a real
reference."

**Known, accepted, NOT fixed** (second review round, 2026-08-20): matching
is by bare string value only, with no type/schema awareness -- a genuinely
unrelated `result.confidence` or `row["confidence"]` on some completely
different object still matches, because AST alone cannot know what type
`result`/`row` actually is without real type inference. `confidence` is the
one covered token common enough as a generic word/attribute name for this to
be a real risk once that field's own liveness ever actually goes unclean;
the other five are specific enough (`broadcast_lane_age_sec`,
`field_overall_salience`, `heartbeat_mean_ratio`,
`prediction_error_confidence`, `l7_l11_ladder`) that collision is
near-zero. Deliberately not solved here: real type-aware resolution is a
much larger feature (proper static type inference, not an AST pattern
scan) than a lightweight commit gate should attempt, and the failure
direction of leaving it unsolved is the safe one -- worst case is an
unnecessary live-DB round-trip or a wrong block on an unrelated commit,
trivially overridden with `ORION_ALLOW_DEAD_METRIC_WIRE=1`, never a missed
real one.

Scope is "new references" only: parses each staged file's *staged* content
(`git show :<path>`, not the working-tree copy -- what would actually be
committed) via `ast.parse`, then keeps only visitor hits whose line number
falls inside the set of lines this diff actually ADDS (from
`git diff --cached -U0`). A file that already contains one of these tokens
elsewhere, untouched by this diff, cannot retrigger the gate. Restricted to
`.py` files -- the covered tokens are Python attribute/field names, and this
gate does not scan YAML/JSON config for them (a real, documented narrowing,
not an oversight: `orion.metrics.consumers.scan_config()` exists for that
surface but pulling it in here would need its own line-added correlation).

Read-only, real-DB (bounded by orion.metrics.liveness's own
CONNECT_TIMEOUT_SECONDS=5 / STATEMENT_TIMEOUT_MS=10s per query) -- adds
latency to a commit ONLY when the diff actually names one of these six
tokens, which is rare. A commit touching no `.py` file at all short-circuits
before any import (one `git diff --cached` name-only call, a dict lookup) --
found live-driving the real hook: an earlier version always called
build_graph() once anything was staged, paying its ~1s pydantic import cost
even for a README-only commit. A commit that DOES stage `.py` files but
matches no covered token still pays that ~1s (build_graph() has to run to
know the token set at all), but never opens a DB connection. Worst case is
NOT ~15s: `main()` calls
`liveness_for_node()` once per unique matched token, and l7_l11_ladder's own
call internally issues 5 more sequential per-stage queries (orion/metrics/
liveness.py's `_ladder_liveness`) -- a commit whose diff touches several
covered tokens at once (plausible: 4 of the 5 attention_self_model.v1 fields
live in one schema file) can take a low tens-of-seconds. Not batched or
capped here -- same accepted trade-off as the un-batched ladder queries
themselves (acknowledged, not fixed, in the underlying liveness module's own
round-3 review).

Test/eval files are excluded from triggering a block: writing a regression
test that references a dead-metric token (this repo's own
tests/test_metric_liveness.py does exactly this for ratchet_suspect fixtures)
must never be mistaken for "wiring a new consumer".

Usage:
    python3 scripts/check_metric_dead_wiring.py            # git mode (the hook)
    python3 scripts/check_metric_dead_wiring.py --json

Exit codes: 0 = nothing matched, DB unreachable, or every matched metric is
                clean (live/quiet). Also 0 on any internal error -- this is
                an advisory-grade gate layered on a best-effort live DB
                check, not a correctness-critical one; failing open is the
                same posture as this repo's other DB-dependent tooling.
            1 = a new reference to a currently-unclean metric was found. BLOCK.

Escape hatch: ORION_ALLOW_DEAD_METRIC_WIRE=1 (set deliberately per command).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENV_ESCAPE = "ORION_ALLOW_DEAD_METRIC_WIRE"


def _staged_files() -> list[str]:
    # ACMR, not ACM: a file that is both renamed and content-modified in one
    # commit (git's default rename detection) has status R, and --diff-
    # filter=ACM alone silently excludes it entirely -- found by code review
    # 2026-08-20, live-verified with `git mv` + a new line in the moved file.
    # Known residual imprecision, not a coverage gap: `_added_line_numbers`
    # below diffs a single path in isolation (`git diff --cached -U0 --
    # <path>`), which cannot pair a renamed path with its pre-rename name, so
    # a renamed file's *entire* new content reads as "added" rather than just
    # the lines actually changed -- live-verified this over-includes (every
    # line becomes a candidate, including ones untouched by the rename), it
    # does not under-include. Accepted: the failure direction is "checks a
    # few more lines than strictly necessary", never "misses a real new
    # reference".
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _added_line_numbers(path: str) -> set[int]:
    """Line numbers this commit ADDS to `path` (in the new file's numbering).

    Parses `git diff --cached -U0`'s hunk headers to track the running line
    number, gated on having already seen the first '@@' -- NOT on the '+++'/
    '---' file-header lines' own text. Found by code review 2026-08-20: an
    earlier version matched on `line.startswith("+++")`/`("---")` directly,
    which also matches a genuinely-added CONTENT line whose own text happens
    to start with those characters (e.g. an added line `++counter;` renders
    as the raw diff line `+++counter;`), silently dropping it from the scan
    and desyncing every following line number in that hunk. Everything
    before the first '@@' (the `diff --git`/`index`/`---`/`+++` preamble) is
    unconditionally skipped instead, which has no such ambiguity.
    """
    proc = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--", path],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return set()
    out: set[int] = set()
    lineno = 0
    in_hunk = False
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            in_hunk = True
            m = re.search(r"\+(\d+)", line)
            lineno = int(m.group(1)) if m else 0
            continue
        if not in_hunk:
            continue
        if line.startswith("+"):
            out.add(lineno)
            lineno += 1
        elif line.startswith("-"):
            pass  # removed line -- does not consume a '+' line number
    return out


def _staged_content(path: str) -> str | None:
    """The INDEX (staged) version of `path` -- what would actually be
    committed, not the working-tree copy (which may carry further unstaged
    edits on top)."""
    proc = subprocess.run(
        ["git", "show", f":{path}"], capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def find_new_token_references(
    staged_files: list[str], known_tokens: set[str]
) -> dict[str, list[tuple[str, int]]]:
    """token -> [(file, line), ...] for every real, AST-classified reference
    that (a) lives in a newly-added line of a non-test .py file and (b) names
    one of `known_tokens`. Pure function of its inputs plus the module-level
    git/AST helpers above -- no DB access -- so it is directly testable by
    monkeypatching those helpers.

    Only `orion.metrics.consumers.HIGH_CONFIDENCE_KINDS` hits count --
    matches this module's own docstring claim ("attribute read, subscript
    read, dict key, .get()") rather than the code silently accepting every
    hit kind the visitor produces. Found by code review 2026-08-20 (live-
    verified by literally running _MetricVisitor against synthetic
    snippets): an earlier version treated `KIND_LITERAL` (any bare string
    constant, e.g. a comparison target or an unrelated docstring sentence
    that happens to equal the token) and `WRITE_KINDS` (`x["metric"] = ...`,
    `Model(metric=0.5)`, `F(channel="metric")`) as equally blocking. Both
    were wrong in the same direction: low-confidence kinds reintroduce the
    exact false-positive class the AST switch existed to kill, and WRITE_
    KINDS are how you'd actually REVIVE a dead metric -- blocking a new
    writer would be actively backwards, not protective.
    """
    from orion.metrics.consumers import HIGH_CONFIDENCE_KINDS, _MetricVisitor
    from orion.metrics.consumers import _is_test_path as _consumers_is_test_path

    hits: dict[str, list[tuple[str, int]]] = {}
    token_set = frozenset(known_tokens)
    for path in staged_files:
        if _consumers_is_test_path(path) or not path.endswith(".py"):
            continue
        added = _added_line_numbers(path)
        if not added:
            continue
        source = _staged_content(path)
        if source is None:
            continue
        try:
            tree = ast.parse(source, filename=path)
        except (SyntaxError, ValueError):
            continue  # not this gate's job to flag a syntax error
        visitor = _MetricVisitor(token_set)
        visitor.visit(tree)
        for tok, lineno, kind, _callee in visitor.hits:
            if kind in HIGH_CONFIDENCE_KINDS and lineno in added:
                hits.setdefault(tok, []).append((path, lineno))
    return hits


def _emit(blocked: list[dict], as_json: bool) -> None:
    if as_json:
        print(json.dumps({"blocked": blocked}, indent=2))
        return
    for b in blocked:
        print(
            f"check_metric_dead_wiring: BLOCK -- new reference to '{b['token']}' "
            f"(verdict={b['verdict']}, {b['detail']}) at {', '.join(b['sites'])}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    # Wrapped like the rest of main()'s fallible steps: _staged_files() runs
    # `subprocess.run(..., text=True)`, whose default-locale stdout decode
    # can raise on a staged filename with non-UTF-8 bytes -- an uncaught
    # exception here would crash before reaching any of the try/excepts
    # below, and the shell wrapper treats a crash exit identically to a real
    # block. Found by code review 2026-08-20, the same failure class the
    # rest of main() already guards against, just missed at this call site.
    try:
        staged = _staged_files()
    except Exception as exc:
        print(f"check_metric_dead_wiring: SKIP (staged-file listing failed: {exc})", file=sys.stderr)
        _emit([], args.json)
        return 0
    if not staged:
        _emit([], args.json)
        return 0

    # Cheap pre-check before paying for build_graph()'s pydantic import
    # (~1s, on every commit that reaches this point otherwise) -- found
    # while live-driving the real hook: an all-README/YAML/etc commit still
    # imported pydantic every time despite this module's own docstring
    # claiming "near-zero cost" for the no-match case. find_new_token_
    # references() already restricts to .py files, so a diff with none can
    # never produce a hit regardless of what build_graph() would return.
    if not any(f.endswith(".py") for f in staged):
        _emit([], args.json)
        return 0

    try:
        from orion.field.channel_glossary import CLEAN_VERDICTS
        from orion.metrics.lineage import build_graph
        from orion.metrics.liveness import has_registered_source, liveness_for_node, open_readonly_connection
    except Exception as exc:
        # Missing pydantic/psycopg2 under whatever interpreter ran this, or
        # an import-time break elsewhere -- this gate cannot do its job, but
        # that must never be the reason an unrelated commit is blocked.
        print(f"check_metric_dead_wiring: SKIP (import failed: {exc})", file=sys.stderr)
        _emit([], args.json)
        return 0

    try:
        graph = build_graph()
    except Exception as exc:
        print(f"check_metric_dead_wiring: SKIP (build_graph failed: {exc})", file=sys.stderr)
        _emit([], args.json)
        return 0

    # Everything from here through find_new_token_references() (registry
    # filtering, git plumbing, AST parsing) is wrapped -- an uncaught
    # exception anywhere in this stretch must degrade to "skip", not crash
    # the process. A crash exit is what scripts/git_hooks/pre-commit's Gate 4
    # treats as "the gate found something", which blocks an UNRELATED
    # commit -- confirmed live 2026-08-20: exactly this happened for real
    # when this module's own docstring briefly had a syntax error. That one
    # instance is fixed, but code review (correctly) flagged that nothing
    # upstream of the DB call had a blanket guard, only individual known
    # failure points -- this closes that for good rather than one symptom
    # at a time.
    try:
        registered_by_name = {n.name: n for n in graph.nodes.values() if has_registered_source(n)}
        if not registered_by_name:
            _emit([], args.json)
            return 0

        hits = find_new_token_references(staged, set(registered_by_name))
    except Exception as exc:
        print(f"check_metric_dead_wiring: SKIP (token detection failed: {exc})", file=sys.stderr)
        _emit([], args.json)
        return 0

    if not hits:
        _emit([], args.json)
        return 0  # common case: diff doesn't touch any phase-5-covered metric

    conn = open_readonly_connection()
    if conn is None:
        print(
            "check_metric_dead_wiring: SKIP -- diff references a phase-5-covered "
            f"metric ({', '.join(sorted(hits))}) but Postgres is unreachable, so "
            "liveness cannot be confirmed. Not blocking on infra.",
            file=sys.stderr,
        )
        _emit([], args.json)
        return 0

    blocked: list[dict] = []
    try:
        for token in sorted(hits):
            node = registered_by_name.get(token)
            if node is None:
                continue
            try:
                outcome = liveness_for_node(node, conn)
            except Exception as exc:
                print(f"check_metric_dead_wiring: {token}: liveness query failed ({exc}) -- not blocking", file=sys.stderr)
                continue
            if outcome is None:
                continue
            if outcome.verdict not in CLEAN_VERDICTS:
                blocked.append({
                    "token": token,
                    "verdict": outcome.verdict,
                    "detail": outcome.detail,
                    "sites": [f"{f}:{l}" for f, l in hits[token]],
                })
    finally:
        try:
            conn.close()
        except Exception:
            # A close() failure (e.g. the backend already aborted the
            # connection after a statement timeout) must never surface as an
            # uncaught exception here -- the shell hook treats any nonzero/
            # crash exit as "gate failed, block the commit", which is exactly
            # backwards from this module's documented fail-open contract.
            # Found by code review 2026-08-20.
            pass

    _emit(blocked, args.json)

    if not blocked:
        return 0

    if os.environ.get(ENV_ESCAPE) == "1":
        print(f"check_metric_dead_wiring: ALLOWED via {ENV_ESCAPE}=1", file=sys.stderr)
        return 0

    print(
        "\ncheck_metric_dead_wiring: COMMIT BLOCKED -- this commit wires new code to "
        "a metric that phase-5 computed liveness (just re-checked live against "
        "Postgres) currently reports as unclean. Check\n"
        f"  python scripts/check_metric_lineage.py --metric {blocked[0]['token']}\n"
        "before building on top of it (note: that command runs a full-repo "
        "consumer scan and is not fast). If this is intentional (e.g. you are "
        "the one reviving/fixing it):\n"
        f"  {ENV_ESCAPE}=1 git commit ...\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

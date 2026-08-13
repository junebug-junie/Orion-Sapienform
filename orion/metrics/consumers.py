"""Mechanical downstream-consumer discovery: who actually reads a metric.

Why this cannot reuse graphify
------------------------------
The knowledge graph at graphify-out/ carries 81,046 links, 7,984 of them
`references` and 7,039 `uses`. None of them can answer "who reads
cpu_pressure", because **metrics in this codebase are string dict keys, not
symbols**::

    vector["cpu_pressure"]
    node_vectors.get("gpu_pressure", 0.0)
    {"cpu_pressure": 0.3, "availability": 1.0}

Symbol-level extraction cannot see any of those. That is the specific,
verified reason this module exists rather than deferring to the graph.

Why AST and not grep
--------------------
CLAUDE.md 0A bans regex as an architecture. A regex for `cpu_pressure` cannot
distinguish a real read from a mention in a docstring, a log format string, or
a comment -- and blast radius is only useful if it is trustworthy. The AST walk
classifies each hit by *access kind*, so a caller can tell a subscript read
from an incidental string literal.

Known limits, stated rather than papered over:

- Dynamic access (``vector[name_from_config]``) is invisible. Nothing static
  can see it. Such call sites are undercounted, never overcounted.
- A string literal equal to a metric name in unrelated code is reported as
  ``literal`` (the lowest-confidence kind), not silently dropped.
- YAML/JSON config references are found by a separate, plain key/value scan.
"""
from __future__ import annotations

import ast
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]

# Other branches' code, build junk, and vendored deps are not this repo's
# blast radius. .worktrees/ in particular holds dozens of divergent checkouts.
EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".worktrees",
        "__pycache__",
        "node_modules",
        "graphify-out",
        ".claude",
        ".tmp-runtime-logs",
        ".orion-smoke-logs",
        ".verify-run",
        "codex_reviews",
    }
)

SCAN_ROOTS = ("orion", "services", "scripts", "tests", "config")

# Access kinds, ordered most to least confident that this is a real read.
KIND_SUBSCRIPT = "subscript"  # x["metric"]
KIND_GET = "get"  # x.get("metric")
KIND_DICT_KEY = "dict_key"  # {"metric": ...}
KIND_ATTRIBUTE = "attribute"  # x.metric  -- pydantic scalar fields
KIND_COLLECTION = "collection_member"  # ("metric", ...) / ["metric"] / {"metric"}
KIND_COMPARE = "compare"  # x == "metric", "metric" in y
KIND_LITERAL = "literal"  # bare string constant elsewhere
KIND_CONFIG = "config"  # YAML/JSON key or value

# Field-channel and organ metrics are string keys; inner-state pydantic scalar
# fields are attributes. Both are real reads, so both count as high confidence.
# Omitting attribute access systematically undercounted every inner-state
# metric to zero consumers on the first run of this scanner.
#
# collection_member matters most and is the easiest to miss by hand. A metric
# named inside a list/tuple/set constant is registered into a GENERIC consumer
# -- a loop that iterates the collection and never names the metric again.
# Verified live: cpu_pressure has zero explicit non-test key reads, but is a
# real consumer input at orion/field/pressure.py:35 (a channel tuple a reducer
# iterates) and orion/field_coherence.py:9,13 (coherence rule pairs). This is
# the generalized form of the transport_prediction_error retirement miss
# recorded in CLAUDE.md 0A, where a node kept winning budget slots in a
# generic consumer after being "retired" from one named consumer.
HIGH_CONFIDENCE_KINDS = frozenset(
    {KIND_SUBSCRIPT, KIND_GET, KIND_DICT_KEY, KIND_ATTRIBUTE, KIND_COLLECTION}
)


@dataclass(frozen=True)
class ConsumerHit:
    token: str
    path: str  # repo-relative
    line: int
    kind: str
    is_test: bool

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "path": self.path,
            "line": self.line,
            "kind": self.kind,
            "is_test": self.is_test,
        }


def _is_test_path(rel: str) -> bool:
    parts = Path(rel).parts
    return any(p in ("tests", "evals") for p in parts) or Path(rel).name.startswith("test_")


def iter_source_files(
    roots: Iterable[str] = SCAN_ROOTS, repo_root: Path | None = None
) -> Iterator[Path]:
    """Yield every scannable file under `roots`, skipping excluded trees."""
    base = repo_root or REPO_ROOT
    for root in roots:
        root_path = base / root
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in (".py", ".yaml", ".yml", ".json"):
                continue
            # Match against parts RELATIVE to the repo root. Matching absolute
            # parts silently returns zero files whenever the checkout itself
            # lives under an excluded name -- and two live worktree
            # conventions do exactly that (CLAUDE.md 2):
            # `.worktrees/<name>` and `.claude/worktrees/agent-<id>`, the
            # latter created by the Agent tool's own isolation="worktree".
            # A silent empty scan is the precise false negative this tool
            # exists to prevent, so this must stay relative.
            if any(part in EXCLUDED_DIR_NAMES for part in path.relative_to(base).parts):
                continue
            yield path


class _MetricVisitor(ast.NodeVisitor):
    """Collect string-literal metric accesses, classified by access kind."""

    def __init__(self, tokens: frozenset[str]) -> None:
        self.tokens = tokens
        self.hits: list[tuple[str, int, str]] = []
        # Nodes already claimed by a higher-confidence classification, so a
        # subscript's inner Constant is not also reported as a bare literal.
        self._claimed: set[int] = set()

    def _record(self, node: ast.AST, value: str, kind: str) -> None:
        self.hits.append((value, getattr(node, "lineno", 0), kind))
        self._claimed.add(id(node))

    def visit_Subscript(self, node: ast.Subscript) -> None:
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str) and sl.value in self.tokens:
            self._record(sl, sl.value, KIND_SUBSCRIPT)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("get", "setdefault", "pop")
            and node.args
        ):
            first = node.args[0]
            if (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value in self.tokens
            ):
                self._record(first, first.value, KIND_GET)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key in node.keys:
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value in self.tokens
            ):
                self._record(key, key.value, KIND_DICT_KEY)
        self.generic_visit(node)

    def _visit_collection(self, node: ast.List | ast.Tuple | ast.Set) -> None:
        for elt in node.elts:
            if (
                isinstance(elt, ast.Constant)
                and isinstance(elt.value, str)
                and elt.value in self.tokens
            ):
                self._record(elt, elt.value, KIND_COLLECTION)
        self.generic_visit(node)

    visit_List = _visit_collection
    visit_Tuple = _visit_collection
    visit_Set = _visit_collection

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # `frame.recon_error` -- how pydantic scalar fields are actually read.
        # Load context only: `self.pressure = x` is a WRITE, not a consumer,
        # and counting it inflated single-word tokens like `pressure` with
        # unrelated assignments across the repo.
        if node.attr in self.tokens and isinstance(node.ctx, ast.Load):
            self._record(node, node.attr, KIND_ATTRIBUTE)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for operand in [node.left, *node.comparators]:
            if (
                isinstance(operand, ast.Constant)
                and isinstance(operand.value, str)
                and operand.value in self.tokens
            ):
                self._record(operand, operand.value, KIND_COMPARE)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if (
            isinstance(node.value, str)
            and node.value in self.tokens
            and id(node) not in self._claimed
        ):
            self.hits.append((node.value, getattr(node, "lineno", 0), KIND_LITERAL))
        self.generic_visit(node)


def scan_python(path: Path, tokens: frozenset[str], rel: str) -> list[ConsumerHit]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except (SyntaxError, ValueError):
        # A file that does not parse is not a consumer. Reported by the caller
        # via scan_repo()'s unparsed list rather than silently ignored.
        # ValueError matters as much as SyntaxError: ast.parse raises it for
        # embedded NUL bytes, and letting it escape aborted the whole scan.
        raise
    visitor = _MetricVisitor(tokens)
    visitor.visit(tree)
    is_test = _is_test_path(rel)
    return [ConsumerHit(tok, rel, line, kind, is_test) for tok, line, kind in visitor.hits]


def scan_config(path: Path, tokens: frozenset[str], rel: str) -> list[ConsumerHit]:
    """Plain line scan for metric names in YAML/JSON.

    Structured parsing was considered and rejected: these files carry metric
    names as keys, as list items, and inside free-text `meaning:` prose, and a
    parse-then-walk would need a per-schema traversal for each. A line scan
    reports position accurately, which is what a blast-radius report needs.
    """
    hits: list[ConsumerHit] = []
    is_test = _is_test_path(rel)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return hits
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for token in tokens:
            if _contains_whole_token(line, token):
                hits.append(ConsumerHit(token, rel, lineno, KIND_CONFIG, is_test))
    return hits


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _contains_whole_token(line: str, token: str) -> bool:
    """Whole-token containment, so `cpu_pressure` does not match
    `cpu_pressure_ewma` and `pressure` does not match every `*_pressure`.

    A plain substring test overmatched badly, and these hits feed the orphan
    report. Boundary chars are identifier chars, so `orion:substrate:x` still
    matches inside quotes and YAML values.
    """
    start = 0
    n = len(token)
    while True:
        idx = line.find(token, start)
        if idx == -1:
            return False
        before_ok = idx == 0 or not _is_ident_char(line[idx - 1])
        after = idx + n
        after_ok = after >= len(line) or not _is_ident_char(line[after])
        if before_ok and after_ok:
            return True
        start = idx + 1


@dataclass
class ScanResult:
    hits: list[ConsumerHit]
    files_scanned: int
    unparsed: list[str]

    def by_token(self) -> dict[str, list[ConsumerHit]]:
        out: dict[str, list[ConsumerHit]] = defaultdict(list)
        for hit in self.hits:
            out[hit.token].append(hit)
        return dict(out)

    def consumers_for(
        self,
        token: str,
        high_confidence_only: bool = True,
        include_tests: bool = False,
        exclude_paths: Iterable[str] = (),
    ) -> list[ConsumerHit]:
        """Downstream consumers of `token`.

        `exclude_paths` must carry the registry file(s) the metric was
        projected FROM. A registry declaring its own metric is not a consumer
        of it, and counting it inverted the tool's central claim: every organ
        token got a collection_member self-hit from orion/signals/registry.py,
        and for 38 of 57 that self-hit was the ONLY high-confidence non-test
        result -- reporting "blast radius: 1" for a metric with no real
        consumers at all.
        """
        excluded = set(exclude_paths)
        out = [h for h in self.hits if h.token == token and h.path not in excluded]
        if high_confidence_only:
            out = [h for h in out if h.kind in HIGH_CONFIDENCE_KINDS]
        if not include_tests:
            out = [h for h in out if not h.is_test]
        return sorted(out, key=lambda h: (h.path, h.line))


def scan_repo(
    tokens: Iterable[str],
    roots: Iterable[str] = SCAN_ROOTS,
    repo_root: Path | None = None,
) -> ScanResult:
    """Walk the repo once, collecting every access to any token."""
    base = repo_root or REPO_ROOT
    token_set = frozenset(tokens)
    hits: list[ConsumerHit] = []
    unparsed: list[str] = []
    count = 0

    for path in iter_source_files(roots, repo_root=base):
        rel = str(path.relative_to(base))
        count += 1
        if path.suffix == ".py":
            try:
                hits.extend(scan_python(path, token_set, rel))
            except (SyntaxError, ValueError):
                unparsed.append(rel)
        else:
            hits.extend(scan_config(path, token_set, rel))

    return ScanResult(hits=hits, files_scanned=count, unparsed=unparsed)

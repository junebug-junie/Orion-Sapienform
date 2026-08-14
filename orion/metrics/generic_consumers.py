"""Find code that consumes a metric WITHOUT ever naming it.

WHY THIS EXISTS
---------------
`orion.metrics.consumers` finds metrics read as string literals -- the way most
of this codebase reads them. It therefore cannot see a consumer that takes a
whole channel vector and aggregates it:

    def _current_pressure_proxy(vector: dict[str, float]) -> float:
        ...max() over every channel in the vector...

`orion/attention/field_attention/selectors.py:84` does exactly that, and it
feeds attention target selection for every physical host node. So on
2026-08-14 the blast radius for `field_coherence_warning` reported **zero
consumers** while a live autonomy-adjacent path read it every 2-second tick.

That is not a cosmetic reporting gap. `check_metric_lineage.py` publishes a
list of metrics that "feed nothing", and CLAUDE.md 0A's "kill means kill" says
a retired metric gets its producer deleted outright. Acting on that list would
have killed a channel that is genuinely read. The risk was already written
down -- `HIGH_CONFIDENCE_KINDS`' own comment cites the CLAUDE.md 0A incident
"where a node kept winning budget slots in a generic consumer after being
retired from one named consumer" -- it just had no detector behind it.

WHAT IT CLAIMS, AND WHAT IT DOES NOT
------------------------------------
It does NOT claim "channel X is read here". Proving that needs real dataflow.
It claims something weaker and checkable: **these sites consume whole channel
vectors, so no channel is safely retirable without reading them.** That is the
question a retirement decision actually turns on.

Two detectors, reported at DIFFERENT confidence, because they earn different
amounts of belief:

1. **`confirmed`** -- a direct `.node_vectors` / `.capability_vectors`
   enumeration. These are FieldStateV1's channel-bearing attributes, so the
   thing being enumerated is provably a channel vector.
2. **`likely`** -- a function parameter annotated `dict[str, float]` that the
   body iterates or aggregates whole, *in a module that also touches a vector
   attribute or FieldStateV1*. This is what catches `_current_pressure_proxy`,
   whose vector arrives as a parameter with the `.node_vectors` root two call
   hops away in the same file.

The module-scope condition on (2) is doing real work, not decoration.
`dict[str, float]` is a shape, not a meaning: without it the first run
reported 34 sites, and roughly half were dictionaries keyed by topic share,
rank score, or prediction-error domain -- not by channel at all
(`_top_topic_share`, `scorer_top1`, `_js_divergence`). Reporting those as
channel consumers would make the output noise, and noise is how a report gets
ignored, which is the same end state as not having one.

KNOWN MISSES, stated rather than discovered later:

- an unannotated vector parameter (`def f(vector):`) -- no signal to key on
  without type inference;
- a vector stored on an object attribute and iterated in another method;
- generic consumption in a non-Python surface.

So a clean result here means "no generic consumer *found*", never "none
exists". The orphan report says so in those words.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from orion.metrics.consumers import _is_test_path, iter_source_files

# FieldStateV1's channel-bearing attributes. A metric surface whose values live
# in one of these is exposed to whole-vector consumption.
VECTOR_ATTRS: frozenset[str] = frozenset({"node_vectors", "capability_vectors"})

# The surfaces whose metrics live inside those vectors. Only `field_channel`
# today; named so the coupling is visible rather than assumed.
VECTOR_SURFACES: frozenset[str] = frozenset({"field_channel"})

# Builtins that consume a whole collection.
AGGREGATORS: frozenset[str] = frozenset(
    {"max", "min", "sum", "sorted", "any", "all", "len", "list", "set", "tuple", "dict"}
)

# Dict methods that hand back every entry.
ENUMERATORS: frozenset[str] = frozenset({"items", "values", "keys"})


CONFIRMED = "confirmed"
LIKELY = "likely"


@dataclass(frozen=True)
class GenericConsumer:
    path: str  # repo-relative
    line: int
    function: str
    evidence: str  # what was detected, in words
    confidence: str  # CONFIRMED | LIKELY
    is_test: bool


def _annotation_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 - unparse is best-effort on odd nodes
        return ""


def _is_channel_vector_annotation(text: str) -> bool:
    """`dict[str, float]` and its common spellings.

    Deliberately narrow: `dict[str, Any]` and bare `dict` are NOT vectors --
    matching them would flag most of the repo and make the output unreadable,
    which is its own kind of useless.
    """
    flat = text.replace(" ", "").replace("typing.", "")
    for prefix in ("Optional[", "Union["):
        if flat.startswith(prefix) and flat.endswith("]"):
            # Drop the wrapper's own brackets, then any `,None` member. An
            # earlier version used `.rstrip("]")`, which strips EVERY trailing
            # bracket -- `Optional[dict[str,float]]` became `dict[str,float`
            # and matched nothing, so every optional vector parameter was
            # silently invisible.
            flat = flat[len(prefix) : -1].split(",None")[0]
    flat = flat.removesuffix("|None")
    return flat in {
        "dict[str,float]",
        "Dict[str,float]",
        "Mapping[str,float]",
        "MutableMapping[str,float]",
    }


class _GenericVisitor(ast.NodeVisitor):
    def __init__(self, rel: str, is_test: bool, *, module_touches_vectors: bool) -> None:
        self.rel = rel
        self.is_test = is_test
        self.module_touches_vectors = module_touches_vectors
        self.found: list[GenericConsumer] = []
        self._func_stack: list[str] = []

    # -- function bodies -------------------------------------------------
    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._func_stack.append(node.name)
        args = node.args
        vector_params = {
            a.arg
            for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)
            if _is_channel_vector_annotation(_annotation_text(a.annotation))
        }
        if vector_params and self.module_touches_vectors:
            for param in sorted(vector_params):
                use = _generic_use_of(node, param)
                if use is not None:
                    line, how = use
                    self.found.append(
                        GenericConsumer(
                            path=self.rel,
                            line=line,
                            function=".".join(self._func_stack),
                            evidence=f"{how} over `{param}: dict[str, float]`",
                            confidence=LIKELY,
                            is_test=self.is_test,
                        )
                    )
        self.generic_visit(node)
        self._func_stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # -- direct .node_vectors / .capability_vectors ----------------------
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in ENUMERATORS and isinstance(node.value, ast.Attribute):
            inner = node.value
            if inner.attr in VECTOR_ATTRS:
                self.found.append(
                    GenericConsumer(
                        path=self.rel,
                        line=getattr(node, "lineno", 0),
                        function=".".join(self._func_stack) or "<module>",
                        evidence=f".{inner.attr}.{node.attr}() -- enumerates every entry",
                        confidence=CONFIRMED,
                        is_test=self.is_test,
                    )
                )
        self.generic_visit(node)


def _generic_use_of(func: ast.AST, param: str) -> tuple[int, str] | None:
    """Is `param` used as a whole, rather than subscripted by name?

    Returns (lineno, description) for the first generic use, or None. Reporting
    only the first keeps one function from emitting a dozen near-identical
    rows; the point is to name the SITE, not to enumerate every expression in
    it.
    """
    for node in ast.walk(func):
        # param.items() / .values() / .keys()
        if (
            isinstance(node, ast.Attribute)
            and node.attr in ENUMERATORS
            and isinstance(node.value, ast.Name)
            and node.value.id == param
        ):
            return getattr(node, "lineno", 0), f".{node.attr}()"
        # for x in param:
        if isinstance(node, (ast.For, ast.AsyncFor)):
            if isinstance(node.iter, ast.Name) and node.iter.id == param:
                return getattr(node, "lineno", 0), "for-loop"
        # comprehension over param
        if isinstance(node, ast.comprehension):
            # ast.comprehension carries NO lineno -- reporting `file.py:0` for
            # every comprehension made two real sites unciteable in the first
            # run. The iterable expression does carry one.
            if isinstance(node.iter, ast.Name) and node.iter.id == param:
                return getattr(node.iter, "lineno", 0), "comprehension"
        # max(param) / sum(param) / sorted(param) ...
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in AGGREGATORS:
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id == param:
                        return getattr(node, "lineno", 0), f"{node.func.id}()"
    return None


def scan_generic_consumers(
    repo_root: Path, *, include_tests: bool = False
) -> list[GenericConsumer]:
    """Every generic whole-vector consumption site in the repo."""
    out: list[GenericConsumer] = []
    for path in iter_source_files(repo_root=repo_root):
        if path.suffix != ".py":
            continue
        rel = path.relative_to(repo_root).as_posix()
        is_test = _is_test_path(rel)
        if is_test and not include_tests:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except (SyntaxError, ValueError):
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        touches = any(attr in source for attr in VECTOR_ATTRS) or "FieldStateV1" in source
        visitor = _GenericVisitor(rel, is_test, module_touches_vectors=touches)
        visitor.visit(tree)
        out.extend(visitor.found)
    out.sort(key=lambda c: (c.confidence != CONFIRMED, c.path, c.line))
    return out


def surfaces_at_risk(consumers: Iterable[GenericConsumer]) -> frozenset[str]:
    """Which metric surfaces cannot be safely called orphaned right now.

    Only `confirmed` sites gate this. A `likely` site is a prompt to go read
    something, not grounds for suppressing an orphan verdict -- letting a
    heuristic silently erase orphans would trade a false negative for a false
    positive and lose the ratchet's whole point.
    """
    return (
        VECTOR_SURFACES
        if any(c.confidence == CONFIRMED for c in consumers)
        else frozenset()
    )

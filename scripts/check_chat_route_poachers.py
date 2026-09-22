#!/usr/bin/env python3
"""Gate: no non-Hub-chat code path may default or fall back onto the `chat` LLM route.

Why this exists (live evidence, 2026-09-22):

The gateway route `chat` is Juniper's reserved interactive Hub-chat lane:
circe-worker-1, `n_parallel: 1`, unthrottled. Every background classifier,
annotator, digest, or fallback that quietly hands `route="chat"` to the
gateway is queueing behind -- or in front of -- Juniper's own turn on a
single-slot worker. A prior audit found a dozen such "poachers"; 24h of
gateway logs showed direct un-leased calls on a leased backend waiting and
then dying with `capacity_wait_budget_exhausted`.

The rule: a non-Hub-chat caller goes to `quick` / `quick_background` (small
classifiers), `metacog` (needs a 16k window), or through durable admission
(`agent` + burst lanes). It never lands on `chat` by default.

This is the deterministic version of that rule (CLAUDE.md section 4). It is a
line-based regex scan, deliberately not an AST framework. Every hit must be
named in ALLOW below with a real reason; an ALLOW entry that matches nothing
is reported as stale and also fails, so the list cannot rot.

What it scans (never a `.env`):

    orion/**/*.py
    services/*/app/**/*.py
    services/*/scripts/**/*.py
    services/*/settings.py
    services/*/.env_example

excluding tests/, evals/, docs/, graphify-out/, venv/, .venv/, node_modules/.

Shapes it detects (one line at a time; comment-only lines are skipped):

    route="chat" / llm_route="chat" / llm_route_override="chat"
    lane="chat"                       (only when `route` is also on the line)
    or "chat"                         (only when route|lane|profile is on the line)
    getattr(..., "chat")              (the literal "chat" as the default)
    Field("chat" / = "chat"           (LHS identifier contains route|profile)
    return "chat"                     (inside a def whose name has route|lane)
    route in ("chat", ...)            (a hard-coded route fallback order)
    KEY=chat in .env_example          (KEY contains ROUTE or PROFILE)

Allow-list keys are `<repo-relative path>:<enclosing def or env KEY>`; the
def part is `<module>` for module-level lines. `fnmatch` wildcards are
accepted (`*` crosses `/`), so `services/x/**:*` covers a whole service.

Usage:
    python3 scripts/check_chat_route_poachers.py            # gate (exit 1 on fail)
    python3 scripts/check_chat_route_poachers.py --json     # machine-readable
    python3 scripts/check_chat_route_poachers.py --root DIR # scan another tree
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import pathlib
import re
import sys
from dataclasses import asdict, dataclass

REPO = pathlib.Path(__file__).resolve().parents[1]

RULE = (
    "chat is Juniper's reserved Hub lane; route this call to quick/quick_background/metacog, "
    "or through durable admission (agent + burst lanes), or add an allow-list entry with a real reason"
)

# key -> reason. Keys are "path:def-or-KEY"; fnmatch wildcards allowed.
ALLOW: dict[str, str] = {
    # ---- the legitimate Hub chat path -------------------------------------
    "services/orion-cortex-exec/app/executor.py:_default_llm_route_for_step": (
        "Hub-turn adjacent: stance_react / chat_general final-response defaults ARE the Hub "
        "chat path; changing them needs proposal mode"
    ),
    "services/orion-thought/app/bus_listener.py:execute_stance_react_with_lane_fallback": (
        "agent->chat fallback leg of stance_react, a Hub-turn verb; needs proposal mode"
    ),
    "orion/harness/finalize.py:resolve_finalize_llm_lane": (
        "owner rule: the harness finalize route is owned by the Hub turn"
    ),
    # ---- lane-class axis (chat vs background), not the gateway route --------
    "services/orion-cortex-orch/app/conversation_front.py:handle_chat_turn": (
        "payload.lane is the lane-class axis for recall injection inside the Hub chat turn, "
        "not a gateway route"
    ),
    "services/orion-cortex-exec/app/main.py:<module>": (
        "EXEC_LANE picks which listener this exec process runs (chat vs background); "
        "lane-class axis, not a gateway route"
    ),
    # ---- long-context callers: only fixable through durable admission --------
    "orion/curiosity/supervisor.py:<module>": (
        "long-context batch (6000 tokens, 180 s); needs durable admission (follow-up)"
    ),
    "services/orion-cortex-orch/app/workflow_runtime.py:_run_chat_history_compactor_digest": (
        "digest input is up to 30 turns x 3200 chars (~24k tokens), over quick (4k) and "
        "metacog (16k); needs durable admission"
    ),
    "services/orion-context-exec/**:*": (
        "CONTEXT_EXEC_DEFAULT_LLM_PROFILE=chat: long-context investigations; needs durable admission"
    ),
    # ---- multimodal: only the chat worker serves vision --------------------
    "services/orion-juniper-affective-state/**:*": (
        "AFFECT_VISION_LLM_ROUTE=chat: multimodal worker requirement (vision lives on circe-worker-1)"
    ),
    "services/orion-vision-council/**:*": (
        "FOVEAL_LLM_ROUTE=chat: multimodal worker requirement (vision lives on circe-worker-1)"
    ),
    # ---- catalog lookups and constants, not a dispatch -----------------------
    "orion/situational/context.py:*": (
        "ORION_SITUATION_RUNTIME_ROUTE names which route's status to read from the catalog, "
        "not where to send a request"
    ),
    "services/orion-cortex-exec/app/settings.py:<module>": (
        "ORION_SITUATION_RUNTIME_ROUTE code default: catalog lookup, not a dispatch"
    ),
    "services/orion-cortex-exec/.env_example:ORION_SITUATION_RUNTIME_ROUTE": (
        "catalog lookup, not a dispatch"
    ),
    "orion/schemas/situation.py:<module>": (
        "LLM situation snapshot records which route was probed (catalog lookup), not a dispatch"
    ),
    "orion/llm/routes.py:<module>": (
        "CHAT_BURST_LENDS_ROUTE names the route the chat-burst worker lends capacity to; "
        "a constant the Hub reads, not a dispatch default"
    ),
    # ---- gateway last resorts, gated off live ------------------------------
    "services/orion-llm-gateway/app/lane_routes.py:*": (
        "_chat_fallback is gated off by LLM_ALLOW_BACKGROUND_TO_CHAT_FALLBACK=false and allow_chat_fallback"
    ),
    "services/orion-llm-gateway/app/llm_backend.py:*": (
        "`or 'chat'` last resort behind LLM_ROUTE_DEFAULT, which is quick live"
    ),
    "services/orion-llm-gateway/app/anthropic_passthrough.py:*": (
        "`or 'chat'` last resort behind LLM_ROUTE_DEFAULT, which is quick live"
    ),
    "services/orion-llm-gateway/app/openai_passthrough.py:*": (
        "`or 'chat'` last resort behind LLM_ROUTE_DEFAULT, which is quick live"
    ),
    "services/orion-llm-gateway/app/route_catalog.py:*": (
        "`or 'chat'` last resort behind LLM_ROUTE_DEFAULT, which is quick live"
    ),
}

_EXCLUDED_PARTS = {"tests", "evals", "docs", "graphify-out", "venv", ".venv", "node_modules", "__pycache__"}

_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
_ROUTE_KWARG_RE = re.compile(r"""\b(?:llm_route_override|llm_route|route)\s*=\s*["']chat["']""")
_LANE_KWARG_RE = re.compile(r"""\blane\s*=\s*["']chat["']""")
_OR_CHAT_RE = re.compile(r"""\bor\s+["']chat["']""")
_GETATTR_RE = re.compile(r"""getattr\([^()]*?,\s*["']chat["']\s*\)""")
_FIELD_RE = re.compile(r"""Field\(\s*["']chat["']""")
_ASSIGN_RE = re.compile(r"""^\s*([A-Za-z_][\w.]*)\s*(?::[^=]*)?=\s*["']chat["']""")
_LHS_RE = re.compile(r"""^\s*([A-Za-z_][\w.]*)\s*(?::[^=]*)?=""")
_RETURN_RE = re.compile(r"""^\s*return\s+["']chat["']\s*(?:#.*)?$""")
_ROUTE_TUPLE_RE = re.compile(r"""\broute\w*\b[^=]*\bin\s*\(\s*["']chat["']\s*,""")
_ENV_RE = re.compile(r"^[A-Z0-9_]*(?:ROUTE|PROFILE)[A-Z0-9_]*=chat$")
_NAME_HINT_RE = re.compile(r"route|profile", re.IGNORECASE)
_OR_HINT_RE = re.compile(r"route|lane|profile", re.IGNORECASE)
_DEF_HINT_RE = re.compile(r"route|lane", re.IGNORECASE)


@dataclass(frozen=True)
class Hit:
    path: str
    lineno: int
    scope: str
    shape: str
    line: str

    @property
    def key(self) -> str:
        return f"{self.path}:{self.scope}"


def _excluded(rel: pathlib.PurePath) -> bool:
    return any(part in _EXCLUDED_PARTS for part in rel.parts)


def iter_scan_files(root: pathlib.Path) -> list[pathlib.Path]:
    globs = (
        "orion/**/*.py",
        "services/*/app/**/*.py",
        "services/*/scripts/**/*.py",
        "services/*/settings.py",
        "services/*/.env_example",
    )
    seen: set[pathlib.Path] = set()
    for pattern in globs:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if _excluded(rel):
                continue
            seen.add(path)
    return sorted(seen)


def scan_python_text(text: str, rel: str) -> list[Hit]:
    hits: list[Hit] = []
    scope = "<module>"
    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        m_def = _DEF_RE.match(raw)
        if m_def:
            scope = m_def.group(1)
        elif stripped and (raw[0].isalpha() or raw[0] == "_"):
            # Column 0 and starts a statement: module-level code again. A bare
            # `) -> T:` closing a multi-line def header must NOT reset scope.
            scope = "<module>"
        if not stripped or stripped.startswith("#"):
            continue
        # Drop a trailing comment so a comment mentioning `or "chat"` does not trip the scan.
        code = _strip_trailing_comment(raw)
        shape: str | None = None
        if _ROUTE_KWARG_RE.search(code):
            shape = "route_kwarg"
        elif _LANE_KWARG_RE.search(code) and "route" in code:
            shape = "lane_kwarg"
        elif _OR_CHAT_RE.search(code) and _OR_HINT_RE.search(code):
            # `x or "chat"` alone is too loose: `source or "chat"` in
            # orion/hub/runtime_activity.py is a turn-source label, not a route.
            shape = "or_chat"
        elif _GETATTR_RE.search(code):
            shape = "getattr_default"
        elif _FIELD_RE.search(code):
            m_lhs = _LHS_RE.match(code)
            if m_lhs and _NAME_HINT_RE.search(m_lhs.group(1)):
                shape = "field_default"
        elif _ASSIGN_RE.match(code):
            if _NAME_HINT_RE.search(_ASSIGN_RE.match(code).group(1)):
                shape = "assign_default"
        elif _RETURN_RE.match(code):
            if _DEF_HINT_RE.search(scope):
                shape = "return_chat"
        elif _ROUTE_TUPLE_RE.search(code):
            shape = "route_tuple"
        if shape:
            hits.append(Hit(path=rel, lineno=lineno, scope=scope, shape=shape, line=stripped))
    return hits


def scan_env_text(text: str, rel: str) -> list[Hit]:
    hits: list[Hit] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if _ENV_RE.match(line):
            key = line.split("=", 1)[0]
            hits.append(Hit(path=rel, lineno=lineno, scope=key, shape="env_default", line=line))
    return hits


def _strip_trailing_comment(line: str) -> str:
    # Good enough for this scan: cut at the first `#` that is not inside quotes.
    out = []
    quote: str | None = None
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#":
            break
        out.append(ch)
    return "".join(out)


def scan_tree(root: pathlib.Path) -> list[Hit]:
    hits: list[Hit] = []
    for path in iter_scan_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if path.name == ".env_example":
            hits.extend(scan_env_text(text, rel))
        else:
            hits.extend(scan_python_text(text, rel))
    return hits


def classify(hits: list[Hit], allow: dict[str, str]) -> tuple[list[Hit], list[tuple[Hit, str]], list[str]]:
    """Split hits into (unallowed, allowed-with-reason, stale-allow-keys)."""
    unallowed: list[Hit] = []
    allowed: list[tuple[Hit, str]] = []
    used: set[str] = set()
    for hit in hits:
        reason = None
        for pattern, why in allow.items():
            if hit.key == pattern or fnmatch.fnmatchcase(hit.key, pattern):
                reason = why
                used.add(pattern)
                break
        if reason is None:
            unallowed.append(hit)
        else:
            allowed.append((hit, reason))
    stale = [k for k in allow if k not in used]
    return unallowed, allowed, stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path, default=REPO, help="tree to scan (default: repo root)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    hits = scan_tree(root)
    unallowed, allowed, stale = classify(hits, ALLOW)
    ok = not unallowed and not stale

    if args.json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "unallowed": [asdict(h) for h in unallowed],
                    "allowed": [{**asdict(h), "reason": r} for h, r in allowed],
                    "stale_allow": stale,
                },
                indent=2,
            )
        )
        return 0 if ok else 1

    if unallowed:
        print("chat route poacher gate: FAIL")
        print("")
        print(f"  {RULE}")
        print("")
        for hit in unallowed:
            print(f"    {hit.path}:{hit.lineno} [{hit.scope}] ({hit.shape})")
            print(f"        {hit.line}")
    if stale:
        print("chat route poacher gate: STALE allow-list entries (match nothing; remove them):")
        for key in stale:
            print(f"    {key}")
    if not ok:
        return 1
    print(
        f"chat route poacher gate: PASS ({len(hits)} chat-route defaults found, all allow-listed; "
        f"{len(ALLOW)} allow entries in use)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

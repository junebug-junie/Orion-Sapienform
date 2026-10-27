#!/usr/bin/env python3
"""Fail when a tuned env key's value is copied somewhere that has drifted.

Some env keys are load-bearing numbers that get restated in several places --
the service's `.env_example`, its compose default, a `Field(...)` default, and
prose in a comment or README explaining a budget derived from them. Every
restatement is a copy, and copies drift.

Measured, 2026-08-26: `HARNESS_FCC_TIMEOUT_SEC` was raised to 1600 in the live
`.env` while `.env_example`, the compose default, the governor's own
`settings.py` default, two Hub comments, `orion/llm/routes.py` and the
llm-gateway README all still said 900. Nothing failed, because nothing was
checking. That number is now stamped into the FCC sandbox at spawn time
(`orion/harness/fcc_motor.py:_build_subprocess_env`) so a turn can read its own
deadline -- which turns a stale copy from something operators misread into
something Orion is actively told.

The rule is NOT "state it once". Code and compose defaults have to exist. The
rule is ONE OWNER: every literal for the key, anywhere, must equal the value in
the owner file. The gate never hardcodes the number itself, so retuning the key
in its owner is a one-line change that keeps this passing.

Usage:
    python scripts/check_env_key_single_source.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent

# key -> the file whose value is authoritative.
OWNERS: dict[str, str] = {
    "HARNESS_FCC_TIMEOUT_SEC": "services/orion-harness-governor/.env_example",
    "HUB_READING_SEARCH_MIN_SIMILARITY": "services/orion-hub/.env_example",
    "DREAM_SEARCH_MIN_SIMILARITY": "services/orion-dream/.env_example",
    "HUB_CURIOSITY_SEARCH_MIN_SIMILARITY": "services/orion-hub/.env_example",
    # The engaged/idle boundary. Read by the dream's idle gate AND by arousal in
    # orion-durable-runs (Temporal Self rev 4 R3: "reuses the dream setting's existing
    # meaning, so the threshold is not copied"). Two services need the env key; one owns it.
    "DREAM_IDLE_MINUTES": "services/orion-dream/.env_example",
    # The outreach daily cap. Enforced by the Hub's outreach gate; read by orion-durable-runs as
    # the attention:juniper line of the Body organ (body economy Patch A3). One owner.
    "HUB_ENDOGENOUS_OUTREACH_DAILY_CAP": "services/orion-hub/.env_example",
}

# Lines owned by a Python MODULE, not an env file (body economy Patch B, 2026-10-27).
# The rule is stricter than OWNERS: there may be no copy at all, equal or not. A consumer
# imports the constant (or calls the owner's function and takes its defaults). The values
# are read from the owner by AST, never hardcoded here.
#
# Measured before Patch B: the 32 C hot line lived in thermal_gate.py AND as
# ORION_THERMAL_HOT_C / ORION_THERMAL_HOT_REARM_C in orion-thought (settings default, .env_example,
# live .env) AND as `value: 32.0` / `value: 34.0` in config/prediction/series.v1.yaml.
MODULE_OWNED_LINES: dict[str, tuple[str, ...]] = {
    "orion/autonomy/thermal_gate.py": (
        "DEFAULT_HOT_C", "DEFAULT_HOT_REARM_C",
        "DEFAULT_ELEVATED_C", "DEFAULT_ELEVATED_REARM_C",
        "DEFAULT_CRITICAL_C", "DEFAULT_CRITICAL_REARM_C",
    ),
}
# A name that reads as a room-temperature line: a thermal word, and a Celsius suffix. Both the
# name AND an owned value must match, so GPU/CPU ceilings (80/85 C) and unrelated `_c` knobs pass.
_THERMAL_LINE_NAME = re.compile(r"(?i)(hot|crit|elevat|rearm|re_arm|thermal|cabinet|room)\w*_c$")
# Env keys deleted by Patch B. Refused anywhere they reappear (including a live `.env`, which the
# sync script never prunes -- that is exactly where a dead key lingers).
RETIRED_KEYS: dict[str, str] = {
    "ORION_THERMAL_HOT_C": "orion/autonomy/thermal_gate.py DEFAULT_HOT_C",  # env-key-single-source: sample
    "ORION_THERMAL_HOT_REARM_C": "orion/autonomy/thermal_gate.py DEFAULT_HOT_REARM_C",  # env-key-single-source: sample
}
_THERMAL_NAME_IN_TEXT = re.compile(r"(?i)(hot|crit|elevat|rearm|re_arm|thermal|cabinet|room)\w*_c\b")
_ENV_KEY = re.compile(r"^\s*-?\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:=]")
_ENV_ASSIGN = re.compile(r"^\s*-?\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(?:\$\{[A-Za-z0-9_]+:-)?[\"']?([0-9]+(?:\.[0-9]+)?)")

# Line-level opt-out for text that quotes a value deliberately.
SAMPLE_MARKER = "env-key-single-source: sample"

SCANNED_SUFFIXES = {".py", ".md", ".yml", ".yaml"}
# The local `.env` is scanned too, and this is the point rather than an
# afterthought: the 2026-08-26 drift was live `.env` at 1600 against a checked-in
# `.env_example` at 900. A gate reading only committed files would have been
# GREEN throughout that incident, because every committed copy agreed with every
# other -- uniformly stale. CI runners have no `.env`, so there it simply finds
# nothing to compare; the check has teeth exactly where the divergence lives.
#
# A diverged local value is a finding, not a presumed deliberate override --
# `.env_example` is required to state the live INTENDED default.
SCANNED_NAMES = {".env_example", ".env"}

# Historical records, deliberately frozen at what was true when written. A PR
# report or a design spec that silently rewrote itself when a config changed
# would be worse than a stale one.
EXCLUDED_PREFIXES = (
    "docs/superpowers/pr-reports/",
    "docs/superpowers/specs/",
    ".git/",
    "node_modules/",
    "graphify-out/",
    ".worktrees/",
    ".claude/worktrees/",
    # Gitignored agent scratch (task briefs quoting values as of when written).
    ".superpowers/",
)


def _iter_files() -> Iterator[Path]:
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith(EXCLUDED_PREFIXES):
            continue
        if path.suffix in SCANNED_SUFFIXES or path.name in SCANNED_NAMES:
            yield path


def _read_tree() -> list[tuple[Path, str, str]]:
    """(path, rel, text) for every scanned file, read once and shared by both checks."""
    out: list[tuple[Path, str, str]] = []
    for path in _iter_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        out.append((path, path.relative_to(REPO_ROOT).as_posix(), text))
    return out


def _is_local_env(rel: str) -> bool:
    return rel == ".env" or rel.endswith("/.env")


def _same_number(a: str, b: str) -> bool:
    """`900`, `900.0` and `900s` are the same value stated three ways."""
    try:
        return float(a.rstrip("s")) == float(b.rstrip("s"))
    except ValueError:
        return a == b


def _literals(text: str, key: str) -> Iterator[tuple[int, str]]:
    """Every place this text pins the key to a number.

    Deliberately narrow: the number must follow the key directly, through `=`
    or compose's `:-`. Prose like "KEY + OTHER_KEY=300" names the key without
    pinning it and is not a copy of its value.
    """
    patterns = (
        rf"{re.escape(key)}\s*[:]?[-=]\s*([0-9]+(?:\.[0-9]+)?s?)",
        rf"Field\(\s*(?:default\s*=\s*)?([0-9]+(?:\.[0-9]+)?)[^)]*alias=[\"']{re.escape(key)}[\"']",
    )
    lines = text.splitlines()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            line = text.count("\n", 0, match.start()) + 1
            # An inline opt-out for text that quotes a value on purpose -- this
            # gate's own fixtures, or a doc showing what drift looks like.
            # Marker rather than a path exclusion so that a *real* stale default
            # sitting in some other test file is still caught.
            if 0 < line <= len(lines) and SAMPLE_MARKER in lines[line - 1]:
                continue
            yield line, match.group(1)


def _owned_line_values() -> tuple[dict[float, str], list[str]]:
    """value -> "owner NAME", read from each owner module by AST."""
    values: dict[float, str] = {}
    failures: list[str] = []
    for owner_rel, names in MODULE_OWNED_LINES.items():
        owner = REPO_ROOT / owner_rel
        if not owner.is_file():
            failures.append(f"module owner {owner_rel} does not exist")
            continue
        found: dict[str, float] = {}
        for node in ast.parse(owner.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in names and isinstance(node.value.value, (int, float)):
                        found[t.id] = float(node.value.value)
        for name in names:
            if name not in found:
                failures.append(f"module owner {owner_rel} defines no numeric {name}")
            else:
                values.setdefault(found[name], f"{owner_rel} {name}")
    return values, failures


def _number(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    return None


def _py_line_copies(text: str) -> Iterator[tuple[int, str, float]]:
    """(line, name, value) for every place Python binds a thermal-looking name to a number:
    assignments, annotated assignments, parameter defaults, and pydantic `Field(<n>, alias=KEY)`."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        pairs: list[tuple[str, ast.AST | None, int]] = []
        if isinstance(node, ast.Assign):
            for t in node.targets:
                name = t.id if isinstance(t, ast.Name) else t.attr if isinstance(t, ast.Attribute) else None
                if name:
                    pairs.append((name, node.value, node.lineno))
        elif isinstance(node, ast.AnnAssign):
            t = node.target
            name = t.id if isinstance(t, ast.Name) else t.attr if isinstance(t, ast.Attribute) else None
            if name:
                pairs.append((name, node.value, node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            pos = a.posonlyargs + a.args
            for arg, default in zip(pos[len(pos) - len(a.defaults):], a.defaults):
                pairs.append((arg.arg, default, default.lineno))
            for arg, default in zip(a.kwonlyargs, a.kw_defaults):
                if default is not None:
                    pairs.append((arg.arg, default, default.lineno))
        for name, value, line in pairs:
            if isinstance(value, ast.Call) and getattr(value.func, "id", getattr(value.func, "attr", "")) == "Field":
                alias = next((k.value for k in value.keywords if k.arg == "alias"), None)
                default = value.args[0] if value.args else next(
                    (k.value for k in value.keywords if k.arg == "default"), None)
                if isinstance(alias, ast.Constant) and isinstance(alias.value, str) and _THERMAL_LINE_NAME.search(alias.value):
                    name = alias.value
                value = default
            num = _number(value)
            if num is not None and _THERMAL_LINE_NAME.search(name):
                yield line, name, num


def _module_owned_failures(tree: list[tuple[Path, str, str]] | None = None) -> list[str]:
    values, failures = _owned_line_values()
    owners = set(MODULE_OWNED_LINES)
    for path, rel, text in (tree if tree is not None else _read_tree()):
        if rel in owners:
            continue
        lines = text.splitlines()

        def opted_out(line: int) -> bool:
            return 0 < line <= len(lines) and SAMPLE_MARKER in lines[line - 1]

        for key, owner in RETIRED_KEYS.items():
            if key not in text:
                continue
            # A retired key is refused where it is SET or READ, not where prose names it: an env/compose
            # assignment, or the key as a Python string literal (alias=, os.getenv, environ[...]).
            hits: list[int] = []
            if path.suffix == ".py":
                try:
                    hits = [n.lineno for n in ast.walk(ast.parse(text))
                            if isinstance(n, ast.Constant) and n.value == key]
                except SyntaxError:
                    hits = []
            else:
                hits = [i for i, ln in enumerate(lines, 1)
                        if (m := _ENV_KEY.match(ln)) and m.group(1) == key]
            for i in hits:
                if not opted_out(i):
                    failures.append(f"{rel}:{i}: {key} was retired; the line is owned by {owner} -- import it")
        copies: list[tuple[int, str, float]] = []
        # Cheap prefilter before any AST parse: a copy needs a thermal-looking `_c` name in the text.
        if not _THERMAL_NAME_IN_TEXT.search(text):
            continue
        if path.suffix == ".py":
            copies = list(_py_line_copies(text))
        elif path.suffix in {".yml", ".yaml"} or path.name in SCANNED_NAMES:
            for i, ln in enumerate(lines, 1):
                m = _ENV_ASSIGN.match(ln)
                if m and _THERMAL_LINE_NAME.search(m.group(1)):
                    copies.append((i, m.group(1), float(m.group(2))))
        for line, name, num in copies:
            if num in values and not opted_out(line) and name not in RETIRED_KEYS:
                failures.append(f"{rel}:{line}: {name} = {num:g} copies the line owned by {values[num]} -- import it")
    return failures


def main() -> int:
    tree = _read_tree()
    failures: list[str] = _module_owned_failures(tree)
    for key, owner_rel in OWNERS.items():
        owner = REPO_ROOT / owner_rel
        if not owner.is_file():
            failures.append(f"{key}: owner file {owner_rel} does not exist")
            continue
        owned = list(_literals(owner.read_text(encoding="utf-8"), key))
        if not owned:
            failures.append(f"{key}: owner {owner_rel} states no value for it")
            continue
        expected = owned[0][1]

        for path, rel, text in tree:
            if rel == owner_rel:
                continue
            if key not in text:
                continue
            for line, found in _literals(text, key):
                if not _same_number(found, expected):
                    if _is_local_env(rel):
                        failures.append(
                            f"{rel}:{line}: {key} is LIVE at {found} but "
                            f"{owner_rel} still says {expected} -- the operator "
                            f"contract has drifted from what is running"
                        )
                    else:
                        failures.append(
                            f"{rel}:{line}: {key} stated as {found}, but "
                            f"{owner_rel} says {expected}"
                        )

    if failures:
        print("Env key drift -- one owner per tuned key:\n", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nFix the copy, not the owner, unless you are deliberately "
            "retuning the key.",
            file=sys.stderr,
        )
        return 1

    n_lines = sum(len(v) for v in MODULE_OWNED_LINES.values())
    print(f"OK: {len(OWNERS)} owned env key(s), no drifted copies; "
          f"{n_lines} module-owned line(s), no copies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The "Ask Orion" catalog (static/catalog/orion-abilities.json) cannot drift from what Orion can do.

The catalog is hand-written prose, so it would rot the day someone adds a dig command. These tests read
the real sources by text (no imports, no services) and fail when:

* a dig command is in the dig help (``orion/dig/__main__.py`` HELP), in its dispatcher (``main()``), or
  in the turn-rules line (``orion/dig/brief.py`` DIG_BRIEF_LINE) with no catalog entry and no
  ``hidden`` reason;
* a ``capacity`` sub-verb or a ``body-economy`` section (``orion/physiology/report_text.py`` SECTIONS)
  has no entry;
* an MCP tool the chat motor is told about (``orion/introspect/brief.py``,
  ``orion/world_pulse_read/tools.py``, ``orion/fcc/self_index_brief.py``) or an AI Town server tool
  (``services/orion-ai-town/mcp/orion_aitown_mcp/server.py``) has no entry;
* an entry is malformed: empty text, a bare command as the message, a hint without the command in
  backticks, a hint naming a dig command that does not exist, or a ``source`` file that is gone.

Fix a failure by describing the new thing in the catalog (or adding it to ``hidden`` with a reason).
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

HUB = Path(__file__).resolve().parents[1]
REPO = HUB.parents[1]
CATALOG_PATH = HUB / "static" / "catalog" / "orion-abilities.json"
DIG_MAIN = REPO / "orion" / "dig" / "__main__.py"
DIG_BRIEF = REPO / "orion" / "dig" / "brief.py"
REPORT_TEXT = REPO / "orion" / "physiology" / "report_text.py"
#: Briefs that name MCP tools to the chat motor, as ``mcp_tool(SERVER_CONST, "tool")`` calls.
MCP_BRIEFS = {
    "orion-introspect": REPO / "orion" / "introspect" / "brief.py",
    "orion-reading": REPO / "orion" / "world_pulse_read" / "tools.py",
    "gitnexus": REPO / "orion" / "fcc" / "self_index_brief.py",
}
MCP_SERVER_CONST = {"ORION_INTROSPECT": "orion-introspect", "ORION_READING": "orion-reading", "GITNEXUS": "gitnexus"}
#: AI Town has no brief line; its tools are the ``@mcp.tool()`` functions of its server.
AITOWN_SERVER = REPO / "services" / "orion-ai-town" / "mcp" / "orion_aitown_mcp" / "server.py"
HELP_ALIASES = {"help", "-h", "--help"}


def _catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _entries(cat: dict) -> list[dict]:
    return [e for s in cat["sections"] for e in s["entries"]]


def _covered(cat: dict, kind: str) -> set[str]:
    out = {name for e in _entries(cat) for name in (e.get("covers") or {}).get(kind, [])}
    out |= set((cat.get("hidden") or {}).get(kind, {}))
    return out


def _string_assign(path: Path, name: str) -> str:
    """The literal value of ``NAME = (...)`` / ``NAME = f\"\"\"...\"\"\"`` (f-string parts kept as text)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            v = node.value
            if isinstance(v, ast.Constant):
                return v.value
            if isinstance(v, ast.JoinedStr):
                return "".join(p.value if isinstance(p, ast.Constant) else "{}" for p in v.values)
            return ast.literal_eval(v)
    raise AssertionError(f"{name} not found in {path}")


def dig_commands_from_help() -> set[str]:
    """Top-level commands as the help lists them: the first word of each two-space-indented line."""
    help_text = _string_assign(DIG_MAIN, "HELP")
    # The command block ends where the source-label block ("Label what you say ...") begins; its
    # indented lines (code_then, now, reflection, ...) are labels, not commands.
    block = re.split(r"\n(?=\S)", help_text.split("\n\n", 1)[1], maxsplit=1)[0]
    cmds: set[str] = set()
    for line in block.splitlines():
        m = re.match(r"^  ([a-z][a-z|-]*)\b", line)
        if m:
            cmds |= set(m.group(1).split("|"))
        cmds |= set(re.findall(r"\b([a-z-]+): show", line))  # "list: show open exports."
    return cmds


def dig_commands_from_dispatch() -> tuple[set[str], set[str]]:
    """Every command main() dispatches, and every ``capacity`` sub-verb."""
    src = DIG_MAIN.read_text(encoding="utf-8")
    body = src[src.index("def main("):]
    cmds = set(re.findall(r'ns\.command == "([^"]+)"', body))
    for group in re.findall(r"ns\.command in \(([^)]*)\)", body):
        cmds |= set(re.findall(r'"([^"]+)"', group))
    ref = re.search(r"^REF_COMMANDS = \(([^)]*)\)", src, re.M)
    if ref:
        cmds |= set(re.findall(r'"([^"]+)"', ref.group(1)))
    subs: set[str] = set()
    for line in body.splitlines():
        if 'ns.command == "capacity"' not in line:
            continue
        for group in re.findall(r"a\[0\] in \(([^)]*)\)", line):
            subs |= set(re.findall(r'"([^"]+)"', group))
        subs |= set(re.findall(r'a\[0\] == "([^"]+)"', line))
        if '["help"]' in line:
            subs.add("help")
    return cmds - HELP_ALIASES, subs


def dig_commands_from_brief() -> tuple[set[str], set[str]]:
    """Commands (and body-economy sections) the always-on turn-rules line names."""
    line = _string_assign(DIG_BRIEF, "DIG_BRIEF_LINE")
    cmds, sections = set(), set()
    for cmd, rest in re.findall(r"orion\.dig ([a-z][a-z-]*)((?: [a-z0-9_]+)*)", line):
        if cmd.startswith("<"):
            continue
        cmds.add(cmd)
        if cmd == "body-economy":
            sections |= {w for w in rest.split() if not w.isdigit()}
    return cmds, sections


def body_economy_sections() -> set[str]:
    return set(_string_assign(REPORT_TEXT, "SECTIONS"))


def mcp_tools_told_to_chat() -> set[str]:
    tools: set[str] = set()
    for server, path in MCP_BRIEFS.items():
        assert path.is_file(), f"{path} moved: update MCP_BRIEFS"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "mcp_tool" and len(node.args) == 2
                    and isinstance(node.args[0], ast.Name) and isinstance(node.args[1], ast.Constant)
                    and MCP_SERVER_CONST.get(node.args[0].id) == server):
                tools.add(f"mcp__{server}__{node.args[1].value}")
    assert AITOWN_SERVER.is_file(), f"{AITOWN_SERVER} moved: update AITOWN_SERVER"
    for node in ast.parse(AITOWN_SERVER.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                isinstance(d, ast.Call) and getattr(d.func, "attr", None) == "tool" for d in node.decorator_list):
            tools.add(f"mcp__orion-aitown__{node.name}")
    return tools


# ---- the parsers themselves see real commands (so a parser bug cannot pass everything) ----

def test_parsers_find_the_known_commands():
    helped = dig_commands_from_help()
    dispatched, subs = dig_commands_from_dispatch()
    assert {"body", "portrait", "reminisce", "body-economy", "field-note", "list", "arc", "act"} <= helped
    assert {"open", "close", "capacity", "name-organs", "notes"} <= dispatched
    assert {"claim", "dispute", "write", "withdraw", "help"} <= subs
    assert "body-economy" in dig_commands_from_brief()[0]
    assert {"hosts", "money", "limits"} <= body_economy_sections()
    tools = mcp_tools_told_to_chat()
    assert {"mcp__orion-reading__recommend_reading", "mcp__orion-introspect__dreams", "mcp__gitnexus__query",
            "mcp__orion-aitown__aitown_move_player"} <= tools


# ---- drift gates ----

def test_every_dig_command_has_an_entry():
    cat = _catalog()
    dispatched, _ = dig_commands_from_dispatch()
    missing = sorted((dig_commands_from_help() | dispatched | dig_commands_from_brief()[0]) - _covered(cat, "dig"))
    assert not missing, (f"dig commands with no catalog entry in {CATALOG_PATH.relative_to(REPO)}: {missing}. "
                         "Describe each one (or add it to hidden.dig with a reason).")


def test_every_capacity_sub_verb_has_an_entry():
    _, subs = dig_commands_from_dispatch()
    missing = sorted(subs - _covered(_catalog(), "dig_capacity"))
    assert not missing, f"capacity sub-verbs with no catalog entry: {missing}"


def test_every_body_economy_section_has_an_entry():
    wanted = body_economy_sections() | dig_commands_from_brief()[1]
    missing = sorted(wanted - _covered(_catalog(), "body_economy"))
    assert not missing, f"body-economy sections with no catalog entry: {missing}"


def test_every_chat_mcp_tool_has_an_entry():
    missing = sorted(mcp_tools_told_to_chat() - _covered(_catalog(), "mcp"))
    assert not missing, f"MCP tools the chat motor is told about, with no catalog entry: {missing}"


def test_catalog_covers_nothing_that_does_not_exist():
    """A stale entry (a removed command still advertised) is drift too."""
    cat = _catalog()
    dispatched, subs = dig_commands_from_dispatch()
    real = {
        "dig": dig_commands_from_help() | dispatched,
        "dig_capacity": subs,
        "body_economy": body_economy_sections() | {"overview", "all"},
        "mcp": mcp_tools_told_to_chat(),
    }
    for e in _entries(cat):
        for kind, names in (e.get("covers") or {}).items():
            assert kind in real, f"{e['id']}: unknown covers kind {kind!r}"
            stale = sorted(set(names) - real[kind])
            assert not stale, f"{e['id']} covers {kind} that no longer exist: {stale}"


# ---- entry shape ----

def test_entries_are_well_formed():
    cat = _catalog()
    assert cat["sections"], "empty catalog"
    ids, sec_ids = set(), set()
    dispatched, _ = dig_commands_from_dispatch()
    known_dig = dig_commands_from_help() | dispatched
    for s in cat["sections"]:
        assert s["id"] not in sec_ids and re.fullmatch(r"[a-z0-9-]+", s["id"]), s["id"]
        sec_ids.add(s["id"])
        assert s["title"].strip() and s["entries"], s["id"]
        for e in s["entries"]:
            eid = e["id"]
            assert eid not in ids and re.fullmatch(r"[a-z0-9-]+", eid), eid
            ids.add(eid)
            assert 3 <= len(e["title"]) <= 60, eid
            assert 30 <= len(e["description"]) <= 320, eid
            msg = e["message"].strip()
            # Juniper does not remember command words; Orion routes by topic. A message is a plain ask.
            assert len(msg) >= 20 and not re.match(r"^(python|`|\$|mcp__|/)", msg), f"{eid}: message is a bare command"
            assert "`" not in msg, f"{eid}: put commands in the hint line, not the message"
            hint = e.get("hint")
            if hint:
                assert re.search(r"`[^`]+`", hint), f"{eid}: hint must hold the command in backticks"
                for cmd in re.findall(r"orion\.dig (?:--dig \S+ )?([a-z][a-z-]*)", hint):
                    assert cmd in known_dig, f"{eid}: hint names unknown dig command {cmd!r}"
            src = e.get("source", "")
            assert src, f"{eid}: say where this ability is defined"
            for ref in src.split(";"):
                file_part = ref.strip().split(":")[0].split(" ")[0]
                assert (REPO / file_part).exists(), f"{eid}: source {file_part} does not exist"
    for kind, reasons in (cat.get("hidden") or {}).items():
        for name, reason in reasons.items():
            assert len(reason.strip()) >= 15, f"hidden.{kind}.{name}: give a real reason"


# ---- the template wires it (repo rule section 9: template + asset path) ----

def test_template_has_button_above_send_and_modal_and_assets():
    html = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    stack = re.search(r'<div id="sendStack"[^>]*>(.*?)</div>', html, re.S)
    assert stack, "the Send button lost its stack"
    inner = stack.group(1)
    assert inner.index('id="orionAbilitiesButton"') < inner.index('id="sendButton"'), "Ask Orion must sit above Send"
    for needed in ('id="orionAbilitiesModalRoot"', 'role="dialog"', 'aria-modal="true"', 'id="orionAbilitiesSearch"',
                   'id="orionAbilitiesList"'):
        assert needed in html, needed
    for asset in ("/static/js/orion-abilities.js", "/static/css/orion-abilities.css"):
        m = re.search(re.escape(asset) + r"\?v=\{\{HUB_UI_ASSET_VERSION\}\}", html)
        assert m, f"{asset} is not linked with the cache-busting version"
        assert (HUB / asset.lstrip("/")).is_file(), asset
    js = (HUB / "static" / "js" / "orion-abilities.js").read_text(encoding="utf-8")
    assert '"/static/catalog/orion-abilities.json"' in js and CATALOG_PATH.is_file()


@pytest.mark.parametrize("path", [CATALOG_PATH])
def test_catalog_is_valid_json(path):
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)


def test_catalog_is_not_gitignored():
    """The root .gitignore ignores every ``data/`` folder; the catalog first lived in static/data/ and
    was silently left out of the commit. Keep it somewhere git (and the image) will carry."""
    import shutil
    import subprocess

    if not shutil.which("git") or not (REPO / ".git").exists():
        pytest.skip("no git checkout")
    rc = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", str(CATALOG_PATH)], check=False).returncode
    assert rc == 1, f"{CATALOG_PATH.relative_to(REPO)} is gitignored: move it"

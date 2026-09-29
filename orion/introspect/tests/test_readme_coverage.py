"""Every introspect tool ships documented: governor overview row + responder README section."""
import re
from pathlib import Path

import yaml

from orion.introspect.tools import IntrospectTools
from orion.schemas.introspect import IntrospectToolBindingV1

REPO = Path(__file__).resolve().parents[3]
GOVERNOR_README = REPO / "services/orion-harness-governor/README.md"
CHANNELS = REPO / "orion/bus/channels.yaml"
SECTION = "## orion-introspect: Orion reading back their own records"
INTROSPECT_REQUEST = re.compile(r"^orion:introspect:[^:]+:request$")


def _section() -> str:
    text = GOVERNOR_README.read_text(encoding="utf-8")
    assert SECTION in text, f"missing '{SECTION}' in {GOVERNOR_README}"
    body = text.split(SECTION, 1)[1]
    return body.split("\n## ", 1)[0]


def _rows() -> dict[str, dict[str, str]]:
    rows = {}
    for line in _section().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or not cells[0].startswith("`"):
            continue
        tool = cells[0].strip("`")
        channel = re.search(r"`(orion:[^`]+)`", cells[3])
        service = re.match(r"(orion-[a-z-]+)", cells[2])
        rows[tool] = {
            "service": service.group(1) if service else "",
            "channel": channel.group(1) if channel else "",
            "status": cells[4],
        }
    return rows


def _live() -> dict[str, dict[str, str]]:
    return {t: r for t, r in _rows().items() if r["status"].startswith("Live")}


def _consumers() -> dict[str, list[str]]:
    data = yaml.safe_load(CHANNELS.read_text(encoding="utf-8"))
    return {c["name"]: list(c.get("consumer_services") or []) for c in data["channels"]}


def _listed_tools() -> set[str]:
    names: set[str] = set()
    for context in ("unified_chat", "curiosity"):
        binding = IntrospectToolBindingV1(
            invocation_context=context, parent_run_id="r", parent_trace_id="t", memory_allowed=True,
        )
        names.update(spec.name for spec in IntrospectTools(None, binding).tool_specs())
    return names


def _responder_section(text: str, tool: str) -> str | None:
    heading = f"### Introspect responder: `{tool}`"
    if heading not in text:
        return None
    lines, in_fence = [], False
    for line in text.split(heading, 1)[1].splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and re.match(r"#{1,3} ", line):
            break
        lines.append(line)
    return "\n".join(lines)


def test_listed_tools_and_live_rows_are_the_same_set():
    listed, live = _listed_tools(), set(_live())
    assert listed - live == set(), f"listed by the server but not a Live overview row: {sorted(listed - live)}"
    assert live - listed == set(), f"marked Live but not listed by the server: {sorted(live - listed)}"


def test_live_rows_match_the_bus_catalog_and_responder_readme():
    consumers = _consumers()
    live = _live()
    assert live, "overview table has no Live rows"
    for tool, row in live.items():
        channel, service = row["channel"], row["service"]
        assert channel in consumers, f"`{tool}` channel {channel} is not in orion/bus/channels.yaml"
        assert service in consumers[channel], f"{channel} consumer_services does not include {service}"
        readme = REPO / "services" / service / "README.md"
        section = _responder_section(readme.read_text(encoding="utf-8"), tool)
        assert section is not None, f"{readme} lacks '### Introspect responder: `{tool}`'"
        assert channel in section, f"{readme} responder section for `{tool}` does not name {channel}"


def test_every_introspect_request_channel_is_a_live_row():
    live_channels = {r["channel"] for r in _live().values()}
    for name in _consumers():
        if INTROSPECT_REQUEST.match(name):
            assert name in live_channels, f"{name} is in channels.yaml but not a Live row in the overview"

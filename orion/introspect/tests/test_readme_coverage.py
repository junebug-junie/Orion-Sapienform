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
INTROSPECT_REQUEST = re.compile(r"^orion:introspect:[a-z_]+:request$")


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


def _consumers() -> dict[str, list[str]]:
    data = yaml.safe_load(CHANNELS.read_text(encoding="utf-8"))
    return {c["name"]: list(c.get("consumer_services") or []) for c in data["channels"]}


def _listed_tools() -> list[str]:
    binding = IntrospectToolBindingV1(
        invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t", memory_allowed=True,
    )
    return [spec.name for spec in IntrospectTools(None, binding).tool_specs()]


def test_every_listed_tool_is_a_live_overview_row():
    rows = _rows()
    for tool in _listed_tools():
        assert tool in rows, f"introspect tool `{tool}` has no row in the governor README overview"
        assert rows[tool]["status"].startswith("Live"), f"`{tool}` is listed by the server but not marked Live"


def test_live_rows_match_the_bus_catalog_and_responder_readme():
    consumers = _consumers()
    live = {t: r for t, r in _rows().items() if r["status"].startswith("Live")}
    assert live, "overview table has no Live rows"
    for tool, row in live.items():
        channel, service = row["channel"], row["service"]
        assert channel in consumers, f"`{tool}` channel {channel} is not in orion/bus/channels.yaml"
        assert service in consumers[channel], f"{channel} consumer_services does not include {service}"
        readme = REPO / "services" / service / "README.md"
        text = readme.read_text(encoding="utf-8")
        assert f"### Introspect responder: `{tool}`" in text, f"{readme} lacks 'Introspect responder: `{tool}`'"
        assert channel in text, f"{readme} does not name {channel}"


def test_every_introspect_request_channel_is_a_live_row():
    live_channels = {r["channel"] for r in _rows().values() if r["status"].startswith("Live")}
    for name in _consumers():
        if INTROSPECT_REQUEST.match(name):
            assert name in live_channels, f"{name} is in channels.yaml but not a Live row in the overview"

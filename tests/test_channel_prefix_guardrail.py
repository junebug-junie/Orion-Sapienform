from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"

ALLOWED_PREFIXES = (
    "orion:",
    # Legacy prefixes retained for existing channels; migrate to orion: and remove.
    "orion-exec:",
    "orion-conversation:",
    "orion.spark.",
    "vision.",
)


def _is_allowed_channel(name: str) -> bool:
    return name.startswith(ALLOWED_PREFIXES)


def test_channel_catalog_prefixes() -> None:
    names: list[str] = []
    for line in CHANNELS_YAML.read_text().splitlines():
        match = re.match(r'^\s*-\s*name:\s*"?([^"#]+)"?\s*$', line)
        if match:
            names.append(match.group(1).strip())

    bad = [name for name in names if not _is_allowed_channel(name)]
    assert not bad, f"Channel catalog entries must start with allowed prefixes: {bad}"


def test_literal_publish_subscribe_prefixes() -> None:
    pattern = re.compile(r'\.(publish|subscribe|raw_subscribe)\(\s*(["\'])(.+?)\2')
    violations: list[str] = []

    for path in ROOT.rglob("*.py"):
        text = path.read_text()
        for match in pattern.finditer(text):
            literal = match.group(3)
            if "{" in literal:
                continue
            if not _is_allowed_channel(literal):
                violations.append(f"{path}:{match.start()}:{literal}")

    assert not violations, "Non-orion channel literal(s) found:\n" + "\n".join(violations)

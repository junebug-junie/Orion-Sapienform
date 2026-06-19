from __future__ import annotations

from pathlib import Path

from app.roster import NEVER_REMEDIATE_IDS, ProbeMode, load_roster


def test_load_roster_substitutes_project(tmp_path: Path) -> None:
    roster_file = tmp_path / "roster.yaml"
    roster_file.write_text(
        """
services:
  - id: landing-pad
    heartbeat_name: landing-pad
    compose_dir: orion-landing-pad
    compose_service: orion-landing-pad
    probe:
      mode: redis_and_http
      intake_channels: [orion:pad:rpc:request]
      ready_url: "http://${PROJECT}-landing-pad:8370/ready"
""",
        encoding="utf-8",
    )
    doc = load_roster(str(roster_file), project="orion-athena", node_name="athena")
    assert doc.services[0].probe.ready_url == "http://orion-athena-landing-pad:8370/ready"
    assert doc.services[0].auto_remediate is True


def test_never_remediate_ids_include_notify() -> None:
    assert "notify" in NEVER_REMEDIATE_IDS

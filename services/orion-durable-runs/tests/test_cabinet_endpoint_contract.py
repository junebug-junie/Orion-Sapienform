"""A host-networked Hub cannot be reached through its Compose DNS name."""
from ipaddress import ip_address, ip_network
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from app.settings import Settings


def test_cabinet_defaults_reach_host_networked_hub():
    root = Path(__file__).resolve().parents[3]
    service = root / "services/orion-durable-runs"
    hub = yaml.safe_load((root / "services/orion-hub/docker-compose.yml").read_text())
    assert hub["services"]["hub-app"]["network_mode"] == "host"
    example = dict(line.split("=", 1) for line in (service / ".env_example").read_text().splitlines()
                   if line and not line.startswith("#") and "=" in line)
    compose = yaml.safe_load((service / "docker-compose.yml").read_text())
    declaration = next(value for value in compose["services"]["durable-runs"]["environment"]
                       if value.startswith("DURABLE_RUNS_ELASTIC_CABINET_URL="))
    compose_default = declaration.split(":-", 1)[1][:-1]
    model_default = Settings.model_fields["elastic_cabinet_url"].default
    assert example["DURABLE_RUNS_ELASTIC_CABINET_URL"] == compose_default == model_default
    endpoint = urlsplit(model_default)
    assert ip_address(endpoint.hostname) in ip_network("100.64.0.0/10")
    assert endpoint.path == "/api/cabinet/sensors/latest"

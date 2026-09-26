"""A host-networked Hub cannot be reached through its Compose DNS name.

Moved from orion-durable-runs in stage 4.5: the cabinet thermal check is now the pool's ``thermal``
swap guard (GPU_POOL_CABINET_URL); durable-runs' elastic copy of it was deleted with the elastic
decider."""
from ipaddress import ip_address, ip_network
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from app.settings import Settings


def test_cabinet_defaults_reach_host_networked_hub():
    root = Path(__file__).resolve().parents[3]
    service = root / "services/orion-gpu-pool"
    hub = yaml.safe_load((root / "services/orion-hub/docker-compose.yml").read_text())
    assert hub["services"]["hub-app"]["network_mode"] == "host"
    example = dict(line.split("=", 1) for line in (service / ".env_example").read_text().splitlines()
                   if line and not line.startswith("#") and "=" in line)
    compose = yaml.safe_load((service / "docker-compose.yml").read_text())
    [svc] = compose["services"].values()
    declaration = next(value for value in svc["environment"] if value.startswith("GPU_POOL_CABINET_URL="))
    compose_default = declaration.split(":-", 1)[1][:-1]
    model_default = Settings.model_fields["cabinet_url"].default
    assert example["GPU_POOL_CABINET_URL"] == compose_default == model_default
    endpoint = urlsplit(model_default)
    assert ip_address(endpoint.hostname) in ip_network("100.64.0.0/10")
    assert endpoint.path == "/api/cabinet/sensors/latest"

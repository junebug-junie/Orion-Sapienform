from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.telemetry.phi_encoder import PhiEncoderManifestV1


def test_registry_resolves_phi_encoder_schemas() -> None:
    assert resolve("PhiEncoderManifestV1") is PhiEncoderManifestV1
    assert SCHEMA_REGISTRY["PhiEncoderManifestV1"].model is PhiEncoderManifestV1

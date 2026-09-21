"""Regression test for the 2026-09-21 self-sense-eval crash: orion-hub's
Dockerfile never copied config/field into the image, so
orion.evals.self_sense's module-level FIELD_NODE_IDS load
(config/field/orion_field_topology.v1.yaml) raised FileNotFoundError the
first time anything in this container imported that module -- which nothing
did until the self-sense-eval scheduler line started calling it.

This can't be caught by a normal unit test (the file exists on the host
checkout those tests run against, so the import succeeds there regardless of
what the built image contains) -- the real check was building the image and
importing the module inside it. This test is a cheap, deterministic backstop
against someone removing the COPY line without re-running that build check."""
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"


def test_dockerfile_copies_field_topology_config():
    text = DOCKERFILE.read_text()
    assert "COPY config/field /app/config/field" in text, (
        "orion-hub's Dockerfile must COPY config/field into the image -- "
        "orion.evals.self_sense reads config/field/orion_field_topology.v1.yaml "
        "at import time (module scope), and this image has no full-repo bind "
        "mount to fall back on the way orion-cortex-exec does."
    )

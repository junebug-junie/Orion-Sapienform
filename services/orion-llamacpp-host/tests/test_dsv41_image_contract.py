"""The DeepSeek soak image pin must stay in-repo and off the shared chat tag."""
from pathlib import Path

HOST = Path(__file__).resolve().parents[1]
REPO = HOST.parents[1]


def test_dsv41_dockerfile_pins_volta_fork():
    text = (HOST / "Dockerfile.dsv41-porte").read_text(encoding="utf-8")
    assert "3b6fcfe" in text
    assert "CMAKE_CUDA_ARCHITECTURES=70" in text
    assert "nvidia/cuda:12.8.1-devel-ubuntu24.04" in text
    assert "dsv41-porte-cmath.patch" in text
    assert "orion-llamacpp-host:0.1.0" in text


def test_dsv41_cmath_patch_adds_isfinite_header():
    patch = (HOST / "patches" / "dsv41-porte-cmath.patch").read_text(encoding="utf-8")
    assert "+#include <cmath>" in patch
    assert "llama-moe-stream.cpp" in patch


def test_dsv41_compose_stays_off_shared_image_tag():
    compose = (HOST / "docker-compose.dsv41.yml").read_text(encoding="utf-8")
    assert "Dockerfile.dsv41-porte" in compose
    assert "deepseek-v41-flash-mxfp4-engram-4xv100-32gb-circe-test" in compose
    assert "8099" in compose
    assert "server-cuda-b8740" not in compose
    example = (HOST / ".env_example").read_text(encoding="utf-8")
    assert "DSV41_LLAMACPP_IMAGE=" in example
    assert "llamacpp-dsv41-porte:server-local-volta" in example


def test_shared_dockerfile_default_is_unchanged():
    text = (HOST / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG LLAMACPP_IMAGE_TAG=server-cuda-b8740" in text

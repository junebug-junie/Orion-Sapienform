from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]


def test_falkordb_data_bind_targets_live_redis_data_directory():
    compose = (SERVICE_ROOT / "docker-compose.yml").read_text()

    assert (
        "${FALKORDB_DATA_DIR:-/mnt/graphdb/falkordb}:/var/lib/falkordb/data"
        in compose
    )

import pytest

from app import worker as spark_worker


@pytest.fixture(autouse=True)
def _reset_spark_introspector_worker_singletons():
    spark_worker._INTRO_SEM = None
    spark_worker._INTRO_DROP_SERIALIZER = None
    spark_worker._REDIS_CLIENT = None
    spark_worker._LAST_HEAVY_INTRO_MONO = 0.0
    spark_worker.settings.spark_introspection_idempotency_enable = False
    yield
    spark_worker._INTRO_SEM = None
    spark_worker._INTRO_DROP_SERIALIZER = None
    spark_worker._REDIS_CLIENT = None
    spark_worker._LAST_HEAVY_INTRO_MONO = 0.0

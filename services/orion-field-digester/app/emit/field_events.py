"""v1: projections persisted to Postgres only; bus emit deferred."""


def publish_field_projection(*_args, **_kwargs) -> None:
    return None
